import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
args, unknown_args = parser.parse_known_args()

# Create a directory to which training results are stored
import tinylogdir
log = tinylogdir.LogDir(args.output, mode="timestamp")

import upsilonconf, os, sys, timerun, shutil, imageio, tqdm, types, yaml, math
import numpy as np
from functools import partial
import geoloc
import tinypl as pl

config = upsilonconf.load(args.config)


print("\n###### Loading training dataset ######")

train_dataset = geoloc.data.train.Dataset.from_config(config["data"])

if "test" in config["data"]:
    print("\n###### Loading testing dataset ######")
    test_pv_dataset = geoloc.data.test.PvDataset.from_config(config["data"])
    test_aerial_dataset = geoloc.data.test.AerialDataset.from_config(config["data"])
    test_matches = geoloc.search.metrics.Matches.from_config(config["data"], test_pv_dataset, test_aerial_dataset.cellregion)




# Save some example training inputs
rng = np.random.default_rng()
idxs = rng.choice(len(train_dataset), size=10, replace=False)
for idx in tqdm.tqdm(idxs, desc=f"Saving some example training inputs to {log.dir('train-images')}"):
    sample = train_dataset[idx]

    imageio.imwrite(os.path.join(log.dir("train-images"), f"{idx:012}-000-pv.png"), sample.pv.image)
    for i, aerial_image in enumerate(sample.aerial.images):
        imageio.imwrite(os.path.join(log.dir("train-images"), f"{idx:012}-001-aerial-scale{i + 1}.png"), aerial_image)
    with open(os.path.join(log.dir("train-images"), f"{idx:012}-002-info.txt"), "w") as f:
        f.write(f"idx: {idx}\n")
        f.write(f"pv latlon: {sample.pv.latlon}\n")
        f.write(f"Aerial latlon: {sample.aerial.latlon}\n")



print("\n###### Creating model ######")

import jax
import optax
import jax.tree_util
import einx
import jax.numpy as jnp
import time

print(f"Jax devices: {jax.devices()}")

train_batchsize = config["train.batchsize"]
if train_batchsize % len(jax.devices()) != 0:
    print(f"ERROR: Training batch size ({train_batchsize}) is not a multiple of the number of devices ({len(jax.devices())})")
    sys.exit(-1)





rng = jax.random.PRNGKey(42)
def next_rng():
    global rng
    rng, x = jax.random.split(rng)
    return x

# Create model and parameters
model = geoloc.model.Model.from_config(config["model"])

batch, _ = train_dataset.collate([train_dataset[0]])
params = jax.jit(model.init)({"dropout": next_rng(), "params": next_rng()}, batch) # jit, so that memory for model computation is not allocated
del batch

# Load pretrained params
import weightbridge
params["params"]["aerial_encoder"] = geoloc.model.util._import(config["model.aerial-encoder"] + "_adaptweights")(params["params"]["aerial_encoder"])
params["params"]["pv_encoder"] = geoloc.model.util._import(config["model.pv-encoder"] + "_adaptweights")(params["params"]["pv_encoder"])

import safetensors.numpy
def save_weights(file):
    file = os.path.join(log.dir("weights"), file)
    p = jax.device_get(params)
    p = geoloc.flatten(p)
    safetensors.numpy.save_file(p, file)

optimizer, schedule = geoloc.optimizer.from_config(params, config)
opt_state = optimizer.init(params)

from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

mesh = Mesh(np.asarray(jax.devices()), axis_names=("devices",))
p_rep = P()
p_sh = P("devices")
s_rep = NamedSharding(mesh, p_rep)
s_sh = NamedSharding(mesh, p_sh)

def replicate(x):
    return jax.tree.map(lambda x: jax.device_put(x, s_rep), x)

opt_state = replicate(opt_state)
params = replicate(params)




def loss(params, batch):
    model_output, model_metrics = model.apply(
        params,
        batch,
        rngs={},
    )

    type = config["train.loss.type"] if "type" in config["train.loss"] else "crossentropy"
    if type == "crossentropy":
        loss, metrics = geoloc.loss.crossentropy(
            batch,
            model_output,
            min_distance=config["data.train.min-offset-factor"] * config["data.cell-size-meters"],
            eps=config["train.loss.label-smoothing"],
            decoupled=config["train.loss.decoupled"],
        )
    elif type == "triplet":
        loss, metrics = geoloc.loss.triplet(
            batch,
            model_output,
            min_distance=config["data.train.min-offset-factor"] * config["data.cell-size-meters"],
            safa_fix=config["train.loss.safa-fix"] if "safa-fix" in config["train.loss"] else False,
        )
    else:
        assert False

    for k, v in model_metrics.items():
        metrics[k] = v

    return loss, (metrics, model_output)

@partial(jax.jit, donate_argnums=(0, 1))
@partial(shard_map, mesh=mesh, in_specs=(p_rep, p_rep, p_sh), out_specs=(p_rep, p_rep, p_rep), check_rep=False)
def update_step(params, opt_state, batch):
    step = opt_state[-2].count

    grads, (metrics, model_output) = jax.grad(loss, has_aux=True)(params, batch)

    grads = jax.lax.pmean(grads, axis_name="devices")

    metrics["lr"] = schedule(step) * config["train.learning-rate"]

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    metrics["grad-norm"] = optax.global_norm(grads)

    return metrics, params, opt_state

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=(p_rep, p_sh), out_specs=p_sh, check_rep=False)
def test_step(params, batch):
    model_output, _ = model.apply(
        params,
        batch,
        rngs={},
    )
    return model_output

def test_step_with_pad(params, batch):
    batch, p = geoloc.batch.pad(batch, val_batchsize)
    model_output = test_step(params, batch)
    model_output = geoloc.batch.unpad(model_output, p)
    return model_output











print("\n###### Starting dataloaders ######")

from functools import partial

train_dataset.make_forksafe()

def pipe():
    rng = np.random.default_rng()
    while True:
        yield from rng.permutation(len(train_dataset))
pipe = pipe()
pipe = pl.thread.mutex(pipe)

pipe = active_sampler = geoloc.hem.Sampler(
    pipe,
    train_dataset,
    config,
    test_step=lambda batch: test_step(params, batch),
    workers=32,
    maxsize=4 * 1024 * 1024 * 1024,
)

ringbuffer = pl.process.SharedMemoryRingBuffer(4 * 1024 * 1024 * 1024, allow_pickle=False)
def load_to_shm(inputs):
    batch = [train_dataset.get(*x) for x in inputs]
    batch, metrics = train_dataset.collate(batch)
    batch = ringbuffer.write(batch)
    return batch, metrics
pipe = train_pl1 = pl.process.map(pipe, load_to_shm, workers=16)
@pl.unpack
def from_shm(batch, metrics):
    batch = ringbuffer.read(batch)
    return batch, metrics
pipe = train_pl2 = pl.thread.map(pipe, from_shm, workers=1, maxsize=2)
train_dataloader = pipe

def queue_metrics():
    return {
        "q0": train_pl1.input_fill,
        "q1": train_pl1.fill,
        "q2": train_pl2.fill,
    }























if "test" in config:
    val_batchsize = config["test.batchsize"]
    dl = geoloc.search.scan.DataLoader(val_batchsize, workers=32, maxsize=4 * 1024 * 1024 * 1024)

    val_period_batches = int(config.get("test.period-samples") / train_batchsize)
    if val_batchsize % len(jax.devices()) != 0:
        print(f"ERROR: Validation batch size ({val_batchsize}) is not a multiple of the number of devices ({len(jax.devices())})")
        sys.exit(-1)
    val_epoch = 0
    def validate(train_batch_index):
        global val_epoch
        print(f"Validating epoch {val_epoch + 1}...")

        path = log.dir("validation")
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

        test_step = partial(test_step_with_pad, params)

        pv_features = geoloc.search.scan("pv", test_pv_dataset, test_step, dl)
        aerial_features = geoloc.search.scan("aerial", test_aerial_dataset, test_step, dl)

        pv_features = jax.device_put(pv_features)
        aerial_features = jax.device_put(aerial_features)

        fp_nums = geoloc.search.metrics.compute_fp_nums_gpu(pv_features, aerial_features, test_matches.pvidx_to_aerialidxs)
        del pv_features, aerial_features

        metrics = {}
        for fp_num, r, pvidx_to_aerialidxs in zip(fp_nums, test_matches.recall_radii, test_matches.pvidx_to_aerialidxs):
            metrics |= geoloc.search.metrics.compute_metrics_gpu(fp_num, len(test_aerial_dataset), postfix=f"<{r}m" if r > 0 else "")

        print(metrics)

        import gc, time
        gc.collect()
        time.sleep(10.0)

        print("Validation done")
        val_epoch += 1
else:
    val_period_batches = None





upsilonconf.save(config, os.path.join(log.dir(), "config.yaml"))
print("\n###### Starting training loop ######")


save_last_weights_period = 1000
save_weights_period = 50000
commit_period = 200
for batch_index in range(config.get("train.schedule.total-steps")):
    active_sampler.scan_if_needed(batch_index)

    metrics = {}

    with timerun.Timer() as timer:
        batch, data_metrics = next(train_dataloader)
    metrics["t-fetch"] = timer.duration.timedelta.total_seconds()

    with timerun.Timer() as timer:
        # batch = jax.tree.map(lambda x: jax.device_put(x, s_sh), batch)
        model_metrics, params, opt_state = update_step(
            params,
            opt_state,
            batch,
        )
        model_metrics = jax.device_get(model_metrics)
    metrics["t-gpu"] = timer.duration.timedelta.total_seconds()

    metrics |= model_metrics
    metrics |= data_metrics
    metrics |= queue_metrics()
    metrics["scan-batches"] = active_sampler.batches

    geoloc.print_state(split="train", batch=batch_index, metrics=metrics)

    if val_period_batches is not None and batch_index % val_period_batches == val_period_batches - 1:
        validate(batch_index)

    if batch_index > 0 and batch_index % save_weights_period == 0:
        save_weights(f"weights-{batch_index:09}.safetensors")

    if batch_index > 0 and batch_index % (save_last_weights_period) == 0:
        save_weights("last.safetensors")

    batch_index += 1


validate(batch_index)
save_weights("last.safetensors")
print("###################### Training done ######################")