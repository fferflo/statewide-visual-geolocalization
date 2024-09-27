import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, required=True)
parser.add_argument("--reference", type=str, required=True)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--output", type=str, default=None)
args, unknown_args = parser.parse_known_args()

import upsilonconf
import numpy as np
import os
import time
import tqdm
import yaml
import types

config = upsilonconf.load(os.path.join(args.reference, "config.yaml"))

if args.output is not None:
    import tinylogdir
    log = tinylogdir.LogDir(args.output)
else:
    log = None

print("##### Loading cell region... #####")
import geoloc
t0 = time.time()
cellregion = geoloc.region.CellRegion.from_npz(os.path.join(args.reference, "cellregion.npz"))
print(f"Took {time.time() - t0} sec. Region has {len(cellregion)} cells")



print("##### Loading pv dataset... #####")
if os.path.isfile(os.path.join(args.query, "dataset.json")):
    pv_dataset = geoloc.data.FolderDataset(args.query, None)
    pv_dataset = geoloc.data.StridedDataset(
        pv_dataset,
        stride=args.stride,
    )
elif args.query.endswith(".mp4"):
    assert log is not None
    geoloc.data.video_to_dataset(args.query, log.dir("pv_dataset"), config["data.pv-shape"], workers=os.cpu_count(), stride=args.stride)
    pv_dataset = geoloc.data.FolderDataset(log.dir("pv_dataset"), None)
else:
    assert False



pv_dataset = geoloc.data.test.PvDataset(
    dataset=pv_dataset,
    shape=config["data.pv-shape"],
)

print(f"Localizing pv-dataset with {len(pv_dataset)} images")


print("##### Creating model... #####")
t0 = time.time()
import jax
import optax
import jax.tree_util
import einx
import jax.numpy as jnp
from functools import partial

print(f"Jax devices: {jax.devices()}")

rng = jax.random.PRNGKey(42)
def next_rng():
    global rng
    rng, x = jax.random.split(rng)
    return x

# Create model and parameters
model = geoloc.model.Model.from_config(config["model"])

batch_pv, _ = pv_dataset.collate([pv_dataset[0]])
batch = types.SimpleNamespace(
    pv=batch_pv.pv,
)
params = jax.jit(model.init)({"dropout": next_rng(), "params": next_rng()}, batch) # jit, so that memory for model computation is not allocated
del batch, batch_pv

# Load pretrained params
import safetensors.numpy
import weightbridge

pretrained_params = safetensors.numpy.load_file(os.path.join(args.reference, "model_weights.safetensors"))
pretrained_params = geoloc.unflatten(pretrained_params)
del pretrained_params["params"]["aerial_encoder"]
del pretrained_params["params"]["aerial_decoder"]

params = weightbridge.adapt(pretrained_params, params)

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

params = replicate(params)



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

print(f"Took {time.time() - t0} sec.")

print("##### Predicting pv features #####")

val_batchsize = config["test.batchsize"]
if val_batchsize % len(jax.devices()) != 0:
    print(f"ERROR: Validation batch size ({val_batchsize}) is not a multiple of the number of devices ({len(jax.devices())})")
    sys.exit(-1)

dl = geoloc.search.scan.DataLoader(val_batchsize, workers=32, maxsize=4 * 1024 * 1024 * 1024)

pv_features = geoloc.search.scan("pv", pv_dataset, partial(test_step_with_pad, params), dl)





print("##### Loading faiss index... #####")
t0 = time.time()
import faiss
faiss_index = faiss.read_index(os.path.join(args.reference, "faiss.index"))
faiss_index.verbose = True
faiss_index.hnsw.efSearch = 512
print(f"Took {time.time() - t0} sec.")


def query_faiss(pv_features, k=1):
    t0 = time.time()
    chunk_size = 1024
    aerialidxs = np.full((len(pv_features), k), -1, dtype="int64") # n k

    pbar = tqdm.tqdm(list(range(0, len(pv_features), chunk_size)), desc="Searching index")
    for c in pbar:
        _, aerialidxs_c = faiss_index.search(pv_features[c:c + chunk_size], k=k) # c k
        aerialidxs[c:c + chunk_size] = aerialidxs_c

    time_per_query = (time.time() - t0) / len(pv_features)
    print(f"FAISS: {time_per_query * 1000} ms per query")

    return aerialidxs

print("##### Querying faiss index... #####")

pred_pvidx_to_aerialidx = query_faiss(pv_features, k=100)



if pv_dataset.latlons is not None:
    print("##### Computing metrics... #####")
    matches = geoloc.search.metrics.Matches.from_config(config["data"], pv_dataset, cellregion)

    for i in range(len(matches.recall_radii)):
        r = matches.recall_radii[i]
        x = einx.equal("pv pred_av, pv gt_av -> pv pred_av gt_av", pred_pvidx_to_aerialidx, matches.pvidx_to_aerialidxs[i])
        x = einx.any("pv pred_av [gt_av]", x)
        
        for k in [1, 5, 10, 50, 100]:
            xk = einx.any("pv [pred_av]", x[:, :k])
            recall = np.mean(np.where(xk, 1.0, 0.0))

            print(f"Recall@{k}<{r}m: {recall:.4f}")
        print()
