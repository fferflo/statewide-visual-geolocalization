import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--tiles", type=str, required=True, nargs="+")
parser.add_argument("--geojson", type=str, default=None)
args, unknown_args = parser.parse_known_args()

import upsilonconf, os, sys, timerun, shutil, imageio, tqdm, types, yaml, math
import numpy as np
from functools import partial
import geoloc
import tinypl as pl

with open(os.path.join(args.train, "config.yaml")) as f:
    config = yaml.safe_load(f)
if not "test" in config["data"]:
    config["data"]["test"] = {}
config["data"]["test"]["tiles"] = [{"path": tiles_path} for tiles_path in args.tiles]
if args.geojson is not None:
    config["data"]["test"]["geojson"] = args.geojson
elif "geojson" in config["data"]["test"]:
    del config["data"]["test"]["geojson"]
del config["data"]["train"]
del config["train"]

if os.path.exists(args.output):
    print(f"ERROR: Output directory {args.output} already exists")
    sys.exit(-1)
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "config.yaml"), "w") as f:
    yaml.safe_dump(config, f)
config = upsilonconf.load(os.path.join(args.output, "config.yaml"))



print("\n###### Loading dataset ######")

test_aerial_dataset = geoloc.data.test.AerialDataset.from_config(config["data"])
test_aerial_dataset.cellregion.save(os.path.join(args.output, "cellregion.npz"))

print("\n###### Creating model ######")

import jax
import jax.tree_util
import einx
import jax.numpy as jnp
import time

print(f"Jax devices: {jax.devices()}")

rng = jax.random.PRNGKey(42)
def next_rng():
    global rng
    rng, x = jax.random.split(rng)
    return x

# Create model and parameters
model = geoloc.model.Model.from_config(config["model"])

batch_aerial, _ = test_aerial_dataset.collate([test_aerial_dataset[0]])
batch = types.SimpleNamespace(
    aerial=batch_aerial.aerial,
)
params = jax.jit(model.init)({"dropout": next_rng(), "params": next_rng()}, batch)
del batch, batch_aerial

# Load pretrained params
import safetensors.numpy
import weightbridge

pretrained_params = safetensors.numpy.load_file(os.path.join(args.train, "weights/last.safetensors"))
pretrained_params = geoloc.unflatten(pretrained_params)
del pretrained_params["params"]["pv_encoder"]
del pretrained_params["params"]["pv_decoder"]

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

shutil.copy(os.path.join(args.train, "weights/last.safetensors"), os.path.join(args.output, "model_weights.safetensors"))

print("\n###### Predicting aerial features ######")

val_batchsize = config["test.batchsize"]
if val_batchsize % len(jax.devices()) != 0:
    print(f"ERROR: Validation batch size ({val_batchsize}) is not a multiple of the number of devices ({len(jax.devices())})")
    sys.exit(-1)

dl = geoloc.search.scan.DataLoader(val_batchsize, workers=32, maxsize=4 * 1024 * 1024 * 1024)

aerial_features = np.memmap(os.path.join(args.output, "aerial_features.bin"), dtype="float32", mode="w+", shape=(len(test_aerial_dataset), config["model.embedding-channels"]))
geoloc.search.scan("aerial", test_aerial_dataset, partial(test_step_with_pad, params), dl, aerial_features)

print("\n###### Creating FAISS index ######")
import faiss
index = faiss.index_factory(aerial_features.shape[-1], "HNSW64", faiss.METRIC_INNER_PRODUCT)

index.hnsw.efConstruction = 40
index.verbose = True

index.add(aerial_features)

file = os.path.join(args.output, "faiss.index")
print(f"Saving index to {file}")
faiss.write_index(index, file)
print("Done")