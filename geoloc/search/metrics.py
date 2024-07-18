import jax
import jax.numpy as jnp
import einx
import numpy as np
import tqdm
from functools import partial
import math

class Matches:
    @staticmethod
    def from_config(config, pv_dataset, cellregion):
        recall_radii = config.get("data.test.recall-radii", [25, 50, 100])
        assert all(isinstance(r, int) for r in recall_radii)
        recall_radii = set(r for r in recall_radii if r > config.get("cell-size-meters") / math.sqrt(2))
        if not 0 in recall_radii:
            recall_radii.add(0)
        recall_radii = sorted(recall_radii)

        pvidx_to_aerialidxs = []
        pvidx_to_aerialidxs.append(cellregion.find_matches(pv_dataset.latlons)[..., np.newaxis])
        for r in recall_radii[1:]:
            pvidx_to_aerialidxs.append(cellregion.find_matches_radius(pv_dataset.latlons, radius=r))

        return Matches(recall_radii, pvidx_to_aerialidxs)

    def __init__(self, recall_radii, pvidx_to_aerialidxs):
        self.recall_radii = recall_radii
        self.pvidx_to_aerialidxs = pvidx_to_aerialidxs

@jax.jit
def compute_logits_gpu(pv_features, aerial_features):
    return einx.dot("bg c, ba c -> bg ba", pv_features, aerial_features)

def compute_logits_cpu(pv_features, aerial_features, aerial_chunk_size, verbose=False):
    logits = np.zeros((pv_features.shape[0], aerial_features.shape[0]), dtype="float32")

    pv_features = jnp.asarray(pv_features)
    cas = list(range(0, aerial_features.shape[0], aerial_chunk_size))
    if verbose is not None:
        cas = tqdm.tqdm(cas, "Computing logits")
    for ca in cas:
        logits_ca = compute_logits_gpu(
            pv_features,
            aerial_features[ca:ca + aerial_chunk_size],
        )
        logits[:, ca:ca + aerial_chunk_size] = logits_ca

    return logits

@jax.jit
def _compute_fp_num(logits, pvidx_to_aerialidxs):
    gt_logits = einx.get_at("bg [ba], bg nns -> bg nns", logits, pvidx_to_aerialidxs)
    gt_logits = jnp.where(pvidx_to_aerialidxs >= 0, gt_logits, -jnp.inf)
    gt_logits = einx.max("bg [nns]", gt_logits)

    logits_without_gt = einx.set_at("bg [ba], bg nns, -> bg [ba]", logits, pvidx_to_aerialidxs, -jnp.inf)

    above_gt = einx.greater_equal("bg ba, bg -> bg ba", logits_without_gt, gt_logits)
    fp_num = einx.count_nonzero("bg [ba]", above_gt)
    return fp_num

def compute_fp_nums_gpu(pv_features, aerial_features, pvidx_to_aerialidxs):
    logits = compute_logits_gpu(pv_features, aerial_features)
    return [_compute_fp_num(logits, pvidx_to_aerialidxs[i]) for i in range(len(pvidx_to_aerialidxs))]

def _to_cpu(x):
    return jax.device_put(x, device=jax.devices("cpu")[0])

def compute_fp_nums_cpu(pv_features, aerial_features, pvidx_to_aerialidxs, pv_chunk_size, aerial_chunk_size):
    fp_num = [np.zeros(pv_features.shape[0], dtype="int64") for _ in range(len(pvidx_to_aerialidxs))]
    pvidx_to_aerialidxs = [_to_cpu(x) for x in pvidx_to_aerialidxs]

    cps = list(range(0, pv_features.shape[0], pv_chunk_size))
    if len(cps) > 1:
        cps = tqdm.tqdm(cps, "Computing num-above-gt")
    for cp in cps:
        logits = compute_logits_cpu(cp + np.arange(min(pv_chunk_size, pv_features.shape[0] - cp)), verbose=len(cps) == 1)
        logits = _to_cpu(logits)

        for i in range(len(pvidx_to_aerialidxs)):
            fp_num_cp = _compute_fp_num(logits, pvidx_to_aerialidxs[i][cp:cp + pv_chunk_size])
            fp_num[i][cp:cp + pv_chunk_size] = fp_num_cp

    return fp_num

def _compute_metrics(fp_num, aerial_num, xnp, prefix="", postfix=""):
    metrics = {}

    for n in [1, 2, 3, 4, 5, 10, 20, 50, 100, 1000]:
        metrics[f"{prefix}r@{n}{postfix}"] = xnp.mean(xnp.where(fp_num < n, 1.0, 0.0))
    metrics[f"{prefix}r@1%{postfix}"] = xnp.mean(xnp.where(fp_num < int(0.01 * aerial_num), 1.0, 0.0))
    metrics[f"{prefix}r@0.1%{postfix}"] = xnp.mean(xnp.where(fp_num < int(0.001 * aerial_num), 1.0, 0.0))
    metrics[f"{prefix}r@0.01%{postfix}"] = xnp.mean(xnp.where(fp_num < int(0.0001 * aerial_num), 1.0, 0.0))
    metrics[f"{prefix}mean-k{postfix}"] = xnp.mean(fp_num)
    metrics[f"{prefix}median-k{postfix}"] = xnp.median(fp_num)
    metrics[f"{prefix}median-k%{postfix}"] = xnp.median(fp_num).astype("float32") / aerial_num

    return metrics

compute_metrics_gpu = partial(_compute_metrics, xnp=jnp)
compute_metrics_cpu = partial(_compute_metrics, xnp=np)
