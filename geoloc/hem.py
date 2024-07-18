import queue
import math
import geoloc
import numpy as np
import jax
import jax.numpy as jnp
import einx
import tqdm
import os
import tinypl as pl
from functools import partial
import timerun
import shutil
import imageio
import tiledwebmaps as twm

@partial(jax.jit, static_argnums=(4, 6), donate_argnums=(5,))
def sample_cluster(pv_features, aerial_features, pv_latlons, aerial_latlons, cluster_size, global_mask, min_distance):
    local_mask = global_mask

    log_c = jnp.log(cluster_size)
    def get_logits(pv_idx):
        logits = einx.dot("c, ba c -> ba", pv_features[pv_idx], aerial_features)
        return logits

    def update_masks(new_pv_idx, global_mask, local_mask):
        global_mask = global_mask.at[new_pv_idx].set(False)
        distances_meter = twm.geo.distance(pv_latlons[new_pv_idx][jnp.newaxis], aerial_latlons, np=jnp)
        local_mask = jnp.logical_and(
            distances_meter > min_distance,
            local_mask,
        )
        return global_mask, local_mask

    gt_logits = einx.dot("b c, b c -> b", pv_features, aerial_features)
    cluster_seed_mode = "random"
    if cluster_seed_mode == "random":
        def new_seed(global_mask):
            return jnp.argmax(global_mask)
    elif cluster_seed_mode == "easiest":
        def new_seed(global_mask):
            return jnp.argmax(jnp.where(global_mask, gt_logits, -jnp.inf))
    elif cluster_seed_mode == "hardest":
        def new_seed(global_mask):
            return jnp.argmax(jnp.where(global_mask, -gt_logits, -jnp.inf))
    else:
        assert False

    first_pv_idx = new_seed(global_mask)
    logits = get_logits(first_pv_idx)
    global_mask, local_mask = update_masks(first_pv_idx, global_mask, local_mask)
    carry = (global_mask, local_mask, logits)

    def f(carry, x):
        assert x is None
        global_mask, local_mask, logits = carry

        sample_logits = logits
        sample_logits = jnp.where(local_mask, sample_logits, -100000.0)
        sample_logits = jnp.where(global_mask, sample_logits, -jnp.inf)
        new_pv_idx = jnp.argmax(sample_logits)

        logits = logits + get_logits(new_pv_idx)

        global_mask, local_mask = update_masks(new_pv_idx, global_mask, local_mask)

        carry = (global_mask, local_mask, logits)
        return carry, new_pv_idx

    (global_mask, _, _), cluster = jax.lax.scan(f, carry, None, cluster_size - 1)
    cluster = jnp.concatenate([first_pv_idx[jnp.newaxis], cluster])

    return cluster, global_mask

class Sampler:
    def __init__(self, pipe, dataset, config, test_step, workers, maxsize):
        self.train_batchsize = config["train.batchsize"]

        if "hem" in config["train"]:
            self.batchsize = config["train.hem.batchsize"]

            pipe = pl.partition(pipe, self.batchsize)
            pipe = pl.thread.mutex(pipe)
            self.in_pipe = pipe

            self.ringbuffer = pl.process.SharedMemoryRingBuffer(maxsize, allow_pickle=False, verbose=True)
            def load_to_shm(image_idxs):
                batch = [dataset[image_idx] for image_idx in image_idxs]
                batch, metrics = dataset.collate(batch)
                batch = self.ringbuffer.write(batch)
                return batch, metrics
            pipe = self.scan_pl1 = pl.process.map(pipe, load_to_shm, workers=workers)
            @pl.unpack
            def from_shm(batch, metrics):
                batch = self.ringbuffer.read(batch)
                return batch, metrics
            pipe = self.scan_pl2 = pl.thread.map(pipe, from_shm, workers=1, maxsize=4)
            self.act_pipe = pipe

            self.output = queue.Queue()

            self.cluster_size = config["train.hem.clustersize"]
            assert self.train_batchsize % self.cluster_size == 0, f"train_batchsize ({self.train_batchsize}) must be a multiple of cluster_size ({self.cluster_size})"
            self._batches = config["train.hem.first-scan-batches"]
            self.samples_since_last_raise = 0
            self.has_started_scanning = False
            self.config = config
            self.test_step = test_step
            self.min_distance = config["data.train.min-offset-factor"] * config["data.cell-size-meters"]
            self.enabled = True
        else:
            pipe = pl.partition(pipe, self.train_batchsize)
            pipe = pl.thread.mutex(pipe)
            self.in_pipe = pipe
            self.enabled = False

    def queue_metrics(self):
        return {
            "q0": self.scan_pl1.input_fill,
            "q1": self.scan_pl1.fill,
            "q2": self.scan_pl2.fill,
        }

    @property
    def batches(self):
        return self._batches if self.enabled else 0

    @batches.setter
    def batches(self, value):
        assert self.enabled
        self._batches = value
        num_scanned = self._batches * self.batchsize
        assert num_scanned // self.cluster_size * self.cluster_size == num_scanned

    def scan_if_needed(self, train_batch_index):
        if not self.enabled:
            return

        self.samples_since_last_raise += self.train_batchsize

        scan = self.output.empty() \
            and (self.samples_since_last_raise >= self.config.get("train.hem.raise-after-samples") or self.batches > 1) \
            and train_batch_index >= self.config.get("train.hem.pre-scan-batches", 0)

        if scan:
            print("Scan: Starting")
            self.has_started_scanning = True

            # Raise number of batches per scan
            if self.samples_since_last_raise >= self.config.get("train.hem.raise-after-samples"):
                self.batches = min(math.ceil(self.config.get("train.hem.raise-factor") * self.batches), self.config.get("train.hem.max-scan-batches"))
                self.samples_since_last_raise = 0

            # Predict features for the next `self.batches` batches
            n = self.batchsize * self.batches
            pv_features = np.zeros((n, self.config.get("model.embedding-channels")), dtype=np.float32)
            aerial_features = np.zeros((n, self.config.get("model.embedding-channels")), dtype=np.float32)
            pv_latlons = np.zeros((n, 2), dtype=np.float64)
            aerial_latlons = np.zeros((n, 2), dtype=np.float64)
            aerial_bearings = np.zeros((n,), dtype=np.float32)
            image_indices = np.zeros((n,), dtype=np.int32)
            for i in range(self.batches):
                metrics = {}

                with timerun.Timer() as timer:
                    batch, data_metrics = next(self.act_pipe)
                metrics["t-fetch"] = timer.duration.timedelta.total_seconds()

                with timerun.Timer() as timer:
                    model_output = self.test_step(batch)
                    model_output = jax.device_get(model_output)
                metrics["t-gpu"] = timer.duration.timedelta.total_seconds()

                with timerun.Timer() as timer:
                    pv_features[i * self.batchsize:(i + 1) * self.batchsize] = model_output.pv_features
                    aerial_features[i * self.batchsize:(i + 1) * self.batchsize] = model_output.aerial_features
                    pv_latlons[i * self.batchsize:(i + 1) * self.batchsize] = batch.pv.latlons
                    aerial_latlons[i * self.batchsize:(i + 1) * self.batchsize] = batch.aerial.latlons
                    image_indices[i * self.batchsize:(i + 1) * self.batchsize] = batch.pv.idxs
                    aerial_bearings[i * self.batchsize:(i + 1) * self.batchsize] = batch.aerial.bearings
                metrics["t-update"] = timer.duration.timedelta.total_seconds()

                metrics |= data_metrics
                metrics |= self.queue_metrics()

                geoloc.print_state(split="HEM", batch=i, metrics=metrics, total_batches=self.batches)

            with jax.experimental.enable_x64():
                # Clustering
                cluster_num = pv_features.shape[0] // self.cluster_size
                assert cluster_num * self.cluster_size == pv_features.shape[0]

                pv_features_ = jax.device_put(pv_features)
                aerial_features_ = jax.device_put(aerial_features)
                pv_latlons_ = jax.device_put(pv_latlons)
                aerial_latlons_ = jax.device_put(aerial_latlons)

                clusters = []
                mask = jnp.ones((n,), dtype="bool")
                for _ in tqdm.tqdm(list(range(cluster_num)), desc="Clustering"):
                    cluster, mask = sample_cluster(pv_features_, aerial_features_, pv_latlons_, aerial_latlons_, self.cluster_size, mask, self.min_distance)
                    cluster = np.asarray(cluster)
                    clusters.append(cluster)
                clusters = np.asarray(clusters)

                clusters = np.asarray(clusters)
                assert np.all(clusters >= 0)
                print(f"Scan: Sampled {cluster_num} clusters")

                # Add batches to output queue
                batches = einx.rearrange("(c k) i -> c (k i)", clusters, k=self.train_batchsize // self.cluster_size)
                for batch in batches:
                    self.output.put((
                        [image_indices[i] for i in batch],
                        [aerial_latlons[i] for i in batch],
                        [aerial_bearings[i] for i in batch],
                    ))

            print(f"Scan: Done, now has {self.output.qsize()} batches in queue")

    def __next__(self):
        if not self.enabled or not self.has_started_scanning:
            indices = next(self.in_pipe)
            aerial_latlons = [None for _ in indices]
            aerial_bearings = [None for _ in indices]
        else:
            indices, aerial_latlons, aerial_bearings = self.output.get()
        return zip(indices, aerial_latlons, aerial_bearings)