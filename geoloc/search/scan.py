import tinypl as pl
import jax
import timerun
import geoloc
import numpy as np

class DataLoader:
    def __init__(self, batchsize, workers, maxsize):
        self.batchsize = batchsize
        self.workers = workers
        self.ringbuffer = pl.process.SharedMemoryRingBuffer(maxsize, allow_pickle=False)

    def iter(self, dataset):
        assert self.ringbuffer.num_stored_items == 0
        pipe = iter(range(len(dataset)))
        pipe = pl.partition(pipe, self.batchsize)
        pipe = pl.thread.mutex(pipe)

        def load_to_shm(image_idxs):
            batch = [dataset[image_idx] for image_idx in image_idxs]
            batch, metrics = dataset.collate(batch)
            batch = self.ringbuffer.write(batch)
            return batch, metrics
        pipe = pl1 = pl.process.map(pipe, load_to_shm, workers=self.workers)
        @pl.unpack
        def from_shm(batch, metrics):
            batch = self.ringbuffer.read(batch)
            return batch, metrics
        pipe = pl2 = pl.thread.map(pipe, from_shm, workers=1, maxsize=4)

        def queue_metrics():
            return {
                "q0": pl1.input_fill,
                "q1": pl1.fill,
                "q2": pl2.fill,
            }

        return pipe, queue_metrics

def scan(name, dataset, test_step, dataloader, features=None):
    batch_index = 0
    total_batches = (len(dataset) + dataloader.batchsize - 1) // dataloader.batchsize
    dataloader, queue_metrics = dataloader.iter(dataset)
    while True:
        metrics = {}

        with timerun.Timer() as timer:
            try:
                batch, data_metrics = next(dataloader)
            except StopIteration:
                break
        metrics["t-fetch"] = timer.duration.timedelta.total_seconds()

        with timerun.Timer() as timer:
            model_output = test_step(batch)
            model_output = jax.device_get(model_output)
        metrics["t-gpu"] = timer.duration.timedelta.total_seconds()

        with timerun.Timer() as timer:
            if name == "pv":
                if features is None:
                    features = np.zeros((len(dataset), model_output.pv_features.shape[1]), dtype=np.float32)
                features[batch.pv.idxs] = model_output.pv_features
            elif name == "aerial":
                if features is None:
                    features = np.zeros((len(dataset), model_output.aerial_features.shape[1]), dtype=np.float32)
                features[batch.aerial.idxs] = model_output.aerial_features
            else:
                assert False
        metrics["t-update"] = timer.duration.timedelta.total_seconds()

        metrics |= data_metrics
        metrics |= queue_metrics()

        geoloc.print_state(split=f"scan-{name}", batch=batch_index, metrics=metrics, total_batches=total_batches)
        batch_index += 1

    return features

scan.DataLoader = DataLoader