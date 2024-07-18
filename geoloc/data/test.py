import cv2
import types
import os
import timerun
import jax.tree_util
import geoloc
import tiledwebmaps as twm
import numpy as np
from .dataset import *

class AerialDataset:
    @staticmethod
    def from_config(config):
        tileloaders = []
        for tiles_config in config["test.tiles"]:
            tileloaders.append(twm.from_yaml(tiles_config["path"]))

        if "geojson" in config["test"]:
            region = geoloc.region.Region.from_geojson(config["test.geojson"])
        else:
            region = geoloc.region.Region.from_tiles([c["path"] for c in config["test.tiles"]])
        cell_size_meters = config["cell-size-meters"]
        if config["mercator"]:
            cellregion = geoloc.region.CellRegion.mercator(region, cell_size_meters)
        else:
            cellregion = geoloc.region.CellRegion.consistent(region, cell_size_meters)

        return AerialDataset(
            tileloaders=tileloaders,
            cellregion=cellregion,
            shape=config["aerial-shape"],
            meters_per_pixel=config["meters-per-pixel"],
        )

    def __init__(self, tileloaders, cellregion, shape, meters_per_pixel):
        self.tileloaders = [twm.LRUCached(twm.WithDefault(tileloader), 1000) for tileloader in tileloaders]
        self.cellregion = cellregion
        self.shape = shape
        self.meters_per_pixel = meters_per_pixel

    def make_forksafe(self):
        for tileloader in self.tileloaders:
            tileloader.make_forksafe()

    @property
    def latlons(self):
        return self.cellregion.latlons

    def __len__(self):
        return len(self.cellregion)

    def __getitem__(self, idx):
        metrics = {}

        with timerun.Timer() as timer:
            aerial_images = []
            cell = self.cellregion[idx]
            for mpp in self.meters_per_pixel:
                aerial_images_at_zoom = []
                for tileloader in self.tileloaders:
                    aerial_images_at_zoom.append(tileloader.load(
                        latlon=cell.latlon,
                        bearing=180.0,
                        meters_per_pixel=mpp * cell.scale,
                        shape=self.shape,
                    ))

                if len(aerial_images_at_zoom) > 1:
                    num_white_pixels = [np.count_nonzero(np.all(aerial_image > 240, axis=-1)) for aerial_image in aerial_images_at_zoom]
                    aerial_image = aerial_images_at_zoom[np.argmin(num_white_pixels)]
                else:
                    aerial_image = aerial_images_at_zoom[0]
                aerial_images.append(aerial_image)
            aerial_images = np.stack(aerial_images, axis=0)
        metrics["t-air"] = timer.duration.timedelta.total_seconds()

        return types.SimpleNamespace(
            idx=idx,
            latlon=cell.latlon,
            images=aerial_images,
            metrics=metrics,
        )

    @staticmethod
    def collate(batch):
        with timerun.Timer() as timer:
            metrics = {k: np.mean([sample.metrics[k] for sample in batch]) for k in batch[0].metrics}

            # List of trees -> tree of lists
            batch = jax.tree_util.tree_map(lambda *xs: list(xs), *batch)

            batch = types.SimpleNamespace(
                aerial=types.SimpleNamespace(
                    images=np.asarray(batch.images),
                    latlons=np.asarray(batch.latlon),
                    idxs=np.asarray(batch.idx),
                ),
            )
        metrics["t-acol"] = timer.duration.timedelta.total_seconds()

        return batch, metrics

EARTH_RADIUS_METERS = 6.378137e6
class PvDataset:
    @staticmethod
    def from_config(config):
        dataset = FolderDataset(config["test.path"], None)

        if "bucket-size-meters" in config["test"]:
            dataset = BucketDataset(
                dataset,
                meters_per_bucket=config["test.bucket-size-meters"],
                images_per_bucket=1,
            )

        if "stride" in config["test"]:
            dataset = StridedDataset(
                dataset,
                stride=config["test.stride"],
            )

        return PvDataset(
            dataset=dataset,
            shape=config["pv-shape"],
        )

    @property
    def latlons(self):
        return self.dataset.latlons

    @property
    def timestamps(self):
        return self.dataset.timestamps

    @property
    def sequence_idxs(self):
        return self.dataset.sequence_idxs

    def __init__(self, dataset, shape):
        self.dataset = dataset
        self.shape = shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        metrics = {}

        with timerun.Timer() as timer:
            sample = self.dataset[idx]
            pv_image = cv2.imread(sample.path + f"-{self.shape[0]}.jpg")[..., ::-1]

            padding = np.asarray(self.shape) - np.asarray(pv_image.shape[:2])
            if np.any(padding > 0):
                padding_front = padding // 2
                padding_back = padding - padding_front
                pv_image = np.pad(pv_image, [(padding_front[0], padding_back[0]), (padding_front[1], padding_back[1]), (0, 0)], mode="constant", constant_values=0)
        metrics["t-pv"] = timer.duration.timedelta.total_seconds()

        return types.SimpleNamespace(
            idx=idx,
            image=pv_image,
            metrics=metrics,
        )

    @staticmethod
    def collate(batch):
        with timerun.Timer() as timer:
            metrics = {k: np.mean([sample.metrics[k] for sample in batch]) for k in batch[0].metrics}

            # List of trees -> tree of lists
            batch = jax.tree_util.tree_map(lambda *xs: list(xs), *batch)

            batch = types.SimpleNamespace(
                pv=types.SimpleNamespace(
                    images=np.asarray(batch.image),
                    idxs=np.asarray(batch.idx),
                ),
            )
        metrics["t-pvcol"] = timer.duration.timedelta.total_seconds()

        return batch, metrics
