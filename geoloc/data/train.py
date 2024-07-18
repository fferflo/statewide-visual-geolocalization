import json
import cv2
import types
import os
import timerun
import jax.tree_util
import tiledwebmaps as twm
import numpy as np
import albumentations as A
from .dataset import *

def aug_name(idx):
    return "image" if idx == 0 else f"image{idx + 1}"

class Dataset:
    @staticmethod
    def from_config(config):
        datasets = []
        for folder_config in config["train.list"]:
            tileloader = twm.from_yaml(os.path.join(folder_config["tiles-path"], "layout.yaml"))
            tileloader = twm.WithDefault(tileloader)
            dataset = FolderDataset(folder_config["path"], tileloader)
            datasets.append(dataset)
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)

        if "bucket-size-meters" in config["train"]:
            dataset = BucketDataset(dataset, meters_per_bucket=config["train.bucket-size-meters"])

        return Dataset(
            dataset=dataset,
            pv_shape=config["pv-shape"],
            aerial_shape=config["aerial-shape"],
            meters_per_pixel=config["meters-per-pixel"],
            offset_region_size_meters=config["train.offset-region-size-meters"],
            scale_fn=(lambda latlon: np.cos(np.radians(latlon[0]))) if config["mercator"] else (lambda latlon: 1.0),
        )

    def __init__(self, dataset, pv_shape, aerial_shape, meters_per_pixel, offset_region_size_meters, scale_fn):
        self.dataset = dataset
        self.pv_shape = pv_shape
        self.aerial_shape = aerial_shape
        self.meters_per_pixel = meters_per_pixel
        self.offset_region_size_meters = offset_region_size_meters
        self.scale_fn = scale_fn

        self.augment_color = A.Compose(
            [
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.10, always_apply=True, p=1.0),
            ],
            additional_targets={aug_name(i) : "image" for i in range(len(self.meters_per_pixel))},
        )

    def make_forksafe(self):
        self.dataset.make_forksafe()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, aerial_latlon=None, aerial_bearing=None):
        rng = np.random.default_rng()

        sample = self.dataset[idx]

        metrics = {}

        with timerun.Timer() as timer:
            # Load image
            pv_image = cv2.imread(sample.path + f"-{self.pv_shape[0]}.jpg")[..., ::-1]

            # Apply color augmentation
            pv_image = self.augment_color(image=pv_image)["image"]

            # Pad to pv_shape
            padding = np.asarray(self.pv_shape) - np.asarray(pv_image.shape[:2])
            if np.any(padding > 0):
                padding_front = padding // 2
                padding_back = padding - padding_front
                pv_image = np.pad(pv_image, [(padding_front[0], padding_back[0]), (padding_front[1], padding_back[1]), (0, 0)], mode="constant", constant_values=0)

            # Load latlon
            with open(sample.path + ".json") as f:
                metadata = json.load(f)
            pv_latlon = np.asarray(metadata["latlon"])
        metrics["t-pv"] = timer.duration.timedelta.total_seconds()

        with timerun.Timer() as timer:
            scale = self.scale_fn(pv_latlon)

            # Generate aerial image pose
            if aerial_bearing is None:
                aerial_bearing = rng.uniform(0, 360.0)
            if aerial_latlon is None:
                offset_region_size_meters = self.offset_region_size_meters * scale
                offset_cartesian = 0.5 * rng.uniform(-offset_region_size_meters, offset_region_size_meters, size=(2,))
                offset_distance = np.linalg.norm(offset_cartesian)
                offset_bearing = np.degrees(np.arctan2(offset_cartesian[1], offset_cartesian[0]))
                aerial_latlon = twm.geo.move_from_latlon(pv_latlon, aerial_bearing + offset_bearing, offset_distance)

            # Load aerial images
            aerial_images = []
            for mpp in self.meters_per_pixel:
                aerial_image = sample.tileloader.load(
                    latlon=aerial_latlon,
                    bearing=aerial_bearing + 180.0,
                    meters_per_pixel=mpp * scale,
                    shape=self.aerial_shape,
                )
                aerial_images.append(aerial_image)

            # Apply color augmentations
            aerial_images = {aug_name(i) : aerial_image for i, aerial_image in enumerate(aerial_images)}
            aerial_images = self.augment_color(**aerial_images)
            aerial_images = [aerial_images[aug_name(i)] for i in range(len(self.meters_per_pixel))]

            # Stack images
            aerial_images = np.stack(aerial_images, axis=0)

        metrics["t-air"] = timer.duration.timedelta.total_seconds()

        return types.SimpleNamespace(
            pv=types.SimpleNamespace(
                image=pv_image,
                latlon=pv_latlon,
                idx=idx,
            ),
            aerial=types.SimpleNamespace(
                images=aerial_images,
                latlon=aerial_latlon,
                bearing=aerial_bearing,
            ),
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
                    images=np.asarray(batch.pv.image),
                    latlons=np.asarray(batch.pv.latlon),
                    idxs=np.asarray(batch.pv.idx),
                ),
                aerial=types.SimpleNamespace(
                    images=np.asarray(batch.aerial.images),
                    latlons=np.asarray(batch.aerial.latlon),
                    bearings=np.asarray(batch.aerial.bearing),
                ),
            )

        metrics["t-collate"] = timer.duration.timedelta.total_seconds()

        return batch, metrics
