import json
import os
from collections import defaultdict
import numpy as np
import tqdm
import types

class Dataset:
    def __init__(self):
        self._features = {}

    @property
    def latlons(self):
        return self._get_feature("latlons")

    @property
    def timestamps(self):
        return self._get_feature("timestamps")
    
    @property
    def sequence_idxs(self):
        return self._get_feature("sequence_idxs")

    def _get_feature(self, name):
        if name not in self._features:
            self._features[name] = self._load_feature(name)
        return self._features[name]

class FolderDataset(Dataset):
    def __init__(self, path, tileloader=None):
        super().__init__()
        self.path = path
        with open(os.path.join(self.path, "dataset.json")) as f:
            metadata = json.load(f)
        self.image_directory_levels = metadata["images-directory-levels"]
        self.sequence_directory_levels = metadata["sequences-directory-levels"]

        self.num_sequences = metadata["sequences-num"]
        self.num_images = metadata["images-num"]

        self.tileloader = tileloader

        print(f"Found {self.num_images} images in {self.path}")

    def make_forksafe(self):
        if not self.tileloader is None:
            self.tileloader.make_forksafe()

    def _load_feature(self, name):
        path = os.path.join(self.path, f"{name.replace('_', '-')}.npz")
        if not os.path.exists(path):
            return None
        with np.load(path) as f:
            x = f[name]
        assert x.shape[0] > 0
        if name == "latlons":
            # Check that latitudes are in [-90, 90] and mod longitudes to [-180, 180]
            assert np.all(x[:, 0] >= -90.0) and np.all(x[:, 0] <= 90.0)
            x[:, 1] = np.where(x[:, 1] > 180, x[:, 1] - 360, x[:, 1])
            assert np.all(x[:, 1] >= -180.0) and np.all(x[:, 1] <= 180.0)
        return x

    def __len__(self):
        return self.num_images

    def get_sequence_path(self, idx):
        assert idx >= 0 and idx < self.num_sequences, f"Invalid sequence index {idx} for dataset {self.path}"
        file = f"{idx:012}"[::-1]
        for i in reversed(range(self.sequence_directory_levels)):
            file = file[:i + 1] + "/" + file[i + 1:]
        file = os.path.join(self.path, "sequences", file)
        return file

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.num_images, f"Invalid image index {idx} for dataset {self.path}"

        file = f"{idx:012}"[::-1]
        for i in reversed(range(self.image_directory_levels)):
            file = file[:i + 1] + "/" + file[i + 1:]
        file = os.path.join(self.path, "images", file)

        return types.SimpleNamespace(
            path=file,
            tileloader=self.tileloader,
        )

class FilteredDataset(Dataset):
    def __init__(self, dataset, mask):
        super().__init__()
        self.dataset = dataset
        self.mask = mask
        self.len = np.count_nonzero(mask)
        self.to_unfiltered_idx = np.where(mask)[0]

    def make_forksafe(self):
        self.dataset.make_forksafe()

    def _load_feature(self, name):
        feature = self.dataset._get_feature(name)
        if feature is None:
            return None
        return feature[self.mask]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.len:
            raise IndexError(f"Index {idx} out of bounds for dataset {self.dataset.path}")
        return self.dataset[self.to_unfiltered_idx[idx]]

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.lens = np.asarray([len(dataset) for dataset in datasets])
        self.dataset_starts = np.concatenate([[0], np.cumsum(self.lens)], axis=0)
        self.len = np.sum(self.lens)

    def make_forksafe(self):
        for dataset in self.datasets:
            dataset.make_forksafe()

    def _load_feature(self, name):
        feature = [dataset._get_feature(name) for dataset in self.datasets]
        if any([f is None for f in feature]):
            return None
        return np.concatenate(feature, axis=0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")

        dataset_idx = np.searchsorted(self.dataset_starts, idx, side="right") - 1
        dataset = self.datasets[dataset_idx]
        idx = idx - self.dataset_starts[dataset_idx]

        return dataset[idx]

EARTH_RADIUS_METERS = 6.378137e6
class BucketDataset(Dataset):
    def __init__(self, dataset, meters_per_bucket, images_per_bucket=None):
        super().__init__()
        self.dataset = dataset

        print("Dividing dataset into buckets...")

        # Find unique bucket ids
        bucket_y = np.floor(EARTH_RADIUS_METERS * np.radians(dataset.latlons[:, 0]) / meters_per_bucket).astype("int64")
        bucket_lat = np.degrees(bucket_y * meters_per_bucket / EARTH_RADIUS_METERS)
        radius_at_lat = EARTH_RADIUS_METERS * np.cos(np.radians(bucket_lat))
        bucket_x = np.floor(radius_at_lat * np.radians(dataset.latlons[:, 1]) / meters_per_bucket).astype("int64")
        bucket_ids = np.stack([bucket_y, bucket_x], axis=1)

        # Construct buckets
        x, bucket_idxs = np.unique(bucket_ids, return_inverse=True, axis=0) # image-idx -> bucket-idx
        num_buckets = len(x)
        image_idxs = np.arange(len(dataset))
        assert len(bucket_idxs) == len(image_idxs)
        indices = np.argsort(bucket_idxs)
        bucket_idxs = bucket_idxs[indices]
        image_idxs = image_idxs[indices]

        # Filter by images_per_bucket
        if images_per_bucket is not None:
            new_bucket_idxs = []
            new_image_idxs = []
            n = 0
            last_bucket_idx = None
            for bucket_idx, image_idx in zip(bucket_idxs, image_idxs):
                if last_bucket_idx != bucket_idx:
                    n = 0
                    last_bucket_idx = bucket_idx
                if n < images_per_bucket:
                    new_bucket_idxs.append(bucket_idx)
                    new_image_idxs.append(image_idx)
                    n += 1

            bucket_idxs = np.asarray(new_bucket_idxs)
            image_idxs = np.asarray(new_image_idxs)

        # Compute bucket offsets
        bucket_lens = np.bincount(bucket_idxs)
        assert len(bucket_lens) == num_buckets
        assert np.all(bucket_lens > 0)
        assert images_per_bucket is None or np.max(bucket_lens) <= images_per_bucket
        bucket_starts = np.concatenate([[0], np.cumsum(bucket_lens)], axis=0)

        print(f"Using {len(bucket_starts) - 1} buckets with size {meters_per_bucket}m. Total input images: {len(dataset)}. Total filtered images: {len(image_idxs)}. Images per bucket: mean={np.mean(bucket_lens):.1f} median={np.median(bucket_lens):.1f} min={np.min(bucket_lens)} max={np.max(bucket_lens)}")

        self.image_idxs = image_idxs
        self.bucket_starts = bucket_starts
        self.images_per_bucket = images_per_bucket

        self.used = np.zeros((len(self.image_idxs),), dtype="bool") # Return all images per bucket before repeating images

    def make_forksafe(self):
        self.dataset.make_forksafe()

    def _get_feature(self, name):
        if self.images_per_bucket == 1:
            feature = self.dataset._get_feature(name)
            if feature is None:
                return None
            return feature[self.image_idxs]
        else:
            raise NotImplementedError("Not supported for BucketDataset with images_per_bucket != 1")

    def __len__(self):
        return len(self.bucket_starts) - 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")

        rng = np.random.default_rng()

        bucket_start = self.bucket_starts[idx]
        bucket_end = self.bucket_starts[idx + 1]
        assert bucket_end > bucket_start

        used = self.used[bucket_start:bucket_end]
        if np.count_nonzero(np.logical_not(used)) == 0:
            self.used[bucket_start:bucket_end] = False
            used = self.used[bucket_start:bucket_end]

        valid = np.where(np.logical_not(used))[0] + bucket_start
        assert len(valid) > 0
        i = rng.choice(valid)

        self.used[i] = True
        idx = self.image_idxs[i]

        return self.dataset[idx]

class StridedDataset(Dataset):
    def __init__(self, dataset, stride):
        super().__init__()
        self.dataset = dataset
        self.stride = stride
        print(f"Using subset of dataset with stride {stride}. Total input images: {len(dataset)}. Total filtered images: {len(self)}.")

    def make_forksafe(self):
        self.dataset.make_forksafe()

    def _load_feature(self, name):
        feature = self.dataset._get_feature(name)
        if feature is None:
            return None
        return feature[::self.stride]

    def __len__(self):
        return (len(self.dataset) + self.stride - 1) // self.stride

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        return self.dataset[idx * self.stride]







import os
import multiprocessing
import cv2
import skimage.transform
import tinypl as pl
import numpy as np
import tqdm
import json
from collections import defaultdict

def resize_with_constant_aspect_ratio(image, shape):
    shape_out = np.asarray(shape)
    shape_in = np.asarray(image.shape[:2])

    factor = np.amin(shape_out.astype("float") / shape_in) + 1e-6
    shape_resized = (shape_in * factor).astype("int")
    factor = np.mean(shape_resized.astype("float") / shape_in)
    assert np.all(shape_resized <= shape_out) and np.any(shape_resized == shape_out)

    dtype = image.dtype
    image = skimage.transform.resize(image.astype("float32"), shape_resized, order=1, mode="constant", preserve_range=True, anti_aliasing=True)
    image = image.astype(dtype)

    return image

class DatasetWriter:
    def __init__(self, path, images_num, shape, src_shape=None):
        self.path = path
        self.folders_lock = multiprocessing.Lock()
        self.images_directory_levels = max(len(str(images_num)) - 2, 0)
        self.images_num = images_num
        self.shape = shape
        self.src_shape = src_shape

    def write(self, image, index, latlon=None, timestamp=None, sequence_name=None):
        assert index >= 0 and index < self.images_num
        image = resize_with_constant_aspect_ratio(image, self.shape)

        file = f"{index:012}"[::-1]
        for i in reversed(range(self.images_directory_levels)):
            file = file[:i + 1] + "/" + file[i + 1:]
        file = os.path.join(self.path, "images", file)

        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            with self.folders_lock:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

        cv2.imwrite(file + f"-{self.shape[0]}.jpg", image[:, :, ::-1])

        metadata = {}
        if latlon is not None:
            metadata["latlon"] = [int(latlon[0]), int(latlon[1])]
        if timestamp is not None:
            metadata["timestamp"] = int(timestamp)
        if sequence_name is not None:
            metadata["sequence"] = sequence_name
        with open(file + ".json", "w") as f:
            json.dump(metadata, f)

    def finalize(self):
        sequences = defaultdict(list)
        latlons = np.zeros([self.images_num, 2], dtype="float64")
        timestamps = np.zeros([self.images_num], dtype="uint64")
        for image_idx in tqdm.tqdm(list(range(self.images_num)), "Saving images metadata"):
            file = f"{image_idx:012}"[::-1]
            for i in reversed(range(self.images_directory_levels)):
                file = file[:i + 1] + "/" + file[i + 1:]
            file = os.path.join(self.path, "images", file)

            with open(file + ".json") as f:
                metadata = json.load(f)

            if "latlon" in metadata:
                latlon = np.asarray(metadata["latlon"])
                assert -90.0 <= latlon[0] and latlon[0] <= 90.0, f"Invalid latitude {latlon[0]} for image {image_idx}"
                assert -180.0 <= latlon[1] and latlon[1] <= 180.0, f"Invalid longitude {latlon[1]} for image {image_idx}"
                latlons[image_idx] = latlon
            else:
                latlon = None
                latlons = None

            if "timestamp" in metadata:
                timestamp = int(metadata["timestamp"])
                assert timestamp >= 0, f"Invalid timestamp {timestamp} for image {image_idx}"
                timestamps[image_idx] = timestamp
            else:
                timestamp = None
                timestamps = None

            if "sequence" in metadata:
                sequence_name = metadata["sequence"]
                assert timestamp is not None
                sequences[sequence_name].append((timestamp, image_idx, latlon))
        if latlons is not None:
            np.savez_compressed(os.path.join(self.path, f"latlons.npz"), latlons=latlons)
        if timestamps is not None:
            np.savez_compressed(os.path.join(self.path, f"timestamps.npz"), timestamps=timestamps)

        if len(sequences) > 0:
            # Save per-sequence metadata
            seq_directory_levels = max(len(str(len(sequences))) - 2, 0)
            dest_sequences_path = os.path.join(self.path, "sequences")
            image_seqidx = np.zeros(self.images_num, dtype="int32") - 1
            for sequence_idx, (sequence_name, sequence) in tqdm.tqdm(list(enumerate(sorted(sequences.items()))), "Saving sequences metadata"):
                sequence = sorted(sequence)
                timestamps = [x[0] for x in sequence]
                image_indices = [x[1] for x in sequence]
                latlons = [x[2] for x in sequence]

                image_seqidx[image_indices] = sequence_idx

                file = f"{sequence_idx:012}"[::-1]
                for i in reversed(range(seq_directory_levels)):
                    file = file[:i + 1] + "/" + file[i + 1:]
                file = os.path.join(self.path, "sequences", file)

                directory = os.path.dirname(file)
                os.makedirs(directory, exist_ok=True)

                # Metadata
                metadata = {
                    "name": sequence_name,
                    "t0": timestamps[0],
                    "duration": timestamps[-1] - timestamps[0],
                    "image-indices": image_indices,
                }
                if all([x is not None for x in latlons]):
                    metadata["latlon0"] = [int(latlons[0][0]), int(latlons[0][1])]
                with open(file + ".json", "w") as f:
                    json.dump(metadata, f)
            np.savez_compressed(os.path.join(self.path, f"sequence-idxs.npz"), sequence_idxs=image_seqidx)
        else:
            sequences = []
            seq_directory_levels = 0

        # Metadata for entire dataset
        metadata = {
            "images-num": self.images_num,
            "images-directory-levels": self.images_directory_levels,
            "sequences-num": len(sequences),
            "sequences-directory-levels": seq_directory_levels,
            "orig-resolution": [int(self.src_shape[0]), int(self.src_shape[1])],
        }
        with open(os.path.join(self.path, f"dataset.json"), "w") as f:
            json.dump(metadata, f)

def video_to_dataset(video_file, path, shape, workers=16, stride=1):
    video_loader = cv2.VideoCapture(video_file)
    num_frames = (int(video_loader.get(cv2.CAP_PROP_FRAME_COUNT)) + stride - 1) // stride
    seconds_per_frame = 1.0 / video_loader.get(cv2.CAP_PROP_FPS)

    writer = DatasetWriter(path, num_frames, shape=shape)

    def source():
        index = 0
        while True:
            ret, image = video_loader.read()
            if index == 0:
                writer.src_shape = image.shape[:2]
            if not ret:
                break
            if index % stride == 0:
                yield image, index // stride, seconds_per_frame * index
            index += 1

    pipe = source()
    pipe = pl.thread.mutex(pipe)

    @pl.unpack
    def process(image, index, timestamp):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        writer.write(image, index, timestamp=int(timestamp * 1000), sequence_name=os.path.basename(video_file))
    pipe = pl.process.map(pipe, process, workers=workers)

    for _ in tqdm.tqdm(pipe, total=num_frames, desc="Converting video to dataset"):
        pass

    writer.finalize()