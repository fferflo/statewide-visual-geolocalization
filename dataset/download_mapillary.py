#!/usr/bin/env python3

import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)
parser.add_argument("--token", type=str, required=True)
parser.add_argument("--presize", type=str, default="original")
parser.add_argument("--geojson", type=str, default=None)
parser.add_argument("--tiles", type=str, nargs="*")
parser.add_argument("--image-ids", type=str, default=None)
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--ratelimit1", type=int, default=59000)
parser.add_argument("--ratelimit2", type=int, default=9000)
parser.add_argument("--no-images", action="store_true")
parser.add_argument("--latlons-only", action="store_true")
args = parser.parse_args()

import requests, queue, threading, time, types, shutil, io, json, cv2, shapely.geometry, multiprocessing, skimage.transform, sys
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import tinypl as pl
from scipy.spatial.transform import Rotation
from PIL import Image
import tiledwebmaps as twm

if not args.no_images:
    assert args.width is not None and args.height is not None

shape_out = np.asarray([args.height, args.width])

download_path = os.path.join(args.path, "download")
if not os.path.exists(download_path):
    os.makedirs(download_path)

assert args.presize in ["original", "256", "1024", "2048"]

print("############################################")
print("Make sure to follow the rate limits specified in the Mapillary API documentation: https://www.mapillary.com/developer/api-documentation#rate-limits")
print("The script is currently using the following rate limits:")
print(f"'--ratelimit1 {args.ratelimit1}' requests per minute for graph.mapillary.com/:image_id")
print(f"'--ratelimit2 {args.ratelimit2}' requests per minute for graph.mapillary.com/images")
print("############################################")
assert args.ratelimit1 <= 60000
assert args.ratelimit2 <= 10000

class BoundingBox:
    def __init__(self, min, max, area):
        assert np.all(min < max)
        assert area > 0
        self.min = np.asarray(min)
        self.max = np.asarray(max)
        self.area = area

    @property
    def diameter(self):
        return twm.geo.distance(self.min, self.max)

    def split(self):
        bboxes = []
        mid = (self.min + self.max) / 2
        bboxes.append(BoundingBox(self.min, mid, area=self.area / 4))
        bboxes.append(BoundingBox([self.min[0], mid[1]], [mid[0], self.max[1]], area=self.area / 4))
        bboxes.append(BoundingBox(mid, self.max, area=self.area / 4))
        bboxes.append(BoundingBox([mid[0], self.min[1]], [self.max[0], mid[1]], area=self.area / 4))
        return bboxes

    @property
    def shape(self):
        return shapely.geometry.polygon.Polygon([
            [self.min[1], self.min[0]],
            [self.min[1], self.max[0]],
            [self.max[1], self.max[0]],
            [self.max[1], self.min[0]],
        ])

    def intersects(self, shape):
        return self.shape.intersects(shape)

image_ids_file = os.path.join(download_path, "image_ids.txt")

q = queue.Queue()
if not args.geojson is None:
    assert args.tiles is None or len(args.tiles) == 0
    assert args.image_ids is None
    print(f"Adding area covered by geojson {args.geojson}")

    with open(args.geojson, "r") as f:
        geojson = json.load(f)
    if geojson["type"] == "FeatureCollection":
        assert len(geojson["features"]) == 1
        geojson = geojson["features"][0]

    region = shapely.geometry.shape(geojson["geometry"])
    latlon_min = [region.bounds[1], region.bounds[0]]
    latlon_max = [region.bounds[3], region.bounds[2]]
    assert np.all(latlon_min < latlon_max)

    bbox_intersects = lambda bbox, region=region: bbox.intersects(region)
    def is_in_region(latlon, region=region):
        return region.contains(shapely.geometry.Point([latlon[1], latlon[0]]))

    if not os.path.exists(image_ids_file):
        init_q_size = 1
        q.put(BoundingBox(latlon_min, latlon_max, 1.0))
elif not args.tiles is None and len(args.tiles) > 0:
    assert args.image_ids is None
    import tiledwebmaps as twm

    covers = []
    for tiles_path in args.tiles:
        print(f"Adding area covered by tiles in {tiles_path}")
        layout = twm.Layout.from_yaml(os.path.join(tiles_path, "layout.yaml"))

        zoom = os.listdir(tiles_path)
        zoom = [int(x) for x in zoom if x.isdigit()]
        zoom = np.max(zoom)
        tiles_path = os.path.join(tiles_path, str(zoom))

        tiles = []
        for x in tqdm(os.listdir(tiles_path), desc="Finding tiles"):
            p = os.path.join(tiles_path, x)
            if os.path.isdir(p):
                for y in os.listdir(p):
                    try:
                        tiles.append(np.asarray([int(x), int(y.split(".")[0])]))
                    except ValueError:
                        pass
        tiles = np.asarray(tiles)
        min_tile = np.min(tiles, axis=0)
        max_tile = np.max(tiles, axis=0)
        tiles = tiles - min_tile[np.newaxis]
        cover = np.zeros([max_tile[0] - min_tile[0] + 1, max_tile[1] - min_tile[1] + 1], dtype="bool")
        cover[tiles[:, 0], tiles[:, 1]] = True

        covers.append((cover, layout, zoom, min_tile))

        latlon_min = layout.tile_to_epsg4326(min_tile, zoom=zoom)
        latlon_max = layout.tile_to_epsg4326(max_tile + 1, zoom=zoom)
        latlon_min, latlon_max = np.minimum(latlon_min, latlon_max), np.maximum(latlon_min, latlon_max)
        print(f"Found {len(tiles)} tiles in {tiles_path} between {latlon_min} and {latlon_max}")
        q.put(BoundingBox(latlon_min, latlon_max, 1.0))

    init_q_size = len(args.tiles)

    bbox_intersects = lambda bbox: True
    def is_in_region(latlon):
        for cover, layout, zoom, min_tile in covers:
            tile = layout.epsg4326_to_tile(latlon, zoom=zoom).astype("int32")
            tile -= min_tile
            if np.any(tile < 0) or np.any(tile >= cover.shape):
                continue
            if cover[tile[0], tile[1]]:
                return True
        return False
else:
    assert not args.image_ids is None
    assert os.path.exists(args.image_ids)

    if args.image_ids.endswith(".txt"):
        print(f"Copying image ids from {args.image_ids}")
        shutil.copy(args.image_ids, image_ids_file)
    elif args.image_ids.endswith(".npz"):
        print(f"Reading image ids from {args.image_ids}")
        with np.load(args.image_ids) as f:
            image_ids = f["image_ids"]
        image_ids = np.cumsum(image_ids) # Decode delta encoding

        # Store as text file in download directory
        with open(image_ids_file, "w") as f:
            f.write("\n".join([str(x) for x in image_ids]))
    else:
        raise Exception(f"Unknown image ids file format: {args.image_ids}")

    init_q_size = 0
    bbox_intersects = lambda bbox: True
    is_in_region = lambda latlon: True


if not os.path.exists(image_ids_file):
    print("Retrieving image ids")

    limit = 2000
    fix_threshold = 0.95

    end = False
    def generator():
        while not end:
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                pass
    pipe = generator()
    pipe = pl.thread.mutex(pipe)

    ratelimit = twm.util.Ratelimit(args.ratelimit2 / 60.0, 1.0)
    error_lock = threading.Lock()
    def process(bbox):
        if not bbox_intersects(bbox):
            return [], [], bbox.area, []

        url = f"https://graph.mapillary.com/images?access_token={args.token}&fields=id,computed_geometry&bbox={bbox.min[1]},{bbox.min[0]},{bbox.max[1]},{bbox.max[0]}&limit={limit}"

        for _ in range(20):
            try:
                with ratelimit:
                    response = requests.get(url)
                if response.status_code == 200:
                    infos = response.json()["data"]
                    if len(infos) >= limit * fix_threshold:
                        if bbox.diameter < 1.0:
                            print(f"WARNING: Got too many entries in small bounding box: n={len(infos)} in diameter={bbox.diameter}m latlon_min={bbox.min} latlon_max={bbox.max}\nurl={url}")
                        else:
                            return [], bbox.split(), 0.0, []

                    def is_valid(info):
                        if not "computed_geometry" in info:
                            return False
                        latlon = info["computed_geometry"]["coordinates"]
                        latlon = [latlon[1], latlon[0]]
                        return is_in_region(latlon)
                    infos = [info for info in infos if is_valid(info)]
                    ids = [x["id"] for x in infos]
                    latlons = [np.asarray(x["computed_geometry"]["coordinates"][::-1]) for x in infos]
                    return ids, [], bbox.area, latlons
            except requests.exceptions.RequestException as e:
                pass
            time.sleep(5.0)
        raise Exception(f"Failed http request. status_code={response.status_code} url={url}\n{response.text}")
    pipe = pl.thread.map(pipe, process, workers=args.workers)

    done_area = 0.0
    n = 100000
    with tqdm(total=init_q_size * n) as pbar:
        ids = set()
        if args.latlons_only:
            all_latlons = []
        remaining_boxes = init_q_size
        for ids2, bboxes, area, latlons in pipe:
            remaining_boxes -= 1
            ids.update(ids2)
            if args.latlons_only:
                all_latlons.extend(latlons)
            for bbox in bboxes:
                q.put(bbox)
                remaining_boxes += 1
            done_area += area

            pbar.update(int(done_area * n) - pbar.n)
            pbar.set_description(f"#bbox={q.qsize()} #image={len(ids)}")
            if remaining_boxes == 0:
                end = True

    with open(image_ids_file, "w") as f:
        f.write("\n".join(ids))

    if args.latlons_only:
        all_latlons = np.asarray(all_latlons)
        np.save(os.path.join(download_path, "latlons.npy"), all_latlons)
        print("Saved latlons.")
        sys.exit(0)

with open(image_ids_file, "r") as f:
    ids = f.read()
ids = [x.strip() for x in ids.split("\n") if len(x.strip()) > 0]
ids = np.asarray([int(x) for x in ids])# [:1]
print(f"Found {len(ids)} potential images")






images_directory_levels = max(len(str(len(ids))) - 2, 0)

allowed_camera_types = {"perspective", "fisheye", "brown", "fisheye_opencv", "radial", "simple_radial"}

pipe = enumerate(ids)

ratelimit = twm.util.Ratelimit(0.5 * args.ratelimit1 / 60.0, 1.0)
def rate(x):
    with ratelimit:
        return x
pipe = pl.thread.map(pipe, rate, workers=1)

# Fields: https://www.mapillary.com/developer/api-documentation
fields_str = f"camera_type,thumb_original_url,thumb_{args.presize}_url,captured_at,sequence,computed_geometry,height,width,creator"

fields = fields_str.split(",")
folders_lock = multiprocessing.Lock()
def process(x):
    image_idx, image_id = x
    # https://www.mapillary.com/developer/api-documentation
    url = f"https://graph.mapillary.com/{image_id}?access_token={args.token}&fields={fields_str}"
    for _ in range(3):
        try:
            response = requests.get(url)
            response.raise_for_status()
            info = response.json()

            if not "camera_type" in info:
                return "field missing: camera_type", None

            camera_type = info["camera_type"]
            if not camera_type in allowed_camera_types:
                return f"skipped camera_type: {camera_type}", None

            if not "thumb_original_url" in info:
                return "field missing: thumb_original_url", None
            for f in fields:
                if f != f"thumb_{args.presize}_url":
                    if f not in info:
                        return f"field missing: {f}", None

            latlon = np.asarray(info["computed_geometry"]["coordinates"][::-1])
            if not is_in_region(latlon):
                return "not in region", None

            latlon[1] = np.where(latlon[1] > 180, latlon[1] - 360, latlon[1])
            if not (-90.0 <= latlon[0] and latlon[0] <= 90.0) and (-180.0 <= latlon[1] and latlon[1] <= 180.0):
                return "invalid latlon", None

            timestamp = info["captured_at"]
            sequence_name = info["sequence"]

            if not args.no_images:
                # Download image
                url = info[f"thumb_{args.presize}_url"] if f"thumb_{args.presize}_url" in info else info["thumb_original_url"]
                response = requests.get(url)
                response.raise_for_status()
                image = np.asarray(Image.open(io.BytesIO(response.content)))

                if len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] < 3:
                    return "grayscale", None
                image = image[:, :, :3]

                downloaded_image_resolution = np.asarray([int(image.shape[0]), int(image.shape[1])])
                original_resolution = np.asarray([info["height"], info["width"]])

            if not args.no_images:
                # Resize
                shape_in = np.asarray(image.shape[:2])
                factor = np.amin(shape_out.astype("float") / shape_in) + 1e-6
                shape_resized = (shape_in * factor).astype("int")
                factor = np.mean(shape_resized.astype("float") / shape_in)
                assert np.all(shape_resized <= shape_out) and np.any(shape_resized == shape_out)

                dtype = image.dtype
                image = skimage.transform.resize(image.astype("float32"), shape_resized, order=1, mode="constant", preserve_range=True, anti_aliasing=True)
                image = image.astype(dtype)

            file = f"{image_idx:012}"[::-1]
            for i in reversed(range(images_directory_levels)):
                file = file[:i + 1] + "/" + file[i + 1:]
            file = os.path.join(args.path, "download", "images", file)

            directory = os.path.dirname(file)
            if not os.path.exists(directory):
                with folders_lock:
                    if not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)

            if not args.no_images:
                cv2.imwrite(file + ".jpg", image[:, :, ::-1])

            # Metadata
            metadata = {
                "timestamp": timestamp,
                "latlon": latlon.tolist(),
                "sequence": sequence_name,
                "camera-type": camera_type,
                "image-id": int(image_id),
                "creator": info["creator"]["username"],
            }

            with open(file + ".json", "w") as f:
                json.dump(metadata, f)

            return "success", (sequence_name, timestamp, image_idx, latlon)
        except requests.exceptions.RequestException as e:
            pass
        time.sleep(10.0)
    return "error", None
pipe = pl.process.map(pipe, process, workers=args.workers)

successes = []
results = defaultdict(lambda: 0)
results["success"] = 0
with tqdm(total=len(ids), smoothing=0.1) as pbar:
    for msg, data in pipe:
        if msg == "success":
            successes.append(data)
        results[msg] += 1
        pbar.update(1)
        pbar.set_description(f"Downloading images: #success={results['success']}")
print("Results:")
for k, v in results.items():
    print("   ", k, v)



# Defragment image indices
sequences = defaultdict(list)
new_image_idx = 0
for sequence_name, timestamp, old_image_idx, latlon in tqdm(successes, desc="Defragmenting image indices"):
    in_file = f"{old_image_idx:012}"[::-1]
    for i in reversed(range(images_directory_levels)):
        in_file = in_file[:i + 1] + "/" + in_file[i + 1:]
    in_file = os.path.join(args.path, "download", "images", in_file)

    if os.path.exists(in_file + ".json"):
        out_file = f"{new_image_idx:012}"[::-1]
        for i in reversed(range(images_directory_levels)):
            out_file = out_file[:i + 1] + "/" + out_file[i + 1:]
        out_file = os.path.join(args.path, "images", out_file)

        directory = os.path.dirname(out_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not args.no_images:
            shutil.move(in_file + ".jpg", out_file + f"-{args.height}.jpg")
        shutil.move(in_file + ".json", out_file + ".json")

        sequences[sequence_name].append((timestamp, new_image_idx, latlon))

        new_image_idx += 1
images_num = new_image_idx

print(f"Found {images_num} images in {len(sequences)} sequences")

# Save per-sequence metadata
seq_directory_levels = max(len(str(len(sequences))) - 2, 0)
dest_sequences_path = os.path.join(args.path, "sequences")
image_seqidx = np.zeros(images_num, dtype="int32") - 1
for sequence_idx, (sequence_name, sequence) in tqdm(list(enumerate(sorted(sequences.items()))), "Saving sequences metadata"):
    sequence = sorted(sequence)
    timestamps = [x[0] for x in sequence]
    image_indices = [x[1] for x in sequence]
    latlons = [x[2] for x in sequence]

    image_seqidx[image_indices] = sequence_idx

    file = f"{sequence_idx:012}"[::-1]
    for i in reversed(range(seq_directory_levels)):
        file = file[:i + 1] + "/" + file[i + 1:]
    file = os.path.join(args.path, "sequences", file)

    directory = os.path.dirname(file)
    os.makedirs(directory, exist_ok=True)

    # Metadata
    metadata = {
        "name": sequence_name,
        "t0": timestamps[0],
        "duration": timestamps[-1] - timestamps[0],
        "latlon0": latlons[0].tolist(),
        "image-indices": image_indices,
    }
    with open(file + ".json", "w") as f:
        json.dump(metadata, f)
np.savez_compressed(os.path.join(args.path, f"sequence-idxs.npz"), sequence_idxs=image_seqidx)

# Save per-image metadata
latlons = np.zeros([images_num, 2], dtype="float64")
timestamps = np.zeros([images_num], dtype="uint64")
for image_idx in tqdm(list(range(images_num)), "Saving images metadata"):
    file = f"{image_idx:012}"[::-1]
    for i in reversed(range(images_directory_levels)):
        file = file[:i + 1] + "/" + file[i + 1:]
    file = os.path.join(args.path, "images", file)

    with open(file + ".json") as f:
        metadata = json.load(f)

    if "latlon" in metadata:
        latlon = np.asarray(metadata["latlon"])
        assert -90.0 <= latlon[0] and latlon[0] <= 90.0, f"Invalid latitude {latlon[0]} for image {image_idx}"
        assert -180.0 <= latlon[1] and latlon[1] <= 180.0, f"Invalid longitude {latlon[1]} for image {image_idx}"
        latlons[image_idx] = latlon
    else:
        latlons = None

    if "timestamp" in metadata:
        timestamp = int(metadata["timestamp"])
        assert timestamp >= 0, f"Invalid timestamp {timestamp} for image {image_idx}"
        timestamps[image_idx] = timestamp
    else:
        timestamps = None
np.savez_compressed(os.path.join(args.path, f"latlons.npz"), latlons=latlons)
np.savez_compressed(os.path.join(args.path, f"timestamps.npz"), timestamps=timestamps)

# Metadata for entire dataset
metadata = {
    "images-num": images_num,
    "images-directory-levels": images_directory_levels,
    "sequences-num": len(sequences),
    "sequences-directory-levels": seq_directory_levels,
    "orig-resolution": [args.height, args.width],
}
with open(os.path.join(args.path, f"dataset.json"), "w") as f:
    json.dump(metadata, f)

with open(os.path.join(args.path, "LICENSE"), "w") as f:
    f.write("CC BY-NC-SA https://help.mapillary.com/hc/en-us/articles/115001770409-Licenses")