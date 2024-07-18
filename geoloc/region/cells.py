import tiledwebmaps as twm
import numpy as np
import os
import einx
import tqdm
import tinypl as pl
from sklearn.neighbors import KDTree
import math
import types
import jax.numpy as jnp
import jax
from functools import partial

EARTH_RADIUS_METERS = 6.378137e6

@jax.jit
def _find_matches(query_latlons, reference_latlons, nns_aerialidxs):
    nns_latlons = reference_latlons[nns_aerialidxs] # bg nns 2

    nns_distances_lat = twm.geo.distance(
        einx.rearrange("bg nns, bg -> bg nns (1 + 1)", nns_latlons[..., 0], query_latlons[..., 1]),
        query_latlons[:, jnp.newaxis, :],
        np=jnp,
    )
    nns_distances_lon = twm.geo.distance(
        einx.rearrange("bg, bg nns -> bg nns (1 + 1)", query_latlons[..., 0], nns_latlons[..., 1]),
        query_latlons[:, jnp.newaxis, :],
        np=jnp,
    )
    nns_distances = nns_distances_lat + nns_distances_lon
    nn_idx = jnp.argmin(nns_distances, axis=1) # bg

    nn_aerialidx = einx.get_at("bg [nns], bg -> bg", nns_aerialidxs, nn_idx)

    return nn_aerialidx.astype("int32")

@partial(jax.jit, static_argnums=(3,))
def _find_matches_radius(query_latlons, reference_latlons, nns_aerialidxs, radius):
    nns_latlons = reference_latlons[nns_aerialidxs] # bg nns 2
    nns_distances = twm.geo.distance(
        nns_latlons,
        query_latlons[:, jnp.newaxis, :],
        np=jnp,
    )
    nns_match = nns_distances < radius # bg nns

    nns_aerialidxs = jnp.where(nns_match, nns_aerialidxs, -1)
    nns_aerialidxs = jnp.sort(nns_aerialidxs, axis=1, descending=True)
    match_num = jnp.count_nonzero(nns_match, axis=1)

    return nns_aerialidxs.astype("int32"), jnp.mean(match_num), jnp.median(match_num), jnp.min(match_num), jnp.max(match_num)



class CellRegion:
    @staticmethod
    def consistent(region, cell_size_meters):
        min_patch_y = int(EARTH_RADIUS_METERS * math.radians(region.latlon_min[0]) / cell_size_meters) - 1 # TODO: use np.floor
        max_patch_y = int(EARTH_RADIUS_METERS * math.radians(region.latlon_max[0]) / cell_size_meters) + 2 # TODO: use np.floor

        jobs = list(enumerate(range(min_patch_y, max_patch_y + 1)))

        pipe = jobs

        @pl.unpack
        def process(rowidx, patch_y):
            patch_center_y_meters = patch_y * cell_size_meters
            patch_center_y_lat = math.degrees(patch_center_y_meters / EARTH_RADIUS_METERS)

            radius_at_lat = EARTH_RADIUS_METERS * math.cos(math.radians(patch_center_y_lat))

            min_patch_x = int(radius_at_lat * math.radians(region.latlon_min[1]) / cell_size_meters) - 1 # TODO: use np.floor
            max_patch_x = int(radius_at_lat * math.radians(region.latlon_max[1]) / cell_size_meters) + 2 # TODO: use np.floor

            patch_xs = np.arange(min_patch_x, max_patch_x + 1)

            patch_center_xs_meters = patch_xs * cell_size_meters
            patch_center_xs_lon = np.degrees(patch_center_xs_meters / radius_at_lat)

            patch_corners_y_meters = np.asarray([patch_center_y_meters - cell_size_meters / 2, patch_center_y_meters + cell_size_meters / 2])
            patch_corners_xs_meters = np.asarray([patch_center_xs_meters - cell_size_meters / 2, patch_center_xs_meters + cell_size_meters / 2])

            patch_corners_00_meters = einx.rearrange(", x -> x (1 + 1)", patch_corners_y_meters[0], patch_corners_xs_meters[0])
            patch_corners_01_meters = einx.rearrange(", x -> x (1 + 1)", patch_corners_y_meters[0], patch_corners_xs_meters[1])
            patch_corners_10_meters = einx.rearrange(", x -> x (1 + 1)", patch_corners_y_meters[1], patch_corners_xs_meters[0])
            patch_corners_11_meters = einx.rearrange(", x -> x (1 + 1)", patch_corners_y_meters[1], patch_corners_xs_meters[1])
            patch_corners_meters = np.stack([patch_corners_00_meters, patch_corners_01_meters, patch_corners_11_meters, patch_corners_10_meters], axis=1)

            patch_corners_latlon = np.degrees(patch_corners_meters / np.asarray([EARTH_RADIUS_METERS, radius_at_lat])[np.newaxis, np.newaxis])

            patch_centers = []
            for patch_corners_latlon, patch_center_x_lon in zip(patch_corners_latlon, patch_center_xs_lon):
                if any(region.is_in_region(corner) for corner in patch_corners_latlon):
                    patch_centers.append(np.asarray([patch_center_y_lat, patch_center_x_lon]))

            return rowidx, patch_centers
        pipe = pl.process.map(pipe, process, workers=os.cpu_count())

        patch_centers = []
        for rowidx, patch_centers_ in tqdm.tqdm(pipe, total=len(jobs), desc="Constructing cells"):
            if len(patch_centers_) > 0:
                patch_centers.append((rowidx, patch_centers_))
        patch_centers = sorted(patch_centers)

        patch_centers = [np.asarray(p) for _, p in patch_centers]
        patch_centers_flat = np.concatenate(patch_centers, axis=0)
        row_lens = np.asarray([len(p) for p in patch_centers])

        return CellRegion(patch_centers_flat, row_lens, cell_size_meters, mercator=False)

    @staticmethod
    def mercator(region, cell_size_meters):
        crs_min = twm.proj.epsg4326_to_epsg3857(region.latlon_min)
        crs_max = twm.proj.epsg4326_to_epsg3857(region.latlon_max)

        cell_size_crs = np.linalg.norm(twm.proj.epsg4326_to_epsg3857([0.0, 0.0]) - twm.proj.epsg4326_to_epsg3857(twm.geo.move_from_latlon([0.0, 0.0], 90.0, cell_size_meters)))

        cell_nums = np.ceil((crs_max - crs_min) / cell_size_crs).astype("int") + 3
        cells_y = np.arange(cell_nums[1])

        jobs = list(enumerate(cells_y))

        pipe = jobs

        @pl.unpack
        def process(rowidx, cell_y):
            cells_x = np.arange(cell_nums[0])

            cells_xy = einx.rearrange("x, -> x (1 + 1)", cells_x, cell_y)
            cells_min_crs = crs_min + cells_xy * cell_size_crs
            cells_center_crs = cells_min_crs + 0.5 * cell_size_crs
            cells_max_crs = cells_min_crs + cell_size_crs
            cells_min_latlon = np.asarray([twm.proj.epsg3857_to_epsg4326(c) for c in cells_min_crs])
            cells_center_latlon = np.asarray([twm.proj.epsg3857_to_epsg4326(c) for c in cells_center_crs])
            cells_max_latlon = np.asarray([twm.proj.epsg3857_to_epsg4326(c) for c in cells_max_crs])

            corners_latlon = np.stack([
                cells_min_latlon,
                np.stack([cells_min_latlon[:, 0], cells_max_latlon[:, 1]], axis=-1),
                cells_max_latlon,
                np.stack([cells_max_latlon[:, 0], cells_min_latlon[:, 1]], axis=-1),
            ], axis=1) # x 4 2

            patch_centers = []
            for corners_latlon, center_latlon in zip(corners_latlon, cells_center_latlon):
                if any(region.is_in_region(corner) for corner in corners_latlon):
                    patch_centers.append(np.asarray(center_latlon))

            return rowidx, patch_centers
        pipe = pl.process.map(pipe, process, workers=os.cpu_count())

        patch_centers = []
        for rowidx, patch_centers_ in tqdm.tqdm(pipe, total=len(jobs), desc="Constructing cells"):
            if len(patch_centers_) > 0:
                patch_centers.append((rowidx, patch_centers_))
        patch_centers = sorted(patch_centers)

        patch_centers = [np.asarray(p) for _, p in patch_centers]
        patch_centers_flat = np.concatenate(patch_centers, axis=0)
        row_lens = np.asarray([len(p) for p in patch_centers])

        return CellRegion(patch_centers_flat, row_lens, cell_size_meters, mercator=True)

    @staticmethod
    def from_npz(file):
        data = np.load(file)
        return CellRegion(data["latlons"], data["row_lens"], data["cell_size_meters"], data["mercator"] if "mercator" in data else False)

    def __init__(self, latlons, row_lens, cell_size_meters, mercator):
        self.latlons = latlons
        self.row_lens = row_lens
        self.cell_size_meters = cell_size_meters
        self.mercator = mercator
        self.tree = KDTree(self.latlons, metric="euclidean")

        max_row_len = np.max(self.row_lens)
        self.latlon_matrix = np.full((len(self.row_lens), max_row_len, 2), float("nan"), dtype="float64")
        self.idx_matrix = np.full((len(self.row_lens), max_row_len), -1, dtype="int32")
        self.start_idxs = np.cumsum([0] + self.row_lens.tolist())
        for row_idx in range(len(self.row_lens)):
            start_idx = self.start_idxs[row_idx]
            end_idx = self.start_idxs[row_idx + 1]
            latlons = self.latlons[start_idx:end_idx]
            idxs = np.arange(start_idx, end_idx)

            self.latlon_matrix[row_idx] = np.pad(latlons, ((0, max_row_len - len(latlons)), (0, 0)), constant_values=999.0)
            self.idx_matrix[row_idx] = np.pad(idxs, ((0, max_row_len - len(idxs))), constant_values=-1)

    def save(self, file):
        np.savez(file,
            latlons=self.latlons,
            row_lens=self.row_lens,
            cell_size_meters=self.cell_size_meters,
            mercator=self.mercator,
        )

    def __len__(self):
        return len(self.latlons)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of bounds for {len(self)} cells")
        latlon = self.latlons[idx]
        return types.SimpleNamespace(
            latlon=latlon,
            scale=np.cos(np.radians(latlon[0])) if self.mercator else 1.0,
        )

    @property
    def center_latlon(self):
        return 0.5 * (np.max(self.latlons, axis=0) + np.min(self.latlons, axis=0))

    def find_matches(self, latlons):
        print(f"Finding closest match for {latlons.shape[0]} queries in {len(self)} cells...", flush=True, end="")
        
        # Sample candidates
        nns_aerialidxs = np.asarray(self.tree.query(latlons, k=10, dualtree=True, return_distance=False)) # bg nns

        # Find matches
        with jax.experimental.enable_x64():
            nn_aerialidxs = _find_matches(
                jax.device_put(latlons, jax.devices("cpu")[0]),
                jax.device_put(self.latlons, jax.devices("cpu")[0]),
                jax.device_put(nns_aerialidxs, jax.devices("cpu")[0]),
            )

        return nn_aerialidxs

    def find_matches_radius(self, latlons, radius):
        print(f"Finding match<{radius}m for {latlons.shape[0]} queries in {len(self)} cells...", flush=True, end="")

        # Sample candidates
        k = math.ceil(3 * radius / self.cell_size_meters) ** 2
        nns_aerialidxs = np.asarray(self.tree.query(latlons, k=k, dualtree=True, return_distance=False)) # bg nns

        # Find matches
        with jax.experimental.enable_x64():
            nns_aerialidxs, mean_num, median_num, min_num, max_num = _find_matches_radius(
                jax.device_put(latlons, jax.devices("cpu")[0]),
                jax.device_put(self.latlons, jax.devices("cpu")[0]),
                jax.device_put(nns_aerialidxs, jax.devices("cpu")[0]),
                radius,
            )

        assert max_num < k, "Increase k"
        assert min_num > 0, "No match found for some query"

        nns_aerialidxs = nns_aerialidxs[:, :max_num]

        print(f" done. Got #cells/query: mean={mean_num:.2f}, median={median_num}, max={max_num} < {k}", flush=True)
        return nns_aerialidxs