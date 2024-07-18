import json
import os
import tiledwebmaps as twm
import tqdm
import numpy as np
import shapely.geometry

class Region:
    def __init__(self, latlon_min, latlon_max, is_in_region):
        self.latlon_min = latlon_min
        self.latlon_max = latlon_max
        assert np.all(latlon_min < latlon_max)
        assert latlon_min[0] >= -90.0 and latlon_max[0] <= 90.0
        assert latlon_min[1] >= -180.0 and latlon_max[1] <= 180.0
        assert latlon_max[0] >= -90.0 and latlon_min[0] <= 90.0
        assert latlon_max[1] >= -180.0 and latlon_min[1] <= 180.0
        self.is_in_region = is_in_region

    @staticmethod
    def from_geojson(geojson):
        with open(geojson, "r") as f:
            geojson = json.load(f)
        if geojson["type"] == "FeatureCollection":
            assert len(geojson["features"]) == 1
            geojson = geojson["features"][0]

        region = shapely.geometry.shape(geojson["geometry"])
        latlon_min = np.asarray([region.bounds[1], region.bounds[0]])
        latlon_max = np.asarray([region.bounds[3], region.bounds[2]])
        assert np.all(latlon_min < latlon_max)

        def is_in_region(latlon):
            return region.contains(shapely.geometry.Point([latlon[1], latlon[0]]))

        return Region(latlon_min, latlon_max, is_in_region)

    @staticmethod
    def from_tiles(tiles_paths):
        if isinstance(tiles_paths, str):
            tiles_paths = [tiles_paths]
        covers = []
        latlon_min_all = None
        latlon_max_all = None
        for tiles_path in tiles_paths:
            layout = twm.Layout.from_yaml(os.path.join(tiles_path, "layout.yaml"))

            zoom = os.listdir(tiles_path)
            zoom = [int(x) for x in zoom if x.isdigit()]
            zoom = np.max(zoom)
            tiles_path = os.path.join(tiles_path, str(zoom))

            tiles = []
            for x in tqdm.tqdm(os.listdir(tiles_path), desc="Finding tiles"):
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

            tile_corners = np.stack([
                [min_tile[0], min_tile[1]],
                [min_tile[0], max_tile[1] + 1],
                [max_tile[0] + 1, max_tile[1] + 1],
                [max_tile[0] + 1, min_tile[1]],
            ])
            latlon_corners = np.asarray([layout.tile_to_epsg4326(tile, zoom=zoom) for tile in tile_corners])
            latlon_min = np.min(latlon_corners, axis=0)
            latlon_max = np.max(latlon_corners, axis=0)
            print(f"Found {len(tiles)} tiles in {tiles_path} between {latlon_min} and {latlon_max}")

            if latlon_min_all is None:
                latlon_min_all = latlon_min
                latlon_max_all = latlon_max
            else:
                latlon_min_all = np.minimum(latlon_min_all, latlon_min)
                latlon_max_all = np.maximum(latlon_max_all, latlon_max)
        latlon_min = latlon_min_all
        latlon_max = latlon_max_all

        def is_in_region(latlon):
            for cover, layout, zoom, min_tile in covers:
                tile = layout.epsg4326_to_tile(latlon, zoom=zoom).astype("int32")
                tile -= min_tile
                if np.any(tile < 0) or np.any(tile >= cover.shape):
                    continue
                if cover[tile[0], tile[1]]:
                    return True
            return False

        return Region(latlon_min, latlon_max, is_in_region)