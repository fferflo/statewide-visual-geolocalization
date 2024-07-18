import jax
import jax.numpy as jnp
import numpy as np
import einx
import tiledwebmaps as twm
from functools import partial
import cv2
import skimage.draw

@jax.jit
def get_y(latlon, ys, latlon_rows, idx_rows, row_lens, default, cell_size_meters):
    # Find nearest lat
    lats = latlon_rows[:, 0, 0]

    lat_idx_upper = jnp.searchsorted(lats, latlon[0])
    lat_idx_lower = lat_idx_upper - 1

    distance_lat_upper = twm.geo.distance(
        jnp.stack([lats[lat_idx_upper], latlon[1]]),
        latlon,
        np=jnp,
    )
    distance_lat_lower = twm.geo.distance(
        jnp.stack([lats[lat_idx_lower], latlon[1]]),
        latlon,
        np=jnp,
    )
    distance_lat = jnp.minimum(distance_lat_upper, distance_lat_lower)
    is_upper = distance_lat_upper < distance_lat_lower
    lat_idx = jnp.where(is_upper, lat_idx_upper, lat_idx_lower)

    # Find nearest lon (in lat row)
    lons = latlon_rows[lat_idx, :, 1]

    lon_idx_upper = jnp.searchsorted(lons, latlon[1])
    lon_idx_lower = lon_idx_upper - 1

    distance_lon_upper = twm.geo.distance(
        jnp.stack([latlon[0], lons[lon_idx_upper]]),
        latlon,
        np=jnp,
    )
    distance_lon_lower = twm.geo.distance(
        jnp.stack([latlon[0], lons[lon_idx_lower]]),
        latlon,
        np=jnp,
    )
    distance_lon = jnp.minimum(distance_lon_upper, distance_lon_lower)
    is_upper = distance_lon_upper < distance_lon_lower
    lon_idx = jnp.where(is_upper, lon_idx_upper, lon_idx_lower)

    # Check if cell is valid
    valid = jnp.logical_and(
        jnp.logical_and(0 <= lat_idx_lower, lat_idx_upper < len(lats)),
        jnp.logical_and(0 <= lon_idx_lower, lon_idx_upper < row_lens[lat_idx]),
    )
    eps = 0.1
    valid = jnp.logical_and(
        valid,
        jnp.logical_and(distance_lat <= (0.5 + eps) * cell_size_meters, distance_lon <= (0.5 + eps) * cell_size_meters),
    )

    # Retrieve value
    idx = idx_rows[lat_idx, lon_idx]
    y = ys[idx]
    return jnp.where(valid, y, default)

@partial(jax.jit)
def get_ys(latlons, ys, latlon_rows, idx_rows, row_lens, default, cell_size_meters):
    get_y1 = partial(get_y, ys=ys, latlon_rows=latlon_rows, idx_rows=idx_rows, row_lens=row_lens, default=default, cell_size_meters=cell_size_meters)
    ys = jax.vmap(get_y1)(latlons)
    return ys

class Drawer:
    def __init__(self, cellregion, tileloader, min_latlon, max_latlon, default_y=-1.0, p=1024):
        self.p = p
        self.tileloader = tileloader

        with jax.experimental.enable_x64():
            assert str(cellregion.latlon_matrix.dtype) == "float64"
            latlon_rows = jax.device_put(cellregion.latlon_matrix)
            assert str(latlon_rows.dtype) == "float64"
            idx_rows = jax.device_put(cellregion.idx_matrix)
            row_lens = jax.device_put(cellregion.row_lens)
            self.get_ys = partial(jax.jit(partial(get_ys, default=default_y, cell_size_meters=cellregion.cell_size_meters)), latlon_rows=latlon_rows, idx_rows=idx_rows, row_lens=row_lens)

        meters_per_pixel = twm.geo.distance(min_latlon, max_latlon) / p
        zoom = self.tileloader.get_zoom(0.5 * (min_latlon + max_latlon), meters_per_pixel)

        latlon_corners = np.stack([
            [min_latlon[0], min_latlon[1]],
            [min_latlon[0], max_latlon[1]],
            [max_latlon[0], max_latlon[1]],
            [max_latlon[0], min_latlon[1]],
        ])
        tile_corners = np.asarray([self.tileloader.layout.epsg4326_to_tile(latlon, zoom=zoom) for latlon in latlon_corners]).astype("int")
        min_tile = np.min(tile_corners, axis=0) - 1
        max_tile = np.max(tile_corners, axis=0) + 2

        pixel_corners = np.asarray([self.tileloader.layout.epsg4326_to_pixel(latlon, zoom=zoom) for latlon in latlon_corners]).astype("int")
        min_pixel = np.min(pixel_corners, axis=0)
        max_pixel = np.max(pixel_corners, axis=0)

        image = self.tileloader.load(
            min_tile=min_tile,
            max_tile=max_tile,
            zoom=zoom,
        )
        image_min_pixel = np.minimum(
            self.tileloader.layout.tile_to_pixel(min_tile, zoom=zoom).astype("int"),
            self.tileloader.layout.tile_to_pixel(max_tile, zoom=zoom).astype("int"),
        )
        image_max_pixel = image_min_pixel + np.asarray(image.shape[:2])

        crop_front = min_pixel - image_min_pixel
        crop_back = image_max_pixel - max_pixel
        assert np.all(crop_front >= 0) and np.all(crop_back >= 0)
        image = image[crop_front[0]:image.shape[0] - crop_back[0], crop_front[1]:image.shape[1] - crop_back[1]]

        self.min_pixel = min_pixel
        self.image = image
        self.zoom = zoom

        pixel_corners = np.stack([
            [0, 0],
            [0, image.shape[1]],
            [image.shape[0], image.shape[1]],
            [image.shape[0], 0],
        ], axis=0) + self.min_pixel
        self.latlon_corners = np.asarray([self.tileloader.layout.pixel_to_epsg4326(pixel, zoom=self.zoom) for pixel in pixel_corners])

        pixels = einx.rearrange("x, y -> (x y) (1 + 1)", np.arange(self.image.shape[0]), np.arange(self.image.shape[1])) + self.min_pixel[np.newaxis]
        self.image_latlons = self.tileloader.layout.pixel_to_epsg4326(pixels, zoom=self.zoom).astype("float64")

    def __call__(self, ys):
        with jax.experimental.enable_x64():
            ys = jax.device_put(ys)
            chunk_size = 1024 * 8
            ys_image = np.zeros((self.image.shape[0] * self.image.shape[1],) + ys.shape[1:], dtype=ys.dtype)
            is_ = list(range(0, ys_image.shape[0], chunk_size))
            # is_ = tqdm.tqdm(is_, desc="Drawing")
            for i in is_:
                ys_image[i:i + chunk_size] = self.get_ys(self.image_latlons[i:i + chunk_size], ys)
            ys_image = einx.rearrange("(x y) -> x y", ys_image, x=self.image.shape[0], y=self.image.shape[1])
            ys_image = np.asarray(ys_image)

        return ys_image

    def draw_probs(self, probs, gt_latlon=None, crosshair=False, box=None):
        if crosshair == False:
            crosshair = 0.0
        elif crosshair == True:
            crosshair = 1.0

        probs = probs / np.max(probs)
        ys_image = self(probs)

        mask = ys_image >= 0

        colormap = cv2.applyColorMap(np.arange(256).astype("uint8")[np.newaxis], cv2.COLORMAP_JET)[0, :, ::-1]

        ys_image = ((ys_image - 1e-5) * 255).astype("uint8")
        ys_image = colormap[ys_image]
        ys_image = np.where(mask[..., np.newaxis], ys_image, self.image)

        image = np.copy(self.image)
        image[mask] = 0.5 * image[mask] + 0.5 * ys_image[mask]

        # Draw gt position
        if not gt_latlon is None:
            factor = np.sqrt(image.shape[0] * image.shape[1]) / 400

            if crosshair > 0:
                pixel = self.tileloader.layout.epsg4326_to_pixel(gt_latlon, zoom=self.zoom) - self.min_pixel
                pixel = pixel.astype("int")

                r1 = int(60 * factor * crosshair)
                r2 = int(20 * factor * crosshair)
                b = int(2.5 * factor * crosshair)
                color = np.asarray([215, 215, 215])

                def draw(min_y, max_y, min_x, max_x):
                    pixels = einx.arange("(y x) [2]", y=max_y - min_y, x=max_x - min_x) + np.asarray([min_y, min_x])
                    mask = np.all(np.logical_and(0 <= pixels, pixels < np.asarray(image.shape[:2])[np.newaxis, :]), axis=1)
                    pixels = pixels[mask, :]
                    if len(pixels) > 0:
                        image[pixels[:, 0], pixels[:, 1]] = color

                draw(pixel[0] - r1, pixel[0] - r2, pixel[1] - b, pixel[1] + b)
                draw(pixel[0] + r2, pixel[0] + r1, pixel[1] - b, pixel[1] + b)
                draw(pixel[0] - b, pixel[0] + b, pixel[1] - r1, pixel[1] - r2)
                draw(pixel[0] - b, pixel[0] + b, pixel[1] + r2, pixel[1] + r1)
            
            if not box is None:
                pixel_corners = np.asarray([self.tileloader.layout.epsg4326_to_pixel(latlon, zoom=self.zoom) for latlon in box]) - self.min_pixel

                b = 2.5 * factor
                color = np.asarray([215, 215, 215])

                for i in range(4):
                    corner1 = pixel_corners[i]
                    corner2 = pixel_corners[(i + 1) % 4]
                    min_corner, max_corner = np.minimum(corner1, corner2), np.maximum(corner1, corner2)
                    vec = (max_corner - min_corner)
                    vec = vec / np.linalg.norm(vec) * b / 2
                    side = np.asarray([vec[1], -vec[0]])
                    corners = np.stack([
                        min_corner - vec + side,
                        min_corner - vec - side,
                        max_corner + vec - side,
                        max_corner + vec + side,
                    ], axis=0).astype("int")
                    yy, xx = skimage.draw.polygon(corners[:, 0], corners[:, 1])
                    mask = (yy >= 0) & (yy < image.shape[0]) & (xx >= 0) & (xx < image.shape[1])
                    yy, xx = yy[mask], xx[mask]
                    if len(yy) > 0:
                        image[yy, xx] = color

        return np.asarray(image)
