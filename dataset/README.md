# Download instructions for the dataset

The dataset contains street-view images from

- Mapillary: https://www.mapillary.com (licensed under [CC-BY-SA](https://help.mapillary.com/hc/en-us/articles/115001770409-Licenses) as of July 15, 2024)

and aerial imagery from

- Massachuetts: https://www.mass.gov/orgs/massgis-bureau-of-geographic-information ([in public domain](https://www.mass.gov/info-details/massgis-frequently-asked-questions) as of July 15, 2024, see [this discussion](https://wiki.openstreetmap.org/wiki/MassGIS#Right_to_Use) for more detail)
- Washington DC: https://opendata.dc.gov (licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as of July 15, 2024)
- North Carolina: https://www.nconemap.gov (licensed under [custom license](https://www.nconemap.gov/pages/terms) with "free and unrestricted use policy" as of July 15, 2024)
- Berlin and Brandenburg: https://data.geobasis-bb.de (licensed under [DL-DE->BY-2.0](https://www.govdata.de/dl-de/by-2-0) as of July 15, 2024)
- NRW: https://www.opengeodata.nrw.de/produkte (licensed under [DL-DE->Zero-2.0](https://www.govdata.de/dl-de/zero-2-0) as of July 15, 2024)
- Saxony: https://www.geodaten.sachsen.de (licensed under [DL-DE->BY-2.0](https://www.govdata.de/dl-de/by-2-0) as of July 15, 2024)

This document describes the necessary steps to download street-view and aerial data from these sources.

**License:** The data are licensed under the original licenses from their publishers. The above list contains links to the licenses that were stated on the platforms' websites on July 15, 2024.

**Reproducibility:** The data are hosted by Mapillary and the state's orthophoto providers, and are subject to changes from these platforms. Users on Mapillary can remove their images from the service, and orthophoto providers update the aerial imagery from time to time and in some cases provide only the latest imagery. We choose [a test region where aerial imagery from 2021 is kept available](https://www.mass.gov/info-details/massgis-data-2021-aerial-imagery), and provide the list of Mapillary street-view image IDs that we used from this region.

## 1. Download aerial imagery

1. Install [tiledwebmaps](https://github.com/fferflo/tiledwebmaps): ``pip install tiledwebmaps[scripts]`` (includes dependencies for running the download scripts)
2. Download the imagery for each state using [these download scripts](https://github.com/fferflo/tiledwebmaps/tree/master/python/scripts):
   ```bash
   # Example for download_massgis, replace with download_{openbb|opendc|opennrw|opensaxony|nconemap} for other states

   # 1. Download script
   wget https://github.com/fferflo/tiledwebmaps/blob/master/python/scripts/download_massgis.py

   # 2. Run script
   python download_massgis.py --path PATH_TO_DOWNLOAD_FOLDER --shape TILESIZE
   ```
   The aerial imagery is downloaded to ``PATH_TO_DOWNLOAD_FOLDER`` and stored as tiles with size ``TILESIZE x TILESIZE``. The data is provided by the services as large tiles (e.g. with size 10000x10000 from OpenNRW). We split the images into smaller tiles on disk to speed up the loading of patches on-the-fly later on. We use ``--shape 250``.
3. Load an aerial image patch with arbitrary position, orientation, size and pixel resolution:
   ```python
   import tiledwebmaps as twm
   tileloader = twm.from_yaml("PATH_TO_DOWNLOAD_FOLDER")
   image = tileloader.load(
         latlon=(42.360995, -71.051685), # Latitude and longitude of the center of the image
         bearing=0.0, # Bearing pointing upwards in the image
         meters_per_pixel=0.5, # Metric size per pixel
         shape=(512, 512), # Size in pixels
   )
   ```
   Aerial imagery from Massachusetts is [split into two partially overlapping regions based on the UTM18 and UTM19 map projections](https://www.mass.gov/info-details/massgis-data-2021-aerial-imagery) and must be loaded with:
   ```python
   tileloader = twm.from_yaml("PATH_TO_DOWNLOAD_FOLDER/utm18") # Western part of Massachusetts
   # or
   tileloader = twm.from_yaml("PATH_TO_DOWNLOAD_FOLDER/utm19") # Eastern part of Massachusetts
   ```

## 2. Download street-view images

1. [Create an account on Mapillary and get an access token.](https://www.mapillary.com/developer)
2. Install dependencies from [``requirements.txt``](requirements.txt): ``pip install -r requirements.txt``
3. Download a selection of street-view images using [``download_mapillary.py``](download_mapillary.py):
   ```bash
   python download_mapillary.py --path PATH_TO_DOWNLOAD_FOLDER --width 640 --height 480 --token "MAPILLARY_API_TOKEN" --presize 1024 {--geojson ...|--tiles ...|--image-ids ...}
   ```
   The images are resized such that the width is at most ``width`` and the height is at most ``height`` while retaining their aspect ratio. The ``--presize {256|1024|2048|original}`` argument specifies the original resolution that images are downloaded with before resizing.

   The set of street-view images that will be downloaded can be specified using one of three options:
   1. *Download all images from a geographical region defined by a geojson file.* A geojson file can for example be created using [geojson.io](https://geojson.io/). [``boston100.geojson``](boston100.geojson) defines the region around Boston that we use for our ablation studies.
      ```bash
      --geojson PATH_TO_GEOJSON_FILE
      ```
   2. *Download all images from a geographical region for which aerial image tiles are available.* Multiple tile folders can be given, for example to download images for the UTM18 and UTM19 zones in Massachusetts jointly.
      ```bash
      --tiles PATH_TO_TILEFOLDER1 PATH_TO_TILEFOLDER2 ...
      ```
   3. *Download images using a predefined list of image IDs.* The image IDs used in our work are provided [here](imageids).
      ```bash
      --image-ids PATH_TO_IMAGEID_FILE.npz
      ```

## Issues

Feel free to open an issue in this Github repository if you have any problems downloading or accessing the data.