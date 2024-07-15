Image ids are stored in [delta encoded](https://en.wikipedia.org/wiki/Delta_encoding) format to reduce the size to less than 100MB per file. The following code can be used to read and decode the image ids:

```python
import numpy as np
with np.load("PATH_TO_IMAGEID_FILE.npz") as f:
    image_ids = f["image_ids"]
image_ids = np.cumsum(image_ids) # Decode delta encoding
```

A file containing image ids can be passed to the [``download_mapillary.py``](../download_mapillary.py) script to download the respective images. For example:

```bash
python download_mapillary.py --path PATH_TO_DOWNLOAD_FOLDER --width 640 --height 480 --token "MAPILLARY_API_TOKEN" --presize 1024 --image-ids massgis.npz
```