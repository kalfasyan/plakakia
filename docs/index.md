# plakakia

*πλακάκια, Greek for "tiles"*

**plakakia** splits images into (overlapping) square tiles and carries their labels
along: bounding boxes in YOLO, Pascal VOC or COCO format, or segmentation masks.
It is the preprocessing step you need for training detectors on large images
(aerial, satellite, microscopy, industrial inspection...) at native resolution.

<div class="grid" markdown>

- **Fast**: vectorised NumPy tiling (`sliding_window_view`, one gather per image),
  vectorised box-to-tile assignment, and a multiprocess pipeline.
- **Small**: only `numpy`, `opencv-python-headless`, `pyyaml` and `tqdm`.
- **Any data, any axes**: grayscale, RGB(A), 16-bit, multispectral and hyperspectral
  cubes (`.npy`, HDF5, TIFF), channels-first or last, tiled along the axes you choose.
- **Geospatial**: satellite and drone GeoTIFF tiles keep their CRS and transform, and
  mostly-empty tiles can be skipped. See [Hyperspectral & geospatial](remote-sensing.md).
- **Typed and tested**: fully type-annotated (`py.typed`) and tested on Linux, macOS and
  Windows.

</div>

```python
import numpy as np
from plakakia import tile_image

image = np.zeros((1080, 1920, 3), dtype=np.uint8)
tiles, coords = tile_image(image, tile_size=512, step_size=384, cover_edges=True)
tiles.shape  # (15, 512, 512, 3)
coords[:2]  # [[0, 0, 512, 512], [384, 0, 896, 512]]  (x1, y1, x2, y2)
```

Head to [Getting started](getting-started.md).
