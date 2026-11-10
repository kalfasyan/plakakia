# Getting started

## Installation

```bash
uv add plakakia        # in a uv-managed project
pip install plakakia   # or with pip
```

Python 3.10+ is required. plakakia depends on `opencv-python-headless`; if you also need
OpenCV's GUI functions (`cv2.imshow`), install `opencv-python` in the same environment.

## Online tiling

Use [`tile_image`][plakakia.tiling.tile_image] on any array:

```python
import cv2
from plakakia import tile_image

image = cv2.imread("scene.jpg")
tiles, coords = tile_image(image, tile_size=640, step_size=480, cover_edges=True)

for tile, (x1, y1, x2, y2) in zip(tiles, coords):
    ...  # run a model, save, inspect
```

The returned `tiles` array is a contiguous copy, so tiles can be modified independently.
When you only need the windows (e.g. to stream very large images), use
[`tile_coordinates`][plakakia.tiling.tile_coordinates] and slice the image yourself;
that allocates nothing:

```python
from plakakia import tile_coordinates

for x1, y1, x2, y2 in tile_coordinates(*image.shape[:2], tile_size=640, step_size=480):
    window = image[y1:y2, x1:x2]  # a view, no copy
```

### Tiling boxes

```python
from plakakia import assign_boxes_to_tiles, clip_boxes_to_tile, read_yolo

ann = read_yolo("scene.txt", image_size=image.shape[:2])
assignment = assign_boxes_to_tiles(ann.boxes, coords, min_visibility=0.6)  # (tiles, boxes)

for t, tile in enumerate(tiles):
    idx = assignment[t].nonzero()[0]
    local_boxes = clip_boxes_to_tile(ann.boxes[idx], coords[t])
    labels = ann.labels[idx]
```

## Offline tiling of a dataset

Write a config file and run the CLI; see [Offline tiling](cli.md).

```bash
plakakia --config config.yaml
```

## Interactive demo

```bash
uv run --extra demo streamlit run demo/app.py
```

![demo](assets/demo.png){ width="420" }
