# Small object detection

Insects on a sticky trap, cells on a microscope slide, cars in a satellite scene, cracks on a
wind turbine blade: these objects are often only 10 to 30 pixels wide in an image that is
thousands of pixels wide. Most detectors resize their input to about 640 pixels, and at that
size the objects all but disappear.

<div class="plk-compare" markdown>
<figure markdown>
![A parking lot, blurry after resizing the whole image](assets/figures/why_resized.jpg)
<figcaption>❌ The whole image resized to 640 px: each car is a few pixels.</figcaption>
</figure>
<figure markdown>
![The same parking lot, sharp, from a tile](assets/figures/why_native.jpg)
<figcaption>✅ A 640 px tile at native resolution: every car is clear.</figcaption>
</figure>
</div>

The fix is to **never shrink the image**: train on tiles, run the model on tiles, then put
the detections back on the full image. plakakia does the tiling at every step, with any
model, and the merge is a few lines of NumPy and OpenCV.

```text
1. Train     big images + labels ──► plakakia ──► tiles + labels ──► train
2. Predict   big image ──► tile_image ──► tiles ──► your model ──► boxes per tile
3. Merge     boxes + tile offsets (x1, y1) ──► NMS ──► boxes on the full image
```

## 1. Train on tiles

Tile your labelled dataset once, then train any detector on the tiles as if they were
normal images.

```yaml title="config.yaml"
input_dir_images: data/images
input_dir_annotations: data/labels
input_format_annotations: yolo
tile_size: 640          # your model's input size
step_size: 480          # 25% overlap: objects cut by a border are whole next door
cover_edges: true
min_visibility: 0.6     # keep objects at least 60% inside a tile
save_empty_tiles: true  # background tiles: what is *not* an object
num_workers: -1
```

```bash
plakakia --config config.yaml
```

!!! tip "Choosing the tile size"
    Use the input size of your model, so tiles are not resized again during training.
    Objects should be well under half the tile; if some objects are larger than that,
    also train on downscaled full images, or use a bigger tile.

## 2. Predict on tiles

At inference time, cut the image with the **same** `tile_size`, and run the model on the
tiles as one batch. Any framework works: you only need the boxes, scores and class ids
that it returns for each tile.

=== "Ultralytics YOLO"

    ```python
    import cv2
    from ultralytics import YOLO

    from plakakia import tile_image

    model = YOLO("trained_on_tiles.pt")
    image = cv2.imread("trap_042.jpg")
    tiles, coords = tile_image(image, tile_size=640, step_size=480, cover_edges=True)

    results = model.predict(list(tiles), imgsz=640, conf=0.25, verbose=False)
    per_tile = [
        (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
        for r in results
    ]
    ```

=== "Any model"

    ```python
    import cv2

    from plakakia import tile_image

    image = cv2.imread("trap_042.jpg")
    tiles, coords = tile_image(image, tile_size=640, step_size=480, cover_edges=True)

    # Your detector: one (boxes (N, 4) x1 y1 x2 y2, scores (N,), class_ids (N,)) per tile,
    # in tile pixel coordinates.
    per_tile = [my_detector(tile) for tile in tiles]
    ```

## 3. Merge the detections

Each tile's boxes are in that tile's coordinates. Add the tile's top-left corner to move
them into the full image, then remove duplicates: an object in the overlap between two
tiles is detected twice. Non-maximum suppression (NMS) keeps the best-scoring box of each
group of overlapping boxes, per class.

```python
import cv2
import numpy as np

boxes, scores, classes = [], [], []
for (x1, y1, _, _), (b, s, c) in zip(coords, per_tile):
    boxes.append(np.asarray(b, dtype=np.float32).reshape(-1, 4) + [x1, y1, x1, y1])
    scores.append(np.asarray(s, dtype=np.float32))
    classes.append(np.asarray(c, dtype=np.int32))
boxes, scores, classes = map(np.concatenate, (boxes, scores, classes))

xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]])
keep = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), classes.tolist(), 0.25, 0.5)
boxes, scores, classes = boxes[keep], scores[keep], classes[keep]  # on the full image
```

!!! tip "Objects larger than a tile"
    Add the predictions of the model on the whole image, resized to 640 px, before
    the NMS step. Large objects are found there, and small ones in the tiles.

## On maps

For satellite and drone imagery, read the scene with
`plakakia.io.read_geotiff`. Its profile's `transform` turns pixel
coordinates into map coordinates, so the merged boxes can go straight into QGIS or a
GeoJSON file:

```python
from plakakia.io import read_geotiff

scene, profile = read_geotiff("orthomosaic.tif")  # (bands, H, W)
# ... tile with axes=(1, 2), predict and merge as above ...
corners = [profile.transform * (x, y) for x, y in boxes[:, :2]]  # top-left in map units
```

See [Hyperspectral & geospatial](remote-sensing.md) for tiling GeoTIFFs.

## Checklist

* Same `tile_size` for training and inference.
* 20 to 30% overlap (`step_size` = 0.7 to 0.8 × `tile_size`).
* `cover_edges: true`, so objects at the image border are not missed.
* Some empty tiles in the training set (`save_empty_tiles`).
* NMS after merging, per class.
