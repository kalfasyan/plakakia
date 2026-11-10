# Offline tiling (CLI)

The `plakakia` command tiles a whole dataset (images plus labels or masks) to disk
using all your CPU cores.

```bash
plakakia --config config.yaml            # or: python -m plakakia --config ...
plakakia --config config.yaml -j 8 -q    # 8 workers, no progress bar
```

`make_tiles` is kept as an alias for backwards compatibility.

## Example config

```yaml
--8<-- "examples/config.yaml"
```

## Reference

All keys map one-to-one onto [`TilingConfig`][plakakia.config.TilingConfig].

| Key | Default | Description |
| --- | --- | --- |
| `input_dir_images` | *required* | Directory with the input images. |
| `input_dir_annotations` | *required\** | Directory with labels (\*not needed for `none`). |
| `input_format_annotations` | `yolo` | `yolo`, `pascal_voc`, `coco`, `segmentation` or `none`. |
| `input_extension_images` | `jpg` | Extension of the input images. |
| `output_dir_images` | `output/images` | Where image tiles are written. |
| `output_dir_annotations` | `output/annotations` | Where label/mask tiles are written. |
| `output_format_annotations` | `yolo` | `yolo` or `pascal_voc`. |
| `output_extension_images` | `jpg` | Encoding of the image tiles. |
| `tile_size` | `200` | Tile side in pixels. |
| `step_size` | `50` | Sliding-window stride in pixels. |
| `cover_edges` | `false` | Add border-aligned tiles so no pixels are dropped. |
| `pad_image` / `pad_size` | `false` / `10` | Pad images (and shift boxes) before tiling. |
| `min_visibility` | `1.0` | Minimum visible fraction of a box to keep it in a tile. |
| `save_empty_tiles` | `false` | Also save tiles without boxes. |
| `clear_duplicates` | `false` | Skip tiles whose boxes all appear in an earlier tile. |
| `annotation_mapping` | `{}` | Class name to id mapping (Pascal VOC text labels). |
| `draw_boxes` | `false` | Draw boxes on the saved tiles, for inspection. |
| `num_workers` | `1` | Worker processes; `-1` uses all cores. |

## Input layout

| Format | Expected files |
| --- | --- |
| `yolo` | `labels/<image stem>.txt` |
| `pascal_voc` | `labels/<image stem>.xml` |
| `coco` | exactly one `*.json` in `input_dir_annotations` |
| `segmentation` | `masks/<image stem>.png` (same size as the image) |
| `none` | no labels, every tile is saved |

Images without a label file are treated as having no objects.

## Outputs

Each tile is named after its source image and its coordinates in the (padded) image:

```text
output/images/tile_<stem>_<x1>_<y1>_<x2>_<y2>.jpg
output/annotations/tile_<stem>_<x1>_<y1>_<x2>_<y2>.txt   # or .xml
output/annotations/mask_<stem>_<x1>_<y1>_<x2>_<y2>.png   # segmentation
```

Masks are always written as PNG, because lossy formats would corrupt class ids.
