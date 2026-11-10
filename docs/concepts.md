# Concepts

## Coordinates

Boxes and tiles are `(x1, y1, x2, y2)` in pixels with **exclusive** `x2`/`y2`, i.e.
exactly NumPy slicing: `image[y1:y2, x1:x2]`. Tiles are ordered row-major.

## The tiling grid

Along each axis, tiles start at `0, step, 2*step, ...` as long as they fit. When
`(size - tile_size)` is not a multiple of `step_size`, the remainder at the right and bottom
edges is dropped, unless `cover_edges=True`, which adds one extra tile aligned to the border:

```text
size=11, tile=4, step=3                 cover_edges=True
|0123|                                  |0123|
   |3456|                                  |3456|
      |6789|   (pixel 10 dropped)             |6789|
                                               |789A|
```

Images smaller than `tile_size` produce no tiles (the pipeline logs and skips them).
Use `pad_image` to tile images that are only slightly smaller.

## Box assignment

[`assign_boxes_to_tiles`][plakakia.tiling.assign_boxes_to_tiles] returns a `(tiles, boxes)`
boolean matrix, computed in one vectorised step.

* `min_visibility=1.0` (default) keeps a box only in tiles that fully contain it.
* `min_visibility < 1` also keeps boxes whose visible fraction is at least that value.
  They are clipped to the tile by [`clip_boxes_to_tile`][plakakia.tiling.clip_boxes_to_tile].

## Duplicate removal

With overlapping tiles, the same object appears in several tiles. `clear_duplicates`
visits tiles in order and keeps a tile only if it contains at least one box that no
previously kept tile contains
([`select_unique_tiles`][plakakia.tiling.select_unique_tiles]). Every box therefore survives
in at least one tile, with far fewer tiles overall. Deduplication happens per image.
