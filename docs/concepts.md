# How tiling works

This page explains every idea behind plakakia with a picture. No prior computer-vision
background is needed. Each section names the option that controls it, so you can go
straight to the code.

## Try it

Drag the sliders to tile a 2400 × 1500 aerial image. Red areas are pixels that no tile
covers.

<div class="plk-playground" data-src="../assets/figures/scene.jpg" data-width="2400" data-height="1500">
<canvas width="960" height="600" aria-label="Tiling preview"></canvas>
<div class="plk-controls">
<label><span><code>tile_size</code> = <b class="v-tile">640</b> px</span><input type="range" name="tile" min="128" max="1500" step="16" value="640"></label>
<label><span><code>step_size</code> = <b class="v-step">480</b> px</span><input type="range" name="step" min="64" max="1600" step="16" value="480"></label>
<label class="plk-check"><input type="checkbox" name="cover" checked><code>cover_edges</code></label>
</div>
<div class="plk-stats">
<span>Tiles: <b class="s-n">15</b></span>
<span>Overlap between neighbours: <b class="s-overlap">25%</b></span>
<span>Pixels dropped: <b class="s-dropped">0.0%</b></span>
</div>
<pre><code class="s-code">tiles, coords = tile_image(image, tile_size=640, step_size=480,
                           cover_edges=True)</code></pre>
</div>

Things to try:

* Set `step_size` equal to `tile_size`: the tiles touch but do not overlap.
* Untick `cover_edges` with a large tile: a red strip appears at the right or bottom.
* Make `step_size` larger than `tile_size`: gaps appear between tiles.

## Tiles, tile size and step size

A **tile** is a rectangular window cut from the image. plakakia places the first tile in
the top-left corner, then slides the window to the right by `step_size` pixels, again and
again, then moves down by `step_size` and starts a new row. This is called a
**sliding window**, and `step_size` is also known as the **stride**.

<figure class="plk-fig">
--8<-- "docs/assets/figures/tile_step.svg"
<figcaption>Three tiles of 6 pixels with a step of 4. Neighbouring tiles share
2 pixels: that is the <b>overlap</b>.</figcaption>
</figure>

**Why overlap?** An object that is cut in half by one tile's border sits whole inside the
next tile. A good starting point is a `step_size` of 70 to 80% of `tile_size`. Overlap has
a cost, though: more tiles, and the same object appears in several of them.

`tile_size` and `step_size` can also be pairs, `(height, width)`, for rectangular tiles.

## Coordinates

Every tile is described by four numbers, `(x1, y1, x2, y2)`: the top-left and the
bottom-right corners, in pixels. They follow NumPy, so you can cut a tile out of the image
yourself.

<figure class="plk-fig">
--8<-- "docs/assets/figures/coordinates.svg"
</figure>

Tiles are numbered row by row, from the top-left. Boxes use the same convention.

## The edges

Images rarely divide into whole tiles. When the last tile would stick out of the image,
plakakia stops before it, and the pixels at the right and bottom border stay uncovered.
Set `cover_edges=True` to add one more tile, pushed back so it ends exactly at the border.

<figure class="plk-fig">
--8<-- "docs/assets/figures/cover_edges.svg"
<figcaption>An 11-pixel row, tiles of 4, step 3. Each bar is one tile.</figcaption>
</figure>

Images smaller than `tile_size` produce no tiles at all. Use `pad_image` to add a border
to images that are only slightly too small.

## Boxes at the tile border

A **bounding box** is the rectangle around an object, stored as `(x1, y1, x2, y2)` like a
tile. When an image is tiled, each box must be given to the right tiles, and moved into
the tile's own coordinates.

Boxes that are fully inside a tile are easy. Boxes cut by a tile border are controlled by
`min_visibility`, the fraction of the box's area that must be inside the tile:

<figure class="plk-fig">
--8<-- "docs/assets/figures/min_visibility.svg"
<figcaption>With 1.0 a box must be entirely inside. Lower values keep partly visible
objects, cut at the tile border.</figcaption>
</figure>

In Python, [`assign_boxes_to_tiles`][plakakia.tiling.assign_boxes_to_tiles] returns a
`(tiles, boxes)` table of booleans that answers "is box *j* in tile *i*?" for every pair
at once, and [`clip_boxes_to_tile`][plakakia.tiling.clip_boxes_to_tile] moves the boxes
into tile coordinates.

!!! tip "Choosing `min_visibility`"
    For training, `0.5` to `0.7` teaches the model to recognise partly visible objects.
    Keep `1.0` if half-objects would confuse it, and rely on overlap so that every
    object is whole in at least one tile.

## Removing duplicate tiles

With overlap, one object often lands in two or more tiles. For training, that can mean
a lot of near-identical tiles. Set `clear_duplicates=True` to keep a tile only if it holds
at least one object that no earlier tile already holds.

<figure class="plk-fig">
--8<-- "docs/assets/figures/duplicates.svg"
</figure>

Every object still appears in at least one tile, with far fewer tiles overall. In Python
this is [`select_unique_tiles`][plakakia.tiling.select_unique_tiles].

## Segmentation masks

A **mask** is an image of the same size as the photo, where each pixel holds a class id
(0 for background, 1 for road, 2 for building...). Masks are tiled with exactly the same
windows as the image, so pixel `(y, x)` of a tile always matches pixel `(y, x)` of its mask
tile. Set `input_format_annotations: segmentation` in the [CLI config](cli.md).

## More than three channels

A normal photo has three channels (blue, green, red). Satellite and hyperspectral images
can have dozens or hundreds of **bands**, one per wavelength, forming a **cube**.
plakakia tiles any axes you choose and leaves the others whole:

<figure class="plk-fig">
--8<-- "docs/assets/figures/axes.svg"
<figcaption>Left: spatial tiles that keep every band. Right: windows that are also cut
along the bands.</figcaption>
</figure>

[`tile_array`][plakakia.tiling.tile_array] does this for any number of dimensions, and
[`tile_image`][plakakia.tiling.tile_image] is its 2-D shortcut. The
[hyperspectral guide](remote-sensing.md) has the details.

## Glossary

| Term | Meaning |
| --- | --- |
| Tile | A rectangular window cut from an image. Also called a patch or crop. |
| `tile_size` | Width and height of a tile, in pixels. |
| `step_size`, stride | Distance between the starts of two neighbouring tiles. |
| Overlap | Pixels shared by neighbouring tiles: `tile_size - step_size`. |
| Bounding box | Rectangle around an object, `(x1, y1, x2, y2)`. |
| YOLO, Pascal VOC, COCO | Common file formats for bounding-box labels. |
| Mask | Per-pixel class image for segmentation. |
| Band, channel | One layer of an image: red, green, blue, or one wavelength. |
| Online tiling | Tiling in memory, inside your Python code (`tile_image`). |
| Offline tiling | Tiling a dataset on disk ahead of training (`plakakia --config`). |
