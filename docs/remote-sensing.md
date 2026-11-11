# Hyperspectral, multispectral and geospatial data

plakakia tiles arrays with any number of dimensions, along any axes you choose. You can
use it on satellite scenes (Sentinel-2, Landsat), drone orthomosaics, airborne or lab
hyperspectral cubes, and other multi-band rasters.

Install the optional readers:

```bash
uv add "plakakia[hyperspectral,geo]"   # h5py + tifffile, and rasterio
```

## Choosing the axes

A hyperspectral **cube** is a stack of images, one per wavelength (**band**). You decide
which axes are cut into tiles. Everything else stays whole:

<figure class="plk-fig">
--8<-- "docs/assets/figures/axes.svg"
<figcaption>Left: spatial tiles that keep every band. Right: windows cut along the
bands too, for example 32 bands at a time.</figcaption>
</figure>

Cubes come in different layouts. Tell plakakia which axes are spatial, and any other
axes to tile as well:

| Data | Layout | `axes` |
| --- | --- | --- |
| Images, most `.npy` / HDF5 cubes | `(H, W, B)` | `(0, 1)` (default) |
| GeoTIFF (rasterio), PyTorch, ENVI BSQ | `(B, H, W)` | `(1, 2)` |
| Spatial **and** spectral windows | `(H, W, B)` | `(0, 1, 2)` |
| Spatial and spectral, channels-first | `(B, H, W)` | `(1, 2, 0)` |

```python
import numpy as np
from plakakia import iter_tiles, tile_array, tile_image

cube = np.load("scene.npy", mmap_mode="r")  # (224 bands, 2000, 3000), memory-mapped

# Spatial tiles keeping every band: (T, 224, 256, 256)
tiles, coords = tile_image(cube, tile_size=256, step_size=192, axes=(1, 2), cover_edges=True)

# Spatial x spectral windows: 256x256 pixels, 32 bands, stride 16 bands
tiles, starts = tile_array(cube, (256, 256, 32), (256, 256, 16), axes=(1, 2, 0))
# starts[i] = (y, x, band) start of tile i

# Larger-than-memory scenes: zero-copy views, only touched tiles are read from disk
for (y, x), view in iter_tiles(cube, (512, 512), axes=(1, 2)):
    ...
```

`tile_size` and `step_size` take one value for every axis, or one per axis (which also
gives rectangular tiles).

## File formats

| Extension | Reader | Notes |
| --- | --- | --- |
| `.npy` | NumPy | Memory-mapped on read. |
| `.h5`, `.hdf5`, `.he5` | h5py | Set `hdf5_dataset` when a file holds several datasets. Files are closed after reading. |
| `.tif`, `.tiff` | tifffile (fallback: OpenCV) | Multi-band TIFFs of any dtype. |
| GeoTIFF with `georeference: true` | rasterio | Keeps CRS, transform and nodata. Arrays are `(B, H, W)`. |
| `.png`, `.jpg`, ... | OpenCV | 1, 3 or 4 channels. |

Tiles are written in `output_extension_images`. OpenCV formats cannot hold many bands or
floating-point data, so use `npy`, `h5` or `tif` for cubes.

## Satellite and drone imagery (GeoTIFF)

With `georeference: true`, every tile is written as a GeoTIFF with its own affine
transform, the source CRS and nodata value. Tiles open in QGIS exactly where they belong,
and predictions on tiles can be mapped back to world coordinates.

```yaml
--8<-- "examples/config_geotiff.yaml"
```

* `max_nodata_fraction` drops tiles that are mostly empty, such as the black margins of
  an orthomosaic or the area outside a satellite swath. A pixel counts as nodata when all
  of its bands equal `nodata` (`nan` works too). The fraction is computed for all tiles
  at once with a summed-area table.
* `bands` with `band_axis` picks or reorders bands before tiling, for example `[3, 2, 1]`
  for a Sentinel-2 RGB composite or a subset of hyperspectral bands.
* Boxes (YOLO, Pascal VOC, COCO) and masks refer to the spatial axes and are tiled as
  usual.

## Hyperspectral cubes

```yaml
--8<-- "examples/config_hyperspectral.yaml"
```

When extra axes are tiled, each spatial tile produces one file per spectral window, named
`tile_<stem>_<x1>_<y1>_<x2>_<y2>_ax<axis>-<start>-<stop>`. Every one of those files gets
the spatial tile's labels.
