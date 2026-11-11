"""plakakia: fast image tiling for computer vision datasets."""

from importlib.metadata import PackageNotFoundError, version

from .annotations import (
    Annotations,
    read_coco,
    read_pascal_voc,
    read_yolo,
    write_pascal_voc,
    write_yolo,
)
from .config import TilingConfig
from .io import pad_image, read_array, read_image, write_array, write_image
from .pipeline import Sample, SampleResult, discover_samples, process_sample, run
from .tiling import (
    assign_boxes_to_tiles,
    box_visibility,
    clip_boxes_to_tile,
    iter_tiles,
    normalize_axes,
    select_unique_tiles,
    tile_array,
    tile_coordinates,
    tile_grid,
    tile_image,
    tile_starts,
    window_slices,
)

try:
    __version__ = version("plakakia")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0"

__all__ = [
    "Annotations",
    "Sample",
    "SampleResult",
    "TilingConfig",
    "__version__",
    "assign_boxes_to_tiles",
    "box_visibility",
    "clip_boxes_to_tile",
    "discover_samples",
    "iter_tiles",
    "normalize_axes",
    "pad_image",
    "process_sample",
    "read_array",
    "read_coco",
    "read_image",
    "read_pascal_voc",
    "read_yolo",
    "run",
    "select_unique_tiles",
    "tile_array",
    "tile_coordinates",
    "tile_grid",
    "tile_image",
    "tile_starts",
    "window_slices",
    "write_array",
    "write_image",
    "write_pascal_voc",
    "write_yolo",
]
