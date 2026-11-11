"""Configuration of the offline tiling pipeline."""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

__all__ = ["InputFormat", "OutputFormat", "TilingConfig"]

InputFormat = Literal["yolo", "pascal_voc", "coco", "segmentation", "none"]
OutputFormat = Literal["yolo", "pascal_voc"]

ANNOTATION_EXTENSIONS: dict[str, str] = {
    "yolo": "txt",
    "pascal_voc": "xml",
    "coco": "json",
    "segmentation": "png",
}

# Keys accepted by plakakia <= 0.2 that no longer have an effect.
_DEPRECATED_KEYS = {
    "validate_settings",
    "log",
    "log_folder",
    "input_extension_annotations",
    "format_to_extension",
    "check_partial",
    "partial_overlap_threshold",
}


def _normalize_ext(ext: str) -> str:
    return ext.strip().lstrip(".").lower()


def _per_axis(value: int | Sequence[int], n: int) -> tuple[int, ...]:
    return (int(value),) * n if isinstance(value, int) else tuple(int(v) for v in value)


@dataclass(frozen=True)
class TilingConfig:
    """Settings for :func:`plakakia.pipeline.run`.

    Every field can be set from a YAML file with the same key, see
    :meth:`from_yaml`. Paths are resolved relative to the current working directory.
    """

    input_dir_images: Path
    """Directory with the input images."""
    input_dir_annotations: Path | None = None
    """Directory with the input annotations (not needed for ``input_format_annotations='none'``)."""
    output_dir_images: Path = Path("output/images")
    """Directory where image tiles are written."""
    output_dir_annotations: Path = Path("output/annotations")
    """Directory where annotation (or mask) tiles are written."""

    input_format_annotations: InputFormat = "yolo"
    """One of ``yolo``, ``pascal_voc``, ``coco``, ``segmentation`` (masks) or ``none``."""
    output_format_annotations: OutputFormat = "yolo"
    """One of ``yolo`` or ``pascal_voc``. Ignored for ``segmentation`` and ``none``."""
    input_extension_images: str = "jpg"
    """Extension of the input images: ``jpg``, ``png``, ``tif``, or ``npy`` / ``h5`` for
    N-D arrays such as hyperspectral cubes."""
    output_extension_images: str = "jpg"
    """Extension (and therefore encoding) of the image tiles. Use ``npy``, ``h5`` or ``tif``
    for data OpenCV cannot encode (many bands, floats)."""
    mask_extension: str = "png"
    """Extension of segmentation masks (``png``, ``npy``, ``tif``...). Mask tiles keep this
    format (JPEG masks are written as PNG)."""
    hdf5_dataset: str | None = None
    """Name of the dataset to read from (and write to) HDF5 files. Optional when each
    input file holds a single dataset."""

    georeference: bool = False
    """Read inputs as GeoTIFFs with ``rasterio`` (``plakakia[geo]``) and write every tile as
    a GeoTIFF with its own affine transform, CRS and nodata value, so tiles stay on the
    map. Rasterio arrays are ``(bands, height, width)``: use ``axes: [1, 2]``."""
    bands: Sequence[int] | None = None
    """Keep only these indices along ``band_axis`` before tiling, e.g. ``[3, 2, 1]`` to
    build an RGB composite from Sentinel-2 bands. ``None`` keeps every band."""
    band_axis: int = -1
    """Axis that ``bands`` selects from (``0`` for channels-first / GeoTIFF data)."""
    nodata: float | None = None
    """Pixel value meaning "no data" (``nan`` allowed). A pixel is nodata when all of its
    bands equal it. Defaults to the GeoTIFF's own nodata value when ``georeference`` is set."""
    max_nodata_fraction: float = 1.0
    """Skip tiles with a larger fraction of nodata pixels, e.g. ``0.5`` to drop tiles that are
    mostly outside a drone orthomosaic or satellite swath. ``1.0`` keeps every tile."""

    axes: Sequence[int] = (0, 1)
    """Array axes to tile. The first two are the ``(y, x)`` axes that boxes and masks refer
    to, e.g. ``[1, 2]`` for channels-first ``(B, H, W)`` cubes. Add more axes to tile them
    too, e.g. ``[0, 1, 2]`` to also split the bands of a ``(H, W, B)`` cube."""
    tile_size: int | Sequence[int] = 200
    """Tile size in pixels: one value for every tiled axis, or one per entry of ``axes``."""
    step_size: int | Sequence[int] = 50
    """Stride of the sliding window: one value, or one per entry of ``axes``."""
    cover_edges: bool = False
    """Add border-aligned tiles so the whole image is covered."""
    pad_image: bool = False
    """Pad every image with a constant border before tiling."""
    pad_size: int = 10
    """Border size in pixels when ``pad_image`` is set."""

    min_visibility: float = 1.0
    """Minimum fraction of a box that must be inside a tile to be kept (``1.0`` = fully inside).
    Partially visible boxes are clipped to the tile."""
    annotation_mapping: Mapping[str, int] = field(default_factory=dict)
    """Class name to id mapping (needed for Pascal VOC inputs with text labels)."""
    save_empty_tiles: bool = False
    """Also save tiles that contain no boxes (useful as background samples)."""
    clear_duplicates: bool = False
    """Drop tiles whose boxes are all already contained in a previously saved tile."""
    draw_boxes: bool = False
    """Draw the boxes on the saved tiles (for visual inspection only)."""
    num_workers: int = 1
    """Number of worker processes; ``-1`` uses all CPU cores."""

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "input_dir_images", Path(self.input_dir_images))
        set_(self, "output_dir_images", Path(self.output_dir_images))
        set_(self, "output_dir_annotations", Path(self.output_dir_annotations))
        if self.input_dir_annotations is not None:
            set_(self, "input_dir_annotations", Path(self.input_dir_annotations))
        set_(self, "input_extension_images", _normalize_ext(self.input_extension_images))
        set_(self, "output_extension_images", _normalize_ext(self.output_extension_images))
        set_(self, "mask_extension", _normalize_ext(self.mask_extension))
        set_(self, "annotation_mapping", dict(self.annotation_mapping or {}))
        set_(self, "axes", tuple(int(a) for a in self.axes))
        n = len(self.axes)
        if n < 2 or len(set(self.axes)) != n:
            raise ValueError(f"axes must list at least two distinct axes, got {list(self.axes)}.")
        for name in ("tile_size", "step_size"):
            value = getattr(self, name)
            values = (int(value),) * n if isinstance(value, int) else tuple(int(v) for v in value)
            if len(values) != n:
                raise ValueError(f"{name} needs one value or {n} values (one per axis).")
            if any(v <= 0 for v in values):
                raise ValueError("tile_size and step_size must be positive.")
            set_(self, name, value if isinstance(value, int) else values)
        if self.num_workers in (-1, 0):
            set_(self, "num_workers", os.cpu_count() or 1)

        if self.input_format_annotations not in get_args(InputFormat):
            raise ValueError(
                f"input_format_annotations must be one of {get_args(InputFormat)}, "
                f"got {self.input_format_annotations!r}."
            )
        if self.output_format_annotations not in get_args(OutputFormat):
            raise ValueError(
                f"output_format_annotations must be one of {get_args(OutputFormat)}, "
                f"got {self.output_format_annotations!r}."
            )
        if self.input_format_annotations != "none" and self.input_dir_annotations is None:
            raise ValueError("input_dir_annotations is required unless input_format is 'none'.")
        if self.bands is not None:
            set_(self, "bands", tuple(int(b) for b in self.bands))
            if not self.bands:
                raise ValueError("bands must not be empty.")
        if self.nodata is not None:
            set_(self, "nodata", float(self.nodata))
        if not 0.0 <= self.max_nodata_fraction <= 1.0:
            raise ValueError("max_nodata_fraction must be in [0, 1].")
        if self.max_nodata_fraction < 1.0 and self.nodata is None and not self.georeference:
            raise ValueError("max_nodata_fraction needs a `nodata` value (or georeference).")
        if self.georeference:
            if tuple(self.axes[:2]) != (1, 2):
                raise ValueError(
                    "georeference reads (bands, height, width) arrays: set axes to [1, 2] "
                    "(optionally followed by 0 to tile the bands too)."
                )
            if self.output_extension_images not in {"tif", "tiff"}:
                raise ValueError("georeference requires output_extension_images: tif.")
        if self.draw_boxes and self.axes != (0, 1):
            raise ValueError("draw_boxes is only supported for images tiled with axes [0, 1].")
        if self.pad_size < 0:
            raise ValueError("pad_size must be non-negative.")
        if not 0.0 < self.min_visibility <= 1.0:
            raise ValueError("min_visibility must be in (0, 1].")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1 (or -1 for all cores).")

    @property
    def tile_shape(self) -> tuple[int, ...]:
        """Tile size for each entry of :attr:`axes`."""
        return _per_axis(self.tile_size, len(self.axes))

    @property
    def step_shape(self) -> tuple[int, ...]:
        """Step size for each entry of :attr:`axes`."""
        return _per_axis(self.step_size, len(self.axes))

    @property
    def output_extension_annotations(self) -> str:
        """File extension of the written annotation tiles."""
        if self.input_format_annotations == "segmentation":
            # Lossy formats would corrupt class ids.
            return "png" if self.mask_extension in {"jpg", "jpeg"} else self.mask_extension
        return ANNOTATION_EXTENSIONS[self.output_format_annotations]

    @property
    def label_names(self) -> dict[int, str]:
        """Inverse of :attr:`annotation_mapping`."""
        return {v: k for k, v in self.annotation_mapping.items()}

    def replace(self, **changes: Any) -> TilingConfig:
        """Return a copy with some fields replaced."""
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TilingConfig:
        """Build a config from a mapping, handling keys from older plakakia versions."""
        data = dict(data)
        if data.pop("check_partial", False):
            data.setdefault("min_visibility", data.get("partial_overlap_threshold", 0.8))
        deprecated = sorted(_DEPRECATED_KEYS & data.keys())
        if deprecated:
            warnings.warn(
                f"Ignoring deprecated config keys: {', '.join(deprecated)}.",
                DeprecationWarning,
                stacklevel=2,
            )
        for key in deprecated:
            data.pop(key)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(data.keys() - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}. Valid keys: {sorted(known)}.")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TilingConfig:
        """Load a config from a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: expected a mapping at the top level.")
        return cls.from_dict(data)
