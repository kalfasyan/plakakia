"""Configuration of the offline tiling pipeline."""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Mapping
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
    """One of ``yolo``, ``pascal_voc``, ``coco``, ``segmentation`` (PNG masks) or ``none``."""
    output_format_annotations: OutputFormat = "yolo"
    """One of ``yolo`` or ``pascal_voc``. Ignored for ``segmentation`` and ``none``."""
    input_extension_images: str = "jpg"
    """Extension of the input images (``jpg``, ``png``, ``tif``, ...)."""
    output_extension_images: str = "jpg"
    """Extension (and therefore encoding) of the image tiles."""

    tile_size: int = 200
    """Side of the square tiles, in pixels."""
    step_size: int = 50
    """Stride of the sliding window, in pixels."""
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
        set_(self, "annotation_mapping", dict(self.annotation_mapping or {}))
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
        if self.tile_size <= 0 or self.step_size <= 0:
            raise ValueError("tile_size and step_size must be positive.")
        if self.pad_size < 0:
            raise ValueError("pad_size must be non-negative.")
        if not 0.0 < self.min_visibility <= 1.0:
            raise ValueError("min_visibility must be in (0, 1].")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1 (or -1 for all cores).")

    @property
    def output_extension_annotations(self) -> str:
        """File extension of the written annotation tiles."""
        if self.input_format_annotations == "segmentation":
            return "png"
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
