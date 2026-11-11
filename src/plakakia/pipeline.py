"""Offline tiling of whole datasets (images + annotations) to disk."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .annotations import (
    Annotations,
    read_coco,
    read_pascal_voc,
    read_yolo,
    write_pascal_voc,
    write_yolo,
)
from .config import ANNOTATION_EXTENSIONS, TilingConfig
from .io import GeoProfile, pad_image, read_array, read_geotiff, write_array, write_geotiff
from .tiling import (
    assign_boxes_to_tiles,
    clip_boxes_to_tile,
    normalize_axes,
    select_unique_tiles,
    tile_coordinates,
    tile_grid,
    window_slices,
)

__all__ = [
    "Sample",
    "SampleResult",
    "discover_samples",
    "nodata_fraction",
    "process_sample",
    "run",
]

logger = logging.getLogger(__name__)

BOX_COLOR = (0, 255, 255)  # yellow in BGR


@dataclass(frozen=True)
class Sample:
    """One input image and where its labels come from."""

    image_path: Path
    annotation_path: Path | None = None
    annotations: Annotations | None = None
    """Pre-loaded annotations (used for COCO, where all labels live in one file)."""


@dataclass(frozen=True)
class SampleResult:
    """Outcome of tiling one image."""

    image_path: Path
    num_tiles: int = 0
    num_boxes: int = 0
    skipped: str | None = None


# --------------------------------------------------------------------------- discovery


def _find_single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern!r} file in {directory}, found {len(matches)}."
        )
    return matches[0]


def discover_samples(config: TilingConfig) -> list[Sample]:
    """List the images of a dataset and pair them with their annotations."""
    img_dir = config.input_dir_images
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {img_dir}")
    suffix = f".{config.input_extension_images}"
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix)
    if not images:
        raise FileNotFoundError(f"No '*{suffix}' images found in {img_dir}.")

    fmt = config.input_format_annotations
    if fmt == "none":
        return [Sample(p) for p in images]

    ann_dir = config.input_dir_annotations
    assert ann_dir is not None  # guaranteed by TilingConfig validation
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {ann_dir}")

    if fmt == "coco":
        json_path = _find_single(ann_dir, "*.json")
        logger.info("Reading COCO annotations from %s", json_path)
        coco = read_coco(json_path)
        unlisted = [p.name for p in images if p.name not in coco]
        if unlisted:
            logger.warning("%d images are not listed in %s.", len(unlisted), json_path.name)
        return [Sample(p, annotations=coco.get(p.name, Annotations.empty())) for p in images]

    ext = config.mask_extension if fmt == "segmentation" else ANNOTATION_EXTENSIONS[fmt]
    samples = [Sample(p, ann_dir / f"{p.stem}.{ext}") for p in images]
    missing = [s for s in samples if s.annotation_path and not s.annotation_path.is_file()]
    if fmt == "segmentation" and missing:
        raise FileNotFoundError(
            f"{len(missing)} images have no mask, e.g. {missing[0].annotation_path}."
        )
    if len(missing) == len(samples):
        raise FileNotFoundError(f"No '*.{ext}' annotations matching the images found in {ann_dir}.")
    if missing:
        logger.warning("%d images have no annotation file; treating them as empty.", len(missing))
    return samples


# --------------------------------------------------------------------------- processing


def _load_annotations(sample: Sample, config: TilingConfig, height: int, width: int) -> Annotations:
    if sample.annotations is not None:
        return sample.annotations
    path = sample.annotation_path
    if path is None or not path.is_file():
        return Annotations.empty()
    if config.input_format_annotations == "yolo":
        return read_yolo(path, (height, width))
    return read_pascal_voc(path, config.annotation_mapping)


def _draw(tile: np.ndarray, ann: Annotations) -> np.ndarray:
    tile = np.ascontiguousarray(tile).copy()
    for (x1, y1, x2, y2), label in zip(ann.boxes.tolist(), ann.labels.tolist(), strict=True):
        cv2.rectangle(tile, (x1, y1), (x2 - 1, y2 - 1), BOX_COLOR, 1)
        cv2.putText(
            tile, str(label), (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1
        )
    return tile


def _tile_name(prefix: str, stem: str, coord: np.ndarray) -> str:
    x1, y1, x2, y2 = coord.tolist()
    return f"{prefix}_{stem}_{x1}_{y1}_{x2}_{y2}"


def nodata_fraction(
    image: np.ndarray,
    nodata: float,
    coords: np.ndarray,
    spatial_axes: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Fraction of nodata pixels in each tile, via a summed-area table (O(H*W + T)).

    A pixel is nodata when every value along the non-spatial axes equals ``nodata``
    (``nan`` matches NaN).

    Args:
        image: Array with spatial axes ``spatial_axes`` (``(y, x)``).
        nodata: The nodata value.
        coords: ``(T, 4)`` tile coordinates ``(x1, y1, x2, y2)``.
        spatial_axes: The ``(y, x)`` axes of ``image``.

    Returns:
        ``(T,)`` array of fractions in ``[0, 1]``.
    """
    y_axis, x_axis = spatial_axes
    invalid = np.isnan(image) if np.isnan(nodata) else image == nodata
    other = tuple(d for d in range(image.ndim) if d not in spatial_axes)
    if other:
        invalid = invalid.all(axis=other)
    if y_axis > x_axis:  # the reduction keeps the remaining axes in ascending order
        invalid = invalid.T
    sat = np.zeros((invalid.shape[0] + 1, invalid.shape[1] + 1), dtype=np.int64)
    sat[1:, 1:] = invalid.cumsum(axis=0).cumsum(axis=1)
    x1, y1, x2, y2 = np.asarray(coords).T
    counts = sat[y2, x2] - sat[y1, x2] - sat[y2, x1] + sat[y1, x1]
    return np.asarray(counts / ((y2 - y1) * (x2 - x1)), dtype=np.float64)


def _extra_suffix(axes: tuple[int, ...], starts: list[int], sizes: tuple[int, ...]) -> str:
    return "".join(f"_ax{a}-{s}-{s + t}" for a, s, t in zip(axes, starts, sizes, strict=True))


def process_sample(sample: Sample, config: TilingConfig) -> SampleResult:
    """Tile one image or N-D array (and its labels or mask) and write the results to disk.

    The first two entries of ``config.axes`` are the ``(y, x)`` axes that labels refer to;
    any further axes (e.g. spectral bands) are tiled as well, producing one output file per
    combination of spatial and extra window. Those files share the spatial tile's labels.
    """
    profile: GeoProfile | None = None
    if config.georeference:
        image, profile = read_geotiff(sample.image_path)
    else:
        image = read_array(sample.image_path, hdf5_dataset=config.hdf5_dataset)
    if config.bands is not None:
        image = np.take(image, config.bands, axis=config.band_axis)
    axes = normalize_axes(config.axes, image.ndim)
    y_axis, x_axis = axes[:2]
    extra_axes = axes[2:]
    sizes, steps = config.tile_shape, config.step_shape
    height, width = image.shape[y_axis], image.shape[x_axis]
    fmt = config.input_format_annotations
    stem = sample.image_path.stem
    pad = config.pad_size if config.pad_image else 0
    img_ext = config.output_extension_images

    if pad:
        image = pad_image(image, pad, axes=(y_axis, x_axis))
    coords = tile_coordinates(
        image.shape[y_axis],
        image.shape[x_axis],
        sizes[:2],
        steps[:2],
        cover_edges=config.cover_edges,
    )
    extra_starts = tile_grid(
        [image.shape[a] for a in extra_axes], sizes[2:], steps[2:], cover_edges=config.cover_edges
    ).tolist()
    if len(coords) == 0 or not extra_starts:
        reason = f"shape {image.shape} is smaller than tile {sizes} along axes {list(axes)}"
        logger.warning("Skipping %s: %s.", sample.image_path.name, reason)
        return SampleResult(sample.image_path, skipped=reason)

    nodata = config.nodata
    if nodata is None and profile is not None:
        nodata = profile.nodata
    usable = np.ones(len(coords), dtype=bool)
    if config.max_nodata_fraction < 1.0:
        if nodata is None:
            logger.warning("%s has no nodata value; not filtering.", sample.image_path.name)
        else:
            frac = nodata_fraction(image, float(nodata), coords, (y_axis, x_axis))
            usable = frac <= config.max_nodata_fraction

    def write(path: Path, data: np.ndarray, x1: int, y1: int) -> None:
        if profile is not None and path.suffix.lower() in {".tif", ".tiff"}:
            write_geotiff(path, data, profile.for_window(x1 - pad, y1 - pad))
        else:
            write_array(path, data, hdf5_dataset=config.hdf5_dataset)

    # (height, width, depth) of an output tile, for the Pascal VOC <size> tag.
    depth = 1
    for d in range(image.ndim):
        if d not in (y_axis, x_axis):
            depth *= sizes[axes.index(d)] if d in extra_axes else image.shape[d]
    tile_shape_hw = (sizes[0], sizes[1]) if image.ndim == 2 else (sizes[0], sizes[1], depth)

    def save_tiles(index: int, drawn: np.ndarray | None = None) -> str:
        """Write every extra-axis window of spatial tile ``index``; return its base name."""
        x1, y1, _, _ = coords[index].tolist()
        base = _tile_name("tile", stem, coords[index])
        for extra in extra_starts:
            name = base + _extra_suffix(extra_axes, extra, sizes[2:])
            data = drawn
            if data is None:
                data = image[window_slices((y1, x1, *extra), sizes, axes, image.ndim)]
            write(config.output_dir_images / f"{name}.{img_ext}", data, x1, y1)
        return base

    if fmt == "none":
        for i in np.flatnonzero(usable).tolist():
            save_tiles(i)
        return SampleResult(sample.image_path, num_tiles=int(usable.sum()) * len(extra_starts))

    if fmt == "segmentation":
        assert sample.annotation_path is not None
        mask = read_array(sample.annotation_path, hdf5_dataset=config.hdf5_dataset)
        if mask.shape[:2] != (height, width):
            raise ValueError(
                f"Mask {sample.annotation_path} has size {mask.shape[:2]}, "
                f"image has {(height, width)}."
            )
        mask = pad_image(mask, pad)
        mask_ext = config.output_extension_annotations
        for i in np.flatnonzero(usable).tolist():
            x1, y1, x2, y2 = coords[i].tolist()
            save_tiles(i)
            name = _tile_name("mask", stem, coords[i])
            write(config.output_dir_annotations / f"{name}.{mask_ext}", mask[y1:y2, x1:x2], x1, y1)
        return SampleResult(sample.image_path, num_tiles=int(usable.sum()) * len(extra_starts))

    # Object detection.
    ann = _load_annotations(sample, config, height, width)
    if pad:
        ann = ann.shifted(pad, pad)
    assignment = assign_boxes_to_tiles(ann.boxes, coords, min_visibility=config.min_visibility)
    has_boxes = assignment.any(axis=1)
    keep = select_unique_tiles(assignment) if config.clear_duplicates else has_boxes
    if config.save_empty_tiles:
        keep = keep | ~has_boxes
    keep &= usable

    num_boxes: int = 0
    for t in np.flatnonzero(keep).tolist():
        idx = np.flatnonzero(assignment[t])
        local = Annotations(clip_boxes_to_tile(ann.boxes[idx], coords[t]), ann.labels[idx])
        num_boxes += len(local) * len(extra_starts)
        drawn = None
        if config.draw_boxes:
            x1, y1, x2, y2 = coords[t].tolist()
            drawn = _draw(image[y1:y2, x1:x2], local)
        base = save_tiles(t, drawn)
        for extra in extra_starts:
            name = base + _extra_suffix(extra_axes, extra, sizes[2:])
            ann_path = (
                config.output_dir_annotations / f"{name}.{config.output_extension_annotations}"
            )
            if config.output_format_annotations == "yolo":
                write_yolo(ann_path, local, (sizes[0], sizes[1]))
            else:
                write_pascal_voc(
                    ann_path, local, tile_shape_hw, f"{name}.{img_ext}", config.label_names
                )
    n_tiles = int(keep.sum()) * len(extra_starts)
    return SampleResult(sample.image_path, num_tiles=n_tiles, num_boxes=num_boxes)


# --------------------------------------------------------------------------- runner


def _init_worker() -> None:
    # One OpenCV thread per process avoids oversubscribing the CPU.
    cv2.setNumThreads(1)


def _map(config: TilingConfig, samples: list[Sample]) -> Iterator[SampleResult]:
    func = partial(process_sample, config=config)
    if config.num_workers == 1 or len(samples) == 1:
        yield from map(func, samples)
        return
    workers = min(config.num_workers, len(samples))
    chunksize = max(1, len(samples) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        yield from pool.map(func, samples, chunksize=chunksize)


def run(
    config: TilingConfig,
    samples: Iterable[Sample] | None = None,
    *,
    progress: bool = True,
) -> list[SampleResult]:
    """Tile a whole dataset as described by ``config``.

    Args:
        config: Pipeline settings.
        samples: Samples to process; discovered from ``config`` when omitted.
        progress: Show a progress bar.

    Returns:
        One :class:`SampleResult` per input image, in input order.
    """
    samples = list(samples) if samples is not None else discover_samples(config)
    config.output_dir_images.mkdir(parents=True, exist_ok=True)
    if config.input_format_annotations != "none":
        config.output_dir_annotations.mkdir(parents=True, exist_ok=True)

    results = list(
        tqdm(_map(config, samples), total=len(samples), desc="Tiling", disable=not progress)
    )
    logger.info(
        "Wrote %d tiles with %d boxes from %d images (%d skipped).",
        sum(r.num_tiles for r in results),
        sum(r.num_boxes for r in results),
        len(results),
        sum(r.skipped is not None for r in results),
    )
    return results
