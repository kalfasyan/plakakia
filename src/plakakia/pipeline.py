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
from .io import pad_image, read_image, write_image
from .tiling import assign_boxes_to_tiles, clip_boxes_to_tile, select_unique_tiles, tile_coordinates

__all__ = ["Sample", "SampleResult", "discover_samples", "process_sample", "run"]

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

    ext = ANNOTATION_EXTENSIONS[fmt]
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


def process_sample(sample: Sample, config: TilingConfig) -> SampleResult:
    """Tile one image (and its labels or mask) and write the results to disk."""
    image = read_image(sample.image_path)
    height, width = image.shape[:2]
    fmt = config.input_format_annotations
    stem = sample.image_path.stem
    pad = config.pad_size if config.pad_image else 0
    img_ext = config.output_extension_images

    if pad:
        image = pad_image(image, pad)
    coords = tile_coordinates(
        image.shape[0],
        image.shape[1],
        config.tile_size,
        config.step_size,
        cover_edges=config.cover_edges,
    )
    if len(coords) == 0:
        reason = f"image {width}x{height} is smaller than tile_size={config.tile_size}"
        logger.warning("Skipping %s: %s.", sample.image_path.name, reason)
        return SampleResult(sample.image_path, skipped=reason)

    def save_tile(index: int, data: np.ndarray | None = None) -> str:
        x1, y1, x2, y2 = coords[index].tolist()
        name = _tile_name("tile", stem, coords[index])
        tile = image[y1:y2, x1:x2] if data is None else data
        write_image(config.output_dir_images / f"{name}.{img_ext}", tile)
        return name

    if fmt == "none":
        for i in range(len(coords)):
            save_tile(i)
        return SampleResult(sample.image_path, num_tiles=len(coords))

    if fmt == "segmentation":
        assert sample.annotation_path is not None
        mask = read_image(sample.annotation_path)
        if mask.shape[:2] != (height, width):
            raise ValueError(
                f"Mask {sample.annotation_path} has size {mask.shape[:2]}, "
                f"image has {(height, width)}."
            )
        mask = pad_image(mask, pad)
        for i, (x1, y1, x2, y2) in enumerate(coords.tolist()):
            save_tile(i)
            name = _tile_name("mask", stem, coords[i])
            # Masks are always PNG: lossy formats would corrupt class ids.
            write_image(config.output_dir_annotations / f"{name}.png", mask[y1:y2, x1:x2])
        return SampleResult(sample.image_path, num_tiles=len(coords))

    # Object detection.
    ann = _load_annotations(sample, config, height, width)
    if pad:
        ann = ann.shifted(pad, pad)
    assignment = assign_boxes_to_tiles(ann.boxes, coords, min_visibility=config.min_visibility)
    has_boxes = assignment.any(axis=1)
    keep = select_unique_tiles(assignment) if config.clear_duplicates else has_boxes
    if config.save_empty_tiles:
        keep = keep | ~has_boxes

    num_boxes: int = 0
    for t in np.flatnonzero(keep).tolist():
        idx = np.flatnonzero(assignment[t])
        local = Annotations(clip_boxes_to_tile(ann.boxes[idx], coords[t]), ann.labels[idx])
        num_boxes += len(local)
        x1, y1, x2, y2 = coords[t].tolist()
        tile = image[y1:y2, x1:x2]
        name = save_tile(t, _draw(tile, local) if config.draw_boxes else None)
        ann_path = config.output_dir_annotations / f"{name}.{config.output_extension_annotations}"
        if config.output_format_annotations == "yolo":
            write_yolo(ann_path, local, tile.shape[:2])
        else:
            write_pascal_voc(ann_path, local, tile.shape, f"{name}.{img_ext}", config.label_names)
    return SampleResult(sample.image_path, num_tiles=int(keep.sum()), num_boxes=num_boxes)


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
