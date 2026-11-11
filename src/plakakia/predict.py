"""Tiled inference: run any detector on tiles and merge the boxes on the full image.

Small objects (insects, cells, cars in a satellite scene) vanish when a large image is
resized to a detector's input size. :func:`predict_tiled` keeps the native resolution:
it cuts the image into overlapping tiles, runs the model on batches of tiles, moves
every box back to image coordinates and merges the duplicates, all in one call.

The model is any callable. It receives a batch of tiles, stacked along a new first
axis, of the same array type as the image, and returns one ``(boxes, scores,
class_ids)`` triple per tile, with boxes ``(x1, y1, x2, y2)`` in tile pixels. NumPy
arrays, PyTorch or CuPy tensors (on any device) and lists all work, both as input and
as output.

>>> import numpy as np
>>> def bright_spots(batch):  # a toy "detector": one box per tile around its max
...     out = []
...     for tile in batch:
...         y, x = np.unravel_index(tile.argmax(), tile.shape)
...         out.append(([[x - 2, y - 2, x + 3, y + 3]], [float(tile.max())], [0]))
...     return out
>>> image = np.zeros((300, 400), dtype=np.uint8)
>>> image[150, 200] = 255
>>> dets = predict_tiled(image, bright_spots, tile_size=128, score_threshold=1)
>>> len(dets), dets.boxes.tolist()
(1, [[198.0, 148.0, 203.0, 153.0]])
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from .tiling import SizeLike, _per_axis, normalize_axes, tile_grid, window_slices

__all__ = [
    "Detections",
    "merge_detections",
    "predict_tiled",
    "tile_batches",
]

TileResult = tuple[Any, Any, Any]
"""``(boxes (N, 4), scores (N,), class_ids (N,))`` for one tile, in any array type."""

Model = Callable[[Any], Sequence[TileResult]]
"""A batch of tiles in, one :data:`TileResult` per tile out."""


# --------------------------------------------------------------------------- arrays


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a NumPy array, list, PyTorch tensor (any device) or CuPy array to NumPy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        return np.asarray(x.detach().cpu().numpy())
    if type(x).__module__.startswith("cupy"):
        return np.asarray(x.get())
    return np.asarray(x)


def _stack(arrays: list[Any]) -> Any:
    """Stack arrays along a new first axis with their own library, so they stay on device."""
    first = arrays[0]
    if isinstance(first, np.ndarray):
        return np.stack(arrays)
    if hasattr(first, "__array_namespace__"):  # NumPy 2, CuPy, JAX, ... (array API)
        return first.__array_namespace__().stack(arrays)
    module = sys.modules.get(type(first).__module__.split(".")[0])
    stack = getattr(module, "stack", None)
    return stack(arrays) if callable(stack) else np.stack(arrays)


# --------------------------------------------------------------------------- results


@dataclass
class Detections:
    """Boxes on the full image, with scores and class ids.

    Attributes:
        boxes: ``(N, 4)`` float ``(x1, y1, x2, y2)`` in image pixels.
        scores: ``(N,)`` confidence scores, sorted from high to low after a merge.
        class_ids: ``(N,)`` integer class ids.
        profile: Georeferencing of the image (a :class:`plakakia.io.GeoProfile`), if
            known. It lets :meth:`to_geojson` place the boxes on a map.
    """

    boxes: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.float64)
    )
    scores: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    class_ids: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    profile: Any = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
        if not len(self.boxes) == len(self.scores) == len(self.class_ids):
            raise ValueError(
                f"boxes, scores and class_ids differ in length: {len(self.boxes)}, "
                f"{len(self.scores)}, {len(self.class_ids)}."
            )

    def __len__(self) -> int:
        return len(self.boxes)

    def map_polygons(self, transform: Any = None) -> npt.NDArray[np.float64]:
        """Box corners in map coordinates, ``(N, 5, 2)`` closed rings.

        Args:
            transform: An affine transform from ``(col, row)`` pixels to map coordinates
                (a rasterio/``affine`` ``Affine``, or its 6 coefficients
                ``(a, b, c, d, e, f)``). Defaults to the transform of :attr:`profile`.
        """
        if transform is None:
            if self.profile is None:
                raise ValueError("No transform given and the detections have no profile.")
            transform = self.profile.transform
        a, b, c, d, e, f = (float(transform[i]) for i in range(6))
        x1, y1, x2, y2 = self.boxes.T
        cols = np.stack([x1, x2, x2, x1, x1], axis=-1)
        rows = np.stack([y1, y1, y2, y2, y1], axis=-1)
        return np.stack([a * cols + b * rows + c, d * cols + e * rows + f], axis=-1)

    def to_geojson(
        self,
        transform: Any = None,
        *,
        crs: Any = None,
        class_names: Sequence[str] | dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """A GeoJSON ``FeatureCollection`` with one polygon per box.

        Without a transform or :attr:`profile`, coordinates stay in pixels, which is
        still handy for viewers that read GeoJSON.

        Args:
            transform: See :meth:`map_polygons`. Defaults to the profile's transform.
            crs: Coordinate reference system written as the collection's ``crs`` member
                (read by QGIS and GDAL). Defaults to the profile's CRS.
            class_names: Names for the class ids, added to each feature's properties.

        Example:
            >>> dets = Detections([[0, 0, 10, 5]], [0.9], [1])
            >>> feature = dets.to_geojson(class_names=["car", "truck"])["features"].pop()
            >>> feature["properties"]
            {'score': 0.9, 'class_id': 1, 'class_name': 'truck'}
        """
        has_transform = transform is not None or self.profile is not None
        if has_transform:
            rings = self.map_polygons(transform)
        else:
            rings = self.map_polygons((1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        if crs is None and self.profile is not None:
            crs = self.profile.crs
        features = []
        for ring, score, cid in zip(rings, self.scores, self.class_ids, strict=True):
            props: dict[str, Any] = {"score": float(score), "class_id": int(cid)}
            if class_names is not None:
                props["class_name"] = class_names[int(cid)]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
                    "properties": props,
                }
            )
        collection: dict[str, Any] = {"type": "FeatureCollection", "features": features}
        if crs is not None:
            name = crs.to_string() if hasattr(crs, "to_string") else str(crs)
            collection["crs"] = {"type": "name", "properties": {"name": name}}
        return collection


# --------------------------------------------------------------------------- tiling


def _tile_layout(
    shape: Sequence[int],
    tile_size: SizeLike,
    step_size: SizeLike | None,
    axes: Sequence[int],
    cover_edges: bool,
) -> tuple[tuple[int, int], tuple[int, int], npt.NDArray[np.intp]]:
    """Tile shape (clamped to the image), axes and ``(T, 4)`` coordinates."""
    ay, ax = normalize_axes(axes, len(shape))
    height, width = int(shape[ay]), int(shape[ax])
    th, tw = _per_axis(tile_size, 2, "tile_size")
    if step_size is None:
        sh, sw = max(1, round(th * 0.8)), max(1, round(tw * 0.8))
    else:
        sh, sw = _per_axis(step_size, 2, "step_size")
    # An image smaller than a tile is processed as one (smaller) tile.
    th, tw = min(th, height), min(tw, width)
    starts = tile_grid((height, width), (th, tw), (sh, sw), cover_edges=cover_edges)
    y1, x1 = starts[:, 0], starts[:, 1]
    coords = np.stack((x1, y1, x1 + tw, y1 + th), axis=-1).astype(np.intp)
    return (th, tw), (ay, ax), coords


def tile_batches(
    image: Any,
    tile_size: SizeLike = 640,
    step_size: SizeLike | None = None,
    *,
    batch_size: int = 16,
    axes: Sequence[int] = (0, 1),
    cover_edges: bool = True,
) -> Iterator[tuple[Any, npt.NDArray[np.intp]]]:
    """Yield batches of tiles and their ``(x1, y1, x2, y2)`` coordinates.

    Only one batch is in memory at a time, and the tiles keep the image's array type:
    a PyTorch or CuPy image on a GPU gives batches on the same GPU.

    Args:
        image: Any array with ``.shape`` and slicing (NumPy, PyTorch, CuPy, ...).
        tile_size: Tile side, or ``(height, width)``. Clamped to the image size.
        step_size: Stride, or ``(height, width)``. Defaults to 80% of the tile (20%
            overlap).
        batch_size: Tiles per batch.
        axes: The ``(y, x)`` axes of ``image``, e.g. ``(1, 2)`` for ``(bands, H, W)``.
        cover_edges: Add tiles aligned to the right/bottom borders so every pixel is
            seen by the model.

    Yields:
        ``(batch, coords)``: the tiles stacked along a new first axis, and the
        ``(B, 4)`` NumPy coordinates of each tile.

    Example:
        >>> image = np.zeros((100, 150, 3), dtype=np.uint8)
        >>> shapes = [b.shape for b, _ in tile_batches(image, 64, 64, batch_size=4)]
        >>> shapes
        [(4, 64, 64, 3), (2, 64, 64, 3)]
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
    shape = tuple(image.shape)
    (th, tw), (ay, ax), coords = _tile_layout(shape, tile_size, step_size, axes, cover_edges)
    for start in range(0, len(coords), batch_size):
        chunk = coords[start : start + batch_size]
        tiles = [
            image[window_slices((y1, x1), (th, tw), (ay, ax), len(shape))] for x1, y1, _, _ in chunk
        ]
        yield _stack(tiles), chunk


# --------------------------------------------------------------------------- merging


def _area(boxes: np.ndarray) -> np.ndarray:
    w = np.clip(boxes[..., 2] - boxes[..., 0], 0, None)
    h = np.clip(boxes[..., 3] - boxes[..., 1], 0, None)
    return np.asarray(w * h)


def _overlap_1d(a1: float, a2: float, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Intersection over union of one interval with many."""
    inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None)
    union = np.maximum(a2, b2) - np.minimum(a1, b1)
    return np.asarray(np.divide(inter, union, out=np.zeros_like(inter), where=union > 0))


def _merge_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    seam: np.ndarray,
    iou_threshold: float,
    seam_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy NMS with seam fusion. Returns (kept indices, their possibly fused boxes)."""
    order = np.argsort(-scores, kind="stable")
    boxes, seam = boxes[order].copy(), seam[order].copy()
    area = _area(boxes)
    alive = np.ones(len(boxes), dtype=bool)
    keep = []
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        keep.append(i)
        # Fusing grows box i, which can bring more parts in reach (an object cut into four
        # by a tile corner), so repeat until nothing more fuses.
        while True:
            b = boxes[i]
            # Only boxes that overlap or touch box i can merge with it.
            tail = boxes[i + 1 :]
            near = (
                alive[i + 1 :]
                & (tail[:, 0] <= b[2] + seam_margin)
                & (tail[:, 2] >= b[0] - seam_margin)
                & (tail[:, 1] <= b[3] + seam_margin)
                & (tail[:, 3] >= b[1] - seam_margin)
            )
            rest = np.flatnonzero(near) + i + 1
            if not len(rest):
                break
            r = boxes[rest]
            iw = np.clip(np.minimum(b[2], r[:, 2]) - np.maximum(b[0], r[:, 0]), 0, None)
            ih = np.clip(np.minimum(b[3], r[:, 3]) - np.maximum(b[1], r[:, 1]), 0, None)
            inter = iw * ih
            union = area[i] + area[rest] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            alive[rest[iou > iou_threshold]] = False
            if not seam.any():
                break
            # A box cut by a seam is a part of the full box from a neighbouring tile: it
            # lies (mostly) inside it, so compare by intersection over the smaller box,
            # and only when the smaller box is the one that touches a seam.
            smaller = np.minimum(area[i], area[rest])
            ios = np.divide(inter, smaller, out=np.zeros_like(inter), where=smaller > 0)
            smaller_is_cut = np.where(area[i] <= area[rest], seam[i], seam[rest])
            contained = smaller_is_cut & (ios > iou_threshold)
            # Two parts of an object split exactly at a seam (no overlap) touch along one
            # axis and line up along the other.
            both_cut = seam[i] & seam[rest]
            gap_x = np.maximum(b[0], r[:, 0]) - np.minimum(b[2], r[:, 2])
            gap_y = np.maximum(b[1], r[:, 1]) - np.minimum(b[3], r[:, 3])
            touch_x = (np.abs(gap_x) <= seam_margin) & (
                _overlap_1d(b[1], b[3], r[:, 1], r[:, 3]) > iou_threshold
            )
            touch_y = (np.abs(gap_y) <= seam_margin) & (
                _overlap_1d(b[0], b[2], r[:, 0], r[:, 2]) > iou_threshold
            )
            fuse = alive[rest] & (contained | (both_cut & (touch_x | touch_y)))
            if not fuse.any():
                break
            parts = r[fuse]
            boxes[i, :2] = np.minimum(b[:2], parts[:, :2].min(axis=0))
            boxes[i, 2:] = np.maximum(b[2:], parts[:, 2:].max(axis=0))
            area[i] = _area(boxes[i])
            alive[rest[fuse]] = False
    kept = np.asarray(keep, dtype=np.intp)
    return order[kept], boxes[kept]


def merge_detections(
    boxes: npt.ArrayLike,
    scores: npt.ArrayLike,
    class_ids: npt.ArrayLike,
    *,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
    seam: npt.ArrayLike | None = None,
    seam_margin: float = 2.0,
) -> Detections:
    """Remove duplicate boxes with NMS, and fuse boxes cut by tile seams.

    Args:
        boxes: ``(N, 4)`` ``(x1, y1, x2, y2)`` on the full image.
        scores: ``(N,)`` confidence scores.
        class_ids: ``(N,)`` class ids. Boxes of different classes never merge unless
            ``class_agnostic``.
        iou_threshold: Boxes overlapping a better box by more than this IoU are
            dropped.
        class_agnostic: Merge across classes.
        seam: ``(N,)`` booleans, true for boxes that touch an inner tile border (so
            they may be the cut-off part of a larger object). Such a box is fused into a
            box that contains it (intersection over the smaller box above
            ``iou_threshold``), and two cut boxes that meet at a seam are fused into one.
            The fused box is the union of the parts, with the best score.
        seam_margin: Pixels within which two cut boxes count as touching.

    Returns:
        The merged :class:`Detections`, sorted by score.

    Example:
        >>> merged = merge_detections(
        ...     [[0, 0, 10, 10], [1, 1, 10, 10], [20, 0, 30, 10]], [0.9, 0.8, 0.7], [0, 0, 0]
        ... )
        >>> merged.boxes.tolist(), merged.scores.tolist()
        ([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]], [0.9, 0.7])
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    c = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    cut = np.zeros(len(b), dtype=bool) if seam is None else np.asarray(seam, dtype=bool)
    if not len(b) == len(s) == len(c) == len(cut):
        raise ValueError("boxes, scores, class_ids and seam must have the same length.")
    groups = (
        [np.arange(len(b))] if class_agnostic else [np.flatnonzero(c == k) for k in np.unique(c)]
    )
    kept, fused = [], []
    for idx in groups:
        k, kb = _merge_class(b[idx], s[idx], cut[idx], iou_threshold, seam_margin)
        kept.append(idx[k])
        fused.append(kb)
    if not kept:
        return Detections()
    ki, kb = np.concatenate(kept), np.concatenate(fused)
    order = np.argsort(-s[ki], kind="stable")
    return Detections(kb[order], s[ki][order], c[ki][order])


# --------------------------------------------------------------------------- prediction


def _resize_full(
    image: np.ndarray, axes: tuple[int, int], longest: int
) -> tuple[Any, float, float]:
    """Downscale ``image`` so its longest side is ``longest``. Returns (image, sx, sy)."""
    ay, ax = axes
    height, width = image.shape[ay], image.shape[ax]
    scale = longest / max(height, width)
    if scale >= 1:
        return image, 1.0, 1.0
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    # cv2 wants (H, W, C): move the spatial axes to the front, flatten the rest.
    moved = np.moveaxis(image, (ay, ax), (0, 1))
    rest = moved.shape[2:]
    flat = np.ascontiguousarray(moved.reshape(height, width, -1))
    # cv2.resize only takes a few channels at a time.
    chunks = [
        cv2.resize(flat[..., i : i + 4], (new_w, new_h), interpolation=cv2.INTER_AREA)
        for i in range(0, flat.shape[2], 4)
    ]
    chunks = [ch.reshape(new_h, new_w, -1) for ch in chunks]
    small = np.concatenate(chunks, axis=2).reshape(new_h, new_w, *rest)
    small = np.moveaxis(small, (0, 1), (ay, ax))
    return small, width / new_w, height / new_h


def _collect(
    results: Sequence[TileResult], n_tiles: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(results) != n_tiles:
        raise ValueError(f"The model returned {len(results)} results for {n_tiles} tiles.")
    out = []
    for res in results:
        if len(res) != 3:
            raise ValueError("The model must return (boxes, scores, class_ids) for each tile.")
        b, s, c = (_to_numpy(v) for v in res)
        out.append(
            (
                np.asarray(b, dtype=np.float64).reshape(-1, 4),
                np.asarray(s, dtype=np.float64).reshape(-1),
                np.asarray(c).astype(np.int64).reshape(-1),
            )
        )
    return out


def predict_tiled(
    image: Any,
    model: Model,
    tile_size: SizeLike = 640,
    step_size: SizeLike | None = None,
    *,
    batch_size: int = 16,
    axes: Sequence[int] = (0, 1),
    cover_edges: bool = True,
    full_image: bool | int = False,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
    fuse_seams: bool = True,
    seam_margin: float = 2.0,
    profile: Any = None,
) -> Detections:
    """Detect objects on a large image, tile by tile, and merge the boxes.

    The image is cut into overlapping tiles at native resolution, the model runs on
    batches of tiles, every box is moved from tile to image coordinates, and duplicates
    are merged with class-aware NMS. Boxes cut by a tile border are fused with the
    rest of their object (see :func:`merge_detections`).

    Args:
        image: The full image: a NumPy array, or a PyTorch/CuPy tensor (tiles stay on
            its device). Any layout: use ``axes`` to say where ``(y, x)`` are.
        model: ``model(batch) -> [(boxes, scores, class_ids), ...]``: one result per
            tile of the batch, boxes ``(x1, y1, x2, y2)`` in tile pixels. Any array type.
        tile_size: Tile side, or ``(height, width)``. Use your model's input size.
        step_size: Stride. Defaults to 80% of ``tile_size`` (20% overlap).
        batch_size: Tiles per model call.
        axes: The ``(y, x)`` axes of ``image``: ``(0, 1)`` for ``(H, W, C)``, ``(1, 2)``
            for ``(C, H, W)`` tensors and GeoTIFFs.
        cover_edges: Add tiles aligned to the right/bottom borders, so objects at the
            image border are not missed.
        full_image: Also run the model once on the whole image, downscaled so that its
            longest side is ``max(tile_size)`` (or this many pixels if an int). It finds
            objects larger than a tile. NumPy images only.
        score_threshold: Drop boxes scoring below this before merging.
        iou_threshold: IoU above which overlapping boxes of a class are duplicates.
        class_agnostic: Merge overlapping boxes across classes.
        fuse_seams: Fuse boxes cut by a tile border with the rest of their object.
        seam_margin: Pixels from a tile border within which a box counts as cut.
        profile: The image's :class:`plakakia.io.GeoProfile` (from
            :func:`plakakia.io.read_geotiff`). Stored on the result, so
            :meth:`Detections.to_geojson` returns map coordinates.

    Returns:
        :class:`Detections` on the full image, sorted by score.
    """
    shape = tuple(image.shape)
    (th, tw), (ay, ax), _ = _tile_layout(shape, tile_size, step_size, axes, cover_edges)
    height, width = shape[ay], shape[ax]
    boxes, scores, classes, seams = [], [], [], []
    for batch, coords in tile_batches(
        image,
        (th, tw),
        step_size,
        batch_size=batch_size,
        axes=(ay, ax),
        cover_edges=cover_edges,
    ):
        for (x1, y1, x2, y2), (tb, ts, tc) in zip(
            coords, _collect(model(batch), len(coords)), strict=True
        ):
            keep = ts >= score_threshold
            b, s, c = tb[keep], ts[keep], tc[keep]
            m = seam_margin
            cut = (
                ((b[:, 0] <= m) & (x1 > 0))
                | ((b[:, 1] <= m) & (y1 > 0))
                | ((b[:, 2] >= (x2 - x1) - m) & (x2 < width))
                | ((b[:, 3] >= (y2 - y1) - m) & (y2 < height))
            )
            boxes.append(b + np.array([x1, y1, x1, y1], dtype=np.float64))
            scores.append(s)
            classes.append(c)
            seams.append(cut)

    if full_image:
        if not isinstance(image, np.ndarray):
            raise TypeError("full_image=True needs a NumPy image; convert it with np.asarray.")
        longest = max(th, tw) if full_image is True else int(full_image)
        small, sx, sy = _resize_full(image, (ay, ax), longest)
        ((b, s, c),) = _collect(model(_stack([small])), 1)
        keep = s >= score_threshold
        boxes.append(b[keep] * np.array([sx, sy, sx, sy]))
        scores.append(s[keep])
        classes.append(c[keep])
        seams.append(np.zeros(int(keep.sum()), dtype=bool))

    if not boxes:
        return Detections(profile=profile)
    merged = merge_detections(
        np.concatenate(boxes),
        np.concatenate(scores),
        np.concatenate(classes),
        iou_threshold=iou_threshold,
        class_agnostic=class_agnostic,
        seam=np.concatenate(seams) if fuse_seams else None,
        seam_margin=seam_margin,
    )
    # Fused boxes can grow a little past the image.
    np.clip(merged.boxes[:, 0::2], 0, width, out=merged.boxes[:, 0::2])
    np.clip(merged.boxes[:, 1::2], 0, height, out=merged.boxes[:, 1::2])
    merged.profile = profile
    return merged
