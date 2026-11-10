"""Core, dependency-light tiling primitives.

Everything in this module is pure NumPy: no file I/O and no global state, so the
functions can be used online (e.g. inside a data loader) as well as by the
offline pipeline in :mod:`plakakia.pipeline`.

Conventions
-----------
* Images are ``(H, W)`` or ``(H, W, *C)`` arrays (any dtype, any number of channels).
* Boxes and tile coordinates are ``(x1, y1, x2, y2)`` in pixels, with ``x2``/``y2``
  exclusive, matching NumPy slicing: ``image[y1:y2, x1:x2]``.
* Tiles are ordered row-major (top-to-bottom, then left-to-right).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "assign_boxes_to_tiles",
    "box_visibility",
    "clip_boxes_to_tile",
    "select_unique_tiles",
    "tile_coordinates",
    "tile_image",
    "tile_starts",
]


def _check_sizes(tile_size: int, step_size: int) -> None:
    if int(tile_size) <= 0:
        raise ValueError(f"tile_size must be a positive integer, got {tile_size!r}.")
    if int(step_size) <= 0:
        raise ValueError(f"step_size must be a positive integer, got {step_size!r}.")


def tile_starts(
    length: int, tile_size: int, step_size: int, *, cover_edges: bool = False
) -> npt.NDArray[np.intp]:
    """Return the start offsets of tiles along one axis of size ``length``.

    With ``cover_edges=True`` an extra tile aligned to the end of the axis is added
    when the regular grid would otherwise leave a remainder uncovered.

    >>> tile_starts(10, 4, 3).tolist()
    [0, 3, 6]
    >>> tile_starts(11, 4, 3, cover_edges=True).tolist()
    [0, 3, 6, 7]
    """
    _check_sizes(tile_size, step_size)
    if length < tile_size:
        return np.empty(0, dtype=np.intp)
    starts = np.arange(0, length - tile_size + 1, step_size, dtype=np.intp)
    if cover_edges and starts[-1] + tile_size < length:
        starts = np.append(starts, length - tile_size)
    return starts


def tile_coordinates(
    height: int,
    width: int,
    tile_size: int,
    step_size: int,
    *,
    cover_edges: bool = False,
) -> npt.NDArray[np.intp]:
    """Return ``(T, 4)`` tile coordinates ``(x1, y1, x2, y2)`` for an image of the given size."""
    ys = tile_starts(height, tile_size, step_size, cover_edges=cover_edges)
    xs = tile_starts(width, tile_size, step_size, cover_edges=cover_edges)
    y1, x1 = (a.ravel() for a in np.meshgrid(ys, xs, indexing="ij"))
    return np.stack((x1, y1, x1 + tile_size, y1 + tile_size), axis=-1).astype(np.intp)


def tile_image(
    image: npt.ArrayLike,
    tile_size: int = 250,
    step_size: int = 50,
    *,
    cover_edges: bool = False,
) -> tuple[np.ndarray, npt.NDArray[np.intp]]:
    """Split an image into (possibly overlapping) square tiles.

    Args:
        image: ``(H, W)`` or ``(H, W, *C)`` array. Any dtype and number of channels.
        tile_size: Side length of each square tile in pixels.
        step_size: Stride of the sliding window in pixels. ``step_size < tile_size``
            produces overlapping tiles.
        cover_edges: Add tiles aligned to the right/bottom borders so that every pixel
            is covered, even when ``(size - tile_size)`` is not a multiple of ``step_size``.

    Returns:
        ``(tiles, coordinates)`` where ``tiles`` has shape ``(T, tile_size, tile_size, *C)``
        (a contiguous copy, safe to modify) and ``coordinates`` has shape ``(T, 4)``.
        Tiles are empty (``T == 0``) when the image is smaller than ``tile_size``.

    Example:
        >>> img = np.zeros((512, 512, 3), dtype=np.uint8)
        >>> tiles, coords = tile_image(img, tile_size=128, step_size=64)
        >>> tiles.shape, coords.shape
        ((49, 128, 128, 3), (49, 4))
    """
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError(f"Expected an image with at least 2 dimensions, got shape {arr.shape}.")
    height, width = arr.shape[:2]
    coords = tile_coordinates(height, width, tile_size, step_size, cover_edges=cover_edges)
    if len(coords) == 0:
        return np.empty((0, tile_size, tile_size, *arr.shape[2:]), dtype=arr.dtype), coords

    # Zero-copy (H', W', *C, t, t) view -> (H', W', t, t, *C); a single gather then
    # materialises only the selected windows, already in contiguous layout.
    windows = sliding_window_view(arr, (tile_size, tile_size), axis=(0, 1))
    windows = np.moveaxis(windows, (-2, -1), (2, 3))
    tiles = windows[coords[:, 1], coords[:, 0]]
    return tiles, coords


def box_visibility(boxes: npt.ArrayLike, tiles: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Fraction of each box's area that falls inside each tile.

    Args:
        boxes: ``(N, 4)`` boxes.
        tiles: ``(T, 4)`` tile coordinates.

    Returns:
        ``(T, N)`` array with values in ``[0, 1]``. Degenerate (zero-area) boxes get
        ``1.0`` if they lie within the tile and ``0.0`` otherwise.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    t = np.asarray(tiles, dtype=np.float64).reshape(-1, 4)
    ix1 = np.maximum(t[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(t[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(t[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(t[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        vis = np.where(area > 0, inter / area, 0.0)
    degenerate = area <= 0
    if degenerate.any():
        inside = (ix1 <= ix2) & (iy1 <= iy2)
        vis = np.where(degenerate[None, :], inside.astype(np.float64), vis)
    return vis


def assign_boxes_to_tiles(
    boxes: npt.ArrayLike, tiles: npt.ArrayLike, *, min_visibility: float = 1.0
) -> npt.NDArray[np.bool_]:
    """Decide which boxes belong to which tile.

    Args:
        boxes: ``(N, 4)`` boxes.
        tiles: ``(T, 4)`` tile coordinates.
        min_visibility: Minimum fraction of a box's area that must lie inside a tile for
            the box to be assigned to it. ``1.0`` (default) keeps only fully contained
            boxes; lower values keep partially visible boxes (to be clipped).

    Returns:
        ``(T, N)`` boolean assignment matrix.
    """
    if not 0.0 < min_visibility <= 1.0:
        raise ValueError(f"min_visibility must be in (0, 1], got {min_visibility}.")
    b = np.asarray(boxes).reshape(-1, 4)
    t = np.asarray(tiles).reshape(-1, 4)
    if min_visibility >= 1.0:
        # Exact integer containment test; avoids any floating point slack.
        return (
            (b[None, :, 0] >= t[:, None, 0])
            & (b[None, :, 1] >= t[:, None, 1])
            & (b[None, :, 2] <= t[:, None, 2])
            & (b[None, :, 3] <= t[:, None, 3])
        )
    vis = box_visibility(b, t)
    return (vis >= min_visibility) & (vis > 0)


def clip_boxes_to_tile(boxes: npt.ArrayLike, tile: npt.ArrayLike) -> np.ndarray:
    """Translate boxes into a tile's local frame and clip them to its bounds."""
    b = np.asarray(boxes).reshape(-1, 4)
    x1, y1, x2, y2 = np.asarray(tile).ravel()[:4]
    local: np.ndarray = b - np.array([x1, y1, x1, y1], dtype=b.dtype)
    local[:, [0, 2]] = np.clip(local[:, [0, 2]], 0, x2 - x1)
    local[:, [1, 3]] = np.clip(local[:, [1, 3]], 0, y2 - y1)
    return local


def select_unique_tiles(assignment: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Greedily drop tiles whose boxes are all already covered by earlier tiles.

    Tiles are visited in order; a tile is kept if it contains at least one box that no
    previously kept tile contains. Tiles without boxes are never selected.

    Args:
        assignment: ``(T, N)`` boolean matrix from :func:`assign_boxes_to_tiles`.

    Returns:
        ``(T,)`` boolean mask of tiles to keep.
    """
    a = np.asarray(assignment, dtype=bool)
    keep = np.zeros(a.shape[0], dtype=bool)
    seen = np.zeros(a.shape[1] if a.ndim == 2 else 0, dtype=bool)
    for i in np.flatnonzero(a.any(axis=1)) if a.size else ():
        if (a[i] & ~seen).any():
            keep[i] = True
            seen |= a[i]
    return keep
