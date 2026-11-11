"""Core, dependency-light tiling primitives.

Everything in this module is pure NumPy: no file I/O and no global state, so the
functions can be used online (e.g. inside a data loader) as well as by the
offline pipeline in :mod:`plakakia.pipeline`.

Conventions
-----------
* Arrays can have any number of dimensions and any dtype: ``(H, W)`` grayscale,
  ``(H, W, C)`` colour, ``(H, W, B)`` or ``(B, H, W)`` hyperspectral cubes, volumes...
* :func:`tile_array` tiles along **any** subset of axes, with a tile size and step per
  axis. :func:`tile_image` is the 2-D special case that returns box-style coordinates.
* Boxes and tile coordinates are ``(x1, y1, x2, y2)`` in pixels, with ``x2``/``y2``
  exclusive, matching NumPy slicing: ``image[y1:y2, x1:x2]``.
* Tiles are ordered row-major (the first tiled axis varies slowest).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "assign_boxes_to_tiles",
    "box_visibility",
    "clip_boxes_to_tile",
    "iter_tiles",
    "normalize_axes",
    "select_unique_tiles",
    "tile_array",
    "tile_coordinates",
    "tile_grid",
    "tile_image",
    "tile_starts",
    "window_slices",
]

SizeLike = int | Sequence[int]
"""A single size applied to every axis, or one size per axis."""


def _check_sizes(tile_size: int, step_size: int) -> None:
    if int(tile_size) <= 0:
        raise ValueError(f"tile_size must be a positive integer, got {tile_size!r}.")
    if int(step_size) <= 0:
        raise ValueError(f"step_size must be a positive integer, got {step_size!r}.")


def _per_axis(value: SizeLike, n: int, name: str) -> tuple[int, ...]:
    """Broadcast an int or sequence to ``n`` positive ints."""
    if isinstance(value, int | np.integer):
        values = (int(value),) * n
    else:
        values = tuple(int(v) for v in value)
        if len(values) != n:
            raise ValueError(f"{name} has {len(values)} values but {n} axes are tiled.")
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return values


def normalize_axes(axes: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    """Resolve negative axes and check that ``axes`` are valid and unique.

    >>> normalize_axes((-2, -1), 3)
    (1, 2)
    """
    axes_t = (axes,) if isinstance(axes, int | np.integer) else tuple(axes)
    if not axes_t:
        raise ValueError("At least one axis must be tiled.")
    resolved = []
    for axis in axes_t:
        a = int(axis)
        if not -ndim <= a < ndim:
            raise ValueError(f"axis {a} is out of bounds for an array with {ndim} dimensions.")
        resolved.append(a % ndim)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"axes must be unique, got {axes_t}.")
    return tuple(resolved)


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


def tile_grid(
    shape: Sequence[int],
    tile_shape: SizeLike,
    step: SizeLike | None = None,
    *,
    cover_edges: bool = False,
) -> npt.NDArray[np.intp]:
    """Start offsets of every tile of an N-D grid.

    Args:
        shape: Sizes of the tiled axes.
        tile_shape: Tile size per axis (or one size for all).
        step: Stride per axis (or one for all). Defaults to ``tile_shape`` (no overlap).
        cover_edges: See :func:`tile_starts`.

    Returns:
        ``(T, len(shape))`` array of start offsets, row-major.

    >>> tile_grid((4, 6), (2, 3)).tolist()
    [[0, 0], [0, 3], [2, 0], [2, 3]]
    """
    n = len(shape)
    if n == 0:  # zero axes: a single, empty window
        return np.zeros((1, 0), dtype=np.intp)
    sizes = _per_axis(tile_shape, n, "tile_shape")
    steps = sizes if step is None else _per_axis(step, n, "step")
    starts = [
        tile_starts(length, t, s, cover_edges=cover_edges)
        for length, t, s in zip(shape, sizes, steps, strict=True)
    ]
    grids = np.meshgrid(*starts, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1).astype(np.intp).reshape(-1, n)


def window_slices(
    start: Sequence[int], tile_shape: Sequence[int], axes: Sequence[int], ndim: int
) -> tuple[slice, ...]:
    """Slices selecting the tile at ``start`` (one value per tiled axis) from an array.

    >>> window_slices((10, 20), (5, 5), (1, 2), 3)
    (slice(None, None, None), slice(10, 15, None), slice(20, 25, None))
    """
    slices = [slice(None)] * ndim
    for axis, s, t in zip(axes, start, tile_shape, strict=True):
        slices[axis] = slice(int(s), int(s) + int(t))
    return tuple(slices)


def _resolve(
    arr: np.ndarray,
    tile_shape: SizeLike,
    step: SizeLike | None,
    axes: int | Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if axes is None:
        n = 2 if isinstance(tile_shape, int | np.integer) else len(tile_shape)
        axes = tuple(range(n))
    axes_t = normalize_axes(axes, arr.ndim)
    sizes = _per_axis(tile_shape, len(axes_t), "tile_shape")
    steps = sizes if step is None else _per_axis(step, len(axes_t), "step")
    return axes_t, sizes, steps


def tile_array(
    array: npt.ArrayLike,
    tile_shape: SizeLike,
    step: SizeLike | None = None,
    *,
    axes: int | Sequence[int] | None = None,
    cover_edges: bool = False,
) -> tuple[np.ndarray, npt.NDArray[np.intp]]:
    """Tile an N-D array along any axes.

    Axes that are not tiled are kept whole, in their original position, so that
    ``tiles[i]`` equals ``array[window_slices(starts[i], tile_shape, axes, array.ndim)]``.

    Args:
        array: Any array, e.g. a ``(H, W, B)`` or ``(B, H, W)`` hyperspectral cube.
            Memory-mapped arrays (``np.load(..., mmap_mode="r")``) are read lazily.
        tile_shape: Tile size per tiled axis (or one size for all of them).
        step: Stride per tiled axis. Defaults to ``tile_shape`` (no overlap).
        axes: Axes to tile, in any order. Defaults to the first ``len(tile_shape)`` axes
            (the first two when ``tile_shape`` is an int).
        cover_edges: Add border-aligned tiles so the whole extent of each axis is covered.

    Returns:
        ``(tiles, starts)``: ``tiles`` is a contiguous ``(T, ...)`` copy and ``starts`` is a
        ``(T, len(axes))`` array of tile start offsets, one column per entry of ``axes``.

    Example:
        Tile a channels-first cube spatially and spectrally:

        >>> cube = np.zeros((200, 145, 145), dtype=np.float32)  # (bands, H, W)
        >>> tiles, starts = tile_array(cube, (64, 64, 50), axes=(1, 2, 0))
        >>> tiles.shape  # (T, bands, H, W) with tiled sizes
        (16, 50, 64, 64)
        >>> starts[:2].tolist()  # (y, x, band) per tile
        [[0, 0, 0], [0, 0, 50]]
    """
    arr = np.asarray(array)
    axes_t, sizes, steps = _resolve(arr, tile_shape, step, axes)
    starts = tile_grid([arr.shape[a] for a in axes_t], sizes, steps, cover_edges=cover_edges)
    out_shape = list(arr.shape)
    for axis, size in zip(axes_t, sizes, strict=True):
        out_shape[axis] = size
    if len(starts) == 0:
        return np.empty((0, *out_shape), dtype=arr.dtype), starts

    # sliding_window_view gives a zero-copy view with the window offsets on the original
    # axes and the window contents appended at the end. Reorder it (still a view) to
    # (offsets of tiled axes..., tile in the array's own axis order), then gather once:
    # only the selected windows are materialised, already in contiguous layout.
    windows = sliding_window_view(arr, sizes, axis=axes_t)
    ndim = arr.ndim
    order = [*axes_t, *(ndim + axes_t.index(d) if d in axes_t else d for d in range(ndim))]
    tiles = windows.transpose(order)[tuple(starts.T)]
    return tiles, starts


def iter_tiles(
    array: np.ndarray,
    tile_shape: SizeLike,
    step: SizeLike | None = None,
    *,
    axes: int | Sequence[int] | None = None,
    cover_edges: bool = False,
) -> Iterator[tuple[tuple[int, ...], np.ndarray]]:
    """Lazily yield ``(start, view)`` pairs without copying any data.

    Same arguments as :func:`tile_array`. Prefer this for arrays that do not fit in
    memory (e.g. memory-mapped hyperspectral cubes): only the tile you touch is read.

    >>> arr = np.arange(12).reshape(3, 4)
    >>> for start, view in iter_tiles(arr, (3, 2)):
    ...     print(start, view.ravel())
    (0, 0) [0 1 4 5 8 9]
    (0, 2) [ 2  3  6  7 10 11]
    """
    axes_t, sizes, steps = _resolve(array, tile_shape, step, axes)
    starts = tile_grid([array.shape[a] for a in axes_t], sizes, steps, cover_edges=cover_edges)
    for start in starts.tolist():
        yield tuple(start), array[window_slices(start, sizes, axes_t, array.ndim)]


def tile_coordinates(
    height: int,
    width: int,
    tile_size: SizeLike,
    step_size: SizeLike,
    *,
    cover_edges: bool = False,
) -> npt.NDArray[np.intp]:
    """Return ``(T, 4)`` tile coordinates ``(x1, y1, x2, y2)`` for an image of the given size.

    ``tile_size`` and ``step_size`` are an int (square) or ``(height, width)``.
    """
    th, tw = _per_axis(tile_size, 2, "tile_size")
    sh, sw = _per_axis(step_size, 2, "step_size")
    starts = tile_grid((height, width), (th, tw), (sh, sw), cover_edges=cover_edges)
    y1, x1 = starts[:, 0], starts[:, 1]
    return np.stack((x1, y1, x1 + tw, y1 + th), axis=-1).astype(np.intp)


def tile_image(
    image: npt.ArrayLike,
    tile_size: SizeLike = 250,
    step_size: SizeLike = 50,
    *,
    axes: Sequence[int] = (0, 1),
    cover_edges: bool = False,
) -> tuple[np.ndarray, npt.NDArray[np.intp]]:
    """Split an image into (possibly overlapping) tiles along its two spatial axes.

    Args:
        image: Array with at least 2 dimensions: ``(H, W)``, ``(H, W, C)``, a
            hyperspectral ``(H, W, B)`` cube, or channels-first data with ``axes=(1, 2)``.
        tile_size: Tile side in pixels, or ``(height, width)`` for rectangular tiles.
        step_size: Stride in pixels, or ``(height, width)``. ``step_size < tile_size``
            produces overlapping tiles.
        axes: The ``(y, x)`` axes of ``image``. Other axes (channels, bands) are kept
            whole. Use :func:`tile_array` to tile those too.
        cover_edges: Add tiles aligned to the right/bottom borders so that every pixel
            is covered, even when ``(size - tile_size)`` is not a multiple of ``step_size``.

    Returns:
        ``(tiles, coordinates)`` where ``tiles`` has the shape of ``image`` with the
        spatial axes replaced by the tile size, prefixed by the tile index (a contiguous
        copy, safe to modify), and ``coordinates`` is ``(T, 4)`` ``(x1, y1, x2, y2)``.
        Tiles are empty (``T == 0``) when the image is smaller than ``tile_size``.

    Example:
        >>> img = np.zeros((512, 512, 3), dtype=np.uint8)
        >>> tiles, coords = tile_image(img, tile_size=128, step_size=64)
        >>> tiles.shape, coords.shape
        ((49, 128, 128, 3), (49, 4))
        >>> cube = np.zeros((200, 145, 145), dtype=np.float32)  # (bands, H, W)
        >>> tile_image(cube, 64, 64, axes=(1, 2))[0].shape
        (4, 200, 64, 64)
    """
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError(f"Expected an image with at least 2 dimensions, got shape {arr.shape}.")
    if len(axes) != 2:
        raise ValueError(f"tile_image needs exactly two (y, x) axes, got {tuple(axes)}.")
    th, tw = _per_axis(tile_size, 2, "tile_size")
    steps = _per_axis(step_size, 2, "step_size")
    tiles, starts = tile_array(arr, (th, tw), steps, axes=axes, cover_edges=cover_edges)
    y1, x1 = starts[:, 0], starts[:, 1]
    return tiles, np.stack((x1, y1, x1 + tw, y1 + th), axis=-1).astype(np.intp)


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
