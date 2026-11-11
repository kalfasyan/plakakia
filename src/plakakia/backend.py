"""Run tiling and box suppression on GPUs (PyTorch, CuPy) as well as on the CPU.

plakakia's tiling functions accept NumPy arrays, PyTorch tensors and CuPy arrays. A tensor
on a GPU stays on that GPU: tiles are cut out of a zero-copy strided view of it and
gathered in a single device operation, so the image is never copied to the host. Tile
coordinates are always small NumPy arrays.

Neither PyTorch nor CuPy is a dependency of plakakia. This module only imports them
when it is handed one of their arrays.

>>> import numpy as np
>>> array_namespace(np.zeros(3))
'numpy'
>>> boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]])
>>> nms(boxes, np.array([0.9, 0.8, 0.7]), iou_threshold=0.5).tolist()
[0, 2]
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["Namespace", "array_namespace", "is_device_array", "nms", "to_numpy"]

Namespace = Literal["numpy", "torch", "cupy"]
"""The array libraries plakakia knows how to keep data in."""


def array_namespace(array: object) -> Namespace:
    """Name the library that owns ``array``: ``"torch"``, ``"cupy"`` or ``"numpy"``.

    Anything that is neither a PyTorch tensor nor a CuPy array (lists, memory maps,
    h5py datasets...) counts as ``"numpy"``.
    """
    module = type(array).__module__.split(".", 1)[0]
    if module == "torch":
        return "torch"
    if module == "cupy":
        return "cupy"
    return "numpy"


def is_device_array(array: object) -> bool:
    """Whether ``array`` is a PyTorch tensor or CuPy array (on any device, CPU included)."""
    return array_namespace(array) != "numpy"


def to_numpy(array: object) -> np.ndarray:
    """Return ``array`` as a NumPy array, copying from the GPU when needed."""
    namespace = array_namespace(array)
    if namespace == "torch":
        return np.asarray(array.detach().cpu().numpy())  # type: ignore[attr-defined]
    if namespace == "cupy":
        return np.asarray(array.get())  # type: ignore[attr-defined]
    return np.asarray(array)


def _from_numpy(values: np.ndarray, like: object) -> Any:
    """Move a NumPy array to the library and device of ``like``."""
    namespace = array_namespace(like)
    if namespace == "torch":
        torch = importlib.import_module("torch")
        return torch.as_tensor(values, device=like.device)  # type: ignore[attr-defined]
    if namespace == "cupy":
        cupy = importlib.import_module("cupy")
        with like.device:  # type: ignore[attr-defined]
            return cupy.asarray(values)
    return values


# --------------------------------------------------------------------------- tiling


def strided_windows(
    array: np.ndarray,
    sizes: Sequence[int],
    axes: Sequence[int],
    as_strided: Callable[..., Any],
) -> Any:
    """Zero-copy sliding-window view built from ``array.strides``.

    The result has the same layout as :func:`numpy.lib.stride_tricks.sliding_window_view`:
    each tiled axis shrinks to its number of window positions and the window contents
    are appended at the end, in the order of ``axes``. ``as_strided`` is the library's
    own function (CuPy's), which takes strides in bytes like NumPy's.
    """
    shape = list(array.shape)
    for axis, size in zip(axes, sizes, strict=True):
        shape[axis] -= size - 1
    strides = list(array.strides)
    return as_strided(
        array,
        shape=(*shape, *sizes),
        strides=(*strides, *(strides[a] for a in axes)),
    )


def _torch_windows(tensor: Any, sizes: Sequence[int], axes: Sequence[int]) -> Any:
    # Tensor.unfold appends the window dimension at the end and returns a view, so
    # chaining it over the tiled axes reproduces sliding_window_view's layout.
    windows = tensor
    for axis, size in zip(axes, sizes, strict=True):
        windows = windows.unfold(axis, size, 1)
    return windows


def gather_tiles(
    array: Any, sizes: Sequence[int], axes: Sequence[int], starts: npt.NDArray[np.intp]
) -> Any:
    """Copy the tiles starting at ``starts`` out of ``array`` with one gather.

    ``starts`` is ``(T, len(axes))`` (from :func:`plakakia.tile_grid`). Returns a
    contiguous ``(T, ...)`` array from the same library, on the same device, as
    ``array``; untiled axes keep their position.
    """
    namespace = array_namespace(array)
    ndim = array.ndim
    if len(starts) == 0:  # e.g. the array is smaller than one tile
        out_shape = list(array.shape)
        for axis, size in zip(axes, sizes, strict=True):
            out_shape[axis] = size
        return array[:0].reshape(0, *out_shape)
    # Window offsets of the tiled axes first, then each tile in the array's own axis
    # order: windows.transpose(order)[i, j, ...] is one tile.
    order = [*axes, *(ndim + list(axes).index(d) if d in axes else d for d in range(ndim))]
    if namespace == "torch":
        torch = importlib.import_module("torch")
        windows = _torch_windows(array, sizes, axes).permute(order)
        index = torch.as_tensor(starts.T, device=array.device)
        return windows[tuple(index)].contiguous()
    if namespace == "cupy":
        cupy = importlib.import_module("cupy")
        windows = strided_windows(array, sizes, axes, cupy.lib.stride_tricks.as_strided)
        with array.device:
            index = tuple(cupy.asarray(column) for column in starts.T)
            return cupy.ascontiguousarray(windows.transpose(order)[index])
    windows = sliding_window_view(array, tuple(sizes), axis=tuple(axes))
    return windows.transpose(order)[tuple(starts.T)]


# --------------------------------------------------------------------------- NMS


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> npt.NDArray[np.intp]:
    """Greedy NMS: visit boxes by decreasing score, drop those that overlap a kept one."""
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.intp)


def nms(
    boxes: Any,
    scores: Any,
    iou_threshold: float = 0.5,
    *,
    class_ids: Any | None = None,
) -> Any:
    """Non-maximum suppression, on the GPU when the boxes are on the GPU.

    PyTorch tensors go through :func:`torchvision.ops.batched_nms` when torchvision is
    installed (CUDA, MPS or CPU, wherever the tensors are). Everything else, and PyTorch
    without torchvision, uses a NumPy implementation.

    Args:
        boxes: ``(N, 4)`` boxes ``(x1, y1, x2, y2)``, in any supported array library.
        scores: ``(N,)`` confidence per box.
        iou_threshold: Boxes whose IoU with a higher-scoring kept box is **above** this
            value are dropped.
        class_ids: Optional ``(N,)`` class per box. When given, boxes only suppress
            boxes of the same class.

    Returns:
        Indices of the kept boxes, highest score first, in the input's array library
        and on its device.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}.")
    if array_namespace(boxes) == "torch":
        try:
            ops = importlib.import_module("torchvision.ops")
        except ImportError:
            pass
        else:
            torch = importlib.import_module("torch")
            b = boxes.reshape(-1, 4)
            if not b.is_floating_point():
                b = b.float()
            s = torch.as_tensor(scores, device=b.device).reshape(-1).to(b.dtype)
            if class_ids is None:
                return ops.nms(b, s, iou_threshold)
            c = torch.as_tensor(class_ids, device=b.device).reshape(-1)
            return ops.batched_nms(b, s, c, iou_threshold)

    b = to_numpy(boxes).reshape(-1, 4).astype(np.float64)
    s = to_numpy(scores).reshape(-1).astype(np.float64)
    if len(b) != len(s):
        raise ValueError(f"Got {len(b)} boxes but {len(s)} scores.")
    if class_ids is not None:
        c = to_numpy(class_ids).reshape(-1)
        if len(c) != len(b):
            raise ValueError(f"Got {len(b)} boxes but {len(c)} class ids.")
        # Suppress within each class separately (smaller problems than one big pass),
        # then merge the survivors back into one list sorted by score.
        classes, codes = np.unique(c, return_inverse=True)
        codes = codes.reshape(-1)
        kept = []
        for k in range(len(classes)):
            members = np.flatnonzero(codes == k)
            kept.append(members[_nms_numpy(b[members], s[members], iou_threshold)])
        keep = np.concatenate(kept) if kept else np.empty(0, dtype=np.intp)
        keep = keep[np.argsort(-s[keep], kind="stable")]
    else:
        keep = _nms_numpy(b, s, iou_threshold)
    return _from_numpy(keep.astype(np.intp), boxes)
