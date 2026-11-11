from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest
from numpy.lib.stride_tricks import as_strided, sliding_window_view

from plakakia import iter_tiles, nms, tile_array, tile_image, to_numpy
from plakakia.backend import array_namespace, gather_tiles, is_device_array, strided_windows

CASES = [
    # (shape, tile_shape, step, axes, cover_edges)
    ((37, 53, 3), (16, 16), (8, 12), (0, 1), False),
    ((37, 53, 3), (16, 16), (8, 12), (0, 1), True),
    ((5, 40, 31), (10, 7), None, (1, 2), True),
    ((20, 30, 12), (8, 8, 5), (4, 6, 3), (1, 0, 2), True),
    ((8, 8), (10, 10), None, (0, 1), False),  # smaller than a tile: no tiles
]


def brute_nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = w * h
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if all(iou(boxes[i], boxes[k]) <= thr for k in keep):
            keep.append(int(i))
    return keep


def random_boxes(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    xy = rng.uniform(0, 100, (n, 2))
    wh = rng.uniform(5, 30, (n, 2))
    return np.concatenate([xy, xy + wh], axis=1), rng.uniform(0, 1, n)


# ---------------------------------------------------------------- NumPy / generic


def test_namespace_of_numpy_and_friends() -> None:
    assert array_namespace(np.zeros(2)) == "numpy"
    assert array_namespace([1, 2]) == "numpy"
    assert not is_device_array(np.zeros(2))
    assert isinstance(to_numpy([1, 2]), np.ndarray)


@pytest.mark.parametrize(("shape", "tile", "step", "axes", "cover"), CASES[:4])
def test_strided_windows_match_sliding_window_view(
    shape: tuple[int, ...], tile: tuple[int, ...], step: Any, axes: tuple[int, ...], cover: bool
) -> None:
    # The CuPy path builds its window view from strides; check that recipe with NumPy.
    arr = np.arange(np.prod(shape)).reshape(shape)
    expected = sliding_window_view(arr, tile, axis=axes)
    np.testing.assert_array_equal(strided_windows(arr, tile, axes, as_strided), expected)


def test_gather_tiles_empty_keeps_dtype() -> None:
    out = gather_tiles(np.zeros((4, 4, 3), np.uint8), (8, 8), (0, 1), np.empty((0, 2), np.intp))
    assert out.shape == (0, 8, 8, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("thr", [0.0, 0.3, 0.5, 0.9])
def test_nms_matches_brute_force(rng: np.random.Generator, thr: float) -> None:
    boxes, scores = random_boxes(rng, 60)
    assert nms(boxes, scores, thr).tolist() == brute_nms(boxes, scores, thr)


def test_nms_class_aware(rng: np.random.Generator) -> None:
    boxes, scores = random_boxes(rng, 80)
    classes = rng.integers(0, 3, 80)
    keep = set(nms(boxes, scores, 0.4, class_ids=classes).tolist())
    expected = set()
    for c in range(3):
        idx = np.flatnonzero(classes == c)
        expected |= {int(idx[i]) for i in brute_nms(boxes[idx], scores[idx], 0.4)}
    assert keep == expected


def test_nms_edge_cases() -> None:
    assert nms(np.empty((0, 4)), np.empty(0)).tolist() == []
    # Identical boxes of different classes both survive.
    same = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    assert nms(same, np.array([0.9, 0.8]), 0.5).tolist() == [0]
    assert sorted(nms(same, np.array([0.9, 0.8]), 0.5, class_ids=["a", "b"]).tolist()) == [0, 1]
    with pytest.raises(ValueError, match="iou_threshold"):
        nms(same, np.array([0.9, 0.8]), 1.5)
    with pytest.raises(ValueError, match="scores"):
        nms(same, np.array([0.9]))
    with pytest.raises(ValueError, match="class ids"):
        nms(same, np.array([0.9, 0.8]), class_ids=[0])


# ---------------------------------------------------------------- PyTorch


@pytest.fixture
def torch() -> Any:
    return pytest.importorskip("torch")


def devices(torch: Any) -> list[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out.append("mps")
    return out


@pytest.mark.parametrize(("shape", "tile", "step", "axes", "cover"), CASES)
def test_torch_tile_array_matches_numpy(
    torch: Any,
    shape: tuple[int, ...],
    tile: tuple[int, ...],
    step: Any,
    axes: tuple[int, ...],
    cover: bool,
) -> None:
    arr = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    expected, expected_starts = tile_array(arr, tile, step, axes=axes, cover_edges=cover)
    for device in devices(torch):
        tensor = torch.as_tensor(arr, device=device)
        tiles, starts = tile_array(tensor, tile, step, axes=axes, cover_edges=cover)
        assert isinstance(tiles, torch.Tensor)
        assert tiles.device == tensor.device
        assert tiles.dtype == tensor.dtype
        assert tiles.is_contiguous()
        assert isinstance(starts, np.ndarray)
        np.testing.assert_array_equal(starts, expected_starts)
        np.testing.assert_array_equal(to_numpy(tiles), expected)


def test_torch_tile_image_channels_first(torch: Any) -> None:
    image = torch.rand(3, 100, 120)  # (C, H, W), as torchvision loads it
    tiles, coords = tile_image(image, 32, 24, axes=(1, 2), cover_edges=True)
    ref, ref_coords = tile_image(image.numpy(), 32, 24, axes=(1, 2), cover_edges=True)
    assert tiles.shape == (len(coords), 3, 32, 32)
    np.testing.assert_array_equal(coords, ref_coords)
    np.testing.assert_array_equal(tiles.numpy(), ref)
    # Tiles are a copy: writing to them leaves the image alone.
    tiles.zero_()
    assert image.abs().sum() > 0


def test_torch_iter_tiles_are_views(torch: Any) -> None:
    image = torch.zeros(10, 10)
    for _, view in iter_tiles(image, (5, 5)):
        view.fill_(1)
    assert bool((image == 1).all())


def test_torch_tile_image_rejects_1d(torch: Any) -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        tile_image(torch.zeros(10), 4, 4)


@pytest.mark.parametrize("with_classes", [False, True])
def test_torch_nms_matches_numpy(torch: Any, rng: np.random.Generator, with_classes: bool) -> None:
    boxes, scores = random_boxes(rng, 100)
    classes = rng.integers(0, 4, 100) if with_classes else None
    expected = nms(boxes, scores, 0.5, class_ids=classes).tolist()
    for device in devices(torch):
        keep = nms(
            torch.as_tensor(boxes, device=device),
            torch.as_tensor(scores, device=device),
            0.5,
            class_ids=None if classes is None else torch.as_tensor(classes, device=device),
        )
        assert isinstance(keep, torch.Tensor)
        assert keep.device.type == device
        assert sorted(keep.tolist()) == sorted(expected)


def test_torch_nms_without_torchvision(
    torch: Any, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def no_torchvision(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("torchvision"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", lambda n: no_torchvision(n, fromlist=["x"]))
    boxes, scores = random_boxes(rng, 30)
    keep = nms(torch.as_tensor(boxes), torch.as_tensor(scores), 0.5)
    assert isinstance(keep, torch.Tensor)
    assert keep.tolist() == brute_nms(boxes, scores, 0.5)
