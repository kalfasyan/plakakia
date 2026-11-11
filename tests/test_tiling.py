from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plakakia import (
    assign_boxes_to_tiles,
    box_visibility,
    clip_boxes_to_tile,
    select_unique_tiles,
    tile_coordinates,
    tile_image,
    tile_starts,
)


@pytest.mark.parametrize(
    ("length", "tile", "step", "cover", "expected"),
    [
        (10, 4, 3, False, [0, 3, 6]),
        (11, 4, 3, False, [0, 3, 6]),
        (11, 4, 3, True, [0, 3, 6, 7]),
        (10, 4, 3, True, [0, 3, 6]),  # already aligned: no extra tile
        (4, 4, 1, False, [0]),
        (3, 4, 1, True, []),
    ],
)
def test_tile_starts(length: int, tile: int, step: int, cover: bool, expected: list[int]) -> None:
    assert tile_starts(length, tile, step, cover_edges=cover).tolist() == expected


@pytest.mark.parametrize(("tile", "step"), [(0, 1), (1, 0), (-3, 2)])
def test_invalid_sizes(tile: int, step: int) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        tile_starts(10, tile, step)


def test_tile_coordinates_row_major() -> None:
    coords = tile_coordinates(4, 6, tile_size=2, step_size=2)
    assert coords.tolist() == [
        [0, 0, 2, 2],
        [2, 0, 4, 2],
        [4, 0, 6, 2],
        [0, 2, 2, 4],
        [2, 2, 4, 4],
        [4, 2, 6, 4],
    ]


@pytest.mark.parametrize(
    "shape", [(512, 512, 3), (300, 457), (97, 130, 5), (64, 80, 2, 3)], ids=str
)
@pytest.mark.parametrize("cover_edges", [False, True])
def test_tile_image_matches_slicing(
    rng: np.random.Generator, shape: tuple[int, ...], cover_edges: bool
) -> None:
    image = rng.integers(0, 255, shape, dtype=np.uint8)
    tiles, coords = tile_image(image, tile_size=32, step_size=24, cover_edges=cover_edges)

    assert tiles.shape == (len(coords), 32, 32, *shape[2:])
    assert tiles.dtype == image.dtype
    assert tiles.flags.c_contiguous and tiles.flags.writeable
    for tile, (x1, y1, x2, y2) in zip(tiles, coords, strict=True):
        np.testing.assert_array_equal(tile, image[y1:y2, x1:x2])
    if cover_edges:
        assert coords[:, 2].max() == shape[1]
        assert coords[:, 3].max() == shape[0]


def test_tile_image_is_a_copy(rng: np.random.Generator) -> None:
    image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    original = image.copy()
    tiles, _ = tile_image(image, tile_size=32, step_size=16)
    tiles[0] = 0  # overlapping tiles must not share memory with each other or the input
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(tiles[1][:, :16], original[:32, 16:32])


def test_tile_image_smaller_than_tile() -> None:
    tiles, coords = tile_image(np.zeros((10, 10, 3)), tile_size=20, step_size=5)
    assert tiles.shape == (0, 20, 20, 3)
    assert coords.shape == (0, 4)


def test_tile_image_rejects_1d() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        tile_image(np.zeros(10), tile_size=2, step_size=2)


def test_box_visibility() -> None:
    tiles = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    boxes = np.array([[2, 2, 8, 8], [8, 0, 12, 10], [3, 3, 3, 3]])
    vis = box_visibility(boxes, tiles)
    np.testing.assert_allclose(
        vis,
        [
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )


def test_assign_boxes_full_and_partial() -> None:
    tiles = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])
    boxes = np.array([[2, 2, 8, 8], [8, 0, 12, 10]])
    full = assign_boxes_to_tiles(boxes, tiles)
    assert full.tolist() == [[True, False], [False, True]]
    partial = assign_boxes_to_tiles(boxes, tiles, min_visibility=0.5)
    assert partial.tolist() == [[True, True], [True, True]]


def test_assign_boxes_empty() -> None:
    out = assign_boxes_to_tiles(np.empty((0, 4)), np.array([[0, 0, 5, 5]]))
    assert out.shape == (1, 0)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_assign_boxes_invalid_visibility(value: float) -> None:
    with pytest.raises(ValueError, match="min_visibility"):
        assign_boxes_to_tiles(np.zeros((1, 4)), np.zeros((1, 4)), min_visibility=value)


def test_clip_boxes_to_tile() -> None:
    local = clip_boxes_to_tile(np.array([[8, -2, 12, 4], [11, 11, 14, 14]]), [10, 0, 20, 10])
    assert local.tolist() == [[0, 0, 2, 4], [1, 10, 4, 10]]


def test_select_unique_tiles() -> None:
    assignment = np.array(
        [
            [True, False, False],  # new box 0 -> keep
            [True, False, False],  # only box 0 again -> drop
            [False, False, False],  # empty -> drop
            [True, True, False],  # new box 1 -> keep
            [False, True, True],  # new box 2 -> keep
        ]
    )
    assert select_unique_tiles(assignment).tolist() == [True, False, False, True, True]
    assert select_unique_tiles(np.zeros((3, 0), bool)).tolist() == [False] * 3


# --------------------------------------------------------------------------- N-D tiling

from plakakia import iter_tiles, normalize_axes, tile_array, tile_grid, window_slices  # noqa: E402


@pytest.mark.parametrize(
    ("shape", "tile", "step", "axes"),
    [
        ((40, 50, 7), (16, 16), (8, 12), (0, 1)),  # (H, W, B)
        ((7, 40, 50), (16, 16), (16, 16), (1, 2)),  # (B, H, W)
        ((7, 40, 50), (16, 16), (16, 16), (2, 1)),  # (B, H, W), x before y
        ((40, 50, 30), (16, 16, 10), (16, 16, 5), (0, 1, 2)),  # spatial + spectral
        ((30, 40, 50), (10, 16, 16), (10, 16, 16), (0, 1, 2)),  # bands first, all tiled
        ((4, 40, 50, 3), (16, 16), (16, 16), (-3, -2)),  # (T, H, W, C), negative axes
        ((100,), 32, 16, (0,)),  # 1-D signal
    ],
    ids=str,
)
@pytest.mark.parametrize("cover_edges", [False, True])
def test_tile_array_matches_window_slices(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    tile: int | tuple[int, ...],
    step: int | tuple[int, ...],
    axes: tuple[int, ...],
    cover_edges: bool,
) -> None:
    array = rng.random(shape, dtype=np.float32)
    tiles, starts = tile_array(array, tile, step, axes=axes, cover_edges=cover_edges)
    resolved = normalize_axes(axes, array.ndim)
    sizes = (tile,) * len(axes) if isinstance(tile, int) else tile

    expected_shape = list(shape)
    for a, t in zip(resolved, sizes, strict=True):
        expected_shape[a] = t
    assert tiles.shape == (len(starts), *expected_shape)
    assert starts.shape == (len(tiles), len(axes))
    assert tiles.flags.c_contiguous
    for t, start in zip(tiles, starts, strict=True):
        np.testing.assert_array_equal(t, array[window_slices(start, sizes, resolved, array.ndim)])

    lazy = list(iter_tiles(array, tile, step, axes=axes, cover_edges=cover_edges))
    assert [s for s, _ in lazy] == [tuple(s) for s in starts.tolist()]
    for (_, view), t in zip(lazy, tiles, strict=True):
        assert np.shares_memory(view, array)
        np.testing.assert_array_equal(view, t)


def test_tile_array_defaults_and_memmap(tmp_path: Path) -> None:
    path = tmp_path / "cube.npy"
    np.save(path, np.arange(6 * 8 * 3).reshape(6, 8, 3))
    cube = np.load(path, mmap_mode="r")
    tiles, starts = tile_array(cube, 4)  # first two axes, step = tile size
    assert tiles.shape == (2, 4, 4, 3)
    assert starts.tolist() == [[0, 0], [0, 4]]
    tiles, starts = tile_array(cube, (3, 4, 3))  # three axes inferred from tile_shape
    assert tiles.shape == (4, 3, 4, 3)


def test_tile_array_empty_and_errors() -> None:
    tiles, starts = tile_array(np.zeros((5, 5, 10)), (8, 8))
    assert tiles.shape == (0, 8, 8, 10)
    assert starts.shape == (0, 2)
    with pytest.raises(ValueError, match="has 3 values but 2 axes"):
        tile_array(np.zeros((5, 5)), (2, 2, 2), axes=(0, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        tile_array(np.zeros((5, 5)), 2, axes=(0, 2))
    with pytest.raises(ValueError, match="unique"):
        tile_array(np.zeros((5, 5)), 2, axes=(0, -2))
    with pytest.raises(ValueError, match="At least one axis"):
        normalize_axes((), 2)
    with pytest.raises(ValueError, match="positive"):
        tile_array(np.zeros((5, 5)), (2, 0))


def test_tile_grid_zero_axes() -> None:
    assert tile_grid((), ()).shape == (1, 0)


def test_tile_image_rectangular_and_channels_first(rng: np.random.Generator) -> None:
    image = rng.integers(0, 255, (60, 90, 3), dtype=np.uint8)
    tiles, coords = tile_image(image, tile_size=(20, 30), step_size=(20, 30))
    assert tiles.shape == (9, 20, 30, 3)
    assert coords[1].tolist() == [30, 0, 60, 20]
    cf_tiles, cf_coords = tile_image(image.transpose(2, 0, 1), (20, 30), (20, 30), axes=(1, 2))
    np.testing.assert_array_equal(cf_coords, coords)
    np.testing.assert_array_equal(cf_tiles, tiles.transpose(0, 3, 1, 2))
    with pytest.raises(ValueError, match="exactly two"):
        tile_image(image, 10, 10, axes=(0, 1, 2))
    rect = tile_coordinates(60, 90, (20, 30), (10, 15))
    assert rect[:2].tolist() == [[0, 0, 30, 20], [15, 0, 45, 20]]
