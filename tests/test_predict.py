from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from plakakia import Detections, merge_detections, predict_tiled, tile_batches
from plakakia.io import GeoProfile, read_geotiff, write_geotiff


def oracle(batch: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A perfect "detector": one box per connected blob, class = pixel value - 1.

    Like a real model, it only sees the tile, so objects cut by the tile border get a
    cut-off box.
    """
    out = []
    for tile in batch:
        plane = tile if tile.ndim == 2 else tile[0]
        boxes, scores, classes = [], [], []
        for value in np.unique(plane[plane > 0]):
            n, _, stats, _ = cv2.connectedComponentsWithStats((plane == value).astype(np.uint8))
            for x, y, w, h, _ in stats[1:n]:
                boxes.append([x, y, x + w, y + h])
                scores.append(0.5 + 0.4 * w * h / plane.size)
                classes.append(int(value) - 1)
        out.append((np.reshape(boxes, (-1, 4)), np.array(scores), np.array(classes)))
    return out


def scene(objects: list[tuple[int, int, int, int, int]], shape=(400, 600)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2, cls in objects:
        image[y1:y2, x1:x2] = cls + 1
    return image


def assert_found(dets: Detections, objects: list[tuple[int, int, int, int, int]]) -> None:
    assert len(dets) == len(objects), dets.boxes
    expected = sorted(map(tuple, objects))
    found = sorted((*map(int, b), int(c)) for b, c in zip(dets.boxes, dets.class_ids, strict=True))
    assert found == expected


OBJECTS = [
    (10, 10, 30, 25, 0),
    (115, 50, 140, 70, 1),  # across the x=128 seam of 128 px tiles with no overlap
    (240, 120, 262, 140, 0),  # across x=256 and in the overlap of 20%-overlap tiles
    (300, 250, 320, 262, 1),  # across y=256
    (570, 370, 598, 398, 0),  # at the bottom-right image border
]


@pytest.mark.parametrize("step", [128, 100, None])
def test_predict_tiled_recovers_every_object_once(step: int | None) -> None:
    dets = predict_tiled(scene(OBJECTS), oracle, tile_size=128, step_size=step, batch_size=5)
    assert_found(dets, OBJECTS)
    assert np.all(np.diff(dets.scores) <= 0)


def test_without_seam_fusion_cut_objects_are_split() -> None:
    dets = predict_tiled(scene(OBJECTS), oracle, tile_size=128, step_size=128, fuse_seams=False)
    assert len(dets) > len(OBJECTS)


def test_object_cut_into_four_corners_is_fused() -> None:
    objects = [(110, 110, 150, 145, 0)]  # around the corner of four 128 px tiles
    assert_found(predict_tiled(scene(objects), oracle, 128, 128), objects)


def test_classes_are_kept_apart() -> None:
    objects = [(50, 50, 80, 80, 0), (80, 50, 110, 80, 1)]  # touching, different classes
    assert_found(predict_tiled(scene(objects), oracle, 128, 100), objects)
    merged = predict_tiled(scene(objects), oracle, 128, 100, class_agnostic=True)
    assert len(merged) == 2  # touching but not overlapping: still two boxes


def test_full_image_pass_finds_objects_larger_than_a_tile() -> None:
    objects = [(40, 30, 560, 380, 0), (5, 5, 25, 25, 0)]
    dets = predict_tiled(scene(objects), oracle, tile_size=128, full_image=True)
    assert len(dets) == 2
    np.testing.assert_allclose(dets.boxes[0], objects[0][:4], atol=3)
    np.testing.assert_allclose(dets.boxes[1], objects[1][:4], atol=3)


def test_full_image_downscales_to_tile_size() -> None:
    seen = []

    def spy(batch: np.ndarray) -> list[tuple[list, list, list]]:
        seen.append(batch.shape)
        return [([], [], [])] * len(batch)

    predict_tiled(np.zeros((300, 600, 3), np.uint8), spy, 128, 128, full_image=True)
    assert seen[-1] == (1, 64, 128, 3)
    predict_tiled(np.zeros((300, 600, 3), np.uint8), spy, 128, 128, full_image=300)
    assert seen[-1] == (1, 150, 300, 3)


def test_channels_first_and_many_bands() -> None:
    image = np.repeat(scene(OBJECTS)[None], 7, axis=0)  # (bands, H, W)
    assert_found(predict_tiled(image, oracle, 128, 100, axes=(1, 2)), OBJECTS)
    shapes = []

    def spy(batch: np.ndarray) -> list[tuple[list, list, list]]:
        shapes.append(batch.shape)
        return [([], [], [])] * len(batch)

    predict_tiled(image, spy, 128, axes=(1, 2), full_image=True)
    assert shapes[-1] == (1, 7, 85, 128)


def test_score_threshold_and_empty_results() -> None:
    dets = predict_tiled(scene(OBJECTS), oracle, 128, score_threshold=0.99)
    assert len(dets) == 0
    assert dets.boxes.shape == (0, 4)
    assert len(predict_tiled(np.zeros((50, 50)), oracle, 128)) == 0


def test_image_smaller_than_tile_is_one_tile() -> None:
    objects = [(5, 5, 15, 20, 0)]
    calls = []

    def model(batch: np.ndarray):  # type: ignore[no-untyped-def]
        calls.append(batch.shape)
        return oracle(batch)

    assert_found(predict_tiled(scene(objects, (40, 60)), model, 640), objects)
    assert calls == [(1, 40, 60)]


def test_model_must_return_one_result_per_tile() -> None:
    with pytest.raises(ValueError, match="results for"):
        predict_tiled(np.zeros((256, 256)), lambda b: [], 128)
    with pytest.raises(ValueError, match="boxes, scores, class_ids"):
        predict_tiled(np.zeros((128, 128)), lambda b: [([], [])], 128)


def test_tile_batches() -> None:
    image = np.arange(100 * 150).reshape(100, 150)
    batches = list(tile_batches(image, 64, 64, batch_size=4))  # 2 x 3 tiles, cover_edges
    assert [len(b) for b, _ in batches] == [4, 2]
    for batch, coords in batches:
        for tile, (x1, y1, x2, y2) in zip(batch, coords, strict=True):
            np.testing.assert_array_equal(tile, image[y1:y2, x1:x2])
    coords = np.concatenate([c for _, c in batches])
    assert coords[:, 2].max() == 150
    assert coords[:, 3].max() == 100
    with pytest.raises(ValueError, match="batch_size"):
        next(tile_batches(image, batch_size=0))


def test_tile_batches_keeps_torch_tensors() -> None:
    torch = pytest.importorskip("torch")
    image = torch.zeros((3, 100, 100))
    batch, _ = next(tile_batches(image, 64, 36, axes=(1, 2)))
    assert isinstance(batch, torch.Tensor)
    assert tuple(batch.shape) == (4, 3, 64, 64)


def test_merge_detections() -> None:
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10], [0, 0, 10, 10]]
    merged = merge_detections(boxes, [0.6, 0.9, 0.8], [0, 0, 1])
    assert merged.boxes.tolist() == [[1, 0, 11, 10], [0, 0, 10, 10]]
    assert merged.class_ids.tolist() == [0, 1]
    agnostic = merge_detections(boxes, [0.6, 0.9, 0.8], [0, 0, 1], class_agnostic=True)
    assert agnostic.class_ids.tolist() == [0]
    # A cut box inside a full one is fused only when it is marked as cut.
    nested = [[0, 0, 20, 20], [10, 0, 20, 20]]
    assert len(merge_detections(nested, [0.9, 0.8], [0, 0])) == 2
    assert len(merge_detections(nested, [0.9, 0.8], [0, 0], seam=[False, True])) == 1
    assert len(merge_detections(np.zeros((0, 4)), [], [])) == 0
    with pytest.raises(ValueError, match="same length"):
        merge_detections(boxes, [0.5], [0, 0, 0])


def test_detections_validation() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        Detections([[0, 0, 1, 1]], [0.5, 0.4], [0])


def test_geojson_in_pixels_and_with_a_transform() -> None:
    dets = Detections([[0, 0, 10, 5]], [0.9], [0])
    pixel = dets.to_geojson()
    assert "crs" not in pixel
    assert pixel["features"][0]["geometry"]["coordinates"][0][2] == [10.0, 5.0]
    # 0.5 m pixels, origin at (1000, 2000), north up.
    geo = dets.to_geojson((0.5, 0, 1000, 0, -0.5, 2000), crs="EPSG:32631")
    ring = geo["features"][0]["geometry"]["coordinates"][0]
    assert ring == [[1000, 2000], [1005, 2000], [1005, 1997.5], [1000, 1997.5], [1000, 2000]]
    assert geo["crs"]["properties"]["name"] == "EPSG:32631"
    with pytest.raises(ValueError, match="no profile"):
        dets.map_polygons()


def test_geotiff_detections_on_the_map(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    transform = rasterio.Affine(0.5, 0, 1000, 0, -0.5, 2000)
    crs = rasterio.crs.CRS.from_epsg(32631)
    path = tmp_path / "scene.tif"
    write_geotiff(path, np.repeat(scene(OBJECTS)[None], 3, axis=0), GeoProfile(crs, transform))
    image, profile = read_geotiff(path)

    dets = predict_tiled(image, oracle, 128, axes=(1, 2), profile=profile)
    assert_found(dets, OBJECTS)
    geo = dets.to_geojson(class_names=["car", "truck"])
    assert geo["crs"]["properties"]["name"] == "EPSG:32631"
    rings = {tuple(f["geometry"]["coordinates"][0][0]): f["properties"] for f in geo["features"]}
    x1, y1, _, _, cls = OBJECTS[0]
    assert rings[(1000 + 0.5 * x1, 2000 - 0.5 * y1)]["class_name"] == ["car", "truck"][cls]
