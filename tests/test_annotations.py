from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plakakia import (
    Annotations,
    read_coco,
    read_pascal_voc,
    read_yolo,
    write_pascal_voc,
    write_yolo,
)

BOXES = np.array([[150, 200, 450, 600], [120, 160, 240, 320]])
LABELS = np.array([0, 1])


def test_annotations_validation() -> None:
    with pytest.raises(ValueError, match="2 boxes but 1 labels"):
        Annotations(BOXES, [0])
    assert len(Annotations.empty()) == 0
    assert Annotations(BOXES, LABELS).shifted(5, 10).boxes[0].tolist() == [155, 210, 455, 610]


def test_read_yolo(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("0 0.5 0.5 0.5 0.5\n\n1 0.3 0.3 0.2 0.2\n")
    ann = read_yolo(path, image_size=(800, 600))
    assert ann.boxes.tolist() == BOXES.tolist()
    assert ann.labels.tolist() == [0, 1]


def test_read_yolo_polygon(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("2 0.1 0.2 0.5 0.2 0.3 0.6\n")
    ann = read_yolo(path, image_size=(100, 100))
    assert ann.boxes.tolist() == [[10, 20, 50, 60]]
    assert ann.labels.tolist() == [2]


def test_read_yolo_empty_and_invalid(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    assert len(read_yolo(empty, (10, 10))) == 0
    bad = tmp_path / "bad.txt"
    bad.write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError, match=r"bad\.txt:1"):
        read_yolo(bad, (10, 10))
    with pytest.raises(FileNotFoundError):
        read_yolo(tmp_path / "missing.txt", (10, 10))


def test_yolo_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    ann = Annotations(BOXES, LABELS)
    write_yolo(path, ann, (800, 600))
    back = read_yolo(path, (800, 600))
    np.testing.assert_array_equal(back.boxes, ann.boxes)
    np.testing.assert_array_equal(back.labels, ann.labels)


def test_pascal_voc_roundtrip_with_names(tmp_path: Path) -> None:
    path = tmp_path / "a.xml"
    ann = Annotations(BOXES, LABELS)
    write_pascal_voc(path, ann, (800, 600, 3), "a.jpg", {0: "cat", 1: "dog"})
    text = path.read_text()
    assert "<name>cat</name>" in text and "<width>600</width>" in text
    back = read_pascal_voc(path, {"cat": 0, "dog": 1})
    np.testing.assert_array_equal(back.boxes, ann.boxes)
    np.testing.assert_array_equal(back.labels, ann.labels)


def test_pascal_voc_numeric_and_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "a.xml"
    write_pascal_voc(path, Annotations(BOXES, LABELS), (800, 600), "a.jpg")
    assert read_pascal_voc(path).labels.tolist() == [0, 1]
    write_pascal_voc(path, Annotations(BOXES, LABELS), (800, 600), "a.jpg", {0: "cat"})
    with pytest.raises(KeyError, match="'cat' is not in `annotation_mapping`"):
        read_pascal_voc(path)


def test_read_coco(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    data = {
        "images": [{"id": 1, "file_name": "sub/a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10.2, 20, 30, 40]},
            {"id": 2, "image_id": 1, "category_id": 4, "bbox": [0, 0, 5, 5]},
            {"id": 3, "image_id": 99, "category_id": 4, "bbox": [0, 0, 5, 5]},
        ],
    }
    path.write_text(json.dumps(data))
    coco = read_coco(path)
    assert set(coco) == {"a.jpg", "b.jpg"}
    assert coco["a.jpg"].boxes.tolist() == [[10, 20, 40, 60], [0, 0, 5, 5]]
    assert coco["a.jpg"].labels.tolist() == [3, 4]
    assert len(coco["b.jpg"]) == 0
