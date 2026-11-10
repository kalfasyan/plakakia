from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plakakia import Annotations, write_image, write_pascal_voc, write_yolo

IMAGE_SIZE = (120, 160)  # (height, width)
BOXES = np.array([[10, 10, 30, 30], [70, 40, 100, 70], [130, 90, 150, 110]])
LABELS = np.array([0, 1, 0])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def dataset(tmp_path: Path, rng: np.random.Generator) -> Path:
    """A tiny dataset with 2 images in every supported annotation format."""
    root = tmp_path / "data"
    images = root / "images"
    images.mkdir(parents=True)
    for fmt in ("yolo", "pascal_voc", "coco", "segmentation"):
        (root / fmt).mkdir()

    ann = Annotations(BOXES, LABELS)
    coco = {"images": [], "annotations": [], "categories": [{"id": 0}, {"id": 1}]}
    for i in range(2):
        name = f"img{i}"
        image = rng.integers(0, 255, (*IMAGE_SIZE, 3), dtype=np.uint8)
        write_image(images / f"{name}.png", image)
        write_yolo(root / "yolo" / f"{name}.txt", ann, IMAGE_SIZE)
        write_pascal_voc(
            root / "pascal_voc" / f"{name}.xml",
            ann,
            image.shape,
            f"{name}.png",
            {0: "cat", 1: "dog"},
        )
        mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        for (x1, y1, x2, y2), label in zip(BOXES, LABELS, strict=True):
            mask[y1:y2, x1:x2] = label + 1
        write_image(root / "segmentation" / f"{name}.png", mask)
        coco["images"].append({"id": i, "file_name": f"{name}.png"})
        for j, ((x1, y1, x2, y2), label) in enumerate(zip(BOXES, LABELS, strict=True)):
            coco["annotations"].append(
                {
                    "id": i * 10 + j,
                    "image_id": i,
                    "category_id": int(label),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                }
            )
    (root / "coco" / "instances.json").write_text(json.dumps(coco))
    return root
