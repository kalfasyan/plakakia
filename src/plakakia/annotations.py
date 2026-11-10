"""Readers and writers for bounding-box annotation formats.

Supported inputs: YOLO (``.txt``), Pascal VOC (``.xml``) and COCO (``.json``).
Supported outputs: YOLO and Pascal VOC.

All readers return :class:`Annotations`, a pair of NumPy arrays holding integer pixel
boxes ``(x1, y1, x2, y2)`` and integer class ids.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "Annotations",
    "read_coco",
    "read_pascal_voc",
    "read_yolo",
    "write_pascal_voc",
    "write_yolo",
]

PathLike = str | Path


@dataclass(frozen=True)
class Annotations:
    """Bounding boxes of one image.

    Attributes:
        boxes: ``(N, 4)`` int64 array of ``(x1, y1, x2, y2)`` pixel coordinates.
        labels: ``(N,)`` int64 array of class ids.
    """

    boxes: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.int64).reshape(-1, 4)
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if len(boxes) != len(labels):
            raise ValueError(f"Got {len(boxes)} boxes but {len(labels)} labels.")
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "labels", labels)

    @classmethod
    def empty(cls) -> Annotations:
        """Annotations with no boxes."""
        return cls(np.empty((0, 4), np.int64), np.empty(0, np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def shifted(self, dx: int, dy: int) -> Annotations:
        """Return a copy with all boxes translated by ``(dx, dy)``."""
        return Annotations(self.boxes + np.array([dx, dy, dx, dy]), self.labels.copy())


# --------------------------------------------------------------------------- readers


def read_yolo(path: PathLike, image_size: tuple[int, int]) -> Annotations:
    """Read a YOLO ``.txt`` label file.

    Each line is ``class cx cy w h`` with coordinates normalised to ``[0, 1]``. Lines
    with more values are treated as YOLO segmentation polygons
    (``class x1 y1 x2 y2 ...``) and converted to their bounding box.

    Args:
        path: Label file.
        image_size: ``(height, width)`` of the image the labels refer to.
    """
    height, width = image_size
    labels: list[int] = []
    boxes: list[list[float]] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{path}:{lineno}: expected at least 5 values, got {len(parts)}.")
        values = [float(v) for v in parts[1:]]
        if len(values) == 4:
            cx, cy, w, h = values
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        else:
            xs, ys = values[0::2], values[1::2]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        labels.append(int(float(parts[0])))

    if not boxes:
        return Annotations.empty()
    scale = np.array([width, height, width, height], dtype=np.float64)
    pixel_boxes = np.rint(np.asarray(boxes) * scale).astype(np.int64)
    return Annotations(pixel_boxes, np.asarray(labels))


def _resolve_label(name: str, label_map: Mapping[str, int] | None, source: PathLike) -> int:
    if label_map and name in label_map:
        return int(label_map[name])
    try:
        return int(name)
    except ValueError:
        raise KeyError(
            f"{source}: class name {name!r} is not in `annotation_mapping` "
            f"(known: {sorted(label_map or {})})."
        ) from None


def read_pascal_voc(path: PathLike, label_map: Mapping[str, int] | None = None) -> Annotations:
    """Read a Pascal VOC ``.xml`` file.

    Args:
        path: XML file.
        label_map: Mapping from class names to integer ids. Names that are already
            integers do not need an entry.
    """
    root = ET.parse(path).getroot()
    boxes, labels = [], []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        boxes.append([float(bndbox.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(_resolve_label(name, label_map, path))
    if not boxes:
        return Annotations.empty()
    return Annotations(np.rint(np.asarray(boxes)).astype(np.int64), np.asarray(labels))


def read_coco(path: PathLike) -> dict[str, Annotations]:
    """Read a COCO ``.json`` file.

    Returns:
        Mapping from image file name (``Path(file_name).name``) to its annotations.
        Images without annotations map to empty :class:`Annotations`.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    id_to_name = {img["id"]: Path(img["file_name"]).name for img in data.get("images", [])}
    grouped: dict[str, tuple[list[list[float]], list[int]]] = {
        name: ([], []) for name in id_to_name.values()
    }
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        name = id_to_name.get(ann.get("image_id"))
        if name is None or not bbox:
            continue
        x, y, w, h = bbox
        grouped[name][0].append([x, y, x + w, y + h])
        grouped[name][1].append(int(ann["category_id"]))
    return {
        name: Annotations(np.rint(np.asarray(b, dtype=np.float64)).astype(np.int64), np.asarray(lb))
        if b
        else Annotations.empty()
        for name, (b, lb) in grouped.items()
    }


# --------------------------------------------------------------------------- writers


def write_yolo(path: PathLike, annotations: Annotations, image_size: tuple[int, int]) -> None:
    """Write boxes (in pixel coordinates of an image of ``(height, width)``) as YOLO."""
    height, width = image_size
    b = annotations.boxes.astype(np.float64)
    cx = (b[:, 0] + b[:, 2]) / 2 / width
    cy = (b[:, 1] + b[:, 3]) / 2 / height
    w = (b[:, 2] - b[:, 0]) / width
    h = (b[:, 3] - b[:, 1]) / height
    lines = (
        f"{label} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n"
        for label, x, y, bw, bh in zip(annotations.labels.tolist(), cx, cy, w, h, strict=True)
    )
    Path(path).write_text("".join(lines), encoding="utf-8")


def write_pascal_voc(
    path: PathLike,
    annotations: Annotations,
    image_shape: tuple[int, ...],
    image_filename: str,
    label_names: Mapping[int, str] | None = None,
) -> None:
    """Write boxes as a Pascal VOC ``.xml`` file.

    Args:
        path: Output XML file.
        annotations: Boxes in pixel coordinates of the image.
        image_shape: Shape of the image (``(H, W)`` or ``(H, W, C)``).
        image_filename: Value for the ``<filename>`` tag.
        label_names: Optional mapping from class ids back to names.
    """
    label_names = label_names or {}
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = Path(path).parent.name
    ET.SubElement(root, "filename").text = image_filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_shape[1])
    ET.SubElement(size, "height").text = str(image_shape[0])
    ET.SubElement(size, "depth").text = str(image_shape[2] if len(image_shape) > 2 else 1)
    for (x1, y1, x2, y2), label in zip(
        annotations.boxes.tolist(), annotations.labels.tolist(), strict=True
    ):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label_names.get(label, str(label))
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        for tag, value in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2), strict=True):
            ET.SubElement(bndbox, tag).text = str(value)
    tree = ET.ElementTree(root)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)
