"""Synthetic sample data for the demo app: a drone-style aerial scene and its hyperspectral twin.

Both are generated procedurally (no downloads) from the same material map, so the RGB
image comes with ground-truth boxes and the cube has realistic-looking spectra.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Material ids of the label map.
GRASS, SOIL, ROAD, WATER, ROOF, TREE, CAR = range(7)
MATERIALS = ("grass", "soil", "road", "water", "roof", "tree", "car")

# Display colour of every material (RGB).
PALETTE = np.array(
    [
        (112, 150, 70),  # grass
        (158, 124, 88),  # soil
        (120, 120, 124),  # road
        (52, 96, 130),  # water
        (178, 72, 52),  # roof
        (46, 92, 48),  # tree
        (226, 226, 220),  # car
    ],
    dtype=np.float32,
)

CLASS_NAMES = ("house", "tree", "car")
WAVELENGTHS = np.linspace(400, 1000, 48)  # nm, VNIR range of a typical drone sensor


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def material_spectra(wl: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Reflectance (0-1) of every material at the wavelengths ``wl``: ``(materials, bands)``."""
    green_peak = 0.06 * np.exp(-(((wl - 550) / 35) ** 2))
    red_edge = 0.45 * _sigmoid((wl - 715) / 12)
    vegetation = 0.04 + green_peak + red_edge
    return np.stack(
        [
            vegetation,  # grass
            0.12 + 0.18 * (wl - 400) / 600,  # soil: rises slowly towards the NIR
            0.16 + 0.03 * (wl - 400) / 600,  # road: flat, grey
            0.09 * np.exp(-(wl - 400) / 120) + 0.01,  # water: dark, absorbs the NIR
            0.07 + 0.38 * _sigmoid((wl - 590) / 18),  # roof tiles: red
            0.75 * vegetation,  # tree canopy: darker vegetation
            np.full_like(wl, 0.6),  # car: bright and flat
        ]
    ).astype(np.float32)


@dataclass(frozen=True)
class Scene:
    """A synthetic aerial scene."""

    materials: np.ndarray
    """``(H, W)`` material id of every pixel."""
    shading: np.ndarray
    """``(H, W)`` float brightness texture around 1."""
    boxes: np.ndarray
    """``(N, 4)`` ``(x1, y1, x2, y2)`` boxes of the objects."""
    labels: np.ndarray
    """``(N,)`` class ids into :data:`CLASS_NAMES`."""


def _noise(rng: np.random.Generator, shape: tuple[int, int], blur: float) -> np.ndarray:
    n = rng.standard_normal(shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), blur)
    return n / (n.std() + 1e-6)


def make_scene(height: int = 768, width: int = 1024, seed: int = 3) -> Scene:
    """Draw fields, roads, a pond, houses, trees and cars, and record their boxes."""
    rng = np.random.default_rng(seed)
    m = np.full((height, width), GRASS, np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    labels: list[int] = []

    # Ploughed fields.
    for x1, y1, x2, y2 in [(40, 430, 400, 730), (700, 40, 990, 300)]:
        m[y1:y2, x1:x2] = SOIL
    # Pond.
    cv2.ellipse(m, (830, 600), (130, 85), 15, 0, 360, WATER, -1)
    # Roads: a horizontal one and a curved branch.
    cv2.line(m, (0, 360), (width, 330), ROAD, 46)
    pts = np.array([[520, 345], [540, 220], [600, 110], [640, 0]], np.int32)
    cv2.polylines(m, [pts], False, ROAD, 34)
    road = m == ROAD

    def free(x1: int, y1: int, x2: int, y2: int, pad: int = 6) -> bool:
        win = m[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad]
        return bool(win.size) and bool(np.isin(win, (GRASS,)).all())

    # Houses along the road.
    placed = 0
    while placed < 14:
        w, h = rng.integers(46, 86), rng.integers(40, 70)
        x1 = int(rng.integers(10, width - w - 10))
        y1 = int(rng.choice([rng.integers(230, 300 - h + 40), rng.integers(395, 470)]))
        x2, y2 = x1 + int(w), y1 + int(h)
        if y2 >= height or not free(x1, y1, x2, y2):
            continue
        m[y1:y2, x1:x2] = ROOF
        boxes.append((x1, y1, x2, y2))
        labels.append(0)
        placed += 1

    # Trees.
    placed = 0
    for _ in range(4000):
        if placed >= 40:
            break
        r = int(rng.integers(12, 24))
        cx, cy = int(rng.integers(r, width - r)), int(rng.integers(r, height - r))
        if not free(cx - r, cy - r, cx + r, cy + r, pad=2):
            continue
        cv2.circle(m, (cx, cy), r, TREE, -1)
        boxes.append((cx - r, cy - r, cx + r + 1, cy + r + 1))
        labels.append(1)
        placed += 1

    # Cars on the horizontal road.
    for x in rng.choice(np.arange(20, width - 40, 52), size=9, replace=False):
        x1 = int(x)
        lane_y = round(360 - 30 * x1 / width)
        y1 = lane_y + (-17 if rng.random() < 0.5 else 4)
        x2, y2 = x1 + 26, y1 + 13
        if not road[y1:y2, x1:x2].all():
            continue
        m[y1:y2, x1:x2] = CAR
        boxes.append((x1, y1, x2, y2))
        labels.append(2)

    shading = 1 + 0.08 * _noise(rng, (height, width), 6) + 0.05 * _noise(rng, (height, width), 1)
    # Furrows in the fields and ridges on the roofs.
    rows = 1 + 0.1 * np.sin(np.arange(width) / 3.0)[None, :]
    shading = np.where(m == SOIL, shading * rows, shading)
    for x1, y1, x2, y2 in (b for b, lab in zip(boxes, labels, strict=True) if lab == 0):
        shading[y1:y2, x1 : (x1 + x2) // 2] *= 0.82  # shaded half of a gabled roof
    shading = np.where(m == TREE, shading * (0.8 + 0.25 * _noise(rng, m.shape, 2)), shading)
    return Scene(
        m,
        shading.astype(np.float32),
        np.asarray(boxes, np.int64).reshape(-1, 4),
        np.asarray(labels, np.int64),
    )


def scene_rgb(scene: Scene) -> np.ndarray:
    """The scene as an 8-bit ``(H, W, 3)`` RGB image."""
    rgb = PALETTE[scene.materials] * scene.shading[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def scene_cube(scene: Scene, scale: float = 0.5, seed: int = 0) -> np.ndarray:
    """The scene as a float32 ``(H, W, bands)`` reflectance cube at ``scale`` resolution."""
    h, w = scene.materials.shape
    size = (int(w * scale), int(h * scale))
    mats = cv2.resize(scene.materials, size, interpolation=cv2.INTER_NEAREST)
    shade = cv2.resize(scene.shading, size, interpolation=cv2.INTER_AREA)
    cube = material_spectra()[mats] * shade[..., None]
    noise = np.random.default_rng(seed).normal(0, 0.006, cube.shape).astype(np.float32)
    return np.clip(cube + noise, 0, 1).astype(np.float32)
