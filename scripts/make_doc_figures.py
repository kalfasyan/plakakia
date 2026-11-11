"""Generate the figures used in the documentation.

Every figure is computed with plakakia itself, so the pictures always match what the
library does. Run from the repository root::

    uv run python scripts/make_doc_figures.py

Raster figures (a synthetic aerial scene and its tiles) are written as JPEG/PNG, and
diagrams as SVG. The SVGs are inlined into the pages and use Material for MkDocs CSS
variables, so they follow the light/dark theme.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from plakakia import assign_boxes_to_tiles, clip_boxes_to_tile, tile_image, tile_starts

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"

# Scene and tiling used throughout the docs.
SCENE_H, SCENE_W = 1500, 2400
TILE, STEP = 640, 480
# Tile palette (BGR), the same hues as the SVG diagrams.
TILE_COLORS = [(136, 150, 0), (0, 124, 245), (176, 39, 156), (243, 150, 33)]
BOX_COLOR = (255, 0, 255)  # magenta


# --------------------------------------------------------------------------- scene
def _noise(rng: np.random.Generator, h: int, w: int, scale: int) -> np.ndarray:
    small = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[:h, :w]


def make_scene(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Draw a top-down "aerial" scene with small cars; return ``(image, car_boxes)``."""
    rng = np.random.default_rng(seed)
    h, w = SCENE_H, SCENE_W

    # Grass and fields.
    tex = 0.6 * _noise(rng, h, w, 120) + 0.3 * _noise(rng, h, w, 25) + 0.1 * rng.random((h, w))
    grass = np.stack([60 + 40 * tex, 120 + 60 * tex, 70 + 40 * tex], -1)
    img = grass.astype(np.float32)

    # Fields with stripes.
    for x0, y0, x1, y1, col in [
        (1650, 80, 2330, 620, (90, 170, 190)),
        (80, 1080, 700, 1440, (70, 150, 120)),
    ]:
        patch = img[y0:y1, x0:x1]
        stripes = (np.arange(y1 - y0)[:, None] // 14 % 2) * 12.0
        patch[:] = np.array(col, np.float32) * (0.85 + 0.25 * tex[y0:y1, x0:x1, None])
        patch += stripes[..., None]

    asphalt = (88, 88, 92)

    def road(p0: tuple[int, int], p1: tuple[int, int], width: int) -> None:
        cv2.line(img, p0, p1, asphalt, width)
        cv2.line(img, p0, p1, (210, 210, 210), 2, lineType=cv2.LINE_AA)

    road((0, 760), (w, 760), 70)
    road((1180, 0), (1180, h), 64)
    road((1180, 1120), (w, 1360), 56)

    # Parking lots with rows of cars.
    boxes: list[tuple[int, int, int, int]] = []
    car_cols = [(40, 40, 200), (230, 230, 230), (30, 30, 30), (200, 120, 40), (60, 160, 230)]

    def car(cx: int, cy: int, vertical: bool) -> None:
        cw, ch = (14, 28) if vertical else (28, 14)
        x1, y1 = cx - cw // 2, cy - ch // 2
        x2, y2 = x1 + cw, y1 + ch
        col = car_cols[rng.integers(len(car_cols))]
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), col, -1)
        # Windshield.
        if vertical:
            cv2.rectangle(img, (x1 + 2, y1 + 6), (x2 - 3, y1 + 10), (40, 30, 30), -1)
        else:
            cv2.rectangle(img, (x2 - 11, y1 + 2), (x2 - 7, y2 - 3), (40, 30, 30), -1)
        boxes.append((x1, y1, x2, y2))

    lots = [(220, 180, 1000, 640), (1320, 860, 2250, 1080), (260, 870, 1060, 1000)]
    for x0, y0, x1, y1 in lots:
        cv2.rectangle(img, (x0, y0), (x1, y1), (110, 110, 115), -1)
        vertical = (y1 - y0) > 200
        if vertical:
            for row_y in range(y0 + 30, y1 - 20, 90):
                for cx in range(x0 + 20, x1 - 10, 22):
                    cv2.line(img, (cx - 11, row_y - 18), (cx - 11, row_y + 18), (220, 220, 220), 1)
                    if rng.random() < 0.55:
                        car(cx, row_y, vertical=True)
        else:
            for row_y in range(y0 + 22, y1 - 10, 44):
                for cx in range(x0 + 25, x1 - 20, 36):
                    if rng.random() < 0.5:
                        car(cx, row_y, vertical=False)

    # Cars on the roads.
    for cx in rng.choice(np.arange(40, w - 40, 60), 14, replace=False):
        car(int(cx), 745 if rng.random() < 0.5 else 775, vertical=False)
    for cy in rng.choice(np.arange(40, h - 40, 70), 8, replace=False):
        car(1168 if rng.random() < 0.5 else 1192, int(cy), vertical=True)

    # Buildings.
    for x0, y0, bw, bh, col in [
        (1360, 120, 180, 260, (70, 90, 170)),
        (1700, 1450, 260, 140, (150, 150, 160)),
        (1450, 1180, 120, 90, (60, 80, 150)),
        (60, 30, 120, 110, (150, 150, 160)),
    ]:
        cv2.rectangle(img, (x0 + 8, y0 + 8), (x0 + bw + 8, y0 + bh + 8), (40, 60, 40), -1)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), col, -1)
        cv2.line(img, (x0, y0 + bh // 2), (x0 + bw, y0 + bh // 2), (40, 40, 40), 2)

    # Trees.
    for _ in range(220):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if any(bx0 - 30 < x < bx1 + 30 and by0 - 30 < y < by1 + 30 for bx0, by0, bx1, by1 in lots):
            continue
        if abs(y - 760) < 60 or abs(x - 1180) < 55:
            continue
        r = int(rng.integers(10, 26))
        cv2.circle(img, (x + 5, y + 5), r, (30, 60, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), r, (40, 110 + int(rng.integers(0, 40)), 50), -1, cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), np.array(boxes, dtype=np.int64)


# --------------------------------------------------------------------------- rasters
def _save_jpg(path: Path, img: np.ndarray, quality: int = 86) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])


def scene_figures(scene: np.ndarray, boxes: np.ndarray) -> None:
    """Before/after: the scene with its tile grid, and a few tiles with their boxes."""
    tiles, coords = tile_image(scene, TILE, STEP, cover_edges=True)

    # 1. The input image.
    _save_jpg(OUT / "scene.jpg", cv2.resize(scene, (1200, 750), interpolation=cv2.INTER_AREA))

    # 2. The scene with the tile grid drawn on top.
    scale = 1200 / SCENE_W
    small = cv2.resize(scene, (1200, 750), interpolation=cv2.INTER_AREA).astype(np.float32)
    overlay = small.copy()
    n_cols = len(tile_starts(SCENE_W, TILE, STEP, cover_edges=True))
    for i, (x1, y1, x2, y2) in enumerate(coords.tolist()):
        row, col = divmod(i, n_cols)
        color = TILE_COLORS[(row % 2) * 2 + col % 2]
        p1 = (round(x1 * scale), round(y1 * scale))
        p2 = (round(x2 * scale) - 1, round(y2 * scale) - 1)
        tint = np.zeros_like(overlay)
        cv2.rectangle(tint, p1, p2, color, -1)
        mask = tint.any(-1, keepdims=True)
        overlay = np.where(mask, overlay * 0.82 + tint * 0.18, overlay)
    for i, (x1, y1, x2, y2) in enumerate(coords.tolist()):
        row, col = divmod(i, n_cols)
        color = TILE_COLORS[(row % 2) * 2 + col % 2]
        p1 = (round(x1 * scale) + 1, round(y1 * scale) + 1)
        p2 = (round(x2 * scale) - 2, round(y2 * scale) - 2)
        cv2.rectangle(overlay, p1, p2, color, 2, lineType=cv2.LINE_AA)
    _save_jpg(OUT / "scene_tiled.jpg", np.clip(overlay, 0, 255).astype(np.uint8))

    # 3. Six tiles at native resolution, with their (clipped) boxes.
    assignment = assign_boxes_to_tiles(boxes, coords, min_visibility=0.5)
    order = np.argsort(-assignment.sum(1), kind="stable")
    picked = sorted(order[:6].tolist())
    for k, t in enumerate(picked):
        tile = tiles[t].copy()
        local = clip_boxes_to_tile(boxes[assignment[t]], coords[t])
        for bx1, by1, bx2, by2 in local.tolist():
            cv2.rectangle(tile, (bx1 - 1, by1 - 1), (bx2, by2), BOX_COLOR, 2)
        tile = cv2.resize(tile, (400, 400), interpolation=cv2.INTER_AREA)
        _save_jpg(OUT / f"tile_{k}.jpg", tile)

    # 4. Why tile? The same crop, after resizing the whole image vs. from a tile.
    x0, y0, size = 1500, 880, 170
    native = scene[y0 : y0 + size, x0 : x0 + size]
    factor = 640 / SCENE_W
    shrunk = cv2.resize(scene, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    sx0, sy0, ssize = round(x0 * factor), round(y0 * factor), round(size * factor)
    lowres = shrunk[sy0 : sy0 + ssize, sx0 : sx0 + ssize]
    for name, crop, interp in [
        ("why_resized.jpg", lowres, cv2.INTER_NEAREST),
        ("why_native.jpg", native, cv2.INTER_CUBIC),
    ]:
        _save_jpg(OUT / name, cv2.resize(crop, (440, 440), interpolation=interp))

    print(f"scene: {len(coords)} tiles, {len(boxes)} cars")


# --------------------------------------------------------------------------- svg
# Colours shared by all diagrams. Text and outlines use Material's theme variables.
C = ["#009688", "#f57c00", "#9c27b0", "#2196f3"]
STYLE = """<style>
.fg{fill:var(--md-default-fg-color,#222)}
.mut{fill:var(--md-default-fg-color--light,#666)}
.ln{stroke:var(--md-default-fg-color--light,#666);fill:none}
.bg{fill:var(--md-default-fg-color--lightest,#eee)}
text{font-family:var(--md-text-font-family,sans-serif);font-size:14px}
.sm{font-size:12px}.b{font-weight:700}.mono{font-family:var(--md-code-font-family,monospace)}
</style>"""


def _svg(name: str, width: int, height: int, body: str, title: str) -> None:
    lines = [
        f'<svg class="plk-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"'
        f' role="img" aria-label="{title}">',
        f"<title>{title}</title>",
        STYLE,
        *[ln for ln in body.strip().splitlines() if ln.strip()],
        "</svg>",
    ]
    (OUT / f"{name}.svg").write_text("\n".join(ln.strip() for ln in lines) + "\n")


def _arrow_defs() -> str:
    return (
        '<defs><marker id="{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" '
        'style="fill:var(--md-default-fg-color--light,#666)"/></marker></defs>'
    )


def _dim(x1: float, x2: float, y: float, label: str, mid: str) -> str:
    """Horizontal dimension line with arrows on both ends."""
    return (
        f'<line class="ln" x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke-width="1.5" '
        f'marker-start="url(#{mid})" marker-end="url(#{mid})"/>'
        f'<text class="fg sm" x="{(x1 + x2) / 2}" y="{y - 6}" text-anchor="middle">{label}</text>'
    )


def svg_tile_step() -> None:
    """Tile size, step size and overlap on a 2-D grid."""
    s = 22  # px per "pixel cell"
    ox, oy = 30, 70
    cols, rows, tile, step = 14, 7, 6, 4
    starts = tile_starts(cols, tile, step).tolist()  # [0, 4, 8]
    body = [_arrow_defs().format(id="ats")]
    body.append(
        f'<rect class="bg" x="{ox}" y="{oy}" width="{cols * s}" height="{rows * s}" rx="4"/>'
    )
    grid = 'class="ln" stroke-opacity=".25"'
    body.extend(
        f'<line {grid} x1="{ox + i * s}" y1="{oy}" x2="{ox + i * s}" y2="{oy + rows * s}"/>'
        for i in range(cols + 1)
    )
    body.extend(
        f'<line {grid} x1="{ox}" y1="{oy + j * s}" x2="{ox + cols * s}" y2="{oy + j * s}"/>'
        for j in range(rows + 1)
    )
    for k, x in enumerate(starts):
        c = C[k]
        body.append(
            f'<rect x="{ox + x * s + 2}" y="{oy + 2}" width="{tile * s - 4}" '
            f'height="{tile * s - 4}" rx="3" fill="{c}" fill-opacity=".22" '
            f'stroke="{c}" stroke-width="3"/>'
        )
        body.append(
            f'<text class="b" x="{ox + x * s + 8}" y="{oy + tile * s - 10}" fill="{c}">'
            f"tile {k + 1}</text>"
        )
    # Overlap hatch label.
    ov_x = ox + starts[1] * s
    body.append(
        f'<text class="fg sm" x="{ov_x + s}" y="{oy + rows * s + 22}" text-anchor="middle">'
        "overlap</text>"
    )
    body.append(
        f'<line class="ln" x1="{ov_x + s}" y1="{oy + rows * s + 8}" x2="{ov_x + s}" '
        f'y2="{oy + tile * s - 6}" stroke-width="1.2" marker-end="url(#ats)"/>'
    )
    body.append(_dim(ox, ox + tile * s, oy - 34, f"tile_size = {tile}", "ats"))
    body.append(_dim(ox, ox + step * s, oy - 10, f"step_size = {step}", "ats"))
    # Legend.
    lx = ox + cols * s + 30
    body.append(f'<text class="fg b" x="{lx}" y="{oy + 10}">Sliding window</text>')
    legend = [
        "Each tile is tile_size wide.",
        "The next tile starts step_size",
        "pixels further.",
        "step < tile  →  tiles overlap",
        "step = tile  →  no overlap",
        "step > tile  →  gaps",
    ]
    for i, line in enumerate(legend):
        body.append(f'<text class="mut sm" x="{lx}" y="{oy + 36 + i * 20}">{line}</text>')
    _svg("tile_step", lx + 230, oy + rows * s + 34, "\n".join(body), "Tile size and step size")


def svg_cover_edges() -> None:
    """The regular grid drops the remainder; cover_edges adds a border-aligned tile."""
    s, length, tile, step = 26, 11, 4, 3
    body = [_arrow_defs().format(id="ace")]
    for panel, cover in enumerate([False, True]):
        ox = 20 + panel * (length * s + 60)
        oy = 40
        starts = tile_starts(length, tile, step, cover_edges=cover).tolist()
        title = "cover_edges=True" if cover else "cover_edges=False (default)"
        body.append(f'<text class="fg b mono" x="{ox}" y="{oy - 16}">{title}</text>')
        for i in range(length):
            covered = any(x <= i < x + tile for x in starts)
            fill = 'class="bg"' if covered else 'fill="#e53935" fill-opacity=".35"'
            body.append(
                f'<rect {fill} x="{ox + i * s + 1}" y="{oy}" width="{s - 2}" height="{s - 2}"'
                ' rx="3"/>'
            )
            body.append(
                f'<text class="mut sm mono" x="{ox + i * s + s / 2}" y="{oy + s / 2 + 4}" '
                f'text-anchor="middle">{i}</text>'
            )
        for k, x in enumerate(starts):
            y = oy + s + 10 + k * 16
            c = C[k % len(C)]
            body.append(
                f'<rect x="{ox + x * s + 1}" y="{y}" width="{tile * s - 2}" height="11" rx="5"'
                f' fill="{c}"/>'
            )
        n_extra = len(starts)
        note_y = oy + s + 10 + n_extra * 16 + 18
        note = (
            "the last tile is aligned to the border"
            if cover
            else "pixel 10 is not in any tile (dropped)"
        )
        body.append(f'<text class="mut sm" x="{ox}" y="{note_y}">{note}</text>')
    _svg(
        "cover_edges",
        2 * (length * s + 60),
        40 + s + 10 + 4 * 16 + 30,
        "\n".join(body),
        "cover_edges adds a tile aligned to the border",
    )


def svg_coordinates() -> None:
    """Image coordinate system and tile coordinates (x1, y1, x2, y2)."""
    ox, oy, w, h = 60, 40, 360, 220
    tx1, ty1, tx2, ty2 = 150, 80, 300, 200
    body = [_arrow_defs().format(id="aco")]
    body.append(f'<rect class="bg" x="{ox}" y="{oy}" width="{w}" height="{h}" rx="4"/>')
    body.append(
        f'<line class="ln" x1="{ox}" y1="{oy}" x2="{ox + w + 30}" y2="{oy}" stroke-width="2" '
        'marker-end="url(#aco)"/>'
        f'<line class="ln" x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy + h + 30}" stroke-width="2" '
        'marker-end="url(#aco)"/>'
        f'<text class="fg b" x="{ox + w + 36}" y="{oy + 5}">x</text>'
        f'<text class="fg b" x="{ox - 5}" y="{oy + h + 48}">y</text>'
        f'<text class="mut sm mono" x="{ox - 8}" y="{oy - 8}" text-anchor="end">(0, 0)</text>'
    )
    c = C[0]
    body.append(
        f'<rect x="{tx1}" y="{ty1}" width="{tx2 - tx1}" height="{ty2 - ty1}" rx="2" fill="{c}" '
        f'fill-opacity=".22" stroke="{c}" stroke-width="3"/>'
        f'<circle cx="{tx1}" cy="{ty1}" r="5" fill="{c}"/>'
        f'<circle cx="{tx2}" cy="{ty2}" r="5" fill="{C[1]}"/>'
        f'<text class="fg sm mono" x="{tx1 - 8}" y="{ty1 - 10}" text-anchor="middle">'
        "(x1, y1)</text>"
        f'<text class="fg sm mono" x="{tx2 + 8}" y="{ty2 + 20}" text-anchor="middle">'
        "(x2, y2)</text>"
        f'<text class="b" x="{(tx1 + tx2) / 2}" y="{(ty1 + ty2) / 2 + 5}" text-anchor="middle" '
        f'fill="{c}">tile</text>'
    )
    lx = ox + w + 70
    lines = [
        ("fg b", "One tile = (x1, y1, x2, y2)"),
        ("mut sm", "Origin at the top-left corner."),
        ("mut sm", "x goes right, y goes down."),
        ("mut sm", "x2 and y2 are exclusive, just like"),
        ("mut sm", "NumPy slicing:"),
        ("fg sm mono", "tile = image[y1:y2, x1:x2]"),
    ]
    for i, (cls, text) in enumerate(lines):
        body.append(f'<text class="{cls}" x="{lx}" y="{oy + 20 + i * 24}">{text}</text>')
    _svg("coordinates", lx + 260, oy + h + 60, "\n".join(body), "Tile coordinates")


def svg_min_visibility() -> None:
    """A box cut by the tile border: kept or dropped depending on min_visibility."""
    body: list[str] = []
    panels = [
        ("min_visibility=1.0 (default)", False, "dropped: only 60% of the box is inside"),
        ("min_visibility=0.5", True, "kept and clipped to the tile border"),
    ]
    for p, (title, kept, note) in enumerate(panels):
        ox, oy = 20 + p * 330, 40
        body.append(f'<text class="fg b mono" x="{ox}" y="{oy - 16}">{title}</text>')
        body.append(
            f'<rect x="{ox}" y="{oy}" width="200" height="160" rx="3" fill="{C[0]}" '
            f'fill-opacity=".15" stroke="{C[0]}" stroke-width="3"/>'
            f'<text class="b" x="{ox + 8}" y="{oy + 150}" fill="{C[0]}">tile</text>'
        )
        # Box 100 wide, 60 inside the tile.
        bx, by, bw, bh = ox + 140, oy + 50, 100, 60
        body.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" '
            'stroke="#fbc02d" stroke-width="2" stroke-dasharray="5 4"/>'
        )
        if kept:
            body.append(
                f'<rect x="{bx}" y="{by}" width="60" height="{bh}" fill="#fbc02d" '
                'fill-opacity=".45" stroke="#fbc02d" stroke-width="3"/>'
            )
            mark, color = "✔", "#43a047"
        else:
            mark, color = "✘", "#e53935"
        body.append(
            f'<text x="{bx + 30}" y="{by + 38}" text-anchor="middle" font-size="22" '
            f'fill="{color}">{mark}</text>'
            f'<text class="mut sm" x="{ox}" y="{oy + 190}">{note}</text>'
        )
    _svg("min_visibility", 680, 250, "\n".join(body), "min_visibility")


def svg_duplicates() -> None:
    """Overlapping tiles share objects; clear_duplicates keeps the first tile only."""
    body: list[str] = []
    panels = [("clear_duplicates=False (default)", False), ("clear_duplicates=True", True)]
    for p, (title, dedup) in enumerate(panels):
        ox, oy = 20 + p * 340, 40
        body.append(f'<text class="fg b mono" x="{ox}" y="{oy - 16}">{title}</text>')
        for k, x in enumerate([0, 110]):
            c = C[k]
            dropped = dedup and k == 1
            dash = ' stroke-dasharray="6 5"' if dropped else ""
            opacity = ".06" if dropped else ".2"
            body.append(
                f'<rect x="{ox + x}" y="{oy + k * 8}" width="180" height="150" rx="3" '
                f'fill="{c}" fill-opacity="{opacity}" stroke="{c}" stroke-width="3"{dash}/>'
            )
        body.append(
            f'<rect x="{ox + 125}" y="{oy + 60}" width="40" height="30" fill="#fbc02d" '
            'fill-opacity=".6" stroke="#fbc02d" stroke-width="2"/>'
        )
        text = (
            "tile 2 adds no new object, so it is skipped"
            if dedup
            else "the car is saved twice, in tile 1 and tile 2"
        )
        body.append(f'<text class="mut sm" x="{ox}" y="{oy + 185}">{text}</text>')
        body.append(
            f'<text class="b sm" x="{ox + 8}" y="{oy + 145}" fill="{C[0]}">tile 1</text>'
            f'<text class="b sm" x="{ox + 248}" y="{oy + 153}" fill="{C[1]}">tile 2</text>'
        )
    _svg("duplicates", 680, 240, "\n".join(body), "clear_duplicates")


def _cube(x: float, y: float, w: float, h: float, d: float, fill: str, op: float) -> str:
    """Oblique cube: front face w x h at (x, y), depth d drawn up and to the right."""
    dx, dy = d * 0.6, -d * 0.45
    front = f"{x},{y} {x + w},{y} {x + w},{y + h} {x},{y + h}"
    top = f"{x},{y} {x + dx},{y + dy} {x + w + dx},{y + dy} {x + w},{y}"
    side = f"{x + w},{y} {x + w + dx},{y + dy} {x + w + dx},{y + h + dy} {x + w},{y + h}"
    st = f'fill="{fill}" stroke="{fill}" stroke-width="1.5" stroke-linejoin="round"'
    return (
        f'<polygon points="{top}" {st} fill-opacity="{op * 0.7:.2f}"/>'
        f'<polygon points="{side}" {st} fill-opacity="{op * 0.45:.2f}"/>'
        f'<polygon points="{front}" {st} fill-opacity="{op:.2f}"/>'
    )


def svg_axes() -> None:
    """Hyperspectral cube: spatial tiles vs. spatial x spectral windows."""
    body: list[str] = []
    w, h, d = 150, 150, 120
    titles = [
        ("axes=(0, 1)", "spatial tiles, all bands"),
        ("axes=(0, 1, 2)", "spatial &#215; spectral windows"),
    ]
    for p, (title, sub) in enumerate(titles):
        ox, oy = 30 + p * 330, 110
        body.append(f'<text class="fg b mono" x="{ox}" y="28">{title}</text>')
        body.append(f'<text class="mut sm" x="{ox}" y="46">{sub}</text>')
        # Full cube, faint.
        body.append(_cube(ox, oy, w, h, d, "#9e9e9e", 0.12))
        if p == 0:
            body.append(_cube(ox, oy, w / 2, h / 2, d, C[0], 0.45))
        else:
            body.append(_cube(ox, oy, w / 2, h / 2, d / 3, C[2], 0.55))
        body.append(
            f'<text class="mut sm" x="{ox + w / 2}" y="{oy + h + 22}" text-anchor="middle">'
            "W (axis 1)</text>"
            f'<text class="mut sm" x="{ox - 10}" y="{oy + h / 2}" text-anchor="middle" '
            f'transform="rotate(-90 {ox - 10} {oy + h / 2})">H (axis 0)</text>'
            f'<text class="mut sm" x="{ox + w + d * 0.3 + 14}" y="{oy + h - d * 0.22 + 14}" '
            f'transform="rotate(-37 {ox + w + d * 0.3 + 14} {oy + h - d * 0.22 + 14})">'
            "bands (axis 2)</text>"
        )
    _svg("axes", 660, 300, "\n".join(body), "Tiling a hyperspectral cube along chosen axes")


def svg_pipeline() -> None:
    """Dataset folder -> plakakia -> folder of tiles with labels."""
    body = [_arrow_defs().format(id="api")]

    def folder(x: float, y: float, title: str, lines: list[str], color: str) -> None:
        body.append(
            f'<path d="M{x},{y + 10} h50 l10,-10 h120 v0 l0,0 v10 h0" fill="{color}" '
            'fill-opacity=".5"/>'
            f'<rect x="{x}" y="{y + 10}" width="200" height="{44 + 22 * len(lines)}" rx="6" '
            f'fill="{color}" fill-opacity=".15" stroke="{color}" stroke-width="2"/>'
            f'<text class="fg b" x="{x + 14}" y="{y + 32}">{title}</text>'
        )
        for i, ln in enumerate(lines):
            body.append(f'<text class="mut sm mono" x="{x + 14}" y="{y + 56 + i * 22}">{ln}</text>')

    folder(
        10,
        20,
        "your dataset",
        ["images/  big.jpg", "labels/  big.txt", "(YOLO, VOC, COCO,", " or masks)"],
        C[3],
    )
    folder(
        470,
        20,
        "tiled dataset",
        [
            "images/  tile_big_*.jpg",
            "labels/  tile_big_*.txt",
            "boxes shifted and",
            "clipped per tile",
        ],
        C[0],
    )
    # Middle box.
    body.append(
        f'<rect x="265" y="50" width="150" height="80" rx="10" fill="{C[2]}" fill-opacity=".18" '
        f'stroke="{C[2]}" stroke-width="2"/>'
        '<text class="fg b" x="340" y="84" text-anchor="middle">plakakia</text>'
        '<text class="mut sm mono" x="340" y="106" text-anchor="middle">config.yaml</text>'
        '<line class="ln" x1="214" y1="90" x2="260" y2="90" stroke-width="2" '
        'marker-end="url(#api)"/>'
        '<line class="ln" x1="420" y1="90" x2="465" y2="90" stroke-width="2" '
        'marker-end="url(#api)"/>'
    )
    _svg("pipeline", 680, 185, "\n".join(body), "Offline tiling pipeline")


def main() -> None:
    """Write every figure to ``docs/assets/figures``."""
    OUT.mkdir(parents=True, exist_ok=True)
    scene, boxes = make_scene()
    scene_figures(scene, boxes)
    svg_tile_step()
    svg_cover_edges()
    svg_coordinates()
    svg_min_visibility()
    svg_duplicates()
    svg_axes()
    svg_pipeline()


if __name__ == "__main__":
    main()
