"""Generate the animated README hero: tiled detection of tiny objects, step by step.

Everything in the animation is computed with plakakia on the synthetic aerial scene from
``make_doc_figures.py``: the tile grid comes from :func:`plakakia.tile_image`, and the
final boxes from one :func:`plakakia.predict_tiled` call. The "detector" is a small
OpenCV stand-in (local contrast plus connected components), so the script needs no
model weights or GPU. Run from the repository root::

    uv run python scripts/make_hero.py

It writes ``docs/assets/figures/hero.gif`` and a still of the last frame,
``docs/assets/figures/hero.jpg``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_doc_figures import OUT, SCENE_H, SCENE_W, STEP, TILE, TILE_COLORS, make_scene

from plakakia import Detections, predict_tiled, tile_image, tile_starts

# Canvas layout (pixels). Left: the full image. Right: a native-resolution close-up.
W, H = 880, 470
BG = (17, 22, 29)  # RGB, close to GitHub's dark theme
FG, MUTED = (240, 243, 246), (150, 160, 172)
ACCENT = (255, 64, 200)  # final boxes, magenta like the docs figures
LX, LY, LW = 16, 84, 560
LH = round(LW * SCENE_H / SCENE_W)
RX, RY, RS = LX + LW + 16, 84, 272
SCALE = LW / SCENE_W
ZOOM = (1520, 880, 136)  # native close-up of a parking lot: x0, y0, size


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:  # no DejaVu on this machine: Pillow's built-in font
        return ImageFont.load_default(size)


TITLE, BODY, SMALL, MONO = _font(22, True), _font(15), _font(13), _font(14)


# --------------------------------------------------------------------------- detector
def detect(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Toy car detector: compact blobs that stand out from the local background."""
    background = cv2.medianBlur(tile, 51)
    diff = cv2.absdiff(tile, background).max(-1)
    mask = cv2.morphologyEx((diff > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    boxes, scores = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        lo, hi = min(w, h), max(w, h)
        if lo < 8 or hi > 40 or not 1.4 <= hi / lo <= 3.5 or area < 0.6 * w * h:
            continue
        boxes.append([x, y, x + w, y + h])
        scores.append(min(1.0, float(diff[labels == i].mean()) / 120))
    return np.array(boxes, float).reshape(-1, 4), np.array(scores), np.zeros(len(scores), int)


def model(batch: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [detect(tile) for tile in batch]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), -1)
    area = lambda z: np.prod(z[:, 2:] - z[:, :2], -1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter)


# --------------------------------------------------------------------------- drawing
def _rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return bgr[2], bgr[1], bgr[0]


def _to_left(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (LW, LH), interpolation=cv2.INTER_AREA)


def _crop(img: np.ndarray, x0: int, y0: int, size: int, nearest: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    return cv2.resize(img[y0 : y0 + size, x0 : x0 + size], (RS, RS), interpolation=interp)


def _left_box(box: np.ndarray | list[float]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) * SCALE for v in box)
    return round(LX + x1), round(LY + y1), round(LX + x2), round(LY + y2)


class Frame:
    """One frame: a full-image view on the left, a close-up on the right, and text."""

    def __init__(self, left: np.ndarray, right: np.ndarray | None) -> None:
        self.im = Image.new("RGB", (W, H), BG)
        self.im.paste(Image.fromarray(left[..., ::-1]), (LX, LY))
        if right is not None:
            self.im.paste(Image.fromarray(right[..., ::-1]), (RX, RY))
        self.d = ImageDraw.Draw(self.im)

    def header(self, step: str, title: str, subtitle: str) -> Frame:
        self.d.text((LX, 14), step, font=SMALL, fill=MUTED)
        self.d.text((LX, 30), title, font=TITLE, fill=FG)
        self.d.text((LX, 60), subtitle, font=BODY, fill=MUTED)
        return self

    def right_caption(self, *lines: str) -> Frame:
        for i, line in enumerate(lines):
            font, fill = (BODY, FG) if i == 0 else (SMALL, MUTED)
            self.d.text((RX, RY + RS + 10 + 22 * i), line, font=font, fill=fill)
        return self

    def footer(self, text: str, color: tuple[int, int, int] = FG) -> Frame:
        self.d.text((LX, LY + LH + 12), text, font=MONO, fill=color)
        return self

    def zoom_marker(self, x0: int, y0: int, size: int) -> Frame:
        self.d.rectangle(_left_box([x0, y0, x0 + size, y0 + size]), outline=FG, width=2)
        self.d.rectangle((RX - 1, RY - 1, RX + RS, RY + RS), outline=FG, width=2)
        return self

    def boxes(self, boxes: np.ndarray, color: tuple[int, int, int], width: int = 1) -> Frame:
        for b in boxes:
            self.d.rectangle(_left_box(b), outline=color, width=width)
        return self


def _boxes_on(img: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], t: int) -> None:
    for x1, y1, x2, y2 in np.round(boxes).astype(int).tolist():
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), color, t, cv2.LINE_AA)


def _grid(base: np.ndarray, coords: np.ndarray, n_cols: int, upto: int, hi: int = -1) -> None:
    """Outline the first ``upto`` tiles on a left-view image (in place); ``hi`` in white."""
    for i, (x1, y1, x2, y2) in enumerate(coords[:upto].tolist()):
        row, col = divmod(i, n_cols)
        color = TILE_COLORS[(row % 2) * 2 + col % 2]
        p1 = (round(x1 * SCALE) + 1, round(y1 * SCALE) + 1)
        p2 = (round(x2 * SCALE) - 2, round(y2 * SCALE) - 2)
        cv2.rectangle(base, p1, p2, color, 2, cv2.LINE_AA)
    if hi >= 0:
        x1, y1, x2, y2 = (round(v * SCALE) for v in coords[hi].tolist())
        cv2.rectangle(base, (x1 + 1, y1 + 1), (x2 - 2, y2 - 2), (255, 255, 255), 3, cv2.LINE_AA)


def _tile_color(i: int, n_cols: int) -> tuple[int, int, int]:
    row, col = divmod(i, n_cols)
    return TILE_COLORS[(row % 2) * 2 + col % 2]


# --------------------------------------------------------------------------- story
def build() -> tuple[list[Image.Image], list[int]]:
    scene, gt = make_scene()
    tiles, coords = tile_image(scene, TILE, STEP, cover_edges=True)
    n_cols = len(tile_starts(SCENE_W, TILE, STEP, cover_edges=True))
    per_tile = model(tiles)
    raw = [
        b + np.array([x1, y1, x1, y1])
        for b, (x1, y1, _, _) in zip((r[0] for r in per_tile), coords.tolist(), strict=True)
    ]
    dets: Detections = predict_tiled(scene, model, TILE, STEP)
    match = _iou(gt.astype(float), dets.boxes) > 0.5
    found, false_alarms = int(match.any(1).sum()), int((~match.any(0)).sum())
    n_raw = sum(len(r) for r in raw)
    print(f"{len(coords)} tiles, {n_raw} raw boxes, {len(dets)} merged, {found}/{len(gt)} cars")

    frames: list[Image.Image] = []
    durations: list[int] = []

    def add(frame: Frame, ms: int) -> None:
        frames.append(frame.im)
        durations.append(ms)

    # Display copy: a light median filter removes sensor noise, which GIFs compress badly.
    # The 2x close-ups get a lighter filter, so the cars keep their shape.
    shown, crisp = cv2.medianBlur(scene, 5), cv2.medianBlur(scene, 3)
    left = _to_left(shown)
    zx, zy, zs = ZOOM
    native = _crop(crisp, zx, zy, zs)
    car_h, car_w = 28, 14

    # 1. The input.
    add(
        Frame(left, native)
        .header(
            "THE PROBLEM",
            f"{SCENE_W} × {SCENE_H} image, {len(gt)} tiny cars",
            f"Each car is about {car_w} × {car_h} pixels, under 0.02% of the image.",
        )
        .zoom_marker(zx, zy, zs)
        .right_caption("Full resolution", "what the camera captured"),
        2400,
    )

    # 2. What a detector sees after resizing the whole image to its input size.
    f = TILE / SCENE_W
    small = cv2.resize(scene, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    blocky = _crop(small, round(zx * f), round(zy * f), round(zs * f), nearest=True)
    shrunk_left = cv2.resize(small, (LW, LH), interpolation=cv2.INTER_NEAREST)
    add(
        Frame(shrunk_left, blocky)
        .header(
            "THE PROBLEM",
            f"Resize it to {TILE} px and each car is a smudge",
            f"A detector with a {TILE} × {TILE} input would see "
            f"{round(car_w * f)} × {round(car_h * f)} pixel blobs.",
        )
        .zoom_marker(zx, zy, zs)
        .right_caption(f"Resized to {TILE} px", "same close-up"),
        2600,
    )

    # 3. The tile grid.
    base = left.copy()
    _grid(base, coords, n_cols, len(coords))
    mid = len(coords) // 2
    x1, y1, x2, y2 = coords[mid].tolist()
    _grid(base, coords, n_cols, len(coords), hi=mid)
    add(
        Frame(base, cv2.resize(shown[y1:y2, x1:x2], (RS, RS), interpolation=cv2.INTER_AREA))
        .header(
            "1 · TILE",
            f"Cut it into {len(coords)} overlapping {TILE} × {TILE} tiles",
            "Every tile keeps the full resolution. Overlaps catch the cars on the seams.",
        )
        .right_caption(f"Tile {mid + 1} of {len(coords)}", f"tile_size={TILE}, step_size={STEP}"),
        2000,
    )

    # 4. Detection on every tile; boxes accumulate on the full image.
    acc = 0
    for k in range(len(coords)):
        base = left.copy()
        _grid(base, coords, n_cols, len(coords), hi=k)
        x1, y1, x2, y2 = coords[k].tolist()
        tile_view = shown[y1:y2, x1:x2].copy()
        _boxes_on(tile_view, per_tile[k][0], (255, 255, 255), 3)
        _boxes_on(tile_view, per_tile[k][0], _tile_color(k, n_cols), 2)
        acc += len(raw[k])
        frame = Frame(base, cv2.resize(tile_view, (RS, RS), interpolation=cv2.INTER_AREA))
        for j in range(k + 1):
            frame.boxes(raw[j], _rgb(_tile_color(j, n_cols)))
        add(
            frame.header(
                "2 · DETECT",
                "Run the detector on every tile",
                "Batches of full-resolution tiles go to your model.",
            )
            .right_caption(f"Tile {k + 1} of {len(coords)}: {len(raw[k])} boxes", "")
            .footer(f"boxes so far: {acc}", MUTED),
            260 if k < len(coords) - 1 else 900,
        )

    # 5. Merge: duplicates in the overlaps collapse to one box per car.
    all_raw = np.concatenate(raw)
    owner = np.concatenate([np.full(len(r), i) for i, r in enumerate(raw)])
    dup = (_iou(dets.boxes, all_raw) > 0.3).sum(1)  # raw boxes behind each merged box
    best, best_n = (zx, zy), -1
    for cx, cy in (dets.boxes[dup > 1][:, :2]).astype(int).tolist():
        x0, y0 = int(np.clip(cx - 60, 0, SCENE_W - zs)), int(np.clip(cy - 60, 0, SCENE_H - zs))
        inside = (dets.boxes[:, 0] >= x0) & (dets.boxes[:, 2] <= x0 + zs)
        inside &= (dets.boxes[:, 1] >= y0) & (dets.boxes[:, 3] <= y0 + zs)
        n = int((dup[inside] > 1).sum())
        if n > best_n:
            best, best_n = (x0, y0), n
    ox, oy = best
    z = RS / zs
    for merged in (False, True):
        crop = crisp[oy : oy + zs, ox : ox + zs]
        view = cv2.resize(crop, (RS, RS), interpolation=cv2.INTER_CUBIC)
        shift = np.array([ox, oy, ox, oy])
        if merged:
            _boxes_on(view, (dets.boxes - shift) * z, _rgb(ACCENT), 2)
        else:
            for i in np.unique(owner):
                sel = owner == i
                pad = (i % 3) * 2  # nudge each tile's boxes so the duplicates show
                _boxes_on(
                    view,
                    (all_raw[sel] - shift) * z + [-pad, -pad, pad, pad],
                    _tile_color(int(i), n_cols),
                    2,
                )
        base = left.copy()
        _grid(base, coords, n_cols, len(coords))
        frame = Frame(base, view).zoom_marker(ox, oy, zs)
        if merged:
            frame.boxes(dets.boxes, ACCENT)
            caption = (f"{len(dets)} boxes after merging", "one box per car")
            foot = f"{n_raw} boxes → {len(dets)} (class-aware NMS + seam fusion)"
        else:
            for i in np.unique(owner):
                frame.boxes(all_raw[owner == i], _rgb(_tile_color(int(i), n_cols)))
            caption = ("Where tiles overlap", "the same car is found twice")
            foot = f"{n_raw} boxes, in image coordinates"
        add(
            frame.header(
                "3 · MERGE",
                "Move boxes back and merge duplicates",
                "Overlapping tiles see the same car; plakakia keeps one box.",
            )
            .right_caption(*caption)
            .footer(foot),
            2200,
        )

    # 6. The result.
    final_view = native.copy()
    zz = RS / zs
    _boxes_on(final_view, (dets.boxes - [zx, zy, zx, zy]) * zz, _rgb(ACCENT), 2)
    result = (
        Frame(left, final_view)
        .header(
            "RESULT",
            f"{found} of {len(gt)} cars found, at full resolution",
            "dets = predict_tiled(image, model, tile_size=640)",
        )
        .zoom_marker(zx, zy, zs)
        .boxes(dets.boxes, ACCENT)
        .right_caption("Close-up", "same place as the first frame")
        .footer(
            f"Stand-in OpenCV detector ({false_alarms} false alarms). Plug in any model.", MUTED
        )
    )
    add(result, 4500)
    return frames, durations


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    """One shared palette for every frame, so the colours do not flicker."""
    picks = np.linspace(0, len(frames) - 1, 8).round().astype(int).tolist()
    sample = Image.new("RGB", (W, H * len(picks)))
    for i, k in enumerate(picks):
        sample.paste(frames[k], (0, H * i))
    palette = sample.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    # Pin the flat UI colours, so the background and text stay exact.
    entries = np.array(palette.getpalette()[: 192 * 3]).reshape(-1, 3)
    for color in (BG, FG, MUTED, ACCENT):
        entries[np.abs(entries - color).sum(1).argmin()] = color
    palette.putpalette(entries.flatten().tolist())
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )


def main() -> None:
    frames, durations = build()
    OUT.mkdir(parents=True, exist_ok=True)
    save_gif(frames, durations, OUT / "hero.gif")
    frames[-1].save(OUT / "hero.jpg", quality=88, optimize=True)
    size = (OUT / "hero.gif").stat().st_size / 1e6
    print(f"hero.gif: {len(frames)} frames, {sum(durations) / 1000:.1f} s, {size:.2f} MB")


if __name__ == "__main__":
    main()
