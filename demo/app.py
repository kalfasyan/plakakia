"""plakakia playground: see how an image or a hyperspectral cube is cut into tiles.

Run with ``uv run --extra demo streamlit run demo/app.py``.
"""

from __future__ import annotations

import csv
import io
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from samples import CLASS_NAMES, WAVELENGTHS, make_scene, scene_cube, scene_rgb

import plakakia
from plakakia import (
    Annotations,
    assign_boxes_to_tiles,
    clip_boxes_to_tile,
    read_array,
    read_image,
    read_pascal_voc,
    read_yolo,
    tile_array,
    tile_coordinates,
    tile_image,
    write_yolo,
)

HERE = Path(__file__).parent
LOGO = HERE.parent / "docs" / "assets" / "logo.png"
DOCS = "https://kalfasyan.github.io/plakakia"
REPO = "https://github.com/kalfasyan/plakakia"

IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp", "webp")
ARRAY_TYPES = ("tif", "tiff", "npy", "h5", "hdf5")
GALLERY_SIZE = 48
MAX_DOWNLOAD_BYTES = 400 * 2**20

# Outline colours of the tile grid (RGB): azulejo blues and a warm accent.
GRID_COLORS = ((255, 255, 255), (120, 190, 255), (255, 214, 102))
SELECTED_COLOR = (255, 64, 129)
CLASS_COLORS = ((255, 214, 102), (102, 255, 178), (102, 217, 255), (255, 133, 102))

st.set_page_config(
    page_title="plakakia playground",
    page_icon=str(LOGO) if LOGO.exists() else "🧩",
    layout="wide",
)
if LOGO.exists():
    st.logo(str(LOGO), link=REPO, size="large")


# --------------------------------------------------------------------------- data


@dataclass
class Data:
    """What the app tiles, plus how to look at it."""

    name: str
    array: np.ndarray
    yx_axes: tuple[int, int] = (0, 1)
    band_axis: int | None = None
    """Spectral axis of a hyperspectral cube; ``None`` for ordinary images."""
    wavelengths: np.ndarray | None = None
    boxes: Annotations = field(default_factory=Annotations.empty)
    class_names: tuple[str, ...] = ()
    load_code: str = ""
    labels_file: str = "field.txt"

    @property
    def is_cube(self) -> bool:
        return self.band_axis is not None

    @property
    def height(self) -> int:
        return int(self.array.shape[self.yx_axes[0]])

    @property
    def width(self) -> int:
        return int(self.array.shape[self.yx_axes[1]])

    @property
    def bands(self) -> int:
        return 0 if self.band_axis is None else int(self.array.shape[self.band_axis])

    def class_name(self, label: int) -> str:
        return self.class_names[label] if label < len(self.class_names) else f"class {label}"


@st.cache_data(show_spinner="Generating the sample scene…")
def sample_image() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene = make_scene()
    return scene_rgb(scene), scene.boxes, scene.labels


@st.cache_data(show_spinner="Generating the sample cube…")
def sample_cube() -> np.ndarray:
    return scene_cube(make_scene())


@st.cache_data(show_spinner="Reading your file…", max_entries=4)
def read_upload(name: str, payload: bytes) -> np.ndarray:
    suffix = Path(name).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(payload)
        if suffix.lstrip(".") in IMAGE_TYPES:
            image = read_image(path)
            if image.ndim == 3:  # OpenCV is BGR(A)
                code = cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
                image = cv2.cvtColor(image, code)
            return image
        return np.array(read_array(path, mmap=False))


def read_labels(
    upload: st.runtime.uploaded_file_manager.UploadedFile, hw: tuple[int, int]
) -> Annotations:
    suffix = Path(upload.name).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"labels{suffix}"
        path.write_bytes(upload.getvalue())
        return read_yolo(path, hw) if suffix == ".txt" else read_pascal_voc(path)


def looks_like_image(array: np.ndarray) -> bool:
    return array.ndim == 2 or (
        array.ndim == 3 and array.shape[2] in (3, 4) and array.dtype == np.uint8
    )


def nearest_band(wavelengths: np.ndarray, nm: float) -> int:
    return int(np.abs(wavelengths - nm).argmin())


# --------------------------------------------------------------------------- display helpers


@st.cache_data(max_entries=4)
def band_limits(array: np.ndarray, band_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """2nd/98th percentile of every band, for a contrast stretch that ignores outliers."""
    sample = np.moveaxis(np.asarray(array), band_axis, -1)[::4, ::4].reshape(
        -1, array.shape[band_axis]
    )
    lo, hi = np.nanpercentile(sample.astype(np.float32), [2, 98], axis=0)
    return lo, np.maximum(hi, lo + 1e-6)


def to_rgb(
    array: np.ndarray,
    yx_axes: tuple[int, int],
    band_axis: int | None,
    rgb_bands: tuple[int, int, int] | None = None,
    limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Render an image, a cube or a tile of either as an 8-bit ``(H, W, 3)`` image."""
    if band_axis is None:
        img = np.asarray(array)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        img = img[..., :3]
        if img.dtype != np.uint8:
            lo, hi = np.nanpercentile(img, [1, 99])
            img = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        return np.ascontiguousarray(img.astype(np.uint8))
    cube = np.moveaxis(np.asarray(array), (*yx_axes, band_axis), (0, 1, 2)).astype(np.float32)
    bands = rgb_bands or (cube.shape[2] - 1, cube.shape[2] // 2, 0)
    bands = tuple(min(b, cube.shape[2] - 1) for b in bands)
    rgb = cube[..., list(bands)]
    if limits is not None:
        lo, hi = limits[0][list(bands)].min(), limits[1][list(bands)].max()
    else:
        lo, hi = np.nanpercentile(rgb, [2, 98])
    # One stretch shared by the three channels keeps the colours natural.
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return np.ascontiguousarray((np.nan_to_num(rgb) * 255).astype(np.uint8))


def draw_grid(
    rgb: np.ndarray,
    coords: np.ndarray,
    selected: int,
    boxes: Annotations,
    keep: np.ndarray | None,
) -> np.ndarray:
    """Dim uncovered pixels, outline every tile and highlight the selected one."""
    h, w = rgb.shape[:2]
    covered = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in coords.tolist():
        covered[y1:y2, x1:x2] = True
    out = rgb.copy()
    out[~covered] = (out[~covered] * 0.3).astype(np.uint8)

    thick = max(1, round(max(h, w) / 600))
    for (x1, y1, x2, y2), lab in zip(boxes.boxes.tolist(), boxes.labels.tolist(), strict=True):
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), CLASS_COLORS[lab % len(CLASS_COLORS)], thick)

    grid = out.copy()
    for i, (x1, y1, x2, y2) in enumerate(coords.tolist()):
        cv2.rectangle(grid, (x1, y1), (x2 - 1, y2 - 1), GRID_COLORS[i % len(GRID_COLORS)], thick)
    out = cv2.addWeighted(grid, 0.75, out, 0.25, 0)

    if len(coords):
        x1, y1, x2, y2 = coords[selected].tolist()
        tint = out.copy()
        cv2.rectangle(tint, (x1, y1), (x2 - 1, y2 - 1), SELECTED_COLOR, -1)
        out = cv2.addWeighted(tint, 0.18, out, 0.82, 0)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), SELECTED_COLOR, thick * 3)
        if keep is not None:
            for x1b, y1b, x2b, y2b in boxes.boxes[keep[selected]].tolist():
                cv2.rectangle(out, (x1b, y1b), (x2b - 1, y2b - 1), SELECTED_COLOR, thick * 2)
    return out


def draw_tile_boxes(tile_rgb: np.ndarray, local: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = tile_rgb.copy()
    thick = max(1, round(max(out.shape[:2]) / 150))
    for (x1, y1, x2, y2), lab in zip(local.tolist(), labels.tolist(), strict=True):
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), CLASS_COLORS[lab % len(CLASS_COLORS)], thick)
    return out


def upscale(img: np.ndarray, target: int = 360) -> np.ndarray:
    """Enlarge small tiles with crisp pixels so they are easy to inspect."""
    factor = max(1, target // max(img.shape[:2]))
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def nice_round(value: float, multiple: int = 16) -> int:
    return max(multiple, round(value / multiple) * multiple)


# --------------------------------------------------------------------------- sidebar: data

with st.sidebar:
    st.markdown("### ① Pick your data")
    source = st.radio(
        "Data source",
        ["🛰️ Drone photo", "🌈 Hyperspectral cube", "📁 Your own file"],
        captions=[
            "RGB, 63 labelled houses, trees and cars",
            "48 bands from 400 to 1000 nm",
            "Image, TIFF, .npy or HDF5",
        ],
        label_visibility="collapsed",
    )

data: Data
if source == "🛰️ Drone photo":
    image, boxes, labels = sample_image()
    data = Data(
        "Synthetic drone photo · 1024×768 RGB with 63 labelled objects",
        image,
        boxes=Annotations(boxes, labels),
        class_names=CLASS_NAMES,
        load_code='image = read_image("field.jpg")  # (H, W, 3)\n',
    )
elif source == "🌈 Hyperspectral cube":
    data = Data(
        f"Synthetic hyperspectral cube · 384×512 pixels × {len(WAVELENGTHS)} bands (400-1000 nm)",
        sample_cube(),
        band_axis=2,
        wavelengths=WAVELENGTHS,
        load_code='cube = read_array("field.npy")  # (H, W, bands), memory-mapped\n',
    )
else:
    with st.sidebar:
        upload = st.file_uploader(
            "Image or array",
            type=[*IMAGE_TYPES, *ARRAY_TYPES],
            help="Photos (JPG, PNG…) or N-D arrays: multi-band TIFF, `.npy`, HDF5.",
        )
    if upload is None:
        st.title("Upload something to tile")
        st.info(
            "Drop an **image** (JPG, PNG…) or a **hyperspectral / multispectral array** "
            "(`.npy`, `.h5`, multi-band `.tif`) in the sidebar. Nothing leaves your machine.",
            icon="👈",
        )
        st.stop()
    try:
        array = read_upload(upload.name, upload.getvalue())
    except (ImportError, ValueError, KeyError, OSError) as err:
        st.error(f"Could not read **{upload.name}**: {err}", icon="⚠️")
        st.stop()
    loader = (
        "read_image"
        if Path(upload.name).suffix.lower().lstrip(".") in IMAGE_TYPES
        else "read_array"
    )
    load_code = f'array = {loader}("{upload.name}")  # {tuple(array.shape)} {array.dtype}\n'
    if array.ndim not in (2, 3):
        st.error(
            f"**{upload.name}** has shape {array.shape}. The playground shows 2-D and 3-D data; "
            "use `plakakia.tile_array(array, tile_shape, axes=...)` in Python for more axes.",
            icon="📐",
        )
        st.stop()
    if looks_like_image(array):
        data = Data(
            f"{upload.name} · {array.shape[1]}×{array.shape[0]}", array, load_code=load_code
        )
        with st.sidebar:
            label_file = st.file_uploader(
                "Bounding boxes (optional)",
                type=["txt", "xml"],
                help="A YOLO `.txt` or Pascal VOC `.xml` file for this image.",
            )
        if label_file is not None:
            try:
                data.boxes = read_labels(label_file, (data.height, data.width))
                data.labels_file = label_file.name
            except (ValueError, KeyError, OSError) as err:
                st.sidebar.error(f"Could not read the labels: {err}")
    else:
        guess = int(np.argmin(array.shape))
        with st.sidebar:
            band_axis = st.radio(
                "Which axis holds the bands?",
                [0, 1, 2],
                index=guess,
                horizontal=True,
                format_func=lambda a: f"axis {a} ({array.shape[a]})",
                help="Hyperspectral cubes are stored as (H, W, bands) or (bands, H, W).",
            )
        yx = tuple(a for a in range(3) if a != band_axis)
        data = Data(
            f"{upload.name} · {array.shape} {array.dtype}",
            array,
            yx_axes=(yx[0], yx[1]),
            band_axis=band_axis,
            load_code=load_code,
        )

# --------------------------------------------------------------------------- sidebar: tiling

short_side = min(data.height, data.width)
with st.sidebar:
    st.markdown("### ② Shape the tiles")
    max_tile = max(8, short_side)
    tile = st.slider(
        "Tile size (px)",
        min_value=min(16, max_tile),
        max_value=max_tile,
        value=min(max_tile, nice_round(short_side / 3)),
        step=1 if max_tile < 64 else 8,
        help="Side of each square tile. Pick the input size of your model, e.g. 640 for YOLO.",
    )
    overlap = st.slider(
        "Overlap",
        0,
        75,
        25,
        step=5,
        format="%d%%",
        help="How much neighbouring tiles share. Overlap keeps objects on a tile border "
        "whole in at least one tile.",
    )
    step = max(1, round(tile * (1 - overlap / 100)))
    cover_edges = st.toggle(
        "Cover the edges",
        value=True,
        help="Add tiles aligned to the right and bottom borders so no pixel is dropped.",
    )

    band_tile = band_step = 0
    rgb_bands: tuple[int, int, int] | None = None
    if data.is_cube:
        if data.bands > 1 and st.toggle(
            "Also split the bands",
            key="split_bands",
            help="Tile the spectral axis too, e.g. to train on groups of neighbouring bands.",
        ):
            band_tile = st.slider("Bands per tile", 1, data.bands, max(1, data.bands // 4))
            band_step = band_tile
        with st.expander("🎨 Display colours"):
            if data.wavelengths is not None:
                default = tuple(nearest_band(data.wavelengths, nm) for nm in (640, 550, 460))
                fmt = lambda b: f"{data.wavelengths[b]:.0f} nm"  # noqa: E731
            else:
                default = (int(data.bands * 0.8), data.bands // 2, int(data.bands * 0.2))
                fmt = lambda b: f"band {b}"  # noqa: E731
            r = st.select_slider("Red", range(data.bands), default[0], format_func=fmt)
            g = st.select_slider("Green", range(data.bands), default[1], format_func=fmt)
            b = st.select_slider("Blue", range(data.bands), default[2], format_func=fmt)
            rgb_bands = (r, g, b)
            st.caption("Only changes how the cube is drawn, not the tiles.")

    min_visibility = 1.0
    if len(data.boxes):
        st.markdown("### ③ Boxes")
        min_visibility = (
            st.slider(
                "Keep a box if this much of it is inside the tile",
                10,
                100,
                100,
                step=10,
                format="%d%%",
                help="100% keeps only whole objects. Lower values keep cut objects, clipped to "
                "the tile edge.",
            )
            / 100
        )

    st.divider()
    st.caption(f"plakakia {plakakia.__version__} · [Docs]({DOCS}) · [GitHub]({REPO})")

# --------------------------------------------------------------------------- tiling

coords = tile_coordinates(data.height, data.width, tile, step, cover_edges=cover_edges)
n_spatial = len(coords)
n_band_chunks = 1
if band_tile:
    n_band_chunks = len(plakakia.tile_starts(data.bands, band_tile, band_step))
n_tiles = n_spatial * n_band_chunks

keep: np.ndarray | None = None
if len(data.boxes):
    keep = assign_boxes_to_tiles(data.boxes.boxes, coords, min_visibility=min_visibility)

limits = band_limits(data.array, data.band_axis) if data.band_axis is not None else None
preview = to_rgb(data.array, data.yx_axes, data.band_axis, rgb_bands, limits)

covered = np.zeros((data.height, data.width), bool)
for x1, y1, x2, y2 in coords.tolist():
    covered[y1:y2, x1:x2] = True
coverage = float(covered.mean())

# --------------------------------------------------------------------------- header

logo_col, title_col = st.columns([1, 12], vertical_alignment="center")
if LOGO.exists():
    logo_col.image(str(LOGO), width=72)
title_col.title("plakakia playground")
st.markdown(
    "Big images do not fit into a neural network. **plakakia** cuts them into tiles, "
    "keeps your bounding boxes in sync, and works on hyperspectral cubes along any axis. "
    "Move the sliders on the left and watch the grid change."
)
st.caption(data.name)

tile_shape = f"{tile}×{tile}" + (f"×{band_tile}" if band_tile else "")
cols = st.columns(5 if keep is not None else 4)
cols[0].metric("Tiles", f"{n_tiles:,}", border=True, help="Number of tiles produced.")
cols[1].metric("Tile shape", tile_shape, border=True, help="Height × width (× bands).")
cols[2].metric(
    "Step",
    f"{step} px",
    border=True,
    help="Distance between the starts of two neighbouring tiles: tile size × (1 − overlap).",
)
cols[3].metric(
    "Pixels covered",
    f"{coverage:.0%}",
    border=True,
    help="Share of the image inside at least one tile. Turn on 'Cover the edges' to reach 100%.",
)
if keep is not None:
    kept = int(keep.any(axis=0).sum())
    cols[4].metric(
        "Objects kept",
        f"{kept}/{len(data.boxes)}",
        border=True,
        help="Objects that end up in at least one tile with the current visibility threshold.",
    )

if n_spatial == 0:
    st.warning("The tile is larger than the image. Make the tiles smaller.", icon="📏")
    st.stop()

# --------------------------------------------------------------------------- tabs

tab_names = ["🗺️ Tile grid", "🧱 Tiles"]
if data.is_cube:
    tab_names.append("📈 Spectra")
tab_names.append("🐍 Code")
tabs = dict(zip(tab_names, st.tabs(tab_names), strict=True))


def tile_slices(index: int) -> list[slice]:
    x1, y1, x2, y2 = coords[index].tolist()
    sl = [slice(None)] * data.array.ndim
    sl[data.yx_axes[0]], sl[data.yx_axes[1]] = slice(y1, y2), slice(x1, x2)
    return sl


def tile_rgb(index: int) -> np.ndarray:
    """RGB rendering of one tile, with the whole image's contrast so tiles look consistent."""
    window = data.array[tuple(tile_slices(index))]
    return to_rgb(window, data.yx_axes, data.band_axis, rgb_bands, limits)


def band_group_gray(index: int, group: int) -> np.ndarray:
    """Mean of one group of bands of a tile, as a grey image."""
    assert data.band_axis is not None
    b0 = group * band_step
    sl = tile_slices(index)
    sl[data.band_axis] = slice(b0, b0 + band_tile)
    window = np.moveaxis(data.array[tuple(sl)], (*data.yx_axes, data.band_axis), (0, 1, 2))
    mean = window.astype(np.float32).mean(axis=2, keepdims=True)
    lim = None
    if limits is not None:
        lim = (
            limits[0][b0 : b0 + band_tile].min(keepdims=True),
            limits[1][b0 : b0 + band_tile].max(keepdims=True),
        )
    return to_rgb(mean, (0, 1), 2, (0, 0, 0), lim)


with tabs["🗺️ Tile grid"]:
    left, right = st.columns([3, 1.25], gap="large")
    with right:
        st.markdown("**Inspect a tile**")
        selected = (
            st.slider(
                "Tile",
                0,
                n_spatial - 1,
                min(n_spatial - 1, n_spatial // 2),
                label_visibility="collapsed",
            )
            if n_spatial > 1
            else 0
        )
        chunk = 0
        if band_tile and n_band_chunks > 1:
            chunk = st.slider("Band group", 0, n_band_chunks - 1, 0)
        x1, y1, x2, y2 = coords[selected].tolist()
        view = band_group_gray(selected, chunk) if band_tile else tile_rgb(selected)
        n_in = 0
        if keep is not None:
            idx = keep[selected]
            n_in = int(idx.sum())
            local = clip_boxes_to_tile(data.boxes.boxes[idx], coords[selected])
            view = draw_tile_boxes(view, local, data.boxes.labels[idx])
        st.image(upscale(view), width="stretch")
        st.markdown(f"Tile **#{selected}** covers `x {x1}–{x2}`, `y {y1}–{y2}`")
        if band_tile:
            b0 = chunk * band_step
            st.markdown(f"Showing the mean of bands `{b0}–{min(b0 + band_tile, data.bands) - 1}`")
        if keep is not None:
            names = [data.class_name(int(c)) for c in data.boxes.labels[keep[selected]]]
            summary = ", ".join(f"{names.count(n)} {n}" for n in dict.fromkeys(names))
            st.markdown(
                f"**{n_in}** object{'s' if n_in != 1 else ''}" + (f": {summary}" if summary else "")
            )
    with left:
        st.image(draw_grid(preview, coords, selected, data.boxes, keep), width="stretch")
        notes = [
            ":violet-badge[Selected tile]",
            ":blue-badge[Tile outlines]",
        ]
        if keep is not None:
            notes.append(":orange-badge[Bounding boxes]")
        if coverage < 1:
            notes.append(":gray-badge[Dimmed = not in any tile]")
        st.markdown(" ".join(notes))

with tabs["🧱 Tiles"]:
    only_boxes = keep is not None and st.toggle("Only tiles that contain objects", value=False)
    indices = np.arange(n_spatial)
    if only_boxes and keep is not None:
        indices = indices[keep.any(axis=1)]
    shown = indices[:GALLERY_SIZE]
    st.caption(
        f"Showing {len(shown)} of {len(indices)} tiles"
        + (f", each one also split into {n_band_chunks} band groups." if band_tile else ".")
        + (" Boxes are already shifted and clipped to each tile." if keep is not None else "")
    )
    thumbs, captions = [], []
    for i in shown.tolist():
        thumb = tile_rgb(i)
        if keep is not None:
            idx = keep[i]
            thumb = draw_tile_boxes(
                thumb, clip_boxes_to_tile(data.boxes.boxes[idx], coords[i]), data.boxes.labels[idx]
            )
        thumbs.append(cv2.resize(thumb, (160, 160), interpolation=cv2.INTER_AREA))
        captions.append(f"#{i}" + (f" · {int(keep[i].sum())} obj" if keep is not None else ""))
    if thumbs:
        st.image(thumbs, caption=captions, width=160)
    else:
        st.info(
            "No tile contains a whole object. Lower the visibility threshold or enlarge the tiles."
        )

    tile_bytes = (
        n_tiles
        * tile
        * tile
        * max(1, band_tile or data.bands or (3 if data.array.ndim == 3 else 1))
    )
    tile_bytes *= data.array.dtype.itemsize

    def make_zip() -> bytes:
        """All tiles (PNG for images, NPY for cubes), YOLO labels and a coordinates table."""
        if data.is_cube:
            axes = (*data.yx_axes, data.band_axis) if band_tile else data.yx_axes
            sizes = (tile, tile, band_tile) if band_tile else (tile, tile)
            steps = (step, step, band_step) if band_tile else (step, step)
            tiles, starts = tile_array(data.array, sizes, steps, axes=axes, cover_edges=cover_edges)
        else:
            tiles, xyxy = tile_image(data.array, tile, step, cover_edges=cover_edges)
            starts = xyxy[:, [1, 0]]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            table = io.StringIO()
            writer = csv.writer(table)
            writer.writerow(["tile", "y", "x", *(["band"] if band_tile else [])])
            for i, (t, s) in enumerate(zip(tiles, starts.tolist(), strict=True)):
                writer.writerow([i, *s])
                if data.is_cube:
                    npy = io.BytesIO()
                    np.save(npy, t)
                    zf.writestr(f"tiles/tile_{i:05d}.npy", npy.getvalue())
                    continue
                bgr = (
                    t
                    if t.ndim == 2
                    else cv2.cvtColor(
                        t, cv2.COLOR_RGBA2BGRA if t.shape[2] == 4 else cv2.COLOR_RGB2BGR
                    )
                )
                zf.writestr(f"tiles/tile_{i:05d}.png", cv2.imencode(".png", bgr)[1].tobytes())
                if keep is not None:
                    idx = keep[i]
                    local = clip_boxes_to_tile(data.boxes.boxes[idx], coords[i])
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / "labels.txt"
                        write_yolo(path, Annotations(local, data.boxes.labels[idx]), (tile, tile))
                        zf.writestr(f"labels/tile_{i:05d}.txt", path.read_text())
            zf.writestr("tiles.csv", table.getvalue())
        return buf.getvalue()

    if tile_bytes > MAX_DOWNLOAD_BYTES:
        st.caption(
            f"Download disabled: {n_tiles:,} tiles would take about {tile_bytes / 2**20:,.0f} MB. "
            "Use larger tiles, less overlap, or the CLI for big jobs."
        )
    else:
        st.download_button(
            f"⬇️ Download all {n_tiles:,} tiles (.zip)",
            data=make_zip,
            file_name="plakakia_tiles.zip",
            mime="application/zip",
            type="primary",
            help="PNG tiles (NPY for cubes), YOLO labels when boxes are present, and tiles.csv "
            "with the position of every tile.",
        )

if data.is_cube:
    with tabs["📈 Spectra"]:
        cube_hwb = np.moveaxis(np.asarray(data.array), (*data.yx_axes, data.band_axis), (0, 1, 2))
        x1, y1, x2, y2 = coords[selected].tolist()
        axis_name = "Wavelength (nm)" if data.wavelengths is not None else "Band"
        x_values = data.wavelengths if data.wavelengths is not None else np.arange(data.bands)
        frame = pd.DataFrame(
            {
                axis_name: x_values,
                "Whole image": np.nanmean(cube_hwb, axis=(0, 1)),
                f"Tile #{selected}": np.nanmean(cube_hwb[y1:y2, x1:x2], axis=(0, 1)),
            }
        )
        c1, c2 = st.columns([1, 1.4], gap="large")
        with c1:
            band = st.slider(
                "Browse the cube band by band",
                0,
                data.bands - 1,
                data.bands // 2,
                format="band %d",
            )
            if data.wavelengths is not None:
                st.caption(f"Band {band} ≈ {data.wavelengths[band]:.0f} nm")
            gray = cube_hwb[..., band].astype(np.float32)
            lo, hi = (
                (limits[0][band], limits[1][band])
                if limits is not None
                else (gray.min(), gray.max())
            )
            gray = (np.clip((gray - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(
                gray, (x1, y1), (x2 - 1, y2 - 1), SELECTED_COLOR, max(1, data.width // 300)
            )
            st.image(gray, width="stretch")
        with c2:
            st.markdown(
                f"**Mean spectrum** of tile #{selected} (pick it in *Tile grid*) "
                "vs. the whole image"
            )
            st.line_chart(
                frame,
                x=axis_name,
                y=["Whole image", f"Tile #{selected}"],
                color=["#9aa5b1", "#ff4081"],
            )
            st.caption(
                "Every tile keeps all bands unless you split them: "
                f"`tiles.shape == (T, {tile}, {tile}, {band_tile or data.bands})`."
            )

with tabs["🐍 Code"]:
    st.markdown("**Do the same in Python**")
    if data.is_cube:
        if band_tile:
            axes = (*data.yx_axes, data.band_axis)
            call = (
                "tiles, starts = tile_array(\n"
                "    cube,\n"
                f"    tile_shape=({tile}, {tile}, {band_tile}),\n"
                f"    step=({step}, {step}, {band_step}),\n"
                f"    axes={axes},  # (y, x, bands)\n"
                f"    cover_edges={cover_edges},\n"
                ")\n"
                f"# {n_tiles} tiles; starts holds the (y, x, band) offset of each one"
            )
            imports = "from plakakia import read_array, tile_array"
        else:
            axes_arg = "" if data.yx_axes == (0, 1) else f", axes={data.yx_axes}"
            call = (
                f"tiles, coords = tile_image(cube, tile_size={tile}, step_size={step}"
                f"{axes_arg}, cover_edges={cover_edges})\n"
                f"# {n_tiles} tiles, each with all {data.bands} bands; coords are (x1, y1, x2, y2)"
            )
            imports = "from plakakia import read_array, tile_image"
        load = data.load_code.replace("array =", "cube =")
    else:
        call = (
            f"tiles, coords = tile_image(image, tile_size={tile}, step_size={step}, "
            f"cover_edges={cover_edges})\n# tiles.shape == ({n_tiles}, {tile}, {tile}"
            + (f", {data.array.shape[2]}" if data.array.ndim == 3 else "")
            + "); coords are (x1, y1, x2, y2)"
        )
        imports = "from plakakia import read_image, tile_image"
        load = data.load_code.replace("array =", "image =")
        if keep is not None:
            imports = (
                "from plakakia import (\n    assign_boxes_to_tiles,\n    clip_boxes_to_tile,\n"
                "    read_image,\n    read_yolo,\n    tile_image,\n)"
            )
            call += (
                f'\n\nlabels = read_yolo("{data.labels_file}", image.shape[:2])\n'
                "keep = assign_boxes_to_tiles(labels.boxes, coords, "
                f"min_visibility={min_visibility})\n"
                "for tile_pixels, tile_xyxy, mask in zip(tiles, coords, keep):\n"
                "    boxes = clip_boxes_to_tile(labels.boxes[mask], tile_xyxy)  # in tile pixels"
            )
    st.code(f"{imports}\n\n{load}{call}\n", language="python")

    st.markdown("**…or tile a whole folder from the command line**")
    yaml_lines = [
        "input_dir_images: data/images",
        f"input_extension_images: {'npy' if data.is_cube else 'jpg'}",
    ]
    if keep is not None:
        yaml_lines += ["input_dir_annotations: data/labels", "input_format_annotations: yolo"]
    else:
        yaml_lines.append("input_format_annotations: none")
    if data.is_cube:
        yaml_lines.append("output_extension_images: npy")
        axes_list = [*data.yx_axes, *([data.band_axis] if band_tile else [])]
        yaml_lines.append(f"axes: {axes_list}")
        if band_tile:
            yaml_lines += [
                f"tile_size: [{tile}, {tile}, {band_tile}]",
                f"step_size: [{step}, {step}, {band_step}]",
            ]
        else:
            yaml_lines += [f"tile_size: {tile}", f"step_size: {step}"]
    else:
        yaml_lines += [f"tile_size: {tile}", f"step_size: {step}"]
    yaml_lines.append(f"cover_edges: {str(cover_edges).lower()}")
    if keep is not None:
        yaml_lines.append(f"min_visibility: {min_visibility}")
    st.code("\n".join(yaml_lines) + "\n", language="yaml")
    st.code("plakakia --config config.yaml", language="bash")
    st.caption(f"More in the [getting-started guide]({DOCS}/getting-started/).")
