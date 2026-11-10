"""Interactive tiling explorer.

Run with ``uv run --extra demo streamlit run demo/app.py``.
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from plakakia import tile_image

st.set_page_config(page_title="plakakia tiling explorer", layout="centered")
st.title("plakakia tiling explorer")

upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
if upload is None:
    st.info("Upload an image to get started.")
    st.stop()

image = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

with st.sidebar:
    st.header("Tiling")
    scale = st.slider("Resize (%)", 5, 100, 100) / 100
    max_side = max(1, int(min(height, width) * scale))
    tile_size = st.slider("Tile size", 8, max_side, min(256, max_side))
    step_size = st.slider("Step size", 8, max(8, tile_size * 2), tile_size)
    cover_edges = st.checkbox("Cover edges", value=True)

if scale != 1:
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

tiles, coords = tile_image(image, tile_size, step_size, cover_edges=cover_edges)

overlay = image.copy()
for tile, (x1, y1, x2, y2) in zip(tiles, coords.tolist(), strict=True):
    # Draw each tile outline in the inverse of its mean colour so it is always visible.
    color = tuple(int(255 - c) for c in tile.reshape(-1, tile.shape[-1]).mean(axis=0))
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 1)

col1, col2, col3 = st.columns(3)
col1.metric("Image", f"{image.shape[1]}x{image.shape[0]}")
col2.metric("Tiles", len(tiles))
covered = 0.0
if len(coords):
    covered = (coords[:, 2].max() * coords[:, 3].max()) / (image.shape[0] * image.shape[1])
col3.metric("Covered area", f"{covered:.0%}")

st.image(overlay, use_container_width=True)
if len(tiles):
    st.subheader("First tiles")
    st.image(list(tiles[:12]), width=96)
