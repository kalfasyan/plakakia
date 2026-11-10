"""Image reading/writing helpers built on OpenCV.

Files are decoded from bytes (``np.fromfile`` + ``cv2.imdecode``) rather than with
``cv2.imread`` so that non-ASCII paths work on every platform, and failures raise
exceptions instead of silently returning ``None``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

__all__ = ["pad_image", "read_image", "write_image"]

PathLike = str | Path


def read_image(path: PathLike, *, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Read an image from disk.

    By default the image is read unchanged, which preserves grayscale, alpha and
    16-bit data. Colour images are returned in OpenCV's BGR channel order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_image(path: PathLike, image: np.ndarray, params: Sequence[int] = ()) -> None:
    """Write an image to disk; the format is inferred from the file extension.

    Raises:
        OSError: If OpenCV cannot encode the image in the requested format.
    """
    path = Path(path)
    message = f"Could not encode image {path} (shape={image.shape}, dtype={image.dtype})."
    try:
        ok, buffer = cv2.imencode(path.suffix, image, list(params))
    except cv2.error as err:
        raise OSError(message) from err
    if not ok:
        raise OSError(message)
    buffer.tofile(path)


def pad_image(image: np.ndarray, pad: int, value: float = 0) -> np.ndarray:
    """Pad an image with a constant border of ``pad`` pixels on every side.

    Works for any number of channels (unlike ``cv2.copyMakeBorder``).
    """
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}.")
    if pad == 0:
        return image
    widths = [(pad, pad), (pad, pad)] + [(0, 0)] * (image.ndim - 2)
    return np.pad(image, widths, mode="constant", constant_values=value)
