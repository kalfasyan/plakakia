"""Image and N-D array reading/writing.

Images go through OpenCV. Files are decoded from bytes (``np.fromfile`` +
``cv2.imdecode``) rather than with ``cv2.imread`` so that non-ASCII paths work on every
platform, and failures raise exceptions instead of silently returning ``None``.

N-D arrays (hyperspectral cubes, multi-band rasters) are read from ``.npy``, HDF5 and
TIFF files, and GeoTIFFs can be read and written with their georeferencing through
``rasterio``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

__all__ = [
    "GeoProfile",
    "pad_image",
    "read_array",
    "read_geotiff",
    "read_image",
    "write_array",
    "write_geotiff",
    "write_image",
]

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
    message = (
        f"Could not encode image {path} (shape={image.shape}, dtype={image.dtype}). "
        "Use a .npy, .h5 or .tif extension for multi-band or floating point data."
    )
    try:
        ok, buffer = cv2.imencode(path.suffix, image, list(params))
    except cv2.error as err:
        raise OSError(message) from err
    if not ok:
        raise OSError(message)
    buffer.tofile(path)


def pad_image(
    image: np.ndarray, pad: int, value: float = 0, *, axes: Sequence[int] = (0, 1)
) -> np.ndarray:
    """Pad an array with a constant border of ``pad`` pixels along ``axes``.

    Works for any number of dimensions and channels (unlike ``cv2.copyMakeBorder``).
    """
    if pad < 0:
        raise ValueError(f"pad must be non-negative, got {pad}.")
    if pad == 0:
        return image
    resolved = {a % image.ndim for a in axes}
    widths = [(pad, pad) if d in resolved else (0, 0) for d in range(image.ndim)]
    return np.pad(image, widths, mode="constant", constant_values=value)


# --------------------------------------------------------------------------- N-D arrays

NUMPY_EXTENSIONS = {".npy"}
TIFF_EXTENSIONS = {".tif", ".tiff"}
HDF5_EXTENSIONS = {".h5", ".hdf5", ".he5"}
DEFAULT_HDF5_DATASET = "data"


def _optional_import(module: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"Reading this file type needs `{module}`: install it with "
            f"`pip install plakakia[{extra}]` (or `uv add 'plakakia[{extra}]'`)."
        ) from err


def read_array(path: PathLike, *, hdf5_dataset: str | None = None, mmap: bool = True) -> np.ndarray:
    """Read an image or N-D array (e.g. a hyperspectral cube) from disk.

    The reader is chosen from the extension:

    * ``.npy``: NumPy, memory-mapped by default so only the tiles you use are read.
    * ``.h5`` / ``.hdf5`` / ``.he5``: HDF5 via ``h5py`` (``plakakia[hyperspectral]``).
      ``hdf5_dataset`` names the dataset to read; it may be omitted when the file holds
      a single dataset. The file is closed before returning.
    * ``.tif`` / ``.tiff``: ``tifffile`` when installed (multi-band TIFF/GeoTIFF),
      otherwise OpenCV.
    * anything else: OpenCV, see :func:`read_image`.

    The array is returned as stored; no axis is assumed to hold bands.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix in NUMPY_EXTENSIONS:
        array: np.ndarray = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        return array
    if suffix in HDF5_EXTENSIONS:
        h5py = _optional_import("h5py", "hyperspectral")
        with h5py.File(path, "r") as f:
            key = hdf5_dataset
            if key is None:
                names: list[str] = []

                def collect(name: str, obj: Any) -> None:
                    if hasattr(obj, "shape"):
                        names.append(name)

                f.visititems(collect)
                if len(names) != 1:
                    raise ValueError(
                        f"{path} contains {len(names)} datasets {names}; "
                        "set `hdf5_dataset` to choose one."
                    )
                key = names[0]
            if key not in f:
                raise KeyError(f"{path} has no dataset {key!r}.")
            return np.asarray(f[key][()])
    if suffix in TIFF_EXTENSIONS:
        try:
            tifffile = importlib.import_module("tifffile")
        except ImportError:
            return read_image(path)
        return np.asarray(tifffile.imread(path))
    return read_image(path)


def write_array(path: PathLike, array: np.ndarray, *, hdf5_dataset: str | None = None) -> None:
    """Write an array with the writer matching the extension (see :func:`read_array`).

    ``.npy`` and HDF5 accept any shape and dtype; TIFF uses ``tifffile`` when installed;
    other extensions go through OpenCV (:func:`write_image`).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in NUMPY_EXTENSIONS:
        np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    elif suffix in HDF5_EXTENSIONS:
        h5py = _optional_import("h5py", "hyperspectral")
        with h5py.File(path, "w") as f:
            f.create_dataset(hdf5_dataset or DEFAULT_HDF5_DATASET, data=array)
    elif suffix in TIFF_EXTENSIONS:
        try:
            tifffile = importlib.import_module("tifffile")
        except ImportError:
            write_image(path, array)
        else:
            tifffile.imwrite(path, np.ascontiguousarray(array))
    else:
        write_image(path, array)


# --------------------------------------------------------------------------- GeoTIFF


@dataclass(frozen=True)
class GeoProfile:
    """Georeferencing of a raster: coordinate reference system, affine transform, nodata.

    ``transform`` maps ``(col, row)`` pixel coordinates to map coordinates.
    """

    crs: Any
    transform: Any
    nodata: float | None = None

    def for_window(self, col_off: int, row_off: int) -> GeoProfile:
        """Profile of a sub-window starting at pixel ``(col_off, row_off)``."""
        rasterio = _optional_import("rasterio", "geo")
        a, b, c, d, e, f = (self.transform[i] for i in range(6))
        # Origin moves to the window's top-left pixel; scale and rotation are unchanged.
        shifted = rasterio.Affine(
            a, b, c + a * col_off + b * row_off, d, e, f + d * col_off + e * row_off
        )
        return GeoProfile(self.crs, shifted, self.nodata)


def read_geotiff(path: PathLike) -> tuple[np.ndarray, GeoProfile]:
    """Read a (multi-band) GeoTIFF with ``rasterio`` (``plakakia[geo]``).

    Returns:
        ``(array, profile)`` where ``array`` is ``(bands, height, width)``, rasterio's
        native layout, so tile it with ``axes=(1, 2)``.
    """
    rasterio = _optional_import("rasterio", "geo")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with rasterio.open(path) as src:
        array = src.read()
        profile = GeoProfile(src.crs, src.transform, src.nodata)
    return np.asarray(array), profile


def write_geotiff(path: PathLike, array: np.ndarray, profile: GeoProfile) -> None:
    """Write a ``(bands, height, width)`` (or ``(height, width)``) array as a GeoTIFF."""
    rasterio = _optional_import("rasterio", "geo")
    data = array[np.newaxis] if array.ndim == 2 else array
    if data.ndim != 3:
        raise ValueError(f"GeoTIFFs must be (bands, height, width), got shape {array.shape}.")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        crs=profile.crs,
        transform=profile.transform,
        nodata=profile.nodata,
        compress="deflate",
    ) as dst:
        dst.write(data)
