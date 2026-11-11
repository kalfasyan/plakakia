from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plakakia import pad_image, read_image, write_image


@pytest.mark.parametrize("shape", [(20, 30), (20, 30, 3), (20, 30, 4)])
def test_png_roundtrip_unicode_path(tmp_path: Path, shape: tuple[int, ...]) -> None:
    image = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    path = tmp_path / "πλακάκι.png"
    write_image(path, image)
    np.testing.assert_array_equal(read_image(path), image)


def test_16bit_roundtrip(tmp_path: Path) -> None:
    image = np.arange(600, dtype=np.uint16).reshape(20, 30) * 100
    write_image(tmp_path / "a.png", image)
    np.testing.assert_array_equal(read_image(tmp_path / "a.png"), image)


def test_read_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    (tmp_path / "junk.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Could not decode"):
        read_image(tmp_path / "junk.png")


def test_write_error(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Could not encode"):
        write_image(tmp_path / "a.unknownext", np.zeros((4, 4), np.uint8))


def test_pad_image() -> None:
    image = np.ones((4, 5, 7))
    padded = pad_image(image, 2)
    assert padded.shape == (8, 9, 7)
    assert padded[0].sum() == 0
    np.testing.assert_array_equal(padded[2:-2, 2:-2], image)
    assert pad_image(image, 0) is image
    with pytest.raises(ValueError, match="non-negative"):
        pad_image(image, -1)


# --------------------------------------------------------------------------- N-D arrays

import h5py  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from plakakia import read_array, write_array  # noqa: E402
from plakakia.io import GeoProfile, read_geotiff, write_geotiff  # noqa: E402

CUBE = np.random.default_rng(1).random((12, 14, 9), dtype=np.float32)


@pytest.mark.parametrize("ext", ["npy", "h5", "tif"])
def test_array_roundtrip(tmp_path: Path, ext: str) -> None:
    path = tmp_path / f"cube.{ext}"
    write_array(path, CUBE)
    out = read_array(path)
    np.testing.assert_array_equal(out, CUBE)


def test_npy_is_memory_mapped(tmp_path: Path) -> None:
    write_array(tmp_path / "a.npy", CUBE)
    assert isinstance(read_array(tmp_path / "a.npy"), np.memmap)
    assert not isinstance(read_array(tmp_path / "a.npy", mmap=False), np.memmap)


def test_hdf5_dataset_selection(tmp_path: Path) -> None:
    path = tmp_path / "multi.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("cube", data=CUBE)
        f.create_dataset("meta/wavelengths", data=np.arange(9))
    with pytest.raises(ValueError, match="2 datasets"):
        read_array(path)
    np.testing.assert_array_equal(read_array(path, hdf5_dataset="cube"), CUBE)
    np.testing.assert_array_equal(read_array(path, hdf5_dataset="meta/wavelengths"), np.arange(9))
    with pytest.raises(KeyError, match="no dataset 'nope'"):
        read_array(path, hdf5_dataset="nope")
    # The file is closed after reading, so it can be reopened for writing.
    write_array(path, CUBE[..., :2], hdf5_dataset="cube")
    np.testing.assert_array_equal(read_array(path, hdf5_dataset="cube"), CUBE[..., :2])


def test_read_array_falls_back_to_opencv(tmp_path: Path) -> None:
    image = np.random.default_rng(0).integers(0, 255, (10, 10, 3), dtype=np.uint8)
    write_array(tmp_path / "a.png", image)
    np.testing.assert_array_equal(read_array(tmp_path / "a.png"), image)
    with pytest.raises(FileNotFoundError):
        read_array(tmp_path / "missing.npy")


def test_pad_image_axes() -> None:
    padded = pad_image(np.ones((3, 4, 5)), 1, axes=(1, 2))
    assert padded.shape == (3, 6, 7)


def test_geotiff_roundtrip_and_window(tmp_path: Path) -> None:
    profile = GeoProfile(rasterio.CRS.from_epsg(32631), from_origin(500000, 4000000, 10, 10), 0)
    data = (CUBE.transpose(2, 0, 1) * 1000).astype(np.uint16)  # (bands, H, W)
    write_geotiff(tmp_path / "a.tif", data, profile)
    out, read_profile = read_geotiff(tmp_path / "a.tif")
    np.testing.assert_array_equal(out, data)
    assert read_profile.crs == profile.crs
    assert read_profile.nodata == 0
    window = read_profile.for_window(3, 2)
    assert window.transform.c == 500000 + 3 * 10
    assert window.transform.f == 4000000 - 2 * 10
    with pytest.raises(ValueError, match="bands, height, width"):
        write_geotiff(tmp_path / "b.tif", np.zeros((1, 2, 3, 4)), profile)
