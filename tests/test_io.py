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
