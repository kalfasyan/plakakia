from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from plakakia import __version__
from plakakia.cli import main


def write_config(dataset: Path, tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
input_dir_images: {dataset / "images"}
input_dir_annotations: {dataset / "yolo"}
input_extension_images: png
output_dir_images: {tmp_path / "out" / "images"}
output_dir_annotations: {tmp_path / "out" / "annotations"}
tile_size: 40
step_size: 40
"""
    )
    return path


def test_cli_runs(dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(write_config(dataset, tmp_path)), "--quiet", "-j", "1"]) == 0
    assert "tiles from 2 images" in capsys.readouterr().out
    assert any((tmp_path / "out" / "images").iterdir())


def test_cli_reports_errors(tmp_path: Path) -> None:
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 1


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_python_dash_m() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "plakakia", "--help"], capture_output=True, text=True, check=True
    )
    assert "--config" in out.stdout
