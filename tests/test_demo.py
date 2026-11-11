"""Smoke test for the Streamlit demo app (skipped unless the `demo` extra is installed).

The app runs in a subprocess: Streamlit's runtime starts background threads that would
make later ``fork``-based multiprocessing tests deadlock if it ran in the pytest process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

APP = Path(__file__).parents[1] / "demo" / "app.py"

SCRIPT = textwrap.dedent(
    """
    import sys
    from streamlit.testing.v1 import AppTest

    PHOTO, CUBE = "🛰️ Drone photo", "🌈 Hyperspectral cube"


    def start(source):
        at = AppTest.from_file(sys.argv[1], default_timeout=60)
        at.run()
        at.radio[0].set_value(source).run()
        assert not at.exception, at.exception
        return at


    def n_tiles(at):
        assert at.metric[0].label == "Tiles"
        return int(at.metric[0].value.replace(",", ""))


    # Both samples render and show runnable code.
    for source in (PHOTO, CUBE):
        at = start(source)
        assert n_tiles(at) > 0
        assert "from plakakia import" in at.code[0].value

    # Larger tiles give fewer tiles.
    at = start(PHOTO)
    before = n_tiles(at)
    next(s for s in at.slider if s.label == "Tile size (px)").set_value(512).run()
    assert not at.exception, at.exception
    assert n_tiles(at) < before

    # Splitting the bands switches the snippet to tile_array.
    at = start(CUBE)
    next(t for t in at.toggle if t.label == "Also split the bands").set_value(True).run()
    assert not at.exception, at.exception
    assert "tile_array" in at.code[0].value
    """
)


def test_demo_app_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(APP)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
