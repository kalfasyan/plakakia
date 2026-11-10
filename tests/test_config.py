from __future__ import annotations

import os
from pathlib import Path

import pytest

from plakakia import TilingConfig


def test_defaults_and_normalisation(tmp_path: Path) -> None:
    cfg = TilingConfig(
        input_dir_images=str(tmp_path),  # type: ignore[arg-type]
        input_dir_annotations=tmp_path,
        input_extension_images=".JPG",
        num_workers=-1,
    )
    assert cfg.input_dir_images == tmp_path
    assert cfg.input_extension_images == "jpg"
    assert cfg.num_workers == (os.cpu_count() or 1)
    assert cfg.output_extension_annotations == "txt"
    assert cfg.replace(output_format_annotations="pascal_voc").output_extension_annotations == "xml"
    assert (
        cfg.replace(input_format_annotations="segmentation").output_extension_annotations == "png"
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input_format_annotations": "csv"}, "input_format_annotations"),
        ({"output_format_annotations": "coco"}, "output_format_annotations"),
        ({"input_dir_annotations": None}, "input_dir_annotations is required"),
        ({"tile_size": 0}, "positive"),
        ({"pad_size": -1}, "non-negative"),
        ({"min_visibility": 0}, "min_visibility"),
        ({"num_workers": -2}, "num_workers"),
    ],
)
def test_validation(tmp_path: Path, changes: dict[str, object], message: str) -> None:
    kwargs: dict[str, object] = {"input_dir_images": tmp_path, "input_dir_annotations": tmp_path}
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        TilingConfig(**kwargs)  # type: ignore[arg-type]


def test_from_yaml_legacy_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
input_dir_images: {tmp_path}
input_dir_annotations: {tmp_path}
tile_size: 64
validate_settings: true
log: false
check_partial: true
partial_overlap_threshold: 0.6
annotation_mapping: {{cat: 0, dog: 1}}
"""
    )
    with pytest.warns(DeprecationWarning, match="validate_settings"):
        cfg = TilingConfig.from_yaml(path)
    assert cfg.tile_size == 64
    assert cfg.min_visibility == 0.6
    assert cfg.label_names == {0: "cat", 1: "dog"}


def test_from_dict_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown config keys: \\['tile_sise'\\]"):
        TilingConfig.from_dict({"input_dir_images": tmp_path, "tile_sise": 3})


def test_from_yaml_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        TilingConfig.from_yaml(path)


def test_example_config_loads() -> None:
    example = Path(__file__).parents[1] / "examples" / "config.yaml"
    cfg = TilingConfig.from_yaml(example)
    assert cfg.tile_size > 0
