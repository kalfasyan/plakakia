from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plakakia import (
    TilingConfig,
    discover_samples,
    read_image,
    read_pascal_voc,
    read_yolo,
    run,
    write_image,
)


def make_config(dataset: Path, tmp_path: Path, **kwargs: object) -> TilingConfig:
    fmt = kwargs.pop("input_format_annotations", "yolo")
    defaults: dict[str, object] = {
        "input_dir_images": dataset / "images",
        "input_dir_annotations": None if fmt == "none" else dataset / str(fmt),
        "input_format_annotations": fmt,
        "input_extension_images": "png",
        "output_extension_images": "png",
        "output_dir_images": tmp_path / "out" / "images",
        "output_dir_annotations": tmp_path / "out" / "annotations",
        "tile_size": 40,
        "step_size": 20,
        "annotation_mapping": {"cat": 0, "dog": 1},
    }
    defaults.update(kwargs)
    return TilingConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt", ["yolo", "pascal_voc", "coco"])
def test_detection_formats_agree(dataset: Path, tmp_path: Path, fmt: str) -> None:
    cfg = make_config(dataset, tmp_path / fmt, input_format_annotations=fmt)
    results = run(cfg, progress=False)

    assert len(results) == 2
    assert all(r.num_tiles > 0 and r.skipped is None for r in results)
    images = sorted(cfg.output_dir_images.glob("*.png"))
    labels = sorted(cfg.output_dir_annotations.glob("*.txt"))
    assert [p.stem for p in images] == [p.stem for p in labels]
    assert len(images) == sum(r.num_tiles for r in results)
    # Every box in every tile lies inside the tile.
    for label in labels:
        ann = read_yolo(label, (40, 40))
        assert len(ann) > 0
        assert (ann.boxes >= 0).all() and (ann.boxes <= 40).all()


def test_tile_pixels_and_boxes_are_consistent(dataset: Path, tmp_path: Path) -> None:
    cfg = make_config(dataset, tmp_path)
    run(cfg, progress=False)
    source = read_image(dataset / "images" / "img0.png")
    for path in cfg.output_dir_images.glob("tile_img0_*.png"):
        x1, y1, x2, y2 = map(int, path.stem.split("_")[-4:])
        np.testing.assert_array_equal(read_image(path), source[y1:y2, x1:x2])


def test_pascal_voc_output(dataset: Path, tmp_path: Path) -> None:
    cfg = make_config(dataset, tmp_path, output_format_annotations="pascal_voc")
    run(cfg, progress=False)
    xmls = list(cfg.output_dir_annotations.glob("*.xml"))
    assert xmls
    text = xmls[0].read_text()
    assert "<name>cat</name>" in text or "<name>dog</name>" in text
    assert len(read_pascal_voc(xmls[0], cfg.annotation_mapping)) > 0


def test_clear_duplicates_keeps_every_box(dataset: Path, tmp_path: Path) -> None:
    full = run(make_config(dataset, tmp_path / "a", step_size=5), progress=False)
    dedup_cfg = make_config(dataset, tmp_path / "b", step_size=5, clear_duplicates=True)
    dedup = run(dedup_cfg, progress=False)
    assert sum(r.num_tiles for r in dedup) < sum(r.num_tiles for r in full)
    # 3 boxes per image, each must survive in at least one tile.
    assert all(r.num_boxes >= 3 for r in dedup)


def test_save_empty_tiles_and_cover_edges(dataset: Path, tmp_path: Path) -> None:
    cfg = make_config(dataset, tmp_path, save_empty_tiles=True, cover_edges=True, step_size=50)
    results = run(cfg, progress=False)
    # 120x160 image, tile 40, step 50: starts y=[0,50,80], x=[0,50,100,120]
    assert all(r.num_tiles == 12 for r in results)
    empty = [p for p in cfg.output_dir_annotations.glob("*.txt") if not p.read_text()]
    assert empty


def test_partial_boxes_are_clipped(dataset: Path, tmp_path: Path) -> None:
    cfg = make_config(dataset, tmp_path, min_visibility=0.3)
    results = run(cfg, progress=False)
    full = run(make_config(dataset, tmp_path / "full"), progress=False)
    assert sum(r.num_boxes for r in results) > sum(r.num_boxes for r in full)


def test_padding_shifts_boxes(dataset: Path, tmp_path: Path) -> None:
    cfg = make_config(dataset, tmp_path, pad_image=True, pad_size=10, draw_boxes=True)
    run(cfg, progress=False)
    # Box [10,10,30,30] becomes [20,20,40,40] in the padded frame -> fits tile (0,0,40,40).
    ann = read_yolo(cfg.output_dir_annotations / "tile_img0_0_0_40_40.txt", (40, 40))
    assert [20, 20, 40, 40] in ann.boxes.tolist()


def test_segmentation(dataset: Path, tmp_path: Path) -> None:
    cfg = make_config(dataset, tmp_path, input_format_annotations="segmentation")
    results = run(cfg, progress=False)
    masks = sorted(cfg.output_dir_annotations.glob("mask_*.png"))
    assert (
        len(masks)
        == sum(r.num_tiles for r in results)
        == len(list(cfg.output_dir_images.glob("*.png")))
    )
    source = read_image(dataset / "segmentation" / "img1.png")
    x1, y1, x2, y2 = map(int, masks[-1].stem.split("_")[-4:])
    np.testing.assert_array_equal(read_image(masks[-1]), source[y1:y2, x1:x2])


def test_no_annotations(dataset: Path, tmp_path: Path) -> None:
    cfg = make_config(dataset, tmp_path, input_format_annotations="none")
    results = run(cfg, progress=False)
    assert all(r.num_tiles == 5 * 7 for r in results)
    assert not cfg.output_dir_annotations.exists()


def test_multiprocessing_matches_serial(dataset: Path, tmp_path: Path) -> None:
    serial = run(make_config(dataset, tmp_path / "s"), progress=False)
    parallel = run(make_config(dataset, tmp_path / "p", num_workers=2), progress=False)
    assert [(r.num_tiles, r.num_boxes) for r in serial] == [
        (r.num_tiles, r.num_boxes) for r in parallel
    ]


def test_small_images_are_skipped(dataset: Path, tmp_path: Path) -> None:
    results = run(make_config(dataset, tmp_path, tile_size=500), progress=False)
    assert all(r.skipped and "smaller than tile_size" in r.skipped for r in results)


def test_discovery_errors(dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"No '\*\.jpg' images"):
        discover_samples(make_config(dataset, tmp_path, input_extension_images="jpg"))
    with pytest.raises(FileNotFoundError, match=r"No '\*\.txt' annotations"):
        discover_samples(make_config(dataset, tmp_path, input_dir_annotations=dataset / "coco"))
    with pytest.raises(FileNotFoundError, match="Image directory does not exist"):
        discover_samples(make_config(dataset, tmp_path, input_dir_images=tmp_path / "nope"))


def test_missing_annotation_is_treated_as_empty(dataset: Path, tmp_path: Path) -> None:
    (dataset / "yolo" / "img1.txt").unlink()
    results = run(make_config(dataset, tmp_path), progress=False)
    assert results[0].num_tiles > 0
    assert results[1].num_tiles == 0


def test_mask_size_mismatch(dataset: Path, tmp_path: Path) -> None:

    write_image(dataset / "segmentation" / "img0.png", np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError, match="has size"):
        run(make_config(dataset, tmp_path, input_format_annotations="segmentation"), progress=False)
