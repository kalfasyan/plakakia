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
    tile_coordinates,
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
    assert all(r.skipped and "smaller than tile" in r.skipped for r in results)


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


# --------------------------------------------------------------------------- hyperspectral

import h5py  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from plakakia import Annotations, read_array, write_array, write_yolo  # noqa: E402
from plakakia.io import GeoProfile, read_geotiff, write_geotiff  # noqa: E402
from plakakia.pipeline import nodata_fraction  # noqa: E402


@pytest.fixture
def cubes(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Two channels-first (bands=12, H=60, W=80) float cubes with YOLO labels."""
    root = tmp_path / "cubes"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(2):
        write_array(root / "images" / f"cube{i}.npy", rng.random((12, 60, 80), dtype=np.float32))
        write_yolo(
            root / "labels" / f"cube{i}.txt",
            Annotations([[5, 5, 15, 15], [45, 25, 70, 50]], [0, 1]),
            (60, 80),
        )
    return root


def cube_config(cubes: Path, tmp_path: Path, **kwargs: object) -> TilingConfig:
    defaults: dict[str, object] = {
        "input_dir_images": cubes / "images",
        "input_dir_annotations": cubes / "labels",
        "input_extension_images": "npy",
        "output_extension_images": "npy",
        "output_dir_images": tmp_path / "out" / "images",
        "output_dir_annotations": tmp_path / "out" / "labels",
        "axes": [1, 2],
        "tile_size": 20,
        "step_size": 20,
        "save_empty_tiles": True,
    }
    defaults.update(kwargs)
    return TilingConfig(**defaults)  # type: ignore[arg-type]


def test_channels_first_cube(cubes: Path, tmp_path: Path) -> None:
    cfg = cube_config(cubes, tmp_path)
    results = run(cfg, progress=False)
    assert [r.num_tiles for r in results] == [12, 12]  # 3 x 4 spatial tiles each
    source = np.load(cubes / "images" / "cube0.npy")
    tile = np.load(cfg.output_dir_images / "tile_cube0_40_20_60_40.npy")
    np.testing.assert_array_equal(tile, source[:, 20:40, 40:60])
    ann = read_yolo(cfg.output_dir_annotations / "tile_cube0_0_0_20_20.txt", (20, 20))
    assert ann.boxes.tolist() == [[5, 5, 15, 15]]


def test_spectral_tiling(cubes: Path, tmp_path: Path) -> None:
    cfg = cube_config(cubes, tmp_path, axes=[1, 2, 0], tile_size=[20, 20, 4], step_size=[20, 20, 4])
    results = run(cfg, progress=False)
    assert [r.num_tiles for r in results] == [36, 36]  # 12 spatial x 3 spectral windows
    source = np.load(cubes / "images" / "cube1.npy")
    tile = np.load(cfg.output_dir_images / "tile_cube1_20_0_40_20_ax0-8-12.npy")
    np.testing.assert_array_equal(tile, source[8:12, 0:20, 20:40])
    labels = list(cfg.output_dir_annotations.glob("tile_cube1_0_0_20_20_ax0-*.txt"))
    assert len(labels) == 3


def test_band_selection(cubes: Path, tmp_path: Path) -> None:
    cfg = cube_config(cubes, tmp_path, bands=[3, 2, 1], band_axis=0)
    run(cfg, progress=False)
    source = np.load(cubes / "images" / "cube0.npy")
    tile = np.load(cfg.output_dir_images / "tile_cube0_0_0_20_20.npy")
    np.testing.assert_array_equal(tile, source[[3, 2, 1], :20, :20])


def test_hdf5_inputs(cubes: Path, tmp_path: Path) -> None:
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    for npy in (cubes / "images").glob("*.npy"):
        with h5py.File(h5_dir / f"{npy.stem}.h5", "w") as f:
            f.create_dataset("reflectance", data=np.load(npy))
            f.create_dataset("wavelengths", data=np.arange(12))
    cfg = cube_config(
        cubes,
        tmp_path,
        input_dir_images=h5_dir,
        input_extension_images="h5",
        output_extension_images="h5",
        hdf5_dataset="reflectance",
    )
    results = run(cfg, progress=False)
    assert sum(r.num_tiles for r in results) == 24
    tile = read_array(cfg.output_dir_images / "tile_cube0_0_0_20_20.h5", hdf5_dataset="reflectance")
    np.testing.assert_array_equal(tile, np.load(cubes / "images" / "cube0.npy")[:, :20, :20])


def test_multiband_needs_array_format(cubes: Path, tmp_path: Path) -> None:
    cfg = cube_config(cubes, tmp_path, output_extension_images="jpg")
    with pytest.raises(OSError, match=r"Use a \.npy, \.h5 or \.tif"):
        run(cfg, progress=False)


def test_nodata_fraction_matches_brute_force(rng: np.random.Generator) -> None:
    image = rng.random((3, 30, 40))
    image[:, :, :10] = 0  # left quarter has no data
    image[0, 5, 20] = 0  # one band only: still valid
    coords = tile_coordinates(30, 40, 10, 5)
    frac = nodata_fraction(image, 0.0, coords, (1, 2))
    expected = [(image[:, y1:y2, x1:x2] == 0).all(axis=0).mean() for x1, y1, x2, y2 in coords]
    np.testing.assert_allclose(frac, expected)
    image[:, :, :10] = np.nan
    np.testing.assert_allclose(nodata_fraction(image, np.nan, coords, (1, 2)), expected)
    np.testing.assert_allclose(
        nodata_fraction(image.transpose(0, 2, 1), np.nan, coords, (2, 1)), expected
    )


def test_geotiff_pipeline(tmp_path: Path, rng: np.random.Generator) -> None:
    src = tmp_path / "geo" / "images"
    src.mkdir(parents=True)
    data = rng.integers(1, 1000, (4, 40, 60), dtype=np.uint16)  # 4-band scene
    data[:, :, :20] = 0  # outside the swath
    crs = rasterio.CRS.from_epsg(32631)
    write_geotiff(src / "scene.tif", data, GeoProfile(crs, from_origin(600000, 5000000, 10, 10), 0))

    cfg = TilingConfig(
        input_dir_images=src,
        input_extension_images="tif",
        input_format_annotations="none",
        output_dir_images=tmp_path / "out",
        output_extension_images="tif",
        georeference=True,
        axes=(1, 2),
        tile_size=20,
        step_size=20,
        max_nodata_fraction=0.5,
    )
    results = run(cfg, progress=False)
    assert results[0].num_tiles == 4  # 2 x 3 tiles, the left column is all nodata

    tile, profile = read_geotiff(cfg.output_dir_images / "tile_scene_40_20_60_40.tif")
    np.testing.assert_array_equal(tile, data[:, 20:40, 40:60])
    assert profile.crs == crs
    assert profile.nodata == 0
    assert (profile.transform.c, profile.transform.f) == (600000 + 40 * 10, 5000000 - 20 * 10)
    assert not (cfg.output_dir_images / "tile_scene_0_0_20_20.tif").exists()


def test_georeference_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="axes to \\[1, 2\\]"):
        TilingConfig(input_dir_images=tmp_path, input_format_annotations="none", georeference=True)
    with pytest.raises(ValueError, match="output_extension_images: tif"):
        TilingConfig(
            input_dir_images=tmp_path,
            input_format_annotations="none",
            georeference=True,
            axes=(1, 2),
        )
    with pytest.raises(ValueError, match="needs a `nodata` value"):
        TilingConfig(
            input_dir_images=tmp_path, input_format_annotations="none", max_nodata_fraction=0.5
        )
