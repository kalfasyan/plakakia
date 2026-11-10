"""Benchmark the offline pipeline and `tile_image` on a synthetic YOLO dataset.

Usage::

    uv run python benchmarks/bench_pipeline.py --images 200 --size 1024 --workers 1 4
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from plakakia import Annotations, TilingConfig, run, tile_image, write_image, write_yolo


def make_dataset(root: Path, n_images: int, size: int, n_boxes: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(n_images):
        image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        xy = rng.integers(0, size - 64, (n_boxes, 2))
        wh = rng.integers(8, 64, (n_boxes, 2))
        boxes = np.concatenate([xy, xy + wh], axis=1)
        write_image(root / "images" / f"{i:05d}.jpg", image)
        write_yolo(
            root / "labels" / f"{i:05d}.txt", Annotations(boxes, np.zeros(n_boxes)), image.shape[:2]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--boxes", type=int, default=20)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    image = np.zeros((4096, 4096, 3), np.uint8)
    start = time.perf_counter()
    for _ in range(10):
        tile_image(image, args.tile_size, args.step_size)
    per_call = (time.perf_counter() - start) / 10 * 1e3
    print(f"tile_image 4096x4096x3 tile={args.tile_size} step={args.step_size}: {per_call:.1f} ms")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root / "data", args.images, args.size, args.boxes)
        for workers in args.workers:
            out = root / f"out{workers}"
            config = TilingConfig(
                input_dir_images=root / "data" / "images",
                input_dir_annotations=root / "data" / "labels",
                output_dir_images=out / "images",
                output_dir_annotations=out / "labels",
                tile_size=args.tile_size,
                step_size=args.step_size,
                num_workers=workers,
            )
            start = time.perf_counter()
            results = run(config, progress=False)
            elapsed = time.perf_counter() - start
            tiles = sum(r.num_tiles for r in results)
            print(
                f"pipeline workers={workers}: {args.images} images -> {tiles} tiles "
                f"in {elapsed:.2f}s ({args.images / elapsed:.1f} img/s)"
            )
            shutil.rmtree(out)


if __name__ == "__main__":
    main()
