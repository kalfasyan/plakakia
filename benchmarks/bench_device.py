"""Benchmark device tiling and NMS against the NumPy path.

Times ``tile_image`` on a NumPy image and on PyTorch tensors (CPU, and CUDA or MPS when
available), plus ``nms`` in NumPy and through torchvision. GPU timings synchronize the
device, so they include the whole kernel and not just its launch.

Usage::

    uv run python benchmarks/bench_device.py --size 4096 --tile 640 --step 512
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from plakakia import nms, tile_image


def timeit(fn: Callable[[], Any], sync: Callable[[], None], repeat: int) -> float:
    fn()  # warm up (CUDA context, kernel caches)
    sync()
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096, help="image side in pixels")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--step", type=int, default=512)
    parser.add_argument("--boxes", type=int, default=20_000, help="boxes for the NMS benchmark")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (args.size, args.size, 3), dtype=np.uint8)
    xy = rng.uniform(0, args.size, (args.boxes, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(4, 64, (args.boxes, 2))], axis=1)
    scores = rng.uniform(0, 1, args.boxes)
    classes = rng.integers(0, 10, args.boxes)

    def noop() -> None:
        pass

    rows: list[tuple[str, str, float]] = []

    def tile_np() -> Any:
        return tile_image(image, args.tile, args.step, cover_edges=True)

    def nms_np() -> Any:
        return nms(boxes, scores, 0.5, class_ids=classes)

    rows.append(("tile_image", "numpy", timeit(tile_np, noop, args.repeat)))
    rows.append(("nms", "numpy", timeit(nms_np, noop, args.repeat)))

    try:
        import torch  # noqa: PLC0415 - optional dependency
    except ImportError:
        print("PyTorch is not installed: only the NumPy path was timed.")
        torch = None

    if torch is not None:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        for device in devices:
            sync = {
                "cuda": torch.cuda.synchronize,
                "mps": getattr(torch, "mps", None) and torch.mps.synchronize,
            }.get(device) or noop
            # Channels-first, as a GPU pipeline would hold it.
            t_image = torch.as_tensor(image, device=device).permute(2, 0, 1).contiguous()
            t_boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
            t_scores = torch.as_tensor(scores, dtype=torch.float32, device=device)
            t_classes = torch.as_tensor(classes, device=device)
            rows.append(
                (
                    "tile_image",
                    f"torch:{device}",
                    timeit(
                        lambda t=t_image: tile_image(
                            t, args.tile, args.step, axes=(1, 2), cover_edges=True
                        ),
                        sync,
                        args.repeat,
                    ),
                )
            )
            rows.append(
                (
                    "nms",
                    f"torch:{device}",
                    timeit(
                        lambda b=t_boxes, s=t_scores, c=t_classes: nms(b, s, 0.5, class_ids=c),
                        sync,
                        args.repeat,
                    ),
                )
            )

    tiles, _ = tile_np()
    print(
        f"image {args.size}x{args.size}x3 uint8, {len(tiles)} tiles of {args.tile}px "
        f"(step {args.step}); NMS on {args.boxes} boxes, 10 classes; best of {args.repeat}\n"
    )
    base = {op: t for op, backend, t in rows if backend == "numpy"}
    print(f"{'operation':<12}{'backend':<14}{'time (ms)':>12}{'vs numpy':>10}")
    for op, backend, t in rows:
        print(f"{op:<12}{backend:<14}{t * 1e3:>12.2f}{base[op] / t:>9.1f}x")


if __name__ == "__main__":
    main()
