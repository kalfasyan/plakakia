"""Command line interface: ``plakakia --config config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from . import __version__
from .config import TilingConfig
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="plakakia",
        description="Tile an image dataset (and its YOLO/VOC/COCO labels or masks) to disk.",
    )
    parser.add_argument("-c", "--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "-j", "--num-workers", type=int, help="Override num_workers (-1 = all cores)."
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Hide the progress bar.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.captureWarnings(True)
    try:
        config = TilingConfig.from_yaml(args.config)
        if args.num_workers is not None:
            config = config.replace(num_workers=args.num_workers)
        start = time.perf_counter()
        results = run(config, progress=not args.quiet)
    except (OSError, ValueError, KeyError) as err:
        logging.getLogger("plakakia").error("%s", err)
        return 1
    elapsed = time.perf_counter() - start
    tiles = sum(r.num_tiles for r in results)
    print(f"Done: {tiles} tiles from {len(results)} images in {elapsed:.2f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
