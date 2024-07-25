#!/usr/bin/env python
# coding: utf-8

# Author: Ioannis Kalfas (@kalfasyan) (kalfasyan at gmail dot com)

import argparse
import multiprocessing as mp
import os
import random
import shutil
from pathlib import Path
from time import perf_counter

import yaml
from tqdm import tqdm

from plakakia.settings import Settings
from plakakia.tiling import clear_duplicates, process_tiles

random.seed(3)

def process_tiles_wrapper(args):
    """Wrapper function for the process_tile function."""
    t, input_image, input_annotation, settings = args
    return process_tiles(t, input_image, input_annotation, settings)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config.yaml file')
    args = parser.parse_args()
    print(100*'-')
    config_path=args.config

    start_time = perf_counter()

    if config_path is None:
        print(f"No config file provided. Using default file.")
        # If no config_path is provided, use the default config.yaml file in the package
        package_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plakakia')
        config_file = os.path.join(package_dir, 'config_example.yaml')
    else:
        config_file = config_path

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create a settings object
    settings = Settings(**config)

    # Create a process pool with the desired number of workers
    with mp.Pool(processes=settings.num_workers) as pool:
        # Prepare the arguments for each task
        args = [(t, input_image, input_annotation, settings) \
            for t, (input_image, input_annotation) in \
                enumerate(zip(settings.input_images, settings.input_annotations))]

        # Submit the tasks to the pool
        results = pool.map(process_tiles_wrapper, args)

    end_time = perf_counter() - start_time

    print(f"Finished making tiles! Elapsed time: {end_time:.2f} seconds")

    if settings.clear_duplicates:
        clear_duplicates(settings)

if __name__ == '__main__':
    # Call the main function with the provided config path
    main()