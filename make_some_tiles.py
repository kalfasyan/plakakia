#!/usr/bin/env python
# coding: utf-8

# Author:

import concurrent.futures
import random
import shutil
from pathlib import Path
from time import perf_counter

import yaml
from tqdm import tqdm

from settings import Settings
from utils_tiling import process_tile

random.seed(3)

# Delete the tiles and annotations folders if they exist
[shutil.rmtree(x) if Path(x).exists() else None for x in [
               'tiles/', 'output/', 'annotations/', 'images/', 'logs/']]

start_time = perf_counter()

# Read the settings from the config.yaml file
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Create a settings object
settings = Settings(**config)

with concurrent.futures.ThreadPoolExecutor(max_workers=settings.num_workers) as executor:
    futures = []
    for t, (input_image, input_annotation) in enumerate(zip(settings.input_images, settings.input_annotations)):
        future = executor.submit(process_tile, t, input_image, input_annotation, settings)
        futures.append(future)

    for future in tqdm(concurrent.futures.as_completed(futures), desc='Exporting tiles and annotations..', total=len(futures)):
        result = future.result()
    
end_time = perf_counter() - start_time

print(f"Elapsed time: {end_time:.2f} seconds")