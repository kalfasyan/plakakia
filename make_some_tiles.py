#!/usr/bin/env python
# coding: utf-8

# Author:

import multiprocessing as mp
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

def process_tile_wrapper(args):
    t, input_image, input_annotation, settings = args
    return process_tile(t, input_image, input_annotation, settings)

# Create a process pool with the desired number of workers
with mp.Pool(processes=settings.num_workers) as pool:
    # Prepare the arguments for each task
    args = [(t, input_image, input_annotation, settings) for t, (input_image, input_annotation) in enumerate(zip(settings.input_images, settings.input_annotations))]
    
    # Submit the tasks to the pool
    results = pool.map(process_tile_wrapper, args)
    
# Process the results as needed
for result in tqdm(results, desc='Exporting tiles and annotations..', total=len(results)):
    # do something with the result
    pass
    
end_time = perf_counter() - start_time

print(f"Elapsed time: {end_time:.2f} seconds")