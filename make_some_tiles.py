#!/usr/bin/env python
# coding: utf-8

import random
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from tqdm import tqdm
import yaml

from settings import Settings
from utils_tiling import (add_border, get_boxes_inside_tiles,
                          perform_quality_checks,
                          read_coordinates_from_annotations, save_annotations,
                          save_boxes, tile_image)

random.seed(3)

# Delete the tiles and annotations folders if they exist
[shutil.rmtree(x) if Path(x).exists() else None for x in [
               'tiles/', 'output/', 'annotations/', 'images/', 'logs/']]

# Read the settings from the config.yaml file
with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Create a settings object
settings = Settings(**config)

# Set the logger
logger=settings.logger

for t, (input_image, input_annotation) in tqdm(enumerate(zip(settings.input_images,
                                                             settings.input_annotations)),
                                               desc='Exporting tiles and annotations..',
                                               total=len(settings.input_images)):
    start_time=perf_counter()

    # Get the file name
    file_name=Path(input_image).stem
    logger.info("Processing file: %s", file_name)

    # Read the image
    IMAGE_FILENAME=str(Path(settings.input_dir_images)\
                         .joinpath(f"{file_name}.{settings.input_extension_images}"))
    image=cv2.imread(IMAGE_FILENAME)

    # Pad the image if needed
    image=add_border(image, 
                     settings=settings, 
                     color=[0, 0, 0],  # BGR format
                     ) if settings.pad_image else image
    image_shape=image.shape

    # export_yolo_annotation_from_csv(filename=file_name, output_dir=settings.input_dir_annotations)

    # Read the coordinates of the bounding boxes from the annotation files
    all_bboxes_coords, box_classes=read_coordinates_from_annotations(path=input_annotation,
                                                                       image_shape=image_shape,
                                                                       settings=settings)

    # Split the image into tiles and get the coordinates of the tiles
    tiles, coordinates=tile_image(image.copy(), settings=settings)

    logger.info("%d tiles created in total", len(tiles))
    logger.info("%.2fGB of memory used for the tiles",
                tiles.nbytes / (1024*1024*1024)) # 1GB = 1024MB = 1024*1024KB = 1024*1024*1024B

    bounding_boxes=np.array([np.array(i) for i in all_bboxes_coords])
    logger.info("%d bounding boxes in total", len(bounding_boxes))

    # Get the bounding boxes inside the tiles
    boxes_in_tiles=get_boxes_inside_tiles(bounding_boxes=bounding_boxes,
                                          tile_coordinates=coordinates,
                                          settings=settings)

    logger.info("%d tiles that are populated with bounding boxes",
                len([i for i in boxes_in_tiles if len(i)]))

    # Generate the tiles with the bounding boxes
    df_results=save_boxes(filename=file_name,
                            tiles=tiles,
                            coordinates=coordinates,
                            boxes_in_tiles=boxes_in_tiles,
                            box_classes=box_classes,
                            settings=settings)

    # Save the annotations in Pascal VOC format or YOLO format
    save_annotations(df_results,
                     filename=file_name,
                     settings=settings,
                     disable_progress_bar=True)

    # Check if all the bboxes are saved
    perform_quality_checks(df_results,
                           bounding_boxes,
                           settings=settings)

    # Measure the elapsed time
    end_time=perf_counter()
    logger.info("Elapsed time: %.2f seconds", end_time-start_time)
