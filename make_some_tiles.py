#!/usr/bin/env python
# coding: utf-8

# TODO: fix classes
import numpy as np
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
from time import perf_counter
from utils_tiling import *
from settings import Settings
import random
random.seed(3)

# # Delete the tiles and annotations folders if they exist
# [shutil.rmtree(x) if Path(x).exists() else None for x in ['tiles/', 'output/', 'annotations/', 'images/', 'logs/']]

settings = Settings(
    input_extension_images='jpg',
    # pad_image=False, 
    tile_size=250, 
    step_size=100, 
    # check_partial=False, 
    # partial_overlap_threshold=0.8,
    input_dir_images='input/images',
    input_dir_annotations='input/annotations',
    input_format_annotations='yolo',
    output_dir_images='output/images',
    output_dir_annotations='output/annotations',
    output_format_annotations='yolo',
    draw_boxes=True,
    log=True,
    log_folder='logs',
)

# Set the logger
logger = settings.logger

# Get the list of input images and annotations
input_images, input_annotations = get_input_lists(settings)

for t, (input_image, input_annotation) in tqdm(enumerate(zip(input_images, input_annotations)), 
                                               desc='Processing images', 
                                               total=len(input_images)):
    start_time = perf_counter()

    file_name = Path(input_image).stem
    logger.info(f"Processing file: {file_name}") \
        if settings.log else None

    image_filename = str(Path(settings.input_dir_images)\
                         .joinpath(f"{file_name}.{settings.input_extension_images}"))
    image = cv2.imread(image_filename)
    image = add_border(image, 
                        top=settings.pad_size, 
                        bottom=settings.pad_size, 
                        left=settings.pad_size, 
                        right=settings.pad_size, 
                        color=[0,0,0], # BGR format
                        ) if settings.pad_image else image
    image_shape = image.shape

    # export_yolo_annotation_from_csv(filename=file_name, output_dir=settings.input_dir_annotations)

    ''' Read the coordinates of the bounding boxes from the annotation files '''
    all_bboxes_coords, box_classes = read_coordinates_from_annotations(path=input_annotation, 
                                                                       image_shape=image_shape, 
                                                                       settings=settings)

    ''' Split the image into tiles and get the coordinates of the tiles '''
    tiles,coordinates = tile_image(image.copy(), 
                                tile_size=settings.tile_size, 
                                step_size=settings.step_size)

    logger.info(f"{len(tiles)} tiles created in total") if settings.log else None
    logger.info(f"{tiles.nbytes / (1024*1024*1024):.2f}GB of memory used for the tiles") \
        if settings.log else None

    bounding_boxes = np.array([np.array(i) for i in all_bboxes_coords])
    logger.info(f"{len(bounding_boxes)} bounding boxes in total") \
        if settings.log else None

    ''' Get the bounding boxes inside the tiles '''
    boxes_in_tiles = get_boxes_inside_tiles(bounding_boxes=bounding_boxes, 
                                            tile_coordinates=coordinates, 
                                            partial_boxes=settings.check_partial, 
                                            overlap_threshold=settings.partial_overlap_threshold)

    logger.info(f"{len([i for i in boxes_in_tiles if len(i)])} tiles that are populated with bounding boxes") \
        if settings.log else None

    ''' Generate the tiles with the bounding boxes '''
    df_results = save_boxes(filename=file_name,
                            tiles=tiles,                         
                            coordinates=coordinates, 
                            boxes_in_tiles=boxes_in_tiles, 
                            box_classes=box_classes, 
                            draw_boxes=settings.draw_boxes,
                            output_dir=settings.output_dir_images)

    ''' Save the annotations in Pascal VOC format or YOLO format '''
    save_annotations(df_results, 
                     filename=file_name, 
                     settings=settings, 
                     disable_progress_bar=True)

    ''' Check if all the bboxes are saved '''
    perform_quality_checks(df_results, 
                           bounding_boxes, 
                           settings=settings)

    end_time = perf_counter()
    logger.info(f"Elapsed time: {end_time-start_time:.2f} seconds") \
        if settings.log else None
