import logging
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_annotations import (convert_yolo_to_xyxy,
                               read_coordinates_from_annotations,
                               save_annotations)

logger = logging.getLogger(__name__)

def add_border(image, settings, color=[0, 0, 0]):
    """ Add border to an image. """

    top=settings.pad_size
    bottom=settings.pad_size
    left=settings.pad_size
    right=settings.pad_size

    if isinstance(image, str):
        image = cv2.imread(image)

    # Create border
    border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return border

def tile_image(image, tile_size=250, step_size=50):
    """ Tile an image into overlapping tiles. """

    # Compute the number of rows and columns of tiles
    rows = (image.shape[0] - tile_size) // step_size + 1
    cols = (image.shape[1] - tile_size) // step_size + 1

    # Compute the shape and strides of the tile view
    tile_shape = (rows, cols, tile_size, tile_size, image.shape[2])
    tile_strides = (step_size * image.strides[0], step_size * image.strides[1], *image.strides)

    # Create a view of the input image with the tile shape and strides
    tile_view = np.lib.stride_tricks.as_strided(image, shape=tile_shape, strides=tile_strides)

    # Reshape the tile view to a flat array of tiles
    tiles = tile_view.reshape(-1, tile_size, tile_size, image.shape[2])

    # Compute the corresponding tile indices and pixel coordinates
    indices = np.arange(tiles.shape[0])
    i, j = np.unravel_index(indices, (rows, cols))
    x_1 = j * step_size
    y_1 = i * step_size
    x_2 = x_1 + tile_size
    y_2 = y_1 + tile_size

    # Stack the tile indices and coordinates into a single array
    coordinates = np.stack((x_1, y_1, x_2, y_2), axis=-1)

    return tiles, coordinates

def get_boxes_inside_tiles(bounding_boxes,
                           tile_coordinates,
                           settings):
    """ Get the bounding boxes that are inside the tiles. """

    partial_boxes=settings.check_partial
    overlap_threshold=settings.partial_overlap_threshold

    boxes_inside_tiles = [[] for _ in range(len(tile_coordinates))]

    for i, tile_coord in enumerate(tile_coordinates):
        if partial_boxes:
            # Create a boolean mask indicating which boxes partially overlap with the tile
            mask = is_partial_square_inside_array(bounding_boxes,
                                                  tile_coord,
                                                  overlap_threshold=overlap_threshold)
        else:
            # Create a boolean mask indicating which boxes are completely inside the tile
            mask = is_square_inside_array(bounding_boxes, tile_coord)

        # Add the boxes that satisfy the condition to the corresponding tile
        boxes_inside_tiles[i] = bounding_boxes[mask].tolist()

    return boxes_inside_tiles

def is_partial_square_inside_array(bounding_boxes, tile_coord, overlap_threshold=None):
    """ Check if a square is partially inside an array. """
    # Compute the coordinates of the intersection between the box and the tile
    x_1 = np.maximum(bounding_boxes[:, 0], tile_coord[0])
    y_1 = np.maximum(bounding_boxes[:, 1], tile_coord[1])
    x_2 = np.minimum(bounding_boxes[:, 2], tile_coord[2])
    y_2 = np.minimum(bounding_boxes[:, 3], tile_coord[3])

    # Compute the areas of the intersection and the box
    intersection_area = (x_2 - x_1) * (y_2 - y_1)
    box_area = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * \
        (bounding_boxes[:, 3] - bounding_boxes[:, 1])

    # Compute the overlap between the box and the tile
    overlap = intersection_area / box_area

    # Return a boolean mask indicating which boxes have overlap above the threshold
    return overlap > overlap_threshold

def is_square_inside_array(bounding_boxes, tile_coord):
    """ Check if a square is completely inside an array. """
    # Return a boolean mask indicating which boxes are inside the tile
    return np.logical_and.reduce((
        bounding_boxes[:, 0] >= tile_coord[0],
        bounding_boxes[:, 1] >= tile_coord[1],
        bounding_boxes[:, 2] <= tile_coord[2],
        bounding_boxes[:, 3] <= tile_coord[3]
    ))

def save_boxes(tiles=np.array([]),
               filename=None,
               coordinates=np.array([]),
               boxes_in_tiles=[],
               box_classes=[],
               settings=None,
               disable_progress_bar=True):
    '''
    Save the tiles with the boxes drawn on them.
    '''

    # Initialize an array to store the class and coordinates of the boxes and the tile coordinates
    results = np.zeros((0, 13), dtype=np.int32)

    # Loop through each tile and save it with a name that includes the tile coordinates
    for i, (tile, tile_coord, boxes) in tqdm(enumerate(zip(tiles, coordinates, boxes_in_tiles)),
                                             desc="Saving tiles",
                                             total=len(tiles),
                                             disable=disable_progress_bar):
        # Save boxes only if there are boxes in the tile
        if len(boxes_in_tiles[i]) == 0:
            continue

        # Draw the boxes on the tile with yellow borders
        for b, box in enumerate(boxes):
            new_x1 = box[0] - tile_coord[0]
            new_y1 = box[1] - tile_coord[1]
            new_x2 = box[2] - tile_coord[0]
            new_y2 = box[3] - tile_coord[1]

            if settings.draw_boxes:
                cv2.rectangle(tile, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 255), 2)
                # Add the class on top of the rectangle
                cv2.putText(tile,
                            str(box_classes[b]),
                            (new_x1, new_y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Stack the class+coordinates of the boxes w/ results array and tile coordinates
            results = np.vstack((results,
                                 [tile_coord[0], tile_coord[1], tile_coord[2], tile_coord[3],
                                  box_classes[b],
                                  new_x1, new_y1, new_x2, new_y2,
                                  box[0], box[1], box[2], box[3]]))

        # Save the tile with the tile coordinates in the filename
        extension = settings.output_extension_images
        output_dir = settings.output_dir_images
        file_path = f"tile_{filename}_{tile_coord[0]}_{tile_coord[1]}_{tile_coord[2]}_{tile_coord[3]}.{extension}"
        save_path = Path(output_dir) / Path(file_path)

        cv2.imwrite(save_path.as_posix(), tile)

    # Create a dataframe with the results
    results_df = pd.DataFrame(results, 
                              columns=['tile_x1','tile_y1','tile_x2','tile_y2',
                                        'box_class',
                                        'box_x1','box_y1','box_x2','box_y2',
                                        'old_box_x1','old_box_y1','old_box_x2','old_box_y2'])

    # Save the dataframe as a parquet file
    if settings.clear_duplicates:
        results_df.to_parquet(Path(settings.output_dir_duplicates) / f"tile_{filename}.parquet", 
                              index=False)

    return results_df

def plot_example_tile_with_yolo_annotation(settings=None):
    """ Plot an example tile with the corresponding YOLO annotation. """

    # Get all image files from the tiles folder
    tile_imagepaths = list(Path(settings.output_dir_images).glob('*.{settings.output_extension_images}'))

    # Randomly select a tile from tile_imagepaths list
    img_selection  = random.choice(tile_imagepaths)
    logger.info(img_selection)
    assert Path(img_selection).exists(), "does not exist"
    tile = cv2.imread(str(img_selection))

    # Read the corresponding annotation file
    annotation_selection = Path(settings.output_dir_annotations) / f"{img_selection.stem}.txt"
    logger.info(annotation_selection)
    assert Path(annotation_selection).exists(), "does not exist"
    with open(annotation_selection, mode='r', encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            yolo_x = float(line[1])
            yolo_y = float(line[2])
            yolo_w = float(line[3])
            yolo_h = float(line[4])
            x_1, y_1, x_2, y_2 = convert_yolo_to_xyxy(yolo_x,
                                                      yolo_y,
                                                      yolo_w,
                                                      yolo_h,
                                                      tile.shape[0],
                                                      tile.shape[1])
            cv2.rectangle(tile, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)

    # Plot the tile in RGB
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    plt.imshow(tile_rgb)
    plt.show()

def process_tile(t, input_image, input_annotation, settings=None):
    """ The main function to process a tile. """
    # Get the file name
    file_name = Path(input_image).stem

    # Read the image
    extension = settings.input_extension_images
    image_filename = str(Path(settings.input_dir_images).joinpath(f"{file_name}.{extension}"))
    image = cv2.imread(image_filename)

    # Pad the image if needed
    if settings.pad_image:
        image = add_border(image, settings=settings, color=[0, 0, 0])
    image_shape = image.shape

    # Read the coordinates of the bounding boxes from the annotation files
    bounding_boxes, box_classes = read_coordinates_from_annotations(img_path=input_image,
                                                                    ant_path=input_annotation,
                                                                    image_shape=image_shape,
                                                                    settings=settings)
    # Split the image into tiles and get the coordinates of the tiles
    tiles, coordinates = tile_image(image.copy(),
                                    tile_size=settings.tile_size,
                                    step_size=settings.step_size)

    # Get the bounding boxes inside the tiles
    if bounding_boxes.shape[0] > 0:
        boxes_in_tiles = get_boxes_inside_tiles(bounding_boxes=bounding_boxes,
                                                tile_coordinates=coordinates,
                                                settings=settings)
    else:
        return t

    # Generate the tiles with the bounding boxes
    df_results = save_boxes(filename=file_name,
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
    return t

def clear_duplicates(settings):
    """ Clear the duplicate tiles. """

    # Gather all the results from the different processes
    # since settings.duplicates is set to True
    # the results are saved in parquet files
    # in the settings.output_dir_duplicates folder
    all_subs = []
    results = Path(settings.output_dir_duplicates).glob("*.parquet")
    for file in tqdm(results, desc="Gathering results for duplicate removal.."):
        sub = pd.read_parquet(file)
        sub['filename'] = file.stem
        all_subs.append(sub)
        # Delete the file
        file.unlink()
    results_df = pd.concat(all_subs, ignore_index=True)

    # Format the filename to match the format of the saved 
    # images and annotations
    results_df['filename'] = results_df['filename']+"_"+ \
                            results_df['tile_x1'].astype(str)+"_"+ \
                                results_df['tile_y1'].astype(str)+"_"+ \
                                    results_df['tile_x2'].astype(str)+"_"+ \
                                        results_df['tile_y2'].astype(str)

    # Define two sets to store all unique filenames and 
    # filenames without duplicates so that we only keep the latter.
    all_filenames = set(results_df['filename'].unique().tolist())
    no_duplicates = set(results_df[
        ~results_df[
            ['old_box_x1', 'old_box_y1', 'old_box_x2', 'old_box_y2']
            ].duplicated()].filename.tolist())

    # Loop through all images and annotations and remove duplicates
    for filename in tqdm(all_filenames, desc="Removing duplicates..", total=len(all_filenames)):
        if filename in no_duplicates:
            continue
        else:
            annotation_file = Path(settings.output_dir_annotations) / \
                f"{filename}.{settings.output_extension_annotations}"
            image_file = Path(settings.output_dir_images) / \
                f"{filename}.{settings.output_extension_images}"
            annotation_file.unlink()
            image_file.unlink()
