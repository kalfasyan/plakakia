import logging
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# TODO: fix this
def read_pascalvoc_coordinates_from_xml(filename=str, settings=None):
    ''' Read coordinates from PascalVOC xml file. '''
    tree = ET.parse(filename) # type: ignore
    root = tree.getroot()

    boxes, classes = [], []

    for obj in root.iter('object'):
        class_name = settings.annotation_mapping[obj.find('name').text] if isinstance(obj.find('name').text, str) else str(obj.find('name').text)

        # Get bounding box coordinates
        xmlbox = obj.find('bndbox')
        x_1 = int(xmlbox.find('xmin').text) # type: ignore
        y_1 = int(xmlbox.find('ymin').text) # type: ignore
        x_2 = int(xmlbox.find('xmax').text) # type: ignore
        y_2 = int(xmlbox.find('ymax').text) # type: ignore

        # Append to boxes and classes lists
        boxes.append([x_1, y_1, x_2, y_2])
        classes.append(class_name)

    return boxes, classes

def read_yolo_coordinates_from_txt(path=None, image_shape=(), settings=None):
    """ Read coordinates from YOLO txt file. """
    with open(path, mode='r', encoding="utf-8") as file: # type: ignore
        lines = file.readlines()

    image_height, image_width, _ = image_shape

    # Transform yolo_x, yolo_y, yolo_width, yolo_height to x_1, y_1, x_2, y_2
    boxes, classes = [], []

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        line = line.split(' ')
        class_id = int(line[0])
        classes.append(class_id)

        x = float(line[1])
        y = float(line[2])
        w = float(line[3])
        h = float(line[4])
        x_1 = int((x - w/2) * image_width)
        y_1 = int((y - h/2) * image_height)
        x_2 = int((x + w/2) * image_width)
        y_2 = int((y + h/2) * image_height)
        boxes.append([x_1, y_1, x_2, y_2])
    return boxes, classes

def read_coordinates_from_annotations(path=None, image_shape=None, settings=None):
    """ Read coordinates from annotations. """
    if settings.input_format_annotations == 'yolo':
        boxes, classes = read_yolo_coordinates_from_txt(path, image_shape, settings)
    elif settings.input_format_annotations == 'pascal_voc':
        boxes, classes = read_pascalvoc_coordinates_from_xml(path, settings)
    else:
        raise ValueError(f"Annotation format {settings.input_format_annotations} not supported")

    box_classes = [int(i) for i in classes]

    return np.array(boxes), box_classes

def export_yolo_annotation_from_csv(filename=None, output_dir=None):
    """ Export YOLO annotation from csv file. """
    csv_filename = f"df_{filename}.csv"
    dataframe = pd.read_csv(f"{csv_filename}")
    dataframe = dataframe[dataframe.user_verification].copy().reset_index()
    dataframe['prediction_verified'] = np.random.randint(10,size=len(dataframe)).tolist()
    dataframe[['prediction_verified','yolo_x','yolo_y','yolo_width','yolo_height']]\
        .to_csv(f"{output_dir}/{filename}.txt", index=False, header=False, sep=' ')

def tile_image(image, settings=None):
    """ Tile an image into overlapping tiles. """

    # Read the tile size and step size from the settings
    tile_size=settings.tile_size
    step_size=settings.step_size

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

    draw_boxes = settings.draw_boxes
    output_dir_images = settings.output_dir_images
    output_dir_duplicates = settings.output_dir_duplicates

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

            if draw_boxes:
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
        cv2.imwrite(f"{output_dir_images}/tile_{filename}_{tile_coord[0]}_{tile_coord[1]}_{tile_coord[2]}_{tile_coord[3]}.png",
            tile)

    # Create a dataframe with the results
    results_df = pd.DataFrame(results, columns=['tile_x1','tile_y1','tile_x2','tile_y2',
                                        'box_class',
                                        'box_x1','box_y1','box_x2','box_y2',
                                        'old_box_x1','old_box_y1','old_box_x2','old_box_y2'])
    
    # Save the dataframe as a parquet file
    if settings.clear_duplicates:
        results_df.to_parquet(Path(output_dir_duplicates) / f"tile_{filename}.parquet", index=False)

    return results_df

def save_yolo_annotations_from_df(dataframe,
                                  filename=None,
                                  settings=None,
                                  disable_progress_bar=True):
    """
    Save YOLO annotations from a dataframe containing the tile coordinates and the bounding boxes.
    """
    # Compute the coordinates of the center of the box and the width and height of the box
    x_1, y_1, x_2, y_2 = dataframe.box_x1, dataframe.box_y1, dataframe.box_x2, dataframe.box_y2
    image_width = dataframe['tile_x2'] - dataframe['tile_x1']
    image_height = dataframe['tile_y2'] - dataframe['tile_y1']
    dataframe['yolo_x'] = (x_1 + x_2) / 2 / image_width
    dataframe['yolo_y'] = (y_1 + y_2) / 2 / image_height
    dataframe['yolo_w'] = (x_2 - x_1) / image_width
    dataframe['yolo_h'] = (y_2 - y_1) / image_height

    group = dataframe.groupby(['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2'])

    for i, sub in tqdm(group,
                       desc='Saving YOLO annotations',
                       disable=disable_progress_bar,
                       total=len(group.count())):
        with open(f"{settings.output_dir_annotations}/tile_{filename}_{i[0]}_{i[1]}_{i[2]}_{i[3]}.txt",
                  mode="a+",
                  encoding="utf-8") as file:
            for _, row in sub.iterrows():
                file.write(f"{int(row['box_class'])} {row['yolo_x']} {row['yolo_y']} {row['yolo_w']} {row['yolo_h']}\n")

def save_to_pascal_voc_from_df(dataframe,
                               filename=None,
                               settings=None,
                               disable_progress_bar=True):
    """
    Saves a dataframe containing bounding box information in Pascal VOC format.
    """
    # fix this function

    f_n = filename

    # Group by tile and iterate over groups
    group = dataframe.groupby(["tile_x1", "tile_y1", "tile_x2", "tile_y2"])
    for _, tile_df in tqdm(group,
                           desc="Saving Pascal VOC annotations",
                           disable=disable_progress_bar,
                           total=len(group.count())):
        # Create the XML structure
        tile_x1 = tile_df["tile_x1"].iloc[0]
        tile_y1 = tile_df["tile_y1"].iloc[0]
        tile_x2 = tile_df["tile_x2"].iloc[0]
        tile_y2 = tile_df["tile_y2"].iloc[0]

        tile_name = f"tile_{f_n}_{tile_x1}_{tile_y1}_{tile_x2}_{tile_y2}"
        assert 'at' not in str(tile_name), "Something went wrong with the tile name."

        annotation = ET.Element("annotation")
        folder = ET.SubElement(annotation, "folder")
        folder.text = settings.output_dir_annotations
        filename = ET.SubElement(annotation, "filename")
        filename.text = f"{tile_name}.png"
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(np.abs(tile_x2 - tile_x1))
        height = ET.SubElement(size, "height")
        height.text = str(np.abs(tile_y2 - tile_y1))
        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        # Iterate over rows in the group and add bounding box information
        for _, row in tile_df.iterrows():
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = str(row["box_class"])
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(np.abs(row["box_x1"] - row["tile_x1"]))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(np.abs(row["box_y1"] - row["tile_y1"]))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(np.abs(row["box_x2"] - row["tile_x1"]))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(np.abs(row["box_y2"] - row["tile_y1"]))

        # Write the XML to a file
        xml_tile_name = f"{tile_name}.xml"
        output_path = os.path.join(settings.output_dir_annotations, xml_tile_name)
        tree = ET.ElementTree(annotation)
        tree.write(output_path,
                   encoding="utf-8",
                   xml_declaration=True,
                   short_empty_elements=False,
                   method="xml")

def save_annotations(dataframe=None, filename=None, settings=None, disable_progress_bar=True):
    """ Save the annotations in the format specified in the settings. """
    if settings.output_format_annotations == 'yolo':
        save_yolo_annotations_from_df(dataframe,
                                      filename=filename,
                                      settings=settings,
                                      disable_progress_bar=disable_progress_bar)
    elif settings.output_format_annotations == 'pascal_voc':
        save_to_pascal_voc_from_df(dataframe,
                                   filename=filename,
                                   settings=settings,
                                   disable_progress_bar=disable_progress_bar)
    else:
        raise ValueError("The output format of the annotations is not valid.")

def perform_quality_checks(dataframe, bounding_boxes, settings):
    ''' Check if all the bboxes are saved '''
    saved_bboxes_coords = dataframe[['old_box_x1','old_box_y1','old_box_x2','old_box_y2']].values
    unique_rows, row_counts = np.unique(saved_bboxes_coords, axis=0, return_counts=True)
    logger.info(f"{len(saved_bboxes_coords) - len(unique_rows)} \
        bbox(es) saved more than one time.")\
            if settings.log else None
    # Check if the saved_bboxes_coords are a subset of the original_bboxes_coords
    assert np.all(np.isin(saved_bboxes_coords, bounding_boxes)), "Not all bboxes saved."

def convert_yolo_to_xyxy(yolo_x, yolo_y, yolo_w, yolo_h, image_width, image_height):
    """ Convert YOLO format to XYXY format. """
    x_1 = int((yolo_x - yolo_w/2) * image_width)
    y_1 = int((yolo_y - yolo_h/2) * image_height)
    x_2 = int((yolo_x + yolo_w/2) * image_width)
    y_2 = int((yolo_y + yolo_h/2) * image_height)
    return x_1, y_1, x_2, y_2

def plot_example_tile_with_yolo_annotation(settings=None):
    """ Plot an example tile with the corresponding YOLO annotation. """

    # Get all image files from the tiles folder
    tile_imagepaths = list(Path(settings.output_dir_images).glob('*.png'))

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
    # Get the file name
    file_name = Path(input_image).stem

    # Read the image
    IMAGE_FILENAME = str(Path(settings.input_dir_images).joinpath(f"{file_name}.{settings.input_extension_images}"))
    image = cv2.imread(IMAGE_FILENAME)

    # Pad the image if needed
    if settings.pad_image:
        image = add_border(image, settings=settings, color=[0, 0, 0])
    image_shape = image.shape

    # Read the coordinates of the bounding boxes from the annotation files
    bounding_boxes, box_classes = read_coordinates_from_annotations(path=input_annotation,
                                                                     image_shape=image_shape,
                                                                     settings=settings)

    # Split the image into tiles and get the coordinates of the tiles
    tiles, coordinates = tile_image(image.copy(),
                                    settings=settings)

    # Get the bounding boxes inside the tiles
    boxes_in_tiles = get_boxes_inside_tiles(bounding_boxes=bounding_boxes,
                                             tile_coordinates=coordinates,
                                             settings=settings)

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
    all_subs = []
    results = Path(settings.output_dir_duplicates).glob("*.parquet")
    for file in tqdm(results, desc="Gathering results for duplicate removal.."):
        sub = pd.read_parquet(file)
        sub['filename'] = file.stem
        all_subs.append(sub)
        # Delete the file
        file.unlink()
    results_df = pd.concat(all_subs, ignore_index=True)
    results_df.to_csv(Path(settings.output_dir_duplicates) / "all_results.csv", index=False)