import cv2
import numpy as np
import pandas as pd
from time import perf_counter
import logging

logger = logging.getLogger(__name__)


def add_border(image, top=0, bottom=0, left=0, right=0, color=[0, 0, 0]):
    if isinstance(image, str):
        image = cv2.imread(image)

    # Create border
    border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return border

# TODO: fix this
def read_pascalvoc_coordinates_from_xml(fname=None, image_shape=None):
    import xml.etree.ElementTree as ET

    tree = ET.parse(fname)
    root = tree.getroot()
    
    image_height, image_width, _ = image_shape
    
    boxes, classes = [], []
    
    for obj in root.iter('object'):
        try:
            class_name = 0#int(obj.find('name').text)
        except:
            class_name = str(obj.find('name').text)
            #TODO: add class name to class list
        
        # Get bounding box coordinates
        xmlbox = obj.find('bndbox')
        x1 = int(xmlbox.find('xmin').text)
        y1 = int(xmlbox.find('ymin').text)
        x2 = int(xmlbox.find('xmax').text)
        y2 = int(xmlbox.find('ymax').text)
        
        # Append to boxes and classes lists
        boxes.append([x1, y1, x2, y2])
        classes.append(class_name)
        
    return boxes, classes

def get_input_lists(settings):
    from pathlib import Path

    input_images = list(Path(settings.input_dir_images).glob(f"*.{settings.input_extension_images}"))
    input_images = [i.as_posix() for i in input_images]
    input_annotations = list(Path(settings.input_dir_annotations).glob(f"*.{settings.input_extension_annotations}"))
    input_annotations = [i.as_posix() for i in input_annotations]
    assert len(input_images), "No images found."
    assert len(input_annotations), f"No annotations found with extension: {settings.input_extension_annotations}."
    assert len(input_images) == len(input_annotations), "The number of images is not equal to the number of annotations."
    input_annotations.sort()
    input_images.sort()
    return input_images, input_annotations

def read_yolo_coordinates_from_txt(path=None, image_shape=None):
    import os
    with open(path, 'r') as f:
        lines = f.readlines()
    
    image_height, image_width, _ = image_shape
    
    # Transform yolo_x, yolo_y, yolo_width, yolo_height to x1, y1, x2, y2
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
        x1 = int((x - w/2) * image_width)
        y1 = int((y - h/2) * image_height)
        x2 = int((x + w/2) * image_width)
        y2 = int((y + h/2) * image_height)
        boxes.append([x1, y1, x2, y2])
    return boxes, classes

def read_coordinates_from_annotations(path=None, image_shape=None, settings=None):
    if settings.input_format_annotations == 'yolo':
        boxes, classes = read_yolo_coordinates_from_txt(path, image_shape)
    elif settings.input_format_annotations == 'pascal_voc':
        boxes, classes = read_pascalvoc_coordinates_from_xml(path, image_shape)
    else:
        raise ValueError(f"Annotation format {settings.input_format_annotations} not supported")
    
    box_classes = [int(i) for i in classes]
    
    return boxes, box_classes

def export_yolo_annotation_from_csv(filename=None, output_dir=None):
    csv_filename = f"df_{filename}.csv"
    df = pd.read_csv(f"{csv_filename}")
    df = df[df.user_verification].copy().reset_index()
    df['prediction_verified'] = np.random.randint(10,size=len(df)).tolist()
    df[['prediction_verified','yolo_x','yolo_y','yolo_width','yolo_height']]\
        .to_csv(f"{output_dir}/{filename}.txt", index=False, header=False, sep=' ')  

def tile_image(image, tile_size, step_size):
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
    x1 = j * step_size
    y1 = i * step_size
    x2 = x1 + tile_size
    y2 = y1 + tile_size
    
    # Stack the tile indices and coordinates into a single array
    coordinates = np.stack((x1, y1, x2, y2), axis=-1)
        
    return tiles, coordinates

def get_boxes_inside_tiles(bounding_boxes, tile_coordinates, partial_boxes=None, overlap_threshold=None):
    boxes_inside_tiles = [[] for _ in range(len(tile_coordinates))]
    
    for i, tile_coord in enumerate(tile_coordinates):
        if partial_boxes:
            # Create a boolean mask indicating which boxes partially overlap with the tile
            mask = is_partial_square_inside_array(bounding_boxes, tile_coord, overlap_threshold=overlap_threshold)
        else:
            # Create a boolean mask indicating which boxes are completely inside the tile
            mask = is_square_inside_array(bounding_boxes, tile_coord)
        
        # Add the boxes that satisfy the condition to the corresponding tile
        boxes_inside_tiles[i] = bounding_boxes[mask].tolist()
    
    return boxes_inside_tiles

def is_partial_square_inside_array(bounding_boxes, tile_coord, overlap_threshold=None):
    # Compute the coordinates of the intersection between the box and the tile
    x1 = np.maximum(bounding_boxes[:, 0], tile_coord[0])
    y1 = np.maximum(bounding_boxes[:, 1], tile_coord[1])
    x2 = np.minimum(bounding_boxes[:, 2], tile_coord[2])
    y2 = np.minimum(bounding_boxes[:, 3], tile_coord[3])
    
    # Compute the areas of the intersection and the box
    intersection_area = (x2 - x1) * (y2 - y1)
    box_area = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1])
    
    # Compute the overlap between the box and the tile
    overlap = intersection_area / box_area
    
    # Return a boolean mask indicating which boxes have overlap above the threshold
    return overlap > overlap_threshold

def is_square_inside_array(bounding_boxes, tile_coord):
    # Return a boolean mask indicating which boxes are inside the tile
    return np.logical_and.reduce((
        bounding_boxes[:, 0] >= tile_coord[0],
        bounding_boxes[:, 1] >= tile_coord[1],
        bounding_boxes[:, 2] <= tile_coord[2],
        bounding_boxes[:, 3] <= tile_coord[3]
    ))

def save_boxes(tiles=None, filename=None, coordinates=None, boxes_in_tiles=None, box_classes=None, draw_boxes=True, output_dir=None, disable_progress_bar=True):
    '''
    Save the tiles with the boxes drawn on them.
    '''
    from tqdm import tqdm
    start_time = perf_counter()

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
                cv2.putText(tile, str(box_classes[b]), (new_x1, new_y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Stack the class and coordinates of the boxes with the results array and the tile coordinates
            results = np.vstack((results, [tile_coord[0], tile_coord[1], tile_coord[2], tile_coord[3], 
                                           box_classes[b], 
                                           new_x1, new_y1, new_x2, new_y2, 
                                           box[0], box[1], box[2], box[3]]))
            

        # Save the tile with the tile coordinates in the filename
        cv2.imwrite(f"{output_dir}/tile_{filename}_{tile_coord[0]}_{tile_coord[1]}_{tile_coord[2]}_{tile_coord[3]}.png", tile)
    end_time = perf_counter()

    ''' Create a dataframe with the results '''
    df = pd.DataFrame(results, columns=['tile_x1','tile_y1','tile_x2','tile_y2',
                                        'box_class',
                                        'box_x1','box_y1','box_x2','box_y2',
                                        'old_box_x1','old_box_y1','old_box_x2','old_box_y2'])

    # Number of tile images saved
    return df

def save_yolo_annotations_from_df(df, filename=None, output_dir='tiles/', disable_progress_bar=True):
    """
    Save YOLO annotations from a dataframe containing the tile coordinates and the bounding boxes.
    """
    from tqdm import tqdm
    x1, y1, x2, y2 = df.box_x1, df.box_y1, df.box_x2, df.box_y2
    image_width = df['tile_x2'] - df['tile_x1']
    image_height = df['tile_y2'] - df['tile_y1']
    df['yolo_x'] = (x1 + x2) / 2 / image_width
    df['yolo_y'] = (y1 + y2) / 2 / image_height
    df['yolo_w'] = (x2 - x1) / image_width
    df['yolo_h'] = (y2 - y1) / image_height
    
    group = df.groupby(['tile_x1', 'tile_y1', 'tile_x2', 'tile_y2'])

    for i, sub in tqdm(group, 
                       desc='Saving YOLO annotations', 
                       disable=disable_progress_bar,
                       total=len(group.count())):
        with open(f"{output_dir}/tile_{filename}_{i[0]}_{i[1]}_{i[2]}_{i[3]}.txt", "a+") as f:
            for _, row in sub.iterrows():
                f.write(f"{int(row['box_class'])} {row['yolo_x']} {row['yolo_y']} {row['yolo_w']} {row['yolo_h']}\n")

def save_to_pascal_voc_from_df(df, filename=None, output_dir='tiles/', disable_progress_bar=True):
    # TODO: fix this function
    """
    Saves a dataframe containing bounding box information in Pascal VOC format.
    """
    import os
    import pandas as pd
    import xml.etree.cElementTree as ET
    from tqdm import tqdm

    # Group by tile and iterate over groups
    group = df.groupby(["tile_x1", "tile_y1", "tile_x2", "tile_y2"])
    for _, tile_df in tqdm(group, 
                           desc="Saving Pascal VOC annotations", 
                           disable=disable_progress_bar,
                           total=len(group.count())):
        # Create the XML structure
        tile_x1 = tile_df["tile_x1"].iloc[0]
        tile_y1 = tile_df["tile_y1"].iloc[0]
        tile_x2 = tile_df["tile_x2"].iloc[0]
        tile_y2 = tile_df["tile_y2"].iloc[0]

        tile_name = f"tile_{filename}_{tile_x1}_{tile_y1}_{tile_x2}_{tile_y2}"

        annotation = ET.Element("annotation")
        folder = ET.SubElement(annotation, "folder")
        folder.text = output_dir
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
        output_path = os.path.join(output_dir, xml_tile_name)
        tree = ET.ElementTree(annotation)
        tree.write(output_path, encoding="utf-8", xml_declaration=True, short_empty_elements=False, method="xml")

def save_annotations(df=None, filename=None, settings=None, disable_progress_bar=True):
    if settings.output_format_annotations == 'yolo':
        save_yolo_annotations_from_df(df, 
                                      filename=filename,
                                      output_dir=settings.output_dir_annotations, 
                                      disable_progress_bar=disable_progress_bar)
    elif settings.output_format_annotations == 'pascal_voc':
        save_to_pascal_voc_from_df(df, 
                                   filename=filename,
                                   output_dir=settings.output_dir_annotations, 
                                   disable_progress_bar=disable_progress_bar)
    else:
        raise ValueError("The output format of the annotations is not valid.")        

def perform_quality_checks(df, bounding_boxes, settings):
    ''' Check if all the bboxes are saved '''
    saved_bboxes_coords = df[['old_box_x1','old_box_y1','old_box_x2','old_box_y2']].values
    unique_rows, row_counts = np.unique(saved_bboxes_coords, axis=0, return_counts=True)
    logger.info(f"{saved_bboxes_coords.shape[0] - unique_rows.shape[0]} bbox(es) saved more than one time.") if settings.log else None
    # Check if the saved_bboxes_coords are a subset of the original_bboxes_coords
    assert np.all(np.isin(saved_bboxes_coords, bounding_boxes)), "Not all bboxes saved."

def convert_yolo_to_xyxy(yolo_x, yolo_y, yolo_w, yolo_h, image_width, image_height):
    x1 = int((yolo_x - yolo_w/2) * image_width)
    y1 = int((yolo_y - yolo_h/2) * image_height)
    x2 = int((yolo_x + yolo_w/2) * image_width)
    y2 = int((yolo_y + yolo_h/2) * image_height)
    return x1, y1, x2, y2

def plot_example_tile_with_yolo_annotation(settings=None):
    from pathlib import Path
    import random
    import matplotlib.pyplot as plt

    # Get all image files from the tiles folder
    tile_imagepaths = list(Path(settings.output_dir_images).glob('*.png'))

    # Randomly select a tile from tile_imagepaths list
    img_selection  = random.choice(tile_imagepaths)
    logger.info(img_selection) if settings.log else None
    assert Path(img_selection).exists(), "does not exist"
    tile = cv2.imread(str(img_selection))

    # Read the corresponding annotation file
    annotation_selection = Path(settings.output_dir_annotations) / f"{img_selection.stem}.txt"
    logger.info(annotation_selection) if settings.log else None
    assert Path(annotation_selection).exists(), "does not exist"
    with open(annotation_selection, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            yolo_x = float(line[1])
            yolo_y = float(line[2])
            yolo_w = float(line[3])
            yolo_h = float(line[4])
            x1, y1, x2, y2 = convert_yolo_to_xyxy(yolo_x, yolo_y, yolo_w, yolo_h, tile.shape[0], tile.shape[1])
            cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Plot the tile in RGB
    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    plt.imshow(tile_rgb);