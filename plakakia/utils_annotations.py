import pandas as pd
import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

def read_pascalvoc_coordinates_from_xml(filename=str, settings=None):
    ''' Read coordinates from PascalVOC xml file. '''
    tree = ET.parse(filename) # type: ignore
    root = tree.getroot()

    boxes, classes = [], []

    for obj in root.iter('object'):
        class_name = settings.annotation_mapping[obj.find('name').text] \
            if isinstance(obj.find('name').text, str) else str(obj.find('name').text)

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

def read_yolo_coordinates_from_txt(path=None, image_shape=()):
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

def read_coco_coordinates_from_json(filename, dir_images) -> pd.DataFrame:
    """Read coordinates from COCO format json file."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df_anns = pd.DataFrame(data['annotations'])
    df_imgs = pd.DataFrame(data['images'])
    boxes = []
    for annotation in data['annotations']:
        x, y, w, h = annotation['bbox']
        x_1, y_1 = int(x), int(y)
        x_2, y_2 = int(x+w), int(y+h)
        boxes.append([x_1, y_1, x_2, y_2])

    df_merged = pd.merge(df_anns, df_imgs, left_on='image_id', right_on='id')
    df_merged['boxes'] = boxes
    df_merged['file_name'] = df_merged['file_name'].apply(lambda x: (Path(dir_images) / x).as_posix())

    return df_merged

def read_coordinates_from_annotations(img_path=None, 
                                      ant_path=None,
                                      image_shape=None, 
                                      settings=None) -> tuple:
    """ Read coordinates from annotations. """
    if settings.input_format_annotations == 'yolo':
        boxes, classes = read_yolo_coordinates_from_txt(ant_path, image_shape)
    elif settings.input_format_annotations == 'pascal_voc':
        boxes, classes = read_pascalvoc_coordinates_from_xml(ant_path, settings)
    elif settings.input_format_annotations == 'coco':
        boxes = settings.df_coco.query("file_name == @img_path").boxes.tolist()
        classes = settings.df_coco.query("file_name == @img_path").category_id.tolist()
    else:
        raise ValueError(f"Annotation format {settings.input_format_annotations} not supported")

    box_classes = [int(i) for i in classes]

    return np.array(boxes), box_classes

def export_yolo_annotation_from_csv(filename=None, output_dir=None) -> None:
    """ Export YOLO annotation from csv file. """
    csv_filename = f"df_{filename}.csv"
    dataframe = pd.read_csv(f"{csv_filename}")
    dataframe = dataframe[dataframe.user_verification].copy().reset_index()
    dataframe['prediction_verified'] = np.random.randint(10,size=len(dataframe)).tolist()
    dataframe[['prediction_verified','yolo_x','yolo_y','yolo_width','yolo_height']]\
        .to_csv(f"{output_dir}/{filename}.txt", index=False, header=False, sep=' ')

def save_yolo_annotations_from_df(dataframe,
                                  filename=None,
                                  settings=None,
                                  disable_progress_bar=True) -> None:
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
    output_dir = settings.output_dir_annotations

    for i, sub in tqdm(group,
                       desc='Saving YOLO annotations',
                       disable=disable_progress_bar,
                       total=len(group.count())):
        file_name = f"tile_{filename}_{i[0]}_{i[1]}_{i[2]}_{i[3]}.txt"
        with open(Path(output_dir) / file_name, mode="a+", encoding="utf-8") as file:
            for _, row in sub.iterrows():
                file.write(f"{int(row['box_class'])} {row['yolo_x']} {row['yolo_y']} {row['yolo_w']} {row['yolo_h']}\n")

def save_to_pascal_voc_from_df(dataframe,
                               filename=None,
                               settings=None,
                               disable_progress_bar=True) -> None:
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
        filename.text = f"{tile_name}.{settings.output_extension_images}"
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

def save_annotations(dataframe=None, filename=None, settings=None, disable_progress_bar=True) -> None:
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

def convert_yolo_to_xyxy(yolo_x,
                         yolo_y,
                         yolo_w,
                         yolo_h,
                         image_width,
                         image_height):
    """ Convert YOLO format to XYXY format. """
    x_1 = int((yolo_x - yolo_w/2) * image_width)
    y_1 = int((yolo_y - yolo_h/2) * image_height)
    x_2 = int((yolo_x + yolo_w/2) * image_width)
    y_2 = int((yolo_y + yolo_h/2) * image_height)
    return x_1, y_1, x_2, y_2
