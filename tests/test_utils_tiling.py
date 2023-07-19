# content of test_sysexit.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest
import yaml
from lxml import etree as ET

from plakakia.settings import Settings
from plakakia.utils_annotations import (read_coco_coordinates_from_json,
                               read_pascalvoc_coordinates_from_xml,
                               read_yolo_coordinates_from_txt)
from plakakia.utils_tiling import add_border, tile_image

# Read the settings from the config.yaml file
with open('plakakia/config_example.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Create a settings object
settings = Settings(**config)

def test_add_border():
    """
    Test function for the add_border() function.

    Creates a 10x10 numpy array with random values and adds a border of one pixel width to each side using
    the add_border() function. The resulting array should have a shape of (12, 12). This test checks whether
    the resulting shape is as expected.
    """

    tmp = np.random.randn(10, 10)
    settings.pad_size = 1
    tmp = add_border(tmp, settings=settings)
    assert tmp.shape == (12, 12)

import os
import xml.etree.ElementTree as ET


def create_sample_xml():
    """
    Creates a sample XML file with two objects.

    Returns:
    None
    """
    root = ET.Element("annotation")

    obj1 = ET.SubElement(root, "object")
    name1 = ET.SubElement(obj1, "name")
    name1.text = "cat"
    bbox1 = ET.SubElement(obj1, "bndbox")
    xmin1 = ET.SubElement(bbox1, "xmin")
    xmin1.text = "10"
    ymin1 = ET.SubElement(bbox1, "ymin")
    ymin1.text = "20"
    xmax1 = ET.SubElement(bbox1, "xmax")
    xmax1.text = "30"
    ymax1 = ET.SubElement(bbox1, "ymax")
    ymax1.text = "40"

    obj2 = ET.SubElement(root, "object")
    name2 = ET.SubElement(obj2, "name")
    name2.text = "car"
    bbox2 = ET.SubElement(obj2, "bndbox")
    xmin2 = ET.SubElement(bbox2, "xmin")
    xmin2.text = "50"
    ymin2 = ET.SubElement(bbox2, "ymin")
    ymin2.text = "60"
    xmax2 = ET.SubElement(bbox2, "xmax")
    xmax2.text = "80"
    ymax2 = ET.SubElement(bbox2, "ymax")
    ymax2.text = "90"

    tree = ET.ElementTree(root)
    tree.write("sample.xml")

def test_read_pascalvoc_coordinates_from_xml():
    """
    Tests the function read_pascalvoc_coordinates_from_xml by creating a sample XML file, 
    reading it, and comparing the results with expected values. The XML file contains two 
    objects: a cat and a car. The function is tested with a settings object that maps the 
    "cat" class to "animal" and the "car" class to "vehicle".

    Returns:
    None
    """
    create_sample_xml()
    settings.annotation_mapping = {"cat": "animal", "car": "vehicle"}
    boxes, classes = read_pascalvoc_coordinates_from_xml("sample.xml", settings=settings)

    assert boxes == [[10, 20, 30, 40], [50, 60, 80, 90]]
    assert classes == ["animal", "vehicle"]

    os.remove("sample.xml") # delete the sample.xml file once we're done with the test

def test_read_pascalvoc_coordinates_from_xml_invalid_file():
    """
    Tests the function read_pascalvoc_coordinates_from_xml with an invalid file path, 
    expecting a FileNotFoundError to be raised.

    Returns:
    None
    """
    with pytest.raises(FileNotFoundError):
        read_pascalvoc_coordinates_from_xml('nonexistent.xml')

def test_read_yolo_coordinates_from_txt():
    """
    Tests the function read_yolo_coordinates_from_txt by creating a temporary file with 
    some YOLO-formatted data, reading it, and comparing the results with expected values. 
    The YOLO data contains two objects: a bounding box around the center of the image 
    with width and height equal to half the image size, and a smaller bounding box in 
    the top-left corner. The function is tested with an image shape of (800, 600, 3) 
    and no settings object.

    Returns:
    None
    """
    # Create a temporary file with some data in YOLO format
    data = "0 0.5 0.5 0.5 0.5\n1 0.3 0.3 0.2 0.2\n"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        path = tmp_file.name

    # Call the read_yolo_coordinates_from_txt function with the temporary file
    image_shape = (800, 600, 3)
    boxes, classes = read_yolo_coordinates_from_txt(path=path, image_shape=image_shape)

    # Check that the output is as expected
    expected_boxes = [[150, 200, 450, 600], [119, 160, 240, 320]]
    expected_classes = [0, 1]
    assert boxes == expected_boxes
    assert classes == expected_classes

    # Delete the temporary file
    os.remove(path)

def test_read_yolo_coordinates_from_txt_invalid_file():
    """
    Tests the function read_yolo_coordinates_from_txt with an invalid file path, 
    expecting a FileNotFoundError to be raised.

    Returns:
    None
    """
    with pytest.raises(FileNotFoundError):
        read_yolo_coordinates_from_txt('nonexistent.txt')

def test_read_yolo_coordinates_from_txt_invalid_image_shape():
    """
    Tests the function read_yolo_coordinates_from_txt with an invalid image shape, 
    expecting a ValueError to be raised.

    Returns:
    None
    """
    # Create a temporary file with some data in YOLO format
    data = "0 0.5 0.5 0.5 0.5\n1 0.3 0.3 0.2 0.2\n"
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        path = tmp_file.name

    # Call the read_yolo_coordinates_from_txt function with the temporary file
    image_shape = (800, 600)
    with pytest.raises(ValueError):
        read_yolo_coordinates_from_txt(path=path, image_shape=image_shape)

    # Delete the temporary file
    os.remove(path)

def test_tile_image():
    """
    Tests the function tile_image by creating a random input image, tiling it, and 
    comparing the results with expected values. The input image has a size of 512x512 
    and the tiles have a size of 128x128 with a step size of 64. The output tiles and 
    coordinates are checked for correctness.

    Returns:
    None
    """

    # Create a random input image
    image = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)

    # Call the tile_image function with the random image
    tile_size = 128
    step_size = 64

    tiles, coordinates = tile_image(image, tile_size=tile_size, step_size=step_size)

    # Check that the output shape and type is correct
    assert tiles.shape == (49, 128, 128, 3)
    assert coordinates.shape == (49, 4)
    assert tiles.dtype == image.dtype

    # Check that the coordinates correspond to the correct tiles
    for i, (x1, y1, x2, y2) in enumerate(coordinates):
        assert x2 - x1 == tile_size
        assert y2 - y1 == tile_size
        assert x1 % step_size == 0
        assert y1 % step_size == 0
        assert x2 <= image.shape[1]
        assert y2 <= image.shape[0]
        tile = tiles[i]
        assert tile.shape == (tile_size, tile_size, 3)
        assert np.array_equal(tile, image[y1:y2, x1:x2])

def test_tile_image_invalid_image():
    """
    Tests the function tile_image with an invalid image, expecting a ValueError to be 
    raised.

    Returns:
    None
    """
    image = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
    with pytest.raises(IndexError):
        tile_image(image, tile_size=settings.tile_size, step_size=settings.step_size)
