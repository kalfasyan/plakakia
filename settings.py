from pathlib import Path
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler

# Create a dataclass for storing the settings
@dataclass
class Settings():
    # Define the image file extensions
    input_extension_images: str = 'jpg'
    # Set the annotation file extensions
    input_extension_annotations: str = 'txt'
    # Whether to pad the image with a border
    pad_image: bool = False
    # Size of the border to add to the image
    pad_size: int = 10
    # Size of the tile; only square tiles supported for now
    tile_size: int = 200
    # Step size to move the tile in a windowed manner
    step_size: int = 50
    # Check if bounding boxes are partially inside the tile
    check_partial: bool = False
    # Set the overlap threshold for the bounding boxes to be considered partially inside the tile
    partial_overlap_threshold: float = 0.8
    # Set the input directory for the images
    input_dir_images: str = 'input_images'
    # Set the input directory for the annotations
    input_dir_annotations: str = 'input_annotations'
    # Set the input annotation file format
    input_format_annotations: str = 'yolo' # 'yolo' or 'pascal_voc'
    # Set the output directory for the images
    output_dir_images: str = 'output_images'
    # Set the output directory for the annotations
    output_dir_annotations: str = 'output_annotations'
    # Define the annotation file format
    output_format_annotations: str = 'yolo' # 'yolo' or 'pascal_voc'
    # Whether to draw rectangels in the tile images
    draw_boxes: bool = False
    # Set a flag for logging output
    log: bool = False
    # Define a folder for saving the logs
    log_folder: str = 'logs'

    # Define the initialization method of this dataclass
    def __post_init__(self):
        # Create the input/output directories for the images and annotations
        Path(self.input_dir_images).mkdir(parents=True, exist_ok=True)
        Path(self.input_dir_annotations).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_images).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_annotations).mkdir(parents=True, exist_ok=True)
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)

        # Settings the annotations' file extension        
        self.input_extension_annotations = 'txt' if self.input_format_annotations == 'yolo' else 'xml'

        # Logging settings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Create a rotating file handler with 5 files that have up to 5 MB size each
        file_handler = RotatingFileHandler(f'{self.log_folder}/app.log', maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
