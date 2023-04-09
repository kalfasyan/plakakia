import logging
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path

import imagesize
import psutil
from tqdm import tqdm


# Create a dataclass for storing the settings
@dataclass
class Settings():
    """ Settings for the tiling process. """
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
    # Define a mapping for the annotation labels
    annotation_mapping: dict = field(default_factory=dict)
    # Whether to draw rectangels in the tile images
    draw_boxes: bool = False
    # Set a flag for logging output
    log: bool = False
    # Define a folder for saving the logs
    log_folder: str = 'logs'
    # Boolean flag for validating the settings
    validate_settings: bool = True
    # Define the number of threads to use
    num_workers: int = 1
    # Define a flag for removing duplicate tiles
    clear_duplicates: bool = False

    # Define the initialization method of this dataclass
    def __post_init__(self):
        assert Path(self.input_dir_images).exists(), f"{self.input_dir_images} image input directory does not exist."
        assert Path(self.input_dir_annotations).exists(), f"{self.input_dir_annotations} annotations directory does not exist."
        # Create the output directories for the images and annotations
        Path(self.output_dir_images).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_annotations).mkdir(parents=True, exist_ok=True)
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)

        # Settings the annotations' file extension
        self.input_extension_annotations = 'txt' \
            if self.input_format_annotations == 'yolo' else 'xml'

        # Get the list of images and annotations
        self.input_images = list(Path(self.input_dir_images).\
            glob(f"*.{self.input_extension_images}"))
        self.input_images = [i.as_posix() for i in self.input_images]
        self.input_annotations = list(Path(self.input_dir_annotations).\
            glob(f"*.{self.input_extension_annotations}"))
        input_annotations = [i.as_posix() for i in self.input_annotations]

        # Check if there are any images with the given extension
        assert self.input_images, f"No images found with extension: {self.input_extension_images}."
        # Check if there are any annotations with the given extension
        assert len(input_annotations), \
            f"No annotations found with \'input' extension: {self.input_extension_annotations}."
        # Check if the number of annotations is equal to the number of images
        assert len(self.input_images) == len(input_annotations), \
            "The number of images is not equal to the number of annotations."
        # Sort the images and annotations
        self.input_annotations.sort()
        self.input_images.sort()

        if self.validate_settings:
            # Calculate the minimum image dimension
            minimum_image_dim = float('inf')
            for img in tqdm(self.input_images,
                            desc='Validating settings..',
                            total=len(self.input_images)):
                width, height = imagesize.get(img)
                minimum_image_dim = min(minimum_image_dim, width, height)
                # Check if there is an annotation for each image
                assert Path(img).stem in [Path(a).stem for a in self.input_annotations], \
                    f"No annotation found for image {img}."
            # Check if the tile size is smaller than the smallest image dimension
            assert minimum_image_dim >= self.tile_size, \
                f"The tile size is larger than the smallest image dimension: {minimum_image_dim}. Try setting a smaller tile size."

        # Create the inverse mapping for the annotation labels
        self.inv_annotation_mapping = {v: k for k, v in self.annotation_mapping.items()}

        # Logging settings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Create a rotating file handler with 5 files that have up to 5 MB size each
        file_handler = RotatingFileHandler(f'{self.log_folder}/app.log',
                                           maxBytes=5*1024*1024,
                                           backupCount=5)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Set check_partial to False by default
        self.check_partial = False
        
        # Set default value for pad_size
        self.pad_size = 10
        
        self.num_workers = psutil.cpu_count() if self.num_workers == -1 else self.num_workers
        
        if self.clear_duplicates:
            self.output_dir_duplicates = Path(self.output_dir_images).parent / "duplicates"
            Path(f"{self.output_dir_duplicates}").mkdir(parents=True, exist_ok=True)
