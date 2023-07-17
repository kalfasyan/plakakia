![Static Badge](https://badgen.net/github/release/kalfasyan/plakakia)
![Static Badge](https://badgen.net/github/license/kalfasyan/plakakia)
![Static Badge](https://badgen.net/github/stars/kalfasyan/plakakia)
![Static Badge](https://badgen.net/github/open-issues/kalfasyan/plakakia)

# plakakia 
### /πλακάκια  
*Python image tiling library for image processing, object detection, etc.*
    
![Alt text](logo/logo.png?raw=true "This is a \"plakaki\", meaning tile in Greek.")  

## What is this? What is it going to be?
`plakakia` is an efficient image tiling tool designed to handle bounding boxes within images. It divides images into rectangular tiles based on specified parameters, seamlessly handling overlapping tiles. The tool assigns bounding boxes to tiles that fully contain them, and it also offers an option to eliminate duplicate bounding boxes. While the current version only supports fully contained bounding boxes, future updates will include support for partial overlap.  `plakakia` can handle object detection and segmentation datasets.  
  
Currently, the library offers online and offline modes for processing data (refer to the [Usage section](https://github.com/kalfasyan/plakakia#usage) section below for more details):  

- In the offline mode, one can use a config file and run a script once to process all data.
- In the online mode, the `tile_image` function allows processing of images of any dimension.

There are plans to expand `plakakia`'s capabilities in the offline mode to handle images with more than 3 channels.
  
## Performance
To ensure optimal performance, `plakakia` utilizes the `multiprocessing` and `numpy` libraries. This enables efficient processing of thousands of images without the use of nested for-loops commonly used in tiling tasks. For detailed benchmarks on various public datasets, please refer to the information provided below.

  
# Installation

It is **highly** recommended that you create a new virtual environment for the installation:    
 1. Download and install [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) (or [Anaconda](https://www.anaconda.com/products/distribution)). 
 2. Create a virtual environment:  
 `mamba create -n plakakia jupyterlab nb_conda_kernels ipykernel ipywidgets pip -y`  
 3. Activate the environment:  
 `mamba activate plakakia`
4. Run the following command to install the library:  
`pip install plakakia`  

# Usage

### A. Offline tile generation with a config file
    
`make_some_tiles --config path/to/config.yaml`  
  > ⚠️ When executed, the `make_some_tiles` script removes the following folders from the current location: ['tiles/', 'output/', 'annotations/', 'images/', 'logs/']
-    Check an example [`config.yaml`](plakakia/config.yaml).  
  
  
### B. Online tile generation
  
```
from plakakia.utils_tiling import tile_image

tiles, coordinates = tile_image(img, tile_size=100, step_size=100)
```
Shape of original image: (500, 500, 3)  
Shape of tiles array: (25, 100, 100, 3)  
  
![Alt text](logo/tiles.png?raw=true "The result of the tiling process.")  
  

> ⚠️ For more examples, check the [examples](examples/) folder.   
    
# Benchmarks

**Benchmarked on HP Laptop with specs**: AMD Ryzen 5 PRO 6650U; 6 cores; 12 threads; 2.9 GHz

| Dataset | Source | Formats (images/labels) | Number of images | tile_size | step_size | tiles generated | plakakia performance |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Solar Panels v2  | [RoboFlow](https://universe.roboflow.com/roboflow-100/solar-panels-taxvb/dataset/2) | jpg/COCO | 112  | 150 | 50 | 3.075 | 1,11 sec | 
| Traffic Signs  | [Kaggle](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format) | jpg/YOLO | 741  | 300 | 200 | 1.695 | 2,8 sec | 
| Hard Hat Workers v2  | [RoboFlow](https://public.roboflow.com/object-detection/hard-hat-workers/2) | jpg/YOLO | 5.269 | 100 | 50 | 21.678 | 6,94 sec| 
| Microsoft COCO dataset  | [RoboFlow](https://public.roboflow.com/object-detection/microsoft-coco-subset) | jpg/YOLO | 121.408 | 200 | 150 | 177.039 | 3 min 4 sec| 

# TODO list
  
 ☑️ ~~Fix reading of classes from annotations (create a 'mapper' dictionary to map classes to numerical values).~~  
 ☑️ ~~Read settings from a file (e.g. json).~~  
 ☑️ ~~Removing all tiles with duplicate bounding boxes (that appear in other tiles).~~  
 ☑️ ~~Support other annotation formats (e.g. coco).~~ (only input for now)  
 ☑️ ~~Provide tiling functionality without any labels needed.~~  
 ☑️ ~~Add support for segmentation tasks (tile both input images and masks).~~  
 ⬜️ Add less strict (flexible) duplicate removal methods to avoid missing bounding boxes.  
 ⬜️ Consider bounding boxes in tiles if they *partially* belong to one.  
 ⬜️ Support reading annotations from a dataframe/csv file.  
 ⬜️ Make tiles with multidimensional data offline with config file (e.g. hdf5 hyperspectral images).  
   
  
# Want to contribute?
If you want to contribute to this project, please check the [CONTRIBUTING.md](CONTRIBUTING.md) file.
