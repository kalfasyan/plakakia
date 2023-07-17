![Static Badge](https://badgen.net/github/release/kalfasyan/plakakia)
![Static Badge](https://badgen.net/github/license/kalfasyan/plakakia)
![Static Badge](https://badgen.net/github/stars/kalfasyan/plakakia)
![Static Badge](https://badgen.net/github/open-issues/kalfasyan/plakakia)

# plakakia 
### /πλακάκια  
*Python image tiling library for image processing, object detection, etc.*

**DISCLAIMER**: This is a work in progress.  
  
![Alt text](logo/logo.png?raw=true "This is a \"plakaki\", meaning tile in Greek.")  

## What is this?
**`plakakia`** was initially developed to address the need for efficient image tiling while considering bounding boxes within the image. It offers a solution for dividing an image into rectangular tiles based on specified tile_size and step_size parameters. Overlapping tiles are handled seamlessly. Currently, the tool assigns bounding boxes to tiles only if they are fully contained within them. However, future updates will include support for partial overlap.

It is worth noting that the tool may generate duplicate bounding boxes as a result of tiling. To mitigate this, `plakakia` offers an option to eliminate duplicates. However, it is important to be aware that, depending on the chosen `tile_size` and `step_size`, there is a potential risk of missing some bounding boxes entirely due to the strict duplicate deletion method. To address this limitation, I am considering the implementation of more flexible methods in the future.  

## What is it going to be?
Currently, `plakakia` primarily focuses on RGB images in object detection tasks, where the goal is to have tiles that encompass the corresponding bounding boxes. However, I have plans to expand its capabilities to support segmentation tasks as well. This entails tiling both the input images and the associated masks, where each pixel represents a specific category.

In addition, I aim to enhance the tool by providing support for images with more than 3 channels, such as multispectral images. It is important to **note** that `plakakia` already allows online generation of tiles for images with any number of channels (see [examples](examples/) folder). However, offline batch processing is not currently supported.  

So, `plakakia` will hopefully become a versatile tool for tiling images in a variety of tasks using a variety of image formats with the ultimate goal of fast and efficient processing.

## Performance
To ensure optimal performance, `plakakia` extensively utilizes the `multiprocessing` and `numpy` libraries. This enables efficient processing of thousands of images without the use of nested for-loops which is often applied in tiling tasks. For detailed benchmarks on various public datasets, please refer to the information provided below.
  
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
