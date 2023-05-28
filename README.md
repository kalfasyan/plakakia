# plakakia
*Python image tiling library for image processing, object detection, etc.*

**DISCLAIMER**: This is a work in progress.  
  
![Alt text](logo/logo.png?raw=true "This is a \"plakaki\", meaning tile in Greek.")  

The reason for making this tool is to handle image tiling that takes into account bounding boxes that exist inside the image. For now, I've only considered rectangular tiles. An image is divided into tiles based on a given `tile_size` and `step_size`. Overlapping tiles are handled fine, too. Bounding boxes belong to a tile only if they are *fully* inside it, but I'm planning to support partial overlap as well. Note that such methods create duplicate bounding boxes (i.e. a bounding box can appear in more than one tiles). There is an option for avoiding duplicates. Note that depending on the selection of `tile_size` and `step_size`, there is a potential cost of missing some bounding boxes altogether sincce the duplicate deletion method is strictly applied. Might consider adding more flexible methods in the future.  
  
In this package, I employ `multiprocessing` and `numpy` extensively to make this as fast as possible so that one can use it with thousands of images. For benchmarks on some public datasets, see below.  

# Installation

It is **highly** recommended that you create a new virtual environment for the installation:    
 1. Download and install [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) (or [Anaconda](https://www.anaconda.com/products/distribution)). 
 2. Create a virtual environment:  
 `mamba create -n plakakia jupyterlab nb_conda_kernels ipykernel ipywidgets pip -y`  
 3. Activate the environment:  
 `mamba activate plakakia`
4. Run the following command to install the library:  
`pip install git+https://github.com/kalfasyan/plakakia.git`  
**OR** Clone the repository --> `cd plakakia/` --> `pip install .` (don't omit the dot)  
This will use the setup.py file in the current directory to install the plakakia library along with its dependencies.

# Usage

In this section we cover two main use cases for this library.
## A. Offline tile generation with a config file
This scenario covers the case in which you already have a folder with images and annotations.
 - Make sure you have the plakakia library installed in your Python environment. You can refer to the installation instructions mentioned earlier.
 - Open a terminal or command prompt and activate your Python environment (e.g. `mamba activate plakakia`).
 - Run the following command to execute the make_some_tiles script:  
  > `make_some_tiles --config path/to/config.yaml`  

-    Replace *path/to/config.yaml* with the actual path to your configuration file. This file specifies the settings and parameters for the tiling process. Check an example [`config.yaml`](plakakia/config.yaml).  
 - The script will read your `config.yaml` and generate tiles accordingly.  

 Check the output directory you specified in the yaml file for results.  
  
## B. Online tile generation
In this scenario you want to apply tiling on images that you have loaded in memory (e.g. during model inference).  
```
from plakakia.utils_tiling import tile_image
import cv2
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('../logo/logo.png')
# Convert to RGB to plot with matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# <><><><> PLAKAKIA TILING <><><><> 
tiles, coordinates = tile_image(img, tile_size=100, step_size=100)

# Print some basic info
print(f"Shape of original image: {img.shape}")
print(f"Shape of tiles array: {tiles.shape}")
print(f"Some coordinates in x1,y1,x2,y2 format: {coordinates[:5]}")

# Plotting the tiles in a grid
fig, ax = plt.subplots(5, 5, figsize=(10, 10))
fig.suptitle('Tiles of the original image', fontsize=20)
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(tiles[i*5+j])
        ax[i, j].axis('off')
plt.show()
```
Shape of original image: (500, 500, 3)  
Shape of tiles array: (25, 100, 100, 3)  
Some coordinates in x1,y1,x2,y2 format:  
[[  0   0 100 100]  
 [100   0 200 100]  
 [200   0 300 100]]
![Alt text](logo/tiles.png?raw=true "The result of the tiling process.")  


# Benchmarks

**Benchmarked on**: AMD Ryzen 5 PRO 6650U; 6 cores; 12 threads; 2.9 GHz

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
 ⬜️ Add less strict (flexible) duplicate removal methods to avoid missing bounding boxes.  
 ⬜️ Consider bounding boxes in tiles if they *partially* belong to one.  
 ⬜️ Support reading annotations from a dataframe/csv file.  
 ⬜️ Make tiles with multidimensional data (e.g. hdf5 hyperspectral images).  
  
