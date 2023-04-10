# plakakia
*Python image tiling library for image processing, object detection, etc.*

**DISCLAIMER**: This is a work in progress.  
  
![Alt text](logo/logo.png?raw=true "This is a \"plakaki\", meaning tile in Greek.")  

The reason for making this tool is to handle image tiling that takes into account bounding boxes that exist inside the image. For now, I've only considered rectangular tiles. An image is divided into tiles based on a given `tile_size` and `step_size`. Overlapping tiles are handled fine, too. Bounding boxes belong to a tile only if they are *fully* inside it, but I'm planning to support partial overlap as well. Note that such methods create duplicate bounding boxes (i.e. a bounding box can appear in more than one tiles). Soon, there will be options for avoiding duplicates. Of course, depending on the strictness of the duplicate deletion method there is a potential cost of missing some bounding boxes altogether.  
  
In this package, I employ `multiprocessing` and `numpy` extensively to make this as fast as possible so that one can use it with thousands of images. For benchmarks on some public datasets, see below.  

# Requirements
There is a `requirements.txt` file which you can use to create a virtual environment. It is **highly** recommended that you do that before using this repository.  
Here are some recommended steps to follow:  
 1. Download and install [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) (or [Anaconda](https://www.anaconda.com/products/distribution)). 
 2. Create a virtual environment:  
 `mamba create -n plakakia jupyterlab nb_conda_kernels ipykernel ipywidgets pip -y`  
 3. Activate the environment:  
 `mamba activate plakakia`
 4. Install all requirements:  
 `pip install -r requirements.txt`

# Usage

 - The `config.yaml` file is used to define your input/output paths for where you placed your images and annodations and where you want them to be exported.
 - Place all your data in the input folder split into 'images' and 'annotations'.
 - Run `python make_some_tiles.py`

 **NOTE**: For now, only YOLO (`yolo`) and PascalVOC (`pascal_voc`) formats are allowed in the settings. 

**TODO:**  
 ☑️ ~~Fix reading of classes from annotations (create a 'mapper' dictionary to map classes to numerical values).~~  
 ☑️ ~~Read settings from a file (e.g. json).~~  
 ⬜️ Consider bounding boxes in tiles if they *partially* belong to one.  
 ⬜️ Support reading annotations from a dataframe/csv file.  
 ⬜️ Support other annotation formats (e.g. coco).  
 ⬜️ Make tiles with multidimensional data (e.g. hdf5 hyperspectral images).  
 ⬜️ Provide tiling functionality without any labels needed.  

## Benchmarks

| Dataset | Source | Formats (images/labels) | Number of images | tile_size | step_size | tiles generated | plakakia performance |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Traffic Signs  | [Kaggle](https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format) | jpg/YOLO | 741  | 300 | 200 | 1.695 | 2,8 sec | 
| Hard Hat Workers v2  | [RoboFlow](https://public.roboflow.com/object-detection/hard-hat-workers/2) | jpg/YOLO | 5.269 | 100 | 50 | 21.678 | 6,94 sec| 
| Microsoft COCO dataset  | [RoboFlow](https://public.roboflow.com/object-detection/microsoft-coco-subset) | jpg/YOLO | 121.408 | 200 | 150 | 177.039 | 3 min 4 sec| 
