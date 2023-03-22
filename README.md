# plakakia
*Python image tiling library for image processing, object detection, etc.*

**DISCLAIMER**: This is a work in progress.  
  
![Alt text](logo/logo.png?raw=true "This is a \"plakaki\", meaning tile in Greek.")  

The reason for making this tool is to handle image tiling that takes into account bounding boxes that appear around the image. For now, I've only considered rectangular tiles. An image is divided into tiles based on a given `tile_size` and `step_size`. Overlapping tiles can be handled fine. Bounding boxes are assigned to a tile only if they are *fully* inside it, but I'm planning to support partial overlap as well. In any case, you should bear in mind that such methods create duplicate bounding boxes (i.e. a bounding box can appear in more than one tiles). I'm planning to provide some options on how to handle that, e.g. avoid duplicates at the potential cost of missing some bounding boxes depending on the choice of `tile_size` and `step_size`.  
  
Tried to use `numpy` extensively and make this as fast as possible so that one can use it with thousands of images. This is a work in progress. Planning to check out `numba` and `cupy` to see if even larger speedups are possible.

# Requirements
There is a `requirements.txt` file which you can use to create a virtual environment. It is **highly** recommended that you do that before using this repository.  
Here are some recommended steps to follow:  
 1. Download and install [Anaconda](https://www.anaconda.com/). 
 2. Create a virtual environment:  
 `conda create -n plakakia jupyterlab nb_conda_kernels ipykernel ipywidgets pip -y`  
 3. Activate the environment:  
 `conda activate plakakia`
 4. Install all requirements:  
 `pip install -r requirements.txt`

# Usage

 - The dataclass `Settings` can be used to define your input/output paths for where you placed your images and annodations and where you want them to be exported:  
    **`Settings(input_extension_images='jpg',
        # pad_image=False,
        tile_size=250,
        step_size=100,
        input_dir_images='input/images',
        input_dir_annotations='input/annotations',
        input_format_annotations='yolo',
        output_dir_images='output/images',
        output_dir_annotations='output/annotations',
        output_format_annotations='yolo',
        draw_boxes=False,
        log=True,
        log_folder='logs',)`**

 - Place all your data in the input folder split into 'images' and 'annotations'.
 - Run `python make_some_tiles.py`

 **NOTE**: For now, only YOLO (`yolo`) and PascalVOC (`pascal_voc`) formats are allowed in the settings. 
  
**TODO:**  
 ⬜️ Fix reading of classes from annotations (create a 'mapper' dictionary to map classes to numerical values).  
 ⬜️ Read settings from a file (e.g. json).  
 ⬜️ Consider bounding boxes in tiles if they *partially* belong to one.  
 ⬜️ Support reading annotations from a dataframe/csv file.  
 ⬜️ Support other annotation formats (e.g. coco).  
 ⬜️ Make tiles with multidimensional data (e.g. hdf5 hyperspectral images).  
