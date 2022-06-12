# Dataset Generation

This folder contains every script that was used to generate fragment data.
More detailed instructions can be found inside each script itself.

## Prequisites

The blender script makes use of a modified blender build of 2.79b. Only this special build will work.
- Builds: https://drive.google.com/drive/folders/19T2rFXduMOzLDHcvD1t_abYLTwE1IREl
    - in case no builds are available anymore, it may also be build from source: https://github.com/blender/blender/tree/fracture_modifier-master
- Video Tutorial: https://www.youtube.com/watch?v=rCAzkTVyNOw
- Documentation: https://archive.blender.org/wiki/index.php/User:Scorpion81/Fracture_Documentation/

## Usage

The general precedure is as follows:

1. Get 3D meshes in .obj format. They must be watertight, otherwise the script fails.
2. Run one of the fracture scripts inside the ./blender/ folder. There are two versions:
    - `blender_fracture_single.py` takes a list of object file names as input from the command line, thus can be used with other scripts. `command_spawner.py` is an example to run a single blender instance for each object on the ETH Euler cluster job system.
    - `blender_fracture_folder.py` fractures the full content of a folder in a single blender instance (takes very long for many objects)
3. Run `open3d_resample.py` to downsample the obj files and save them into .npy numpy files. It will also output a .html Plotly scatter plot for easy verification of the data.

More detailed instructions of paths and folder structures are located inside each file.

## Surface segmentation

`/surface_segmentation` contains code for labeling the surface of shards. It is not used in the pipeline, but still provided for completeness.

### Usage
- Install the pcl library, i.e. `apt-get install libpcl-dev`
- build the Cmake project: `mkdir build && cd build && cmake .. && make`
- Run `./visualize` for a simple visualization test of the provided fragments
- Run `./regions` to visualize and save the segmented fragments