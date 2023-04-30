# dair3d_displayer

A Python tool for displaying DAIR3D dataset

## Requirements
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/dair3d_displayer.git
   ```
 - Install PyTorch environment with Anaconda (Tested on Ubuntu 16.04 & CUDA 10.2)
   ```
   conda create -n pcdet.v0.5.0 python=3.6
   conda activate pcdet.v0.5.0
   cd dair3d_displayer
   pip install -r requirements.txt
   ```
 - Install visualization tools
   ```
   pip install mayavi
   pip install pyqt5
   pip install open3d-python
   pip install opencv-python
   ```

## DAIR3D Dataset (41.5GB)
 - Download [DAIR3D Dataset](https://thudair.baai.ac.cn/roadtest)
 - Organize the downloaded files as follows
   ```
   dair3d_displayer
   ├── dair_i
   │   │── ImageSets
   │   │   ├──test.txt & train.txt & trainval.txt & val.txt
   │   │── training
   │   │   ├──calib & velodyne & label & image
   │   │── testing
   │   │   ├──calib & velodyne & image
   ├── dair3d_displayer.py
   ```

## Usages
 - Display data in DAIR3D dataset
   ```
   # Show velodyne point clouds and ground truth boxes
   python dair3d_displayer.py --show_gt_boxes
   
   # Show the RGB image and projected ground truth boxes
   python dair3d_displayer.py --onto_image --show_gt_boxes
   ```
