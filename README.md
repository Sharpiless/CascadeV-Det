# CascadeV-Det
Official Code of "[Arxiv 2024] CascadeV-Det: Cascade Point Voting for 3D Object Detection"

Code repository is under construction... 🏗️ 🚧 🔨

This is a [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) implementation.


## Prerequisites
The code is tested with Python3.7, PyTorch == 1.8, CUDA == 11.1, mmdet3d == 0.18.1, mmcv_full == 1.3.18 and mmdet == 2.14. We recommend you to use anaconda to make sure that all dependencies are in place. Note that different versions of the library may cause changes in results.

**Step 1.** Create a conda environment and activate it.
```
conda create --name cascadevdet python=3.7
conda activate cascadevdet
```

**Step 2.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) following the instruction [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

**Step 3.** Prepare SUN RGB-D Data following the procedure [here](https://github.com/open-mmlab/mmdetection3d/tree/master/data/sunrgbd).
