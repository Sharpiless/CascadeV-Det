# Cascade Point Voting: Towards High Quality 3D Object Detection

This is the official implementation of CascadeV-Det.


## Prerequisites
The code is tested with Python3.8, PyTorch == 1.9.0, CUDA == 11.1, mmdet3d == 0.15.0, mmcv_full == 1.4.0, mmsegmentation==0.14.1 and mmdet == 2.14.0. We recommend you to use anaconda to make sure that all dependencies are in place. Note that different versions of the library may cause changes in results.


Install Minkowski Engine.
```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```

Install differentiable IoU.
```
git clone https://github.com/lilanxiao/Rotated_IoU ./rotated_iou
cd ./rotated_iou
git checkout 3bdca6b20d981dffd773507e97f1b53641e98d0a
cp -r ./rotated_iou ${path_to_codebase}/mmdet3d/ops/rotated_iou
cd ${path_to_codebase}/mmdet3d/ops/rotated_iou/cuda_op
python setup.py install
```

Prepare SUN RGB-D Data following the procedure [here](https://github.com/open-mmlab/mmdetection3d/tree/master/data/sunrgbd).

## Getting Started
**Step 1.** First we need to train a image branch [here](https://github.com/haoy945/DeMF).
```shell
Please refer to master branch.
```
Or you can download the pre-trained image branch [here](https://drive.google.com/file/d/1H0SGOSvfYU45ID38CvQohIyAUeAXm3Ra/view?usp=sharing).

**Step 2.**
Specify the path to the pre-trained image branch in [config](configs/fcaf-ca/base.py).

**Step 3.** Train our CascadeV-Det using the following command.
```shell
bash tools/dist_train.sh configs/fcaf-ca/mix_bbox_mask_cascade_lose_top5_5_selective_512_layer2_2gpu.py 2
```
We also provide pre-trained model and log [here](https://drive.google.com/drive/folders/1DVCPN50qKkRRd9Ndtr6CXHYqnOuHs4f5?usp=sharing). Evaluate the pretrained model and you will get the 67.5mAP@0.25 and 51.1mAP@0.5.
```shell
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT tools/test.py --config configs/fcaf-ca/base.py --checkpoint $CHECKPOINT --eval mAP --launcher pytorch ${@:4}
```

## Main Results
We re-implemented VoteNet and ImVoteNet, which are some improvement over the original results.
 Method | Point Backbone | Input | mAP@0.25 | mAP@0.5 |
 :----: | :----: | :----: | :----: | :----: |
 VoteNet | PointNet++ | PC | 60.0 | 41.3 |
 ImVoteNet | PointNet++ | PC+RGB | 64.4 | 43.3 |
 FCAF3D | HDResNet34 | PC | 64.2 (63.8) | 48.9 (48.2) |
 [DeMF(FCAF3D based)](configs/fcaf-ca/base.py) | HDResNet34 | PC+RGB | 67.4 (67.1) | 51.2 (50.5) |


## Citation
If you find this work useful for your research, please cite our paper:
```
@misc{https://doi.org/10.48550/arxiv.2207.10589,
  author    = {Yang, Hao and Shi, Chen and Chen, Yihong and Wang, Liwei},
  title     = {Boosting 3D Object Detection via Object-Focused Image Fusion},
  publisher = {arXiv},
  year      = {2022},
}
```
