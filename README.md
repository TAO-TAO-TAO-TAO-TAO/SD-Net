## SD-Net: A Novel Keypoint Selection and Filtering for 6DoF Pose Estimation in Stacked Scenarios
This is the code of pytorch version for paper: [**A Novel Keypoint Selection and Filtering for 6DoF Pose Estimation in Stacked Scenarios**]


## Environment
Ubuntu 16.04/18.04

python3.6, torch 1.1.0, torchvision 0.3.0, opencv-python, sklearn, h5py, nibabel, et al.

Our backbone PointNet++ is borrowed from [pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Dataset
Siléane dataset is available at [here](http://rbregier.github.io/dataset2017).


Fraunhofer IPA Bin-Picking dataset is available at [here](https://owncloud.fraunhofer.de/index.php/s/AacICuOWQVWDDfP?path=%2F).

## Evaluation metric
The python code of evaluation metric is available at [here](https://github.com/rbregier/pose_recovery_evaluation).

## Illustration of the SD-Net architecture for 6DoF Pose Estimation in stacked scenarios
![Alt text](/SD-Net/images/model.png)
