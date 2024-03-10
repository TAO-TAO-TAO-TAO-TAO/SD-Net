## SD-Net: Symmetric-Aware Keypoint Prediction and Domain Adaptation for 6D Pose Estimation In Bin-picking Scenarios
This is the code of pytorch version for paper: [**Symmetric-Aware Keypoint Prediction and Domain Adaptation for 6D Pose Estimation In Bin-picking Scenarios**]





## Environment
Ubuntu 16.04/18.04

python3.6, torch 1.1.0, torchvision 0.3.0, opencv-python, sklearn, h5py, nibabel, et al.

Our backbone PointNet++ is borrowed from [pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch).

## Dataset
Siléane dataset is available at [here](http://rbregier.github.io/dataset2017).
Parametric dataset is available at [here](https://github.com/lvwj19/ParametricNet).
Fraunhofer IPA Bin-Picking dataset is available at [here](https://owncloud.fraunhofer.de/index.php/s/AacICuOWQVWDDfP?path=%2F).

## Evaluation metric
The python code of evaluation metric is available at [here](https://github.com/rbregier/pose_recovery_evaluation).

## Overview of SD-Net architecture.
Illustration of the SD-Net architecture for 6DoF Pose Estimation in stacked scenarios
![Alt text](/images/model1.png)
We omit the domain adaptation framework, for brevity and more details can be found in ![Alt text](/images/model2.png).



## Citation
If you find our work useful in your research, please consider citing:

@article{din2024SD-Net,
title={SD-Net: Symmetric-Aware Keypoint Prediction and Domain Adaptation for 6D Pose Estimation In Bin-picking Scenarios},
author={Ding-Tao Huang, En-Te Lin, Lipeng Chen2, Li-Fu Liu1, Long Zeng},
journal={arXiv preprint arXiv},
year={2024}
}


## Citation
If you find our work useful in your research, please consider citing:

    @article{lin2023sam,
    title={SAM-6D: Segment Anything Model Meets Zero-Shot 6D Object Pose Estimation},
    author={Lin, Jiehong and Liu, Lihua and Lu, Dekun and Jia, Kui},
    journal={arXiv preprint arXiv:2311.15707},
    year={2023}
    }



## Contact

If you have any questions, please feel free to contact the authors. 

Ding-Tao Huang: [hdt22@mails.tsinghua.edu.cn](hdt22@mails.tsinghua.edu.cn)

En-Te Lin: [linet22@mails.tsinghua.edu.cn](linet22@mails.tsinghua.edu.cn)
