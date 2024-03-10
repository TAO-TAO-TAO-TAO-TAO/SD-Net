## SD-Net: Symmetric-Aware Keypoint Prediction and Domain Adaptation for 6D Pose Estimation In Bin-picking Scenarios
This is the code of pytorch version for paper: [**Symmetric-Aware Keypoint Prediction and Domain Adaptation for 6D Pose Estimation In Bin-picking Scenarios**]


## Qualitative results
Evaluation Siléane dataset
![Alt text](/images/1.gif)
![Alt text](/images/2.gif)
Evaluation Parametric dataset
![Alt text](/images/4.gif)
![Alt text](/images/3.gif)
## Overview of SD-Net architecture.

## Overview of SD-Net architecture.
Illustration of the SD-Net architecture for 6DoF Pose Estimation in stacked scenarios
![Alt text](/images/model1.png)
We omit the domain adaptation framework, for brevity and more details can be found in ![Alt text](/images/model2.png).




## Getting Started

### 1. Preparation
Please clone the repository locally:
```
git clone https://github.com/TAO-TAO-TAO-TAO-TAO/SD-Net.git
```
Install the environment：

Install [Pytorch](https://pytorch.org/get-started/locally/). It is required that you have access to GPUs. The code is tested with Ubuntu 16.04/18.04, CUDA 10.0 and cuDNN v7.4, python3.6.
Our backbone PointNet++ is borrowed from [pointnet2](https://github.com/erikwijmans/Pointnet2_PyTorch).
.Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

    cd tools\Sparepart\train.py
    python train.py install


Install the following Python dependencies (with `pip install`):

    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'
    torch==1.1.0
    torchvision==0.3.0
    sklearn
    h5py
    nibabel


    

### 2. Train SD-Net
    cd tools\Sparepart\train.py
    python train.py install



### 3. Evaluation on the custom data

Dataset
Siléane dataset is available at [here](http://rbregier.github.io/dataset2017).
Parametric dataset is available at [here](https://github.com/lvwj19/ParametricNet).
Fraunhofer IPA Bin-Picking dataset is available at [here](https://owncloud.fraunhofer.de/index.php/s/AacICuOWQVWDDfP?path=%2F).

Evaluation metric
The python code of evaluation metric is available at [here](https://github.com/rbregier/pose_recovery_evaluation).




## Citation
If you find our work useful in your research, please consider citing:

    @article{din2024SD-Net,
    title={SD-Net: Symmetric-Aware Keypoint Prediction and Domain Adaptation for 6D Pose Estimation In Bin-picking Scenarios},
    author={Ding-Tao Huang, En-Te Lin, Lipeng Chen2, Li-Fu Liu1, Long Zeng},
    journal={arXiv preprint arXiv},
    year={2024}
    }



## Contact

If you have any questions, please feel free to contact the authors. 

Ding-Tao Huang: [hdt22@mails.tsinghua.edu.cn](hdt22@mails.tsinghua.edu.cn)

En-Te Lin: [linet22@mails.tsinghua.edu.cn](linet22@mails.tsinghua.edu.cn)

Li-Fu Liu: [llf23@mails.tsinghua.edu.cn](llf23@mails.tsinghua.edu.cn)
