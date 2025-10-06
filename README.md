<div align="center"> 

## ShareCMP: Polarization-Aware RGB-P Semantic Segmentation

</div>

<p align="center">

<a href="https://doi.org/10.1109/TCSVT.2025.3570764">
    <img src="https://img.shields.io/badge/DOI-10.1109/TCSVT.2025.3570764-blue" /></a>

<a href="https://arxiv.org/pdf/2312.03430.pdf">
    <img src="https://img.shields.io/badge/arXiv-2312.03430-rgb(179,27,27)" /></a>

<a href="https://github.com/LEFTeyex/ShareCMP/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/LEFTeyex/ShareCMP" /></a>

</p>

## Introduction

The official implementation of **ShareCMP: Polarization-Aware RGB-P Semantic Segmentation**.

## Installation

This project is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

- Python 3.8
- PyTorch 1.13.1+cu116

**Step 1.** Create a conda virtual environment and activate it.

```bash
conda create -n sharecmp python=3.8 -y
conda activate sharecmp
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

Linux and Windows

```bash
# Wheel CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

```bash
# Conda CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

**Step 3.** Install MMSegmentation and dependent packages.

```bash
pip install -U openmim
mim install mmengine==0.8.5
mim install mmcv==2.0.1
mim install mmsegmentation==1.1.2
pip install -r requirements.txt
```

## Dataset

The data structure UPLight (
download [here](https://drive.google.com/drive/folders/1syjigG5T3CeArglieQud3rNvtrLBLHjT?usp=drive_link)) looks like
below:

```text
# UPLight

data
├── UPLight
│   ├── images_rgb
│   ├── images_rgb_0
│   ├── images_rgb_45
│   ├── images_rgb_90
│   ├── images_rgb_135
│   ├── labels
│   ├── train.txt
│   ├── val.txt
```

## Usage

### Training

```bash
bash tools/dist_train.sh configs/sharecmp/sharecmp_mit-b2_2xb4-200e_uplight-512x612.py 2
```

### Test

The weight .pth of ShareCMP is
available [here](https://drive.google.com/drive/folders/1DliabS_ctGKJXPEmDHJXz4F9jMkJzwAb?usp=drive_link).

```bash
bash tools/dist_test.sh configs/sharecmp/sharecmp_mit-b2_2xb4-200e_uplight-512x612.py sharecmp_mit-b2_2xb4-200e_uplight-512x612.pth 2
```

## Cite

```
@article{liu2025sharecmp,
  title={ShareCMP: Polarization-Aware RGB-P Semantic Segmentation},
  author={Liu, Zhuoyan and Wang, Bo and Wang, Lizhi and Mao, Chenyu and Li, Ye},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={35},
  number={10},
  pages={10316-10329},
  year={2025},
  doi={10.1109/TCSVT.2025.3570764},
}
```