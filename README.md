# Single vs. Multi-Modal: A Comparative Analysis of Object Detection Methods in Autonomous Driving

# Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
    - [Single-Modal](#single-modal)
        - [PointPillars (LiDAR Only)](#pointpillars-lidar-only)
    - [Multi-Modal](#multi-modal)
        - [PointFusion (Early Fusion)](#pointfusion-early-fusion)
        - [MVX-Net (Mid Fusion)](#mvx-net-mid-fusion)
        - [AVOD (Late Fusion)](#avod-late-fusion)


# Introduction
This project compares single-modal and multi-modal object detection methods for autonomous driving using the Argoverse2 dataset. It evaluates how well LiDAR-only, and fused sensor approaches (via early, mid, and late fusion strategies) detect objects like vehicles and pedestrians in diverse urban conditions. By using models such as PointPillars for LiDAR and fusion methods like PointFusion, MVX-Net, and AVOD, the study aims to quantify differences in accuracy, robustness, and inference speed, offering insights into effective sensor fusion strategies for safer autonomous systems.

# Dataset
The Argoverse2 sensor dataset is a large-scale dataset designed for autonomous driving research. It includes high-resolution sensor data from various modalities, including LiDAR and cameras, collected in diverse urban environments.

[Argoverse 2 Link](https://www.argoverse.org/av2.html)

Due to the size of the original dataset (1TB), this study will use a subset of the Argoverse2 dataset, specifically 5% of the data (41,2 GB), which is split into 28 training sequences and 7 test sequences.

To download the dataset, you can use the following command:

```bash
./scripts/download_dataset.sh
```


## Dataset Structure:

```
├── data
│   ├── train
│   │   ├── 0a8a4cfa-4902-3a76-8301-08698d6290a2
│   │   │   ├── calibration
│   │   │   ├── sensors
│   │   │   │   ├── lidar
│   │   │   │   │   ├── 315968251560183000.feather
│   │   │   │   ├── cameras
│   │   │   │   │   ├── ring_front_center
│   │   │   │   │   │   ├── 315968251549927220.jpg
│   │   │   │   │   │   ├── ...
│   │   │   │   │   ├── ring_front_left
│   │   │   │   │   ├── ring_front_right
│   │   │   │   │   ├── ring_rear_left
│   │   │   │   │   ├── ring_rear_right
│   │   │   │   │   ├── ring_side_left
│   │   │   │   │   ├── ring_side_right
│   │   │   ├── annotations.feather
│   │   │   ├── ...
│   ├── test
│   │   ├── ...
```

## Methodology
### Single-Modal:
#### PointPillars (LiDAR Only)

### Multi-Modal
#### PointFusion (Early Fusion)

#### MVX-Net (Mid Fusion)

#### AVOD (Late Fusion)
