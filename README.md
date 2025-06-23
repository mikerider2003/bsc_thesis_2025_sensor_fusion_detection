# Single vs. Multi-Modal: A Comparative Analysis of Object Detection Methods in Autonomous Driving

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
    - [Pre-processing](#pre-processing)
3. [Methodology](#methodology)
    - [Single-Modal](#single-modal)
        - [PointPillars (LiDAR Only)](#pointpillars-lidar-only)
    - [Multi-Modal](#multi-modal)
        - [PointFusion (Early Fusion)](#pointfusion-early-fusion)


## Introduction
This project compares single-modal and multi-modal object detection methods for autonomous driving using the Argoverse2 dataset. It evaluates how well LiDAR-only, and fused sensor approach detect objects like vehicles and pedestrians in diverse urban conditions. By using models such as PointPillars for LiDAR and fusion methods like PointFusion, the study aims to quantify differences in accuracy, robustness, and inference speed, offering insights into effective sensor fusion strategies for safer autonomous systems.

## Dataset
The Argoverse2 sensor dataset is a large-scale dataset designed for autonomous driving research. It includes high-resolution sensor data from various modalities, including LiDAR and cameras, collected in diverse urban environments.

[Argoverse 2 Link](https://www.argoverse.org/av2.html)

Due to the size of the original dataset (1TB), this study will use a subset of the Argoverse2 dataset, specifically 1% of the data (8,05 GB), which is split into 5 training sequences and 2 test sequences.

To download the dataset, you can use the following command:

```bash
./scripts/download_dataset.sh
```


<details>
<summary>Permission Settings</summary>

If you encounter permission issues when running the script, you can edit permissions using the command:

```bash
chmod +x ./scripts/download_data.sh
```
</details>


### Dataset Structure:

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

### Pre-processing
The dataset is pre-processed to extract relevant features and annotations for training and evaluation. 

## Methodology
### Single-Modal
### PointPillars (LiDAR Only)

#### Architecture
1. **Pillar Feature Encoder (PillarFeatureNet)**
   - Input: [P, N, 9] where:
     - P = number of non-empty pillars
     - N = points per pillar (fixed at 100)
     - 9 features per point: [x, y, z, intensity, x_c, y_c, z_c, x_p, y_p]
   - Output: [P, C] feature vectors (typically C = 64)

2. **Pseudo-Image Scattering**
   - Places pillar features into 2D canvas [C, H, W]
   - Uses coordinates: [batch_idx, x_idx, y_idx]

3. **2D CNN Backbone**
   - Processes pseudo-image [B, C, H, W]
   - Outputs high-level features: [B, 6C, H, W]

4. **SSD Detection Head**
   - Predicts:
     - Class scores: [num_anchors * num_classes]
     - Box regression: [num_anchors * 7] → (x, y, z, w, l, h, θ)
     - Orientation: [num_anchors * 2]

#### Key Components
- Pillarization of 3D point clouds
- Efficient 2D convolutions on pillar features
- Anchor-based detection

##### Key Files
- `/src/loaders/loader_Point_Pillars.py`: Handles the data loading and preprocessing for PointPillars, including the pillarization process
- `/src/utils/visualization/pillars.py`: Provides visualization tools for the pillarized point clouds and detection results
- `/src/models/PointPillars.py`: Will contain the model architecture implementation (currently empty, to be implemented)
- `/scripts/run_train_pointpillars.sh`: Will be used for training the PointPillars model (currently empty, to be implemented)

### PointFusion (Early Fusion)
#### Architecture
1. **2D Object Detection**
   - Faster R-CNN with ResNet-50 backbone
   - Processes 7 ring camera images
   - Confidence threshold: 0.9

2. **LiDAR-Camera Alignment**
   - Projects LiDAR points to 2D using calibration matrices
   - Filters points beyond 40m

3. **Feature Extraction**
   - Image branch: CNN with adaptive pooling
   - LiDAR branch: PointNet-style MLP with max pooling

4. **Multi-Modal Fusion**
    - Early fusion of image and LiDAR features
    - Joint classification and regression heads

#### Key Components
- Early fusion of modalities
- Cross-sensor calibration
- Joint classification and regression

##### Key Files
- `/src/loaders/loader_Point_Fusion.py`: Handles the data loading and preprocessing for PointFusion
- `/src/models/PointFusion.py`: Contains the model architecture implementation
- `/scripts/run_train_pointfusion.sh`: Is used for training the PointFusion model

