import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loaders.loader_Point_Pillars import PointPillarsLoader

class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=64):
        super(PillarFeatureNet, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):  # x: [P, N, 9]
        P, N, D = x.shape
        x = x.view(P * N, D)
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(P, N, -1)
        x = torch.max(x, dim=1)[0]  # [P, C]
        return x

class PseudoImageScatter(nn.Module):
    def __init__(self, output_shape, num_features=64):
        """
        Scatter pillar features to a 2D pseudo-image
        
        Args:
            output_shape: Tuple (H, W) defining the output pseudo-image dimensions
            num_features: Number of features per pillar
        """
        super(PseudoImageScatter, self).__init__()
        self.output_shape = output_shape
        self.num_features = num_features
        
    def forward(self, pillar_features, coords):
        """
        Args:
            pillar_features: Tensor of shape [num_pillars, num_features]
            coords: Tensor of shape [num_pillars, 3] with indices (batch_idx, x_idx, y_idx)
        
        Returns:
            pseudo_image: Tensor of shape [1, num_features, H, W]
        """
        H, W = self.output_shape
        
        # Create empty pseudo-image tensor with batch_size=1
        pseudo_image = torch.zeros(
            (1, self.num_features, H, W),
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )
        
        # Only use non-empty pillars
        if pillar_features.shape[0] > 0:
            # Get indices from coords - assuming batch index is always 0
            x_idx = coords[:, 2].long()  # Note: In your implementation, y is at index 2
            y_idx = coords[:, 1].long()  # and x is at index 1 (this follows the paper's convention)
            
            # Filter out invalid coordinates
            valid_mask = (
                (x_idx >= 0) & (x_idx < W) & 
                (y_idx >= 0) & (y_idx < H)
            )
            
            if valid_mask.any():
                y_idx_valid = y_idx[valid_mask]
                x_idx_valid = x_idx[valid_mask]
                features_valid = pillar_features[valid_mask]
                
                # Simple approach - loop through each valid pillar
                for i in range(len(y_idx_valid)):
                    pseudo_image[0, :, y_idx_valid[i], x_idx_valid[i]] = features_valid[i]
        
        return pseudo_image

class CNN_BackBone(nn.Module):
    def __init__(self, in_channels=64):
        super(CNN_BackBone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

# SSD Detection Head for predicting bounding boxes
class SSDDetectionHead(nn.Module):
    def __init__(self, num_classes, in_channels=128):
        super(SSDDetectionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        # For each class, predict 7 values: (x, y, z, length, width, height, θ)
        self.conv2 = nn.Conv2d(256, num_classes * 7, kernel_size=3, padding=1)
        
        # Class prediction
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        bbox_regression = self.conv2(x)
        class_scores = self.conv3(x)
        return bbox_regression, class_scores

class PointPillarsModel(nn.Module):
    def __init__(self, num_classes, voxel_size=(0.3, 0.3), x_range=(-100, 100), y_range=(-100, 100)):
        super(PointPillarsModel, self).__init__()
        
        # Calculate grid dimensions
        self.nx = int(np.floor((x_range[1] - x_range[0]) / voxel_size[0]))
        self.ny = int(np.floor((y_range[1] - y_range[0]) / voxel_size[1]))
        
        # Model components
        self.pfn = PillarFeatureNet(in_channels=9, out_channels=64)
        self.scatter = PseudoImageScatter(output_shape=(self.ny, self.nx), num_features=64)
        self.backbone = CNN_BackBone(in_channels=64)
        self.head = SSDDetectionHead(num_classes=num_classes, in_channels=128)
        
    def forward(self, pillars, coords):
        # Pillar feature encoding
        pillar_features = self.pfn(pillars)  # [P, C]
        
        # Scatter to pseudo image
        pseudo_image = self.scatter(pillar_features, coords)  # [1, C, H, W]
        
        # CNN backbone
        cnn_features = self.backbone(pseudo_image)  # [1, 128, H/4, W/4]
        
        # SSD head
        bbox_preds, cls_scores = self.head(cnn_features)
        
        return bbox_preds, cls_scores

# For testing/debugging
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset and loader
    train_dataset = PointPillarsLoader(dataset_path, split='train')

    # Create a small sample for testing
    processed_samples = train_dataset.process_all_samples(limit=3)
    sample = processed_samples[0]
    pillars, coords = sample["lidar_processed"]

    # Convert numpy arrays to torch tensors
    pillars_tensor = torch.from_numpy(pillars).float()
    coords_tensor = torch.from_numpy(coords).int()

    print(f"Input pillar shape: {pillars.shape}")
    print(f"Coords shape: {coords.shape}")

    # Initialize the model
    model = PointPillarsModel(num_classes=len(train_dataset.classes) + 1)  # +1 for background class

    # Forward pass
    bbox_preds, cls_scores = model(pillars_tensor, coords_tensor)
    print(f"Bounding box predictions shape: {bbox_preds.shape}")
    print(f"Class scores shape: {cls_scores.shape}")



