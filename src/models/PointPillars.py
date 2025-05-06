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
    
    # Calculate grid dimensions based on x_range and y_range from loader
    x_range = (-100, 100)
    y_range = (-100, 100)
    voxel_size = (0.3, 0.3)
    
    nx = int(np.floor((x_range[1] - x_range[0]) / voxel_size[0]))
    ny = int(np.floor((y_range[1] - y_range[0]) / voxel_size[1]))
    
    
    # Create model
    pfn = PillarFeatureNet()
    scatter = PseudoImageScatter(output_shape=(ny, nx), num_features=64)
    
    # Forward pass
    pillar_features = pfn(pillars_tensor)
    pseudo_image = scatter(pillar_features, coords_tensor)

    print(f"Pillar features shape: {pillar_features.shape}")
    print(f"Pseudo-image shape: {pseudo_image.shape}")
    
    # Check if any features were scattered
    non_zero = torch.count_nonzero(pseudo_image)
    print(f"Number of non-zero elements in pseudo-image: {non_zero}")
    
    # Visualize a slice of the pseudo-image if desired
    if pseudo_image.shape[0] > 0:
        # Take first batch, first channel
        sample_slice = pseudo_image[0, 0].detach().cpu().numpy()
        print(f"Sum of values in first channel slice: {np.sum(sample_slice)}")
        print(f"Max value in first channel slice: {np.max(sample_slice)}")
