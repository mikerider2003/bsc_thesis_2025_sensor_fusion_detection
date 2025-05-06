import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def visualize_pseudo_image(pseudo_image, save_path=None, figsize=(10, 8)):
    """
    Visualizes the feature presence in a pseudo-image.
    
    Args:
        pseudo_image (torch.Tensor): Tensor of shape [1, C, H, W] where C is number of channels
        save_path (str, optional): Path to save the visualization. If None, the plot is shown.
        figsize (tuple): Figure size (width, height) in inches
    """
    if isinstance(pseudo_image, torch.Tensor):
        # Move to CPU if on GPU and convert to numpy
        pseudo_image = pseudo_image.detach().cpu().numpy()
    
    # Count non-zero elements in each spatial cell
    non_zero_mask = (pseudo_image[0] != 0).sum(axis=0) > 0
    
    # Create figure with appropriate size
    plt.figure(figsize=figsize)
    
    # Plot binary presence of features with enhanced visibility
    im = plt.imshow(non_zero_mask, cmap='viridis', interpolation='nearest')
    
    plt.title('Pseudo-Image ', fontsize=16)
    plt.grid(False)
    
    # Higher DPI for better quality
    dpi = 150
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Feature presence visualization saved to {save_path}")
    else:
        plt.show()
    
    return plt.gcf()

if __name__ == "__main__":
    import os
    
    
    from src.models.PointPillars import PillarFeatureNet, PseudoImageScatter
    from src.loaders.loader_Point_Pillars import PointPillarsLoader
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset and loader
    train_dataset = PointPillarsLoader(dataset_path, split='train')
    
    # Create a small sample for testing
    processed_samples = train_dataset.process_all_samples(limit=1)
    sample = processed_samples[0]
    pillars, coords = sample["lidar_processed"]
    
    # Convert numpy arrays to torch tensors
    pillars_tensor = torch.from_numpy(pillars).float()
    coords_tensor = torch.from_numpy(coords).int()
    
    # Calculate grid dimensions
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
    
    # Visualize feature density
    visualize_pseudo_image(
        pseudo_image,
        save_path="src/utils/visualization/figures/Figure_2_pseudo_image.png"
    )
    
    print("Visualization complete. Check the figures directory for results.")