import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pyarrow.feather as feather
import pandas as pd
import sys
import os

from src.loaders.loader_Point_Pillars import PointPillarsLoader


def load_point_cloud(lidar_file):
    """Load point cloud from feather file."""
    lidar_data = feather.read_feather(lidar_file)
    points = lidar_data[['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
    return points


def visualize_pillars_2d(pillars, coords, annotations=None, voxel_size=(0.3, 0.3), x_range=(-100, 100), y_range=(-100, 100), 
                         title="2D Pillar Visualization", show_points=True):
    """Visualize pillars in 2D top-down view with matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Set axis limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.grid(True)
    
    # Calculate actual pillar positions in world coordinates
    num_pillars = min(coords.shape[0], np.count_nonzero(np.sum(coords, axis=1)))
    
    # Plot each pillar as a rectangle
    for p in range(num_pillars):
        x_idx, y_idx = coords[p, 0], coords[p, 1]
        
        # Skip if coordinates are all zeros (empty pillar)
        if x_idx == 0 and y_idx == 0 and np.sum(pillars[p]) == 0:
            continue
        
        # Calculate bottom-left corner position
        x_min = x_range[0] + x_idx * voxel_size[0]
        y_min = y_range[0] + y_idx * voxel_size[1]
        
        # Count actual points in this pillar
        num_points = np.count_nonzero(np.sum(pillars[p], axis=1))
        
        # Color based on point density (more points = darker)
        density = min(1.0, num_points / 20)  # Cap at 1.0, assume max 20 points for full darkness
        
        # Draw pillar rectangle
        rect = Rectangle((x_min, y_min), voxel_size[0], voxel_size[1], 
                         fill=True, color=(0, 0, 1-density), alpha=0.5, 
                         ec='black', lw=0.5)
        ax.add_patch(rect)
        
        # Optionally show points in the pillar
        if show_points:
            points_in_pillar = pillars[p, :num_points, :2]  # Get x,y coordinates
            ax.scatter(points_in_pillar[:, 0], points_in_pillar[:, 1], 
                      s=1, color='lightblue', alpha=0.8)
    
    # Plot annotations (bounding boxes) if provided
    if annotations is not None:
        # Define a color map for different categories
        category_colors = {
            'REGULAR_VEHICLE': 'red',
            'PEDESTRIAN': 'green',
            'CYCLIST': 'blue',
            'LARGE_VEHICLE': 'purple',
            # Add more categories as needed
        }
        
        for _, anno in annotations.iterrows():
            # Get box dimensions and position
            length, width = anno['length_m'], anno['width_m']
            tx, ty = anno['tx_m'], anno['ty_m']
            category = anno['category']
            
            # Get orientation from quaternion (simplified for 2D top-down view)
            qw, qx, qy, qz = anno['qw'], anno['qx'], anno['qy'], anno['qz']
            
            # Convert quaternion to yaw angle (rotation around z-axis)
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
            
            # Create rotated rectangle for the bounding box
            color = category_colors.get(category, 'orange')  # Default to orange if category not in dict
            
            # Create corners of rectangle before rotation
            dx, dy = length/2, width/2
            corners = np.array([
                [-dx, -dy],
                [dx, -dy],
                [dx, dy],
                [-dx, dy]
            ])
            
            # Rotation matrix
            rot_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            
            # Rotate corners
            rotated_corners = np.dot(corners, rot_matrix.T)
            
            # Translate corners to the bounding box center
            for i in range(len(rotated_corners)):
                rotated_corners[i][0] += tx
                rotated_corners[i][1] += ty
            
            # Create the polygon and add to plot
            polygon = plt.Polygon(rotated_corners, fill=False, edgecolor=color, linewidth=2, label=category)
            ax.add_patch(polygon)
            
    
    # Add legend if annotations were plotted
    if annotations is not None:
        # Create a set of unique categories for the legend
        unique_categories = set(annotations['category'].tolist())
        # Create proxy artists for the legend
        legend_elements = [plt.Line2D([0], [0], color=category_colors.get(cat, 'orange'), 
                                     lw=2, label=cat) for cat in unique_categories]
        # Add the legend to the plot
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate the visualization tools."""
    from dotenv import load_dotenv
    import sys
    import os

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset and loader
    train_dataset = PointPillarsLoader(dataset_path, split='train')
    
    # Process a sample
    sample = train_dataset.samples[0]
    pillars, coords = train_dataset._process_lidar(sample['lidar_file'], debug=False)
    
    annotations = sample.get('annotations', None)
    print(f"Annotations columns: {annotations.columns}")

    # Load raw point cloud for comparison
    raw_points = load_point_cloud(sample['lidar_file'])
    
    print(f"Raw point cloud shape: {raw_points.shape}")
    print(f"Pillars shape: {pillars.shape}, Coords shape: {coords.shape}")
    
    # Display pillarized point cloud in 2D with annotations
    print("Visualizing pillars in 2D with annotations...")
    visualize_pillars_2d(pillars, coords, annotations=annotations)
    

if __name__ == "__main__":
    main()