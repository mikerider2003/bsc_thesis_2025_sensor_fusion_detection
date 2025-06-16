import os
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import ListedColormap
from typing import List, Dict, Any

import src.loaders.loader as loader

class PointPillarsLoader(loader.ArgoDataset):
    def __init__(self, dataset_path, split='train'):
        super().__init__(dataset_path, split)

        # Initialize class mappings
        self._setup_class_mapping()

    def _setup_class_mapping(self):
        """Create class name to ID mapping based on all annotations"""
        classes = sorted(self._get_class_categories())
        self.class_to_id = {cls: i for i, cls in enumerate(classes)}
        self.id_to_class = {i: cls for cls, i in self.class_to_id.items()}
        self.num_classes = len(self.class_to_id)

    # 01 [Raw Point Cloud]
    def _load_cloud_point(self, lidar_file):

        lidar_data = feather.read_feather(lidar_file)
        points = lidar_data[['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)

        return points
    
    # TODO: NOTE: Change min points per pillar to 5
    # 02 [Pillarization: divide x-y grid]
    def _pillarization(self, points, grid_size=0.2, max_range=(30, 20, 4), 
                   max_points_per_pillar=100, min_points_per_pillar=5):
        """
        Convert point cloud to pillar representation
        Args:
            points: (N, 4) array of [x, y, z, intensity]
            grid_size: Size of each grid cell in meters
            max_range: (x_range, y_range, z_range) in meters
            max_points_per_pillar: Max points per pillar (pad or sample)
            min_points_per_pillar: Min points required to keep a pillar
        Returns:
            pillar_coords: (P, 2) grid indices
            pillar_points: (P, max_points_per_pillar, 4) point features
            pillar_indices: (P,) linearized pillar indices
            grid_dims: (x_bins, y_bins) grid dimensions
        """
        # Define grid boundaries
        x_min, x_max = -max_range[0], max_range[0]
        y_min, y_max = -max_range[1], max_range[1]
        
        # Calculate grid dimensions
        x_bins = int((x_max - x_min) / grid_size)
        y_bins = int((y_max - y_min) / grid_size)

        # Filter points within range
        in_range_mask = (
            (points[:, 0] >= x_min) & 
            (points[:, 0] < x_max) & 
            (points[:, 1] >= y_min) & 
            (points[:, 1] < y_max))        
        points = points[in_range_mask]       

        # Calculate pillar indices
        x_indices = ((points[:, 0] - x_min) / grid_size)
        y_indices = ((points[:, 1] - y_min) / grid_size)
        pillar_ij = np.floor(np.column_stack([x_indices, y_indices])).astype(np.int32)

        # Group points by pillar
        pillar_dict = {}
        for idx, (i, j) in enumerate(pillar_ij):
            if (i < 0) or (i >= x_bins) or (j < 0) or (j >= y_bins):
                continue  # Skip out-of-bound indices
            
            key = (i, j)
            if key not in pillar_dict:
                pillar_dict[key] = []
            pillar_dict[key].append(points[idx])

        # Create lists to store valid pillars
        valid_pillar_coords = []
        valid_pillar_points = []
        
        # Process each pillar and filter by min point count
        for (i, j), pt_list in pillar_dict.items():
            # Skip pillars with insufficient points
            if len(pt_list) < min_points_per_pillar:
                continue
                
            # Create array for this pillar's points
            pillar_arr = np.zeros((max_points_per_pillar, 4), dtype=np.float32)
            num_points = min(len(pt_list), max_points_per_pillar)
            
            # Randomly sample if too many points
            if len(pt_list) > max_points_per_pillar:
                pt_list = np.array(pt_list)[np.random.choice(len(pt_list), max_points_per_pillar, replace=False)]
            
            # Add points to array
            pillar_arr[:num_points] = pt_list[:num_points]
            
            # Store valid pillar
            valid_pillar_coords.append([i, j])
            valid_pillar_points.append(pillar_arr)

        # Convert to numpy arrays
        if valid_pillar_coords:
            pillar_coords = np.array(valid_pillar_coords, dtype=np.int32)
            pillar_points = np.stack(valid_pillar_points, axis=0)
        else:
            # Return empty arrays if no valid pillars
            pillar_coords = np.zeros((0, 2), dtype=np.int32)
            pillar_points = np.zeros((0, max_points_per_pillar, 4), dtype=np.float32)
        
        # Linearize pillar indices
        pillar_indices = pillar_coords[:, 0] * y_bins + pillar_coords[:, 1] if pillar_coords.size > 0 else np.array([], dtype=np.int32)
        
        return pillar_coords, pillar_points, pillar_indices, (x_bins, y_bins)

    def _encode_features(self, pillar_points, pillar_coords, grid_dims, grid_size=0.2):
        # Create mask for valid (non-padded) points
        valid_mask = np.any(pillar_points[..., :3] != 0, axis=-1)
        
        # Compute centroids (P, 3)
        sum_valid = np.sum(pillar_points[..., :3] * valid_mask[..., np.newaxis], axis=1)
        count_valid = np.sum(valid_mask, axis=1)[:, np.newaxis]
        centroid = np.zeros((len(pillar_points), 3), dtype=np.float32)
        non_zero = count_valid.squeeze(1) > 0
        centroid[non_zero] = sum_valid[non_zero] / count_valid[non_zero]
        
        # Compute pillar centers (P, 2)
        pc_x = pillar_coords[:, 0] * grid_size + grid_size/2
        pc_y = (pillar_coords[:, 1] * grid_size) - (grid_dims[1] * grid_size/2) + grid_size/2
        pillar_center = np.stack([pc_x, pc_y], axis=1)
        
        # Initialize enhanced features array (P, N, 9)
        features = np.zeros((*pillar_points.shape[:2], 9), dtype=np.float32)
        
        # Original features (x, y, z, intensity)
        features[..., :4] = pillar_points
        
        # Centroid offsets (xc, yc, zc)
        centroid_offsets = pillar_points[..., :3] - centroid[:, np.newaxis, :]
        features[..., 4:7] = centroid_offsets * valid_mask[..., np.newaxis]
        
        # Pillar center offsets (xp, yp)
        pc_offsets = pillar_points[..., :2] - pillar_center[:, np.newaxis, :]
        features[..., 7:9] = pc_offsets * valid_mask[..., np.newaxis]
        
        return features
    
    def __getitem__(self, idx, debug=False):
        """
        Get a sample from the dataset by index
        Args:
            idx: Index of the sample
        Returns:
            sample: Dictionary containing point cloud, pillar features, and annotations
        """
        sample = super().__getitem__(idx)

        # Load annotations
        annotations = sample['annotations']
        
        # Load point cloud
        points = self._load_cloud_point(sample['lidar_file'])
        
        # Pillarization
        pillar_coords, pillar_points, pillar_indices, grid_dims = self._pillarization(points)
        
        # Encode features
        features = self._encode_features(pillar_points, pillar_coords, grid_dims)
        
        # Convert to tensors
        features = torch.tensor(features, dtype=torch.float32)
        pillar_coords = torch.tensor(pillar_coords, dtype=torch.long)
        pillar_indices = torch.tensor(pillar_indices, dtype=torch.long)
        
        # Prepare annotation tensors
        annotation_tensors = {
            'boxes': torch.zeros((0, 9), dtype=torch.float32),  # [cx, cy, cz, l, w, h, qw, qx, qy, qz]
            'categories': torch.zeros((0,), dtype=torch.long)
        }
        
        if annotations is not None and len(annotations) > 0:
            # Convert annotations to tensors
            box_data = []
            categories = []
            
            for _, row in annotations.iterrows():
                # Bounding box parameters
                box_data.append([
                    row['tx_m'], row['ty_m'], row['tz_m'],  # Center position
                    row['length_m'], row['width_m'], row['height_m'],  # Dimensions
                    row['qw'], row['qx'], row['qy'], row['qz']  # Orientation
                ])
                
                # Category mapping
                cat_str = row['category']
                cat_id = self.class_to_id.get(cat_str, -1)  # -1 for unknown classes
                categories.append(cat_id)
            
            annotation_tensors = {
                'boxes': torch.tensor(box_data, dtype=torch.float32),
                'categories': torch.tensor(categories, dtype=torch.long)
            }

        if debug:
            # Get pillar statistics
            avg_points, min_points, max_points, _ = get_pillar_stats(pillar_points)
            print(f"Points per pillar: avg={avg_points:.2f}, min={min_points}, max={max_points}")

            # Visualize pillars
            visualize_pillars_2d(pillar_coords, pillar_points, grid_size=0.2, max_range=(30, 20, 4), annotations_df=annotations)

        
        return {
            'features': features,               # [num_pillars, num_features]
            'pillar_coords': pillar_coords,     # [num_pillars, 2]
            'pillar_indices': pillar_indices,   # [num_points]
            'grid_dims': torch.tensor(grid_dims),  # [2]
            'annotations': annotation_tensors
        }
    
    # # ONLY FOR Testing remove !!
    # def test_on_sample(self, n=1):
    #     for i in range(n):
    #         sample = self[i]
    #         lidar_file = sample['lidar_file']
    #         print()
    
    #         # 01 [Raw Point Cloud]
    #         points = self._load_cloud_point(lidar_file)
    #         print(f"Sample {i}: Loaded point cloud with shape: {points.shape}\n")

    #         # 02 Pillarization
    #         pillar_coords, pillar_points, pillar_indices, grid_dims = self._pillarization(points)
    #         print(f"Created {len(pillar_coords)} pillars")
    #         print(f"Pillar points shape: {pillar_points.shape}")
    #         print(f"Grid dimensions: {grid_dims}")

    #         # Get pillar statistics
    #         avg_points, min_points, max_points, _ = get_pillar_stats(pillar_points)
    #         print(f"Points per pillar: avg={avg_points:.2f}, min={min_points}, max={max_points}")

    #         # Plot pillars
    #         annotations = sample['annotations']
    #         visualize_pillars_2d(pillar_coords, pillar_points, grid_size=0.2, max_range=(120, 80, 4), annotations_df=annotations)

    #         # 03 Encode features
    #         features = self._encode_features(pillar_points, pillar_coords, grid_dims)
    #         print(f"Encoded features shape: {features.shape}")



def collate_fn(
    batch: List[Dict[str, Any]],
    keep_pillar_indices: bool = False,
) -> Dict[str, Any]:
    """
    Robust collation for PointPillars mini-batches.

    * Adds `batch_idx` to every pillar and every GT box so sample identity
      is never lost.
    * Disambiguates `pillar_indices` by prefixing a frame-offset
      (`batch_idx * x_bins * y_bins`), or drops the field entirely.
    * Verifies all samples share the same grid geometry.
    *
    * Args
    * ----
    *   batch : list produced by ``PointPillarsLoader.__getitem__``.
    *   keep_pillar_indices : if False, the key is removed to save memory.
    *
    * Returns
    * -------
    *   Dict with keys:
    *     - features         : (∑P, N, F)  float32
    *     - pillar_coords    : (∑P, 3)     long   [i, j, batch_idx]
    *     - grid_dims        : (2,)        long
    *     - pillar_indices   : (∑P,)       long   *optional*
    *     - annotations
    *         · boxes        : (∑B, 11)    float32  [cx,cy,cz,l,w,h,qw,qx,qy,qz, batch_idx]
    *         · categories   : (∑B,)       long
    """
    # --- 1. grid geometry must be identical ---------------------------------
    grid_dims = batch[0]['grid_dims']
    if not all(torch.equal(b['grid_dims'], grid_dims) for b in batch):
        raise ValueError("All samples in a mini-batch must share the same grid_dims")

    x_bins, y_bins = grid_dims.tolist()

    # --- 2. pillars ----------------------------------------------------------
    feat_list, coord_list, idx_list = [], [], []
    offset = 0
    for b_idx, sample in enumerate(batch):
        f   = sample['features']                       # (P, N, F)
        ij  = sample['pillar_coords']                  # (P, 2)
        P   = ij.size(0)

        # add batch index
        coord_list.append(torch.cat([ij, torch.full((P, 1), b_idx, dtype=torch.long)], dim=1))
        feat_list.append(f)

        if keep_pillar_indices:
            base = b_idx * x_bins * y_bins             # frame-offset
            idx_list.append(sample['pillar_indices'] + base)

        offset += P

    features      = torch.cat(feat_list, 0)            # (∑P, N, F)
    pillar_coords = torch.cat(coord_list, 0)           # (∑P, 3)

    # --- 3. annotations ------------------------------------------------------
    box_list, cat_list = [], []
    for b_idx, sample in enumerate(batch):
        boxes = sample['annotations']['boxes']         # (B, 10)
        if boxes.numel() == 0:
            continue                                   # allow empty frames
        B = boxes.size(0)
        batch_col = torch.full((B, 1), b_idx, dtype=boxes.dtype)
        box_list.append(torch.cat([boxes, batch_col], dim=1))
        cat_list.append(sample['annotations']['categories'])

    boxes      = torch.cat(box_list, 0) if box_list else boxes.new_zeros((0, 11))
    categories = torch.cat(cat_list, 0) if cat_list else torch.zeros((0,), dtype=torch.long)

    batched: Dict[str, Any] = dict(
        features=features,
        pillar_coords=pillar_coords,
        grid_dims=grid_dims,
        annotations=dict(boxes=boxes, categories=categories),
    )
    if keep_pillar_indices:
        batched['pillar_indices'] = torch.cat(idx_list, 0)

    return batched



def get_pillar_stats(pillar_points):
    """
    Calculate statistics about points per pillar
    Returns:
        avg_points: Average number of points per pillar
        min_points: Minimum number of points in any pillar
        max_points: Maximum number of points in any pillar
        valid_points: Array of point counts per pillar
    """
    # Create mask for valid (non-padded) points
    valid_mask = np.any(pillar_points[..., :3] != 0, axis=-1)
    
    # Count valid points per pillar
    points_per_pillar = np.sum(valid_mask, axis=1)
    
    # Calculate statistics
    avg_points = np.mean(points_per_pillar)
    min_points = np.min(points_per_pillar)
    max_points = np.max(points_per_pillar)
    
    return avg_points, min_points, max_points, points_per_pillar

def visualize_pillars_2d(pillar_coords, pillar_points, grid_size, max_range, annotations_df=None):
    """
    Simplified visualization of pillars, points, and annotations in 2D
    
    Args:
        pillar_coords: (P, 2) array of pillar grid indices
        pillar_points: (P, N, 4) array of point features per pillar
        grid_size: Size of each grid cell in meters
        max_range: (x_range, y_range, z_range) in meters
        annotations_df: DataFrame containing bounding box annotations
    """
    x_min, x_max = -max_range[0], max_range[0]
    y_min, y_max = -max_range[1], max_range[1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Extract all valid points (non-padded)
    all_points = pillar_points.reshape(-1, 4)
    valid_mask = np.any(all_points != 0, axis=1)
    valid_points = all_points[valid_mask]
    
    # Plot all points in a single color
    ax.scatter(
        valid_points[:, 0], 
        valid_points[:, 1], 
        c='blue',          # Single color for all points
        s=1,               # Small dot size
        alpha=0.5,         # Semi-transparent
        edgecolors='none'  # No border
    )
    
    # Draw pillar boundaries
    for i, j in pillar_coords:
        # Calculate real-world coordinates of pillar corner
        x_corner = x_min + i * grid_size
        y_corner = y_min + j * grid_size
        
        rect = Rectangle(
            (x_corner, y_corner),  # Bottom-left corner
            grid_size, grid_size,   # Width and height
            fill=False,             # Transparent fill
            edgecolor='gray',       # Light gray border
            linewidth=1,          # Thin border
            alpha=0.8               # Slightly transparent
        )
        ax.add_patch(rect)
    
    # Add bounding box annotations if DataFrame is provided
    if annotations_df is not None:
        # Define distinct colors for each category
        categories = annotations_df['category'].unique()
        cmap = ListedColormap(plt.cm.tab10.colors[:len(categories)])
        category_colors = {cat: cmap(i) for i, cat in enumerate(categories)}
        
        # Process each bounding box
        for _, row in annotations_df.iterrows():
            # Extract box parameters
            center_x = row['tx_m']
            center_y = row['ty_m']
            length = row['length_m']
            width = row['width_m']
            qw, qx, qy, qz = row[['qw', 'qx', 'qy', 'qz']]
            category = row['category']
            
            # Calculate yaw rotation angle from quaternion
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy**2 + qz**2)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            # Calculate corner points relative to center
            half_l, half_w = length / 2, width / 2
            corners_local = np.array([
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w]
            ])
            
            # Apply rotation
            rot_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            corners_rotated = np.dot(corners_local, rot_matrix.T)
            
            # Translate to world coordinates
            corners_world = corners_rotated + np.array([center_x, center_y])
            
            # Create bounding box polygon
            bbox = Polygon(
                corners_world,
                closed=True,
                fill=False,
                edgecolor=category_colors[category],
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(bbox)
            
        
        # Create legend
        legend_patches = [
            plt.Line2D([0], [0], color=color, lw=2, label=cat)
            for cat, color in category_colors.items()
        ]
        ax.legend(handles=legend_patches, loc='upper right')
    
    # Set plot properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title('Pillar Visualization with Annotations')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()
  

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    

    # Create dataset and loader
    train_dataset = PointPillarsLoader(dataset_path, split='train')

    # Test __getitem__ for custom Dataloader
    # print(train_dataset[0])
    # print(train_dataset.__getitem__(0, debug=True))

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    sample_batch = next(iter(dataloader))
    
    for key in sample_batch:
        print(f"{key}: {sample_batch[key].shape if isinstance(sample_batch[key], torch.Tensor) else type(sample_batch[key])}")

    print(sample_batch['grid_dims']) 

    
    # python -m src.loaders.loader_Point_Pillars