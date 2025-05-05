import os
import src.loaders.loader as loader
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from tqdm import tqdm

class PointPillarsLoader(loader.ArgoDataset):
    def __init__(self, dataset_path, split='train', cameras=None, lidar=True):
        super().__init__(dataset_path, split, cameras, lidar)

    def _process_lidar(self, lidar_file, 
                       max_pillars=12000, 
                       max_points=100, 
                       voxel_size=(0.3, 0.3), 
                       x_range=(-100, 100),
                       y_range=(-100, 100),
                       z_range=(-3, 1),
                       debug=False):
        """
        Load lidar data and convert it to (pillars, coords)
        
        Args:
            lidar_file: Path to the feather file containing LiDAR data
            max_pillars: Maximum number of pillars to generate
            max_points: Maximum number of points per pillar
            voxel_size: (x, y) voxel size in meters
            x_range: (min, max) range in x direction
            y_range: (min, max) range in y direction
            z_range: (min, max) range in z direction
            debug: Whether to print debug information
            
        Returns:
            tuple: (pillars, coords) where:
                - pillars: (max_pillars, max_points, 9) array of pillar features
                - coords: (max_pillars, 3) array of pillar indices
        """
        # Load LiDAR directly as numpy array (faster than converting through DataFrame)
        lidar_data = feather.read_feather(lidar_file)
        points = lidar_data[['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
        
        if debug:
            # Only convert to DataFrame for debugging information
            points_df = pd.DataFrame(points, columns=['x', 'y', 'z', 'intensity'])
            print(points_df.describe())

        # Initialize output arrays
        pillars = np.zeros((max_pillars, max_points, 9), dtype=np.float32)
        coords = np.zeros((max_pillars, 3), dtype=np.int32)

        # 1) Filter by ROI - keep only points within the predefined 3D box
        roi_mask = ((points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
                    (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1]) &
                    (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1]))
        points = points[roi_mask]
        
        if points.shape[0] == 0:
            if debug:
                print("No points within ROI")
            return pillars, coords

        # 2) Compute grid dimensions
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        nx = int(np.floor(x_span / voxel_size[0]))
        ny = int(np.floor(y_span / voxel_size[1]))

        # 3) Compute pillar indices - vectorized operations
        ix = np.floor((points[:, 0] - x_range[0]) / voxel_size[0]).astype(np.int32)
        iy = np.floor((points[:, 1] - y_range[0]) / voxel_size[1]).astype(np.int32)
        
        # Clip stray points
        valid_mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        points = points[valid_mask]
        ix = ix[valid_mask]
        iy = iy[valid_mask]
        
        if points.shape[0] == 0:
            if debug:
                print("No valid points after clipping")
            return pillars, coords

        # 4) Flatten 2D index to 1D and select up to max_pillars
        linear_idx = ix * ny + iy
        
        # Get unique pillar IDs and their counts
        unique_ids, inverse, counts = np.unique(linear_idx, return_inverse=True, return_counts=True)
        
        if debug:
            print(f"Number of non-empty pillars before capping: {unique_ids.shape[0]}")

        # Select pillars (prioritize pillars with more points if exceeding max_pillars)
        num_pillars = min(len(unique_ids), max_pillars)
        if len(unique_ids) > max_pillars:
            # Sort pillars by point count in descending order
            sort_idx = np.argsort(-counts)
            selected = unique_ids[sort_idx[:num_pillars]]
        else:
            selected = unique_ids[:num_pillars]

        # Create a mapping from linear_idx to pillar_id (0 to num_pillars-1)
        pillar_id_map = {lid: pid for pid, lid in enumerate(selected)}
        
        # Process each pillar efficiently
        pillar_point_counts = np.zeros(num_pillars, dtype=np.int32)
        
        for i, (x, y, z, intensity) in enumerate(points):
            lin_idx = linear_idx[i]
            if lin_idx in pillar_id_map:
                p = pillar_id_map[lin_idx]
                pt_idx = pillar_point_counts[p]
                
                if pt_idx < max_points:
                    # For the first point in each pillar, store the coordinates
                    if pt_idx == 0:
                        coords[p] = [ix[i], iy[i], 0]
                    
                    # Efficiently collect points for later processing
                    pillars[p, pt_idx, 0:4] = [x, y, z, intensity]
                    pillar_point_counts[p] += 1
        
        # Compute pillar statistics and relative features
        for p in range(num_pillars):
            n_points = pillar_point_counts[p]
            if n_points == 0:
                continue
                
            # Compute pillar center
            x_idx, y_idx = coords[p, 0], coords[p, 1]
            cx = x_range[0] + (x_idx + 0.5) * voxel_size[0]
            cy = y_range[0] + (y_idx + 0.5) * voxel_size[1]
            
            # Compute mean of points in pillar for relative encoding
            points_in_pillar = pillars[p, :n_points, :3]
            mean_xyz = np.mean(points_in_pillar, axis=0)
            
            # Compute relative features for each point
            for i in range(n_points):
                dx, dy, dz = points_in_pillar[i] - mean_xyz
                fx, fy = points_in_pillar[i, 0] - cx, points_in_pillar[i, 1] - cy
                pillars[p, i, 4:] = [dx, dy, dz, fx, fy]

        return pillars, coords

    def process_all_samples(self, limit=None):
        self.processed_samples = []

        # Determine how many samples to process
        num_samples = min(len(self), limit) if limit is not None else len(self)
        
        for i in tqdm(range(num_samples), desc="Processing samples", ncols=100):
            sample = self.samples[i]
            lidar_file = sample['lidar_file']

            # processed_sample = self._process_lidar(lidar_file=lidar_file)
            # TODO: REMOVE debug=True
            processed_sample = self._process_lidar(lidar_file=lidar_file, debug=True)
            print(f"Processed sample {processed_sample[0].shape}, {processed_sample[1].shape}")

            sample["lidar_processed"] = processed_sample
            self.processed_samples.append(sample)
            
        return self.processed_samples    

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset and loader
    train_dataset = PointPillarsLoader(dataset_path, split='train')
    processed_samples = train_dataset.process_all_samples(limit=1)
    # print(train_dataset[0]["lidar_processed"])

    # Sample format:
    # {
    #     'sequence_path': seq_path,
    #     'timestamp': timestamp,
    #     'lidar_file': os.path.join(lidar_dir, f"{timestamp}.feather"),
    #     'camera_frames': camera_frames,
    #     'annotations': frame_annotations
    #     'lidar_processed': (pillars, coords)
    # }