import os
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ArgoDataset(Dataset):
    def __init__(self, root_dir, split='train', cameras=None, lidar=True, target_classes = {'PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE'}):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.lidar = lidar
        self.target_classes = target_classes

        # Default camera configuration
        if cameras is None:
            self.cameras = [
                'ring_rear_left',
                'ring_side_left',
                'ring_front_left',
                'ring_front_center', 
                'ring_front_right',
                'ring_rear_right',
                'ring_side_right'
            ]
        else:
            self.cameras = cameras
    
        # Find all sequences
        self.sequences = self._get_sequence()
        print(f"Found {len(self.sequences)} sequences in {os.path.join(self.root_dir, self.split)} split.")

        # Load all samples from sequences
        self.samples = self._load_samples_from_sequences()
        print(f"Loaded {len(self.samples)} samples from sequences.")

        self.classes = self._get_class_categories()
        print(f"Found {len(self.classes)} unique classes in the dataset.")
        

    def _get_class_categories(self):
        classes = set()

        # Loop over all samples and get unique classes
        for sample in self.samples:
            annotations = sample['annotations']

            # Extract unique classes from category column
            unique_classes = annotations['category'].unique()
            classes.update(unique_classes)
        
        return classes

    def _get_sequence(self):
        sequences = []
        split_dir = os.path.join(self.root_dir, self.split)
        
        if os.path.exists(split_dir):
            sequence_folders = sorted([
                d for d in os.listdir(split_dir) 
                if os.path.isdir(os.path.join(split_dir, d))
            ])
            sequences = [os.path.join(split_dir, d) for d in sequence_folders]
        return sequences
    
    def _load_samples_from_sequences(self):
        """Load all samples from the dataset sequences."""
        samples = []
        batch_index = -1
        for seq_path in tqdm(self.sequences, desc="Loading sequences", ncols=100):
            # Load annotations
            anno_path = os.path.join(seq_path, 'annotations.feather')
            if os.path.exists(anno_path):
                annotations = feather.read_feather(anno_path)
                
                # Get LiDAR timestamps
                lidar_dir = os.path.join(seq_path, 'sensors', 'lidar')
                if os.path.exists(lidar_dir):
                    lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.feather')]
                    
                    for lidar_file in lidar_files:
                        timestamp = int(lidar_file.split('.')[0])
                        batch_index += 1
                        
                        # Check if corresponding camera images exist
                        camera_frames = {}
                        for camera in self.cameras:
                            camera_dir = os.path.join(seq_path, 'sensors', 'cameras', camera)
                            if os.path.exists(camera_dir):
                                # Find closest camera frame to lidar timestamp
                                camera_files = [int(f.split('.')[0]) for f in os.listdir(camera_dir) if f.endswith('.jpg')]
                                if camera_files:
                                    closest_cam_ts = min(camera_files, key=lambda x: abs(x - timestamp))
                                    camera_frames[camera] = closest_cam_ts
                        
                        # Filter annotations for this timestamp
                        frame_annotations = annotations[annotations['timestamp_ns'] == timestamp]
                        
                        # Filter annotations by target classes if specified
                        if self.target_classes is not None:
                            frame_annotations = frame_annotations[frame_annotations['category'].isin(self.target_classes)]
                        
                        samples.append({
                            'sequence_path': seq_path,
                            'timestamp': timestamp,
                            'batch_index': batch_index,
                            'lidar_file': os.path.join(lidar_dir, f"{timestamp}.feather"),
                            'camera_frames': camera_frames,
                            'annotations': frame_annotations
                        })
        return samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        return sample


# Example DataLoader usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Define target classes
    target_classes = {'PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE'}


    train_dataset = ArgoDataset(
        root_dir=dataset_path, 
        split='train', 
        target_classes=target_classes
    )
    print(f"Training classes: {train_dataset.classes}")
    sample = train_dataset[0]
    print(f"Sample: {sample}")

    test_dataset = ArgoDataset(
        root_dir=dataset_path, 
        split='test', 
        target_classes=target_classes
    )
    print(f"Testing classes: {test_dataset.classes}")

# Sample format:
# {
#     'sequence_path': seq_path,
#     'timestamp': timestamp,
#     'lidar_file': os.path.join(lidar_dir, f"{timestamp}.feather"),
#     'camera_frames': camera_frames,
#     'annotations': frame_annotations
# }

# python -m src.loaders.loader





