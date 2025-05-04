import os
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ArgoDataset(Dataset):
    def __init__(self, root_dir, split='train', cameras=None, lidar=True):
        super().__init__()

        self.root_dir = root_dir
        self.split = split
        self.lidar = lidar

        # Default camera configuration
        if cameras is None:
            self.cameras = [
                'ring_front_center', 
                'ring_front_left',
                'ring_front_right',
                'ring_rear_left',
                'ring_rear_right',
                'ring_side_left',
                'ring_side_right'
            ]
        else:
            self.cameras = cameras
    
        # Find all sequences
        self.sequences = self._get_sequence()
        print(f"Found {len(self.sequences)} sequences in {self.split} split.")

        # Load all samples from sequences
        self.samples = self._load_samples_from_sequences()

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
                        
                        samples.append({
                            'sequence_path': seq_path,
                            'timestamp': timestamp,
                            'lidar_file': os.path.join(lidar_dir, f"{timestamp}.feather"),
                            'camera_frames': camera_frames,
                            'annotations': frame_annotations
                        })
            #             # TODO: REMOVE THIS    
            #             break
            # break
        return samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = {}

        # Save the sequence path and timestamp
        data['sequence_path'] = sample['sequence_path']
        data['timestamp'] = sample['timestamp']

        # Load the lidar data
        data['lidar'] = self.process_lidar(lidar_file=sample['lidar_file'])

        # Load the corresponding camera images
        data['cameras'] = self.process_cameras(camera_frames=sample['camera_frames'], sequence_path=sample['sequence_path'])

        # Load the annotations
        data["annotations"] = sample['annotations']

        return data

    def process_lidar(self, lidar_file):
        """Load LiDAR data from a feather file."""
        # Read the LiDAR data file
        lidar_data = feather.read_feather(lidar_file)

        # Convert to numpy array
        points_np = lidar_data[['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)

        # Convert to torch tensor
        points_torch = torch.from_numpy(points_np)
        return points_torch
    
    def process_cameras(self, camera_frames, sequence_path):
        """Load camera images."""
        camera_images = {}
        for camera, frame in camera_frames.items():
            camera_dir = os.path.join(sequence_path, 'sensors', 'cameras', camera)
            image_path = os.path.join(camera_dir, f"{frame}.jpg")
        
            if os.path.exists(image_path):
                # Load the image
                image = Image.open(image_path).convert('RGB')
                # Convert to numpy array
                image_np = np.array(image)
                # Convert to torch tensor
                image_torch = torch.from_numpy(image_np).permute(2, 0, 1)
            
                camera_images[camera] = image_torch
                # print(f"Camera {camera} image shape: {image_torch.shape}")
    
            else:
                raise FileNotFoundError(f"Image file {image_path} not found.")

        return camera_images
    
    def process_all_samples(self):
        """Process all samples in the dataset."""
        self.processed_samples = []

        for sample in tqdm(range(len(self)), desc="Processing all samples", ncols=100):
            data = self.__getitem__(sample)
            self.processed_samples.append(data)
        
        return self.processed_samples

        


# Example DataLoader usage
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='/src/data')

    
    # Create dataset and loader
    train_dataset = ArgoDataset(dataset_path, split='train')
    processed_samples = train_dataset.process_all_samples()

    print(f"Processed {len(processed_samples)} samples.")

    # for i in train_dataset.samples[0]:
    #     if i == "camera_frames":
    #         for cam in train_dataset.samples[0][i]:
    #             print(f"Camera {cam}: {train_dataset.samples[0][i][cam]}")
    #         continue
    #     print(f"{i}: {train_dataset.samples[0][i]}")


            
    
