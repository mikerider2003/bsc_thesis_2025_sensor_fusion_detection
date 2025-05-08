import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from pyarrow import feather 
from src.loaders.loader import ArgoDataset
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate



class PointFusionloader(ArgoDataset):
    def __init__(self, dataset_path, split='train', cameras=None, lidar=True, target_classes = {'PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE'}, img_size=(388, 512)):
        super().__init__(dataset_path, split, cameras, lidar, target_classes)

        self.processed_samples = []
        self._transform_image = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
            ])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load LiDAR point cloud
        lidar_data = feather.read_feather(sample['lidar_file'])
        
        # Average point 97752.80
        points = lidar_data[['x', 'y', 'z']].to_numpy().astype(np.float32)
        points = self._random_sample_points(points)

        # Load image from all cameras
        images = {}
        for camera_name, image_id in sample['camera_frames'].items():
            img_path = os.path.join(sample['sequence_path'], f"sensors/cameras/{camera_name}/{image_id}.jpg")
            image = Image.open(img_path).convert('RGB')

            images[camera_name] = self._transform_image(image)
        
        # Process annotations
        annotations = self._process_annotations(sample['annotations'])

        return {
            'points': points,
            'images': images,
            # TODO: add annotations back
            'annotations': annotations,
            'timestamp': sample['timestamp'],
            'sequence_path': sample['sequence_path']
        }
    
    def _random_sample_points(self, points, n_points=65536):
        """Sample a fixed number of points from a point cloud."""
        if len(points) == 0:
            return np.zeros((n_points, 3), dtype=np.float32)
        
        if len(points) >= n_points:
            # Random sampling without replacement if we have enough points
            idx = np.random.choice(len(points), n_points, replace=False)
        else:
            # Random sampling with replacement if we don't have enough points
            idx = np.random.choice(len(points), n_points, replace=True)
        
        return points[idx]
    
    def _process_annotations(self, annotations_df):
        """Process annotations to tensor format."""
        if len(annotations_df) == 0:
            return {
                'boxes': torch.zeros((0, 7), dtype=torch.float32),  # x,y,z,l,w,h,heading
                'labels': torch.zeros(0, dtype=torch.int64),
                'num_boxes': 0
            }
        
        # Define class mappings
        class_to_idx = {
            'PEDESTRIAN': 0,
            'REGULAR_VEHICLE': 1,
            'LARGE_VEHICLE': 2,
            'TRUCK': 3
        }

        boxes = []
        labels = []
        
        for _, obj in annotations_df.iterrows():
            # Get center coordinates from translation
            x, y, z = obj['tx_m'], obj['ty_m'], obj['tz_m']
            
            # Get dimensions
            l, w, h = obj['length_m'], obj['width_m'], obj['height_m']

            # Calculate heading from quaternion
            qw, qx, qy, qz = obj['qw'], obj['qx'], obj['qy'], obj['qz']
            heading = (np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)) - np.pi/2) % (2 * np.pi)

            boxes.append([x, y, z, l, w, h, heading])
            labels.append(class_to_idx[obj['category']])
        
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 7), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.int64)
        
        return {
            'boxes': boxes_tensor,  # [N, 7] - x,y,z,l,w,h,heading
            'labels': labels_tensor  # [N]
        }

    def process_samples(self, limit=None):
        """
        Process samples to load images and annotations.
        
        Args:
            limit (int): Limit the number of samples to process.
        """

        max_samples = min(len(self.samples), limit) if limit else len(self.samples)

        for i in tqdm(range(max_samples), desc="Processing samples", ncols=100):
            self.processed_samples.append(self.__getitem__(i))

def custom_collate(batch):
    collated = {}
    for key in batch[0].keys():
        if key == 'annotations':
            # Keep annotations as a list of dictionaries
            collated[key] = [item[key] for item in batch]
        else:
            # Use default collate for other entries
            collated[key] = default_collate([item[key] for item in batch])
    return collated

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    train_set = PointFusionloader(dataset_path, split='train')

    # Inspect the first sample
    sample = train_set[0]

    points = sample['points']
    images = sample['images']
    annotations = sample['annotations']

    print(f"Points shape: {points.shape}")
    for camera_name, image in images.items():
        print(f"Image {camera_name} shape: {image.shape}")
    
    print(f"Annotations: \n\tBoxes shape: {annotations['boxes'].shape},\n\tLabels shape: {annotations['labels'].shape}")

    for camera_name, image in images.items():
        # visualize the image
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.show()

    # avergae number of points per sample
    # train_set.process_samples()
    # avg_points = np.mean([len(sample['points']) for sample in train_set.processed_samples])
    # print(f"Average number of points per sample: {avg_points:.2f}")
    
        


# python -m src.loaders.loader_Point_Fusion

