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
        points = self._random_sample_points(points, n_points=45000)

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
            heading = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)) % (2 * np.pi)

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

def plot_scene(points, annotations):
    """
    Visualize the scene with enhanced camera controls and zoom capability.
    """
    import open3d as o3d

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray for points

    # Create coordinate frame


    # List to hold all geometries
    geometries = [pcd]

    # Define colors for each class
    class_colors = {
        0: [1, 0, 0],   # PEDESTRIAN: Red
        1: [0, 0.5, 1], # REGULAR_VEHICLE: Light blue
        2: [0, 1, 0],   # LARGE_VEHICLE: Green
        3: [1, 1, 0]    # TRUCK: Yellow
    }

    # Add bounding boxes
    if annotations['boxes'].shape[0] > 0:
        boxes = annotations['boxes'].numpy()
        labels = annotations['labels'].numpy()

        for box, label in zip(boxes, labels):
            x, y, z, l, w, h, heading = box
            center = (x, y, z)
            extent = np.array([l, w, h])

            R = np.array([
                [np.cos(heading), -np.sin(heading), 0],
                [np.sin(heading), np.cos(heading), 0],
                [0, 0, 1]
            ])

            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            lines.paint_uniform_color(class_colors.get(int(label), [1, 1, 1]))
            geometries.append(lines)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Scene - Improved Controls', width=1280, height=720)
    
    for geom in geometries:
        vis.add_geometry(geom)

    # Configure render options
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    render_opt.light_on = True
    render_opt.point_size = 3  # Increased point size

    # Get view control and set better camera parameters
    ctr = vis.get_view_control()
    
    # Set initial camera parameters (adjust these based on your data scale)
    camera_params = {
        "front": [0.4, -0.2, 0.2],  # Camera direction
        "lookat": [0, 0, 1],        # Focus point (adjust Z based on average object height)
        "up": [0, 0, 1],            # Up vector
        "zoom": 0.03                # Smaller value = closer zoom
    }
    
    ctr.set_front(camera_params["front"])
    ctr.set_lookat(camera_params["lookat"])
    ctr.set_up(camera_params["up"])
    ctr.set_zoom(camera_params["zoom"])

    # Adjust camera clipping planes (important for proper zoom)
    ctr.set_constant_z_near(0.0001)  # Minimum visible distance
    ctr.set_constant_z_far(1000)  # Maximum visible distance

    # Run visualization
    vis.run()
    vis.destroy_window()

def plot_images(images):
    """
    Visualize images from different cameras.
    
    Args:
        images (dict): Dictionary of images from different cameras. (camera_name: image([3, 388, 512]))
    """
    for camera_name, image in images.items():
        # visualize the image
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.show(title=camera_name)

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

    plot_scene(points, annotations)
    # plot_images(images)
        


# python -m src.loaders.loader_Point_Fusion

