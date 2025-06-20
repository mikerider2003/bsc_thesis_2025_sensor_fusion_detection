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
    def __init__(self, dataset_path, split='train', img_size=(388, 512)):
        super().__init__(dataset_path, split)
        self.img_size = img_size  # Store for calibration adjustment

        self.processed_samples = []
        self._transform_image = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
            ])

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load intrinsics
        intrinsics_path = os.path.join(sample['sequence_path'], 'calibration/intrinsics.feather')
        intrinsics_df = feather.read_feather(intrinsics_path)

        # Load extrinsics
        extrinsics_path = os.path.join(sample['sequence_path'], 'calibration/egovehicle_SE3_sensor.feather')
        extrinsics_df = feather.read_feather(extrinsics_path)
        
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
        
        # Process camera calibrations
        camera_calibrations = self._process_camera_calibrations(intrinsics_df, extrinsics_df)
        
        # Process annotations
        annotations = self._process_annotations(sample['annotations'])

        return {
            'points': points,
            'images': images,
            'annotations': annotations,
            'camera_calibrations': camera_calibrations
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
    
    def _process_camera_calibrations(self, intrinsics_df, extrinsics_df):
        """
        Process camera calibration data and adjust for image resizing.
        """
        calibrations = {}
        
        target_cameras = {
            'ring_rear_left', 'ring_side_left', 'ring_front_left',
            'ring_front_center', 'ring_front_right', 'ring_rear_right', 'ring_side_right'
        }
        
        for _, intrinsic_row in intrinsics_df.iterrows():
            camera_name = intrinsic_row['sensor_name']
            
            if camera_name not in target_cameras:
                continue
                
            # Original image dimensions
            original_height = int(intrinsic_row['height_px'])
            original_width = int(intrinsic_row['width_px'])
            
            # Your resized dimensions (from img_size parameter)
            resized_height, resized_width = self.img_size  # (388, 512)
            
            # Calculate scaling factors
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            
            # Original intrinsic parameters
            fx_original = intrinsic_row['fx_px']
            fy_original = intrinsic_row['fy_px'] 
            cx_original = intrinsic_row['cx_px']
            cy_original = intrinsic_row['cy_px']
            
            # Scale intrinsic parameters for resized image
            fx_scaled = fx_original * scale_x
            fy_scaled = fy_original * scale_y
            cx_scaled = cx_original * scale_x
            cy_scaled = cy_original * scale_y
            
            # Create adjusted intrinsic matrix
            intrinsic_matrix = torch.tensor([
                [fx_scaled,  0, cx_scaled],
                [ 0, fy_scaled, cy_scaled],
                [ 0,  0,  1]
            ], dtype=torch.float32)
            
            # Extrinsic matrix stays the same (world coordinates don't change)
            extrinsic_row = extrinsics_df[extrinsics_df['sensor_name'] == camera_name]
            
            if len(extrinsic_row) > 0:
                extrinsic_data = extrinsic_row.iloc[0]
                
                # Extract quaternion and translation
                qw = extrinsic_data['qw']
                qx = extrinsic_data['qx']
                qy = extrinsic_data['qy']
                qz = extrinsic_data['qz']
                
                tx = extrinsic_data['tx_m']
                ty = extrinsic_data['ty_m']
                tz = extrinsic_data['tz_m']
                
                # Convert quaternion to rotation matrix
                rotation_matrix = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
                
                # Create 4x4 extrinsic matrix (world to camera transformation)
                extrinsic_matrix = torch.eye(4, dtype=torch.float32)
                extrinsic_matrix[:3, :3] = rotation_matrix
                extrinsic_matrix[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
                
            else:
                extrinsic_matrix = torch.eye(4, dtype=torch.float32)
            
            calibrations[camera_name] = {
                'intrinsic': intrinsic_matrix,
                'extrinsic': extrinsic_matrix,
                'distortion': {
                    'k1': intrinsic_row.get('k1', 0.0),
                    'k2': intrinsic_row.get('k2', 0.0), 
                    'k3': intrinsic_row.get('k3', 0.0)
                },
                'image_size': {
                    'height': resized_height,
                    'width': resized_width
                },
                'original_size': {
                    'height': original_height,
                    'width': original_width
                },
                'scale_factors': {
                    'scale_x': scale_x,
                    'scale_y': scale_y
                }
            }
            
        return calibrations

    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            qw, qx, qy, qz: Quaternion components
            
        Returns:
            torch.Tensor: 3x3 rotation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Convert to rotation matrix
        R = torch.tensor([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ], dtype=torch.float32)
        
        return R

# Update custom_collate to handle camera_calibrations
def custom_collate(batch):
    collated = {}
    for key in batch[0].keys():
        if key in ['annotations', 'camera_calibrations']:
            # Keep these as lists since they can vary per sample
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
    test_set = PointFusionloader(dataset_path, split='test')

    # Inspect the first sample
    sample = train_set[0]

    points = sample['points']
    images = sample['images']
    annotations = sample['annotations']
    calibrations = sample['camera_calibrations']

    print(f"Points shape: {points.shape}")
    
    print(f"\nCamera Calibrations:")
    for camera_name, calib in calibrations.items():
        print(f"\n{camera_name}:")
        print(f"  Intrinsic matrix:\n{calib['intrinsic']}")
        print(f"  Extrinsic matrix:\n{calib['extrinsic']}")
        print(f"  Image size: {calib['image_size']}")
        print(f"  Distortion: {calib['distortion']}")
    
    print(f"\nImages:")
    for camera_name, image in images.items():
        print(f"  {camera_name}: {image.shape}")
    
    print(f"\nAnnotations: \n\tBoxes shape: {annotations['boxes'].shape},\n\tLabels shape: {annotations['labels'].shape}")

    # Optional: Remove the visualization calls for now to focus on calibration
    # plot_scene(points, annotations)
    # plot_images(images)
        


# python -m src.loaders.loader_Point_Fusion

