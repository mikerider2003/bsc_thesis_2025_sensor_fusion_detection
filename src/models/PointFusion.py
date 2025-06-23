import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import convert_image_dtype
import torchvision.transforms.functional as TF
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d
import random
import pickle
from datetime import datetime

from src.loaders.loader_Point_Fusion import PointFusionloader

from dotenv import load_dotenv


class Preprocessor():
    def __init__(self, load_path=None):
        """
        Initialize the Preprocessor.
        
        Args:
            load_path (str, optional): Path to a saved preprocessed data file to load.
                                     If None, initializes empty lists.
        """
        if load_path and os.path.exists(load_path):
            self.load_processed_data(load_path)
            print(f"Loaded preprocessed data from: {load_path}")
            print(f"  Rolls: {len(self.rolls)}")
            print(f"  Point clouds: {len(self.point_clouds)}")
            print(f"  Labels: {len(self.labels)}")
            print(f"  Annotations: {len(self.annotations)}")
        else:
            self.rolls = []
            self.point_clouds = []
            self.labels = []
            self.annotations = []
            if load_path:
                print(f"Warning: Could not find file at {load_path}. Initializing empty lists.")
    
    def assign_gt(self):
        """
        Assign ground truth labels to the cloud of points. Compute center of the cloud of points and match it to the closest center of the ground truth boxes. The update self.annotations to be just {boxes: [x, y, z, l, w, h, heading], labels: [label]}
        """
        # Label mapping from string to integer
        label_to_int = {
            'PEDESTRIAN': 0,
            'REGULAR_VEHICLE': 1,
            'LARGE_VEHICLE': 2,
            'TRUCK': 3
        }
        
        new_annotations = []
        
        for i in tqdm.tqdm(range(len(self.point_clouds)), desc="Assigning ground truth"):
            point_cloud = self.point_clouds[i]
            annotation = self.annotations[i]
            current_label = self.labels[i]
            
            # Convert point cloud to numpy if it's a tensor
            if isinstance(point_cloud, torch.Tensor):
                points_np = point_cloud.cpu().numpy()
            else:
                points_np = point_cloud
            
            # Skip empty point clouds
            if len(points_np) == 0 or np.all(points_np == 0):
                # Create empty annotation
                new_annotations.append({
                    'boxes': torch.empty(0, 7),
                    'labels': torch.empty(0, dtype=torch.long)
                })
                continue
            
            # Remove zero-padded points to compute center
            valid_mask = np.any(points_np != 0, axis=1)
            valid_points = points_np[valid_mask]
            
            if len(valid_points) == 0:
                # Create empty annotation
                new_annotations.append({
                    'boxes': torch.empty(0, 7),
                    'labels': torch.empty(0, dtype=torch.long)
                })
                continue
            
            # Compute center of the point cloud
            point_cloud_center = np.mean(valid_points[:, :3], axis=0)  # [x, y, z]
            
            # Extract ground truth boxes from annotation
            gt_boxes = annotation['boxes']
            gt_labels = annotation['labels']
            
            # Convert to numpy if needed
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes_np = gt_boxes.cpu().numpy()
            else:
                gt_boxes_np = gt_boxes
            
            if isinstance(gt_labels, torch.Tensor):
                gt_labels_np = gt_labels.cpu().numpy()
            else:
                gt_labels_np = gt_labels
            
            if len(gt_boxes_np) == 0:
                # No ground truth boxes
                new_annotations.append({
                    'boxes': torch.empty(0, 7),
                    'labels': torch.empty(0, dtype=torch.long)
                })
                continue
            
            # Compute centers of ground truth boxes
            # Assuming gt_boxes format is [x, y, z, l, w, h, heading]
            gt_centers = gt_boxes_np[:, :3]  # Extract x, y, z coordinates
            
            # Find the closest ground truth box center
            distances = np.linalg.norm(gt_centers - point_cloud_center, axis=1)
            closest_idx = np.argmin(distances)
            
            # Get the closest box and its label
            closest_box = gt_boxes_np[closest_idx]
            closest_label = gt_labels_np[closest_idx]
            
            # Convert string label to integer if needed
            if isinstance(closest_label, str):
                closest_label_int = label_to_int.get(closest_label, 0)  # Default to 0 if unknown
            else:
                closest_label_int = closest_label
            
            # Create new annotation with single matched box and label
            new_annotation = {
                'boxes': torch.tensor([closest_box], dtype=torch.float32),
                'labels': torch.tensor([closest_label_int], dtype=torch.long)
            }
            
            new_annotations.append(new_annotation)
        
        # Update self.annotations
        self.annotations = new_annotations
        
        print(f"Assigned ground truth to {len(self.annotations)} samples")

    def save_processed_data(self, save_path=''):
        """
        Save the preprocessed data (rolls, point_clouds, labels, annotations) to a pickle file.
        
        Args:
            save_path (str): Path where to save the data. If empty, creates a timestamped filename.
        """
        if not save_path:
            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"preprocessed_data_{timestamp}.pkl"
        
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Prepare data dictionary
        data_dict = {
            'rolls': self.rolls,
            'point_clouds': self.point_clouds,
            'labels': self.labels,
            'annotations': self.annotations,
            'metadata': {
                'num_samples': len(self.rolls),
                'save_timestamp': datetime.now().isoformat(),
                'data_types': {
                    'rolls': type(self.rolls[0]).__name__ if self.rolls else 'None',
                    'point_clouds': type(self.point_clouds[0]).__name__ if self.point_clouds else 'None',
                    'labels': type(self.labels[0]).__name__ if self.labels else 'None',
                    'annotations': type(self.annotations[0]).__name__ if self.annotations else 'None'
                }
            }
        }
        
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Successfully saved preprocessed data to: {save_path}")
            print(f"  Rolls: {len(self.rolls)} samples")
            print(f"  Point clouds: {len(self.point_clouds)} samples")
            print(f"  Labels: {len(self.labels)} samples")
            print(f"  Annotations: {len(self.annotations)} samples")
            print(f"  File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
            
            return save_path
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return None

    def load_processed_data(self, load_path):
        """
        Load preprocessed data from a pickle file.
        
        Args:
            load_path (str): Path to the saved preprocessed data file.
        
        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            with open(load_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Load the data
            self.rolls = data_dict['rolls']
            self.point_clouds = data_dict['point_clouds']
            self.labels = data_dict['labels']
            self.annotations = data_dict['annotations']
            
            # Print metadata if available
            if 'metadata' in data_dict:
                metadata = data_dict['metadata']
                print(f"Loaded data saved on: {metadata.get('save_timestamp', 'Unknown')}")
                print(f"Number of samples: {metadata.get('num_samples', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def visualize(self):
        """
        Select a random idx. Then visualize the corresponding roll + point_cloud in o3d + annotations boxes in same o3d
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import open3d as o3d
        import numpy as np
        import random
        
        if len(self.rolls) == 0:
            print("No data available for visualization. Make sure to process samples first.")
            return
        
        # Select a random index
        random_idx = random.randint(0, len(self.rolls) - 1)
        
        # Get the corresponding data
        roll = self.rolls[random_idx]  # Cropped image tensor [3, 224, 224]
        point_cloud = self.point_clouds[random_idx]  # Point cloud tensor [N, 3]
        label = self.labels[random_idx]  # String label
        annotations = self.annotations[random_idx]  # Annotations dict
        
        print(f"Visualizing sample {random_idx}:")
        print(f"  Label: {label}")
        print(f"  Point cloud shape: {point_cloud.shape}")
        print(f"  Roll shape: {roll.shape}")
        
        # 1. Visualize the cropped image (roll)
        self._visualize_roll(roll, label)
        
        # 2. Visualize point cloud and annotations in 3D
        self._visualize_3d_scene(point_cloud, annotations)
    
    def _visualize_roll(self, roll, label):
        """
        Visualize the cropped image (roll) with its label.
        
        Args:
            roll (torch.Tensor): The cropped image tensor [3, 224, 224]
            label (str): The object label
        """
        # Convert tensor to numpy and transpose to HWC format
        if isinstance(roll, torch.Tensor):
            image_np = roll.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = roll.transpose(1, 2, 0)
        
        # Ensure values are in [0, 1] range for display
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image_np)
        ax.set_title(f'Cropped Object Image (Roll)\nLabel: {label}', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_3d_scene(self, point_cloud, annotations):
        """
        Visualize the 3D point cloud with annotation bounding boxes.
        
        Args:
            point_cloud (torch.Tensor): Point cloud tensor [N, 3]
            annotations (dict): Annotations containing boxes and labels
        """
        # Convert point cloud to numpy
        if isinstance(point_cloud, torch.Tensor):
            points_np = point_cloud.cpu().numpy()
        else:
            points_np = point_cloud
        
        # Skip if point cloud is empty or padded with zeros
        if len(points_np) == 0 or np.all(points_np == 0):
            print("Point cloud is empty or all zeros. Skipping 3D visualization.")
            return
        
        # Remove zero-padded points
        non_zero_mask = np.any(points_np != 0, axis=1)
        points_np = points_np[non_zero_mask]
        
        if len(points_np) == 0:
            print("No valid points found after removing zeros. Skipping 3D visualization.")
            return
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])
        pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray for points
        
        # List to hold all geometries
        geometries = [pcd]
        
        # Define colors for each class
        class_colors = {
            0: [1, 0, 0],   # PEDESTRIAN: Red
            1: [0, 0.5, 1], # REGULAR_VEHICLE: Light blue
            2: [0, 1, 0],   # LARGE_VEHICLE: Green
            3: [1, 1, 0]    # TRUCK: Yellow
        }
        
        # Add annotation bounding boxes
        if annotations['boxes'].shape[0] > 0:
            boxes = annotations['boxes']
            labels = annotations['labels']
            
            # Convert to numpy if needed
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            for box, label in zip(boxes, labels):
                x, y, z, l, w, h, heading = box
                center = (x, y, z)
                extent = np.array([l, w, h])
                
                # Create rotation matrix from heading
                R = np.array([
                    [np.cos(heading), -np.sin(heading), 0],
                    [np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]
                ])
                
                # Create oriented bounding box
                obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
                lines.paint_uniform_color(class_colors.get(int(label), [1, 1, 1]))
                geometries.append(lines)
        
        # Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        geometries.append(coord_frame)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='3D Point Cloud with Annotations', width=1280, height=720)
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Configure render options
        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        render_opt.light_on = True
        render_opt.point_size = 3
        
        # Set camera view
        ctr = vis.get_view_control()
        ctr.set_front([0.4, -0.2, 0.2])
        ctr.set_lookat([0, 0, 1])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.03)
        
        # Run visualization
        vis.run()
        vis.destroy_window()

    def remove_empty_point_clouds(self):
        """
        Remove empty point clouds from the lists of point_clouds. Remove the corresponding rolls and labels as well.
        """
        non_empty_indices = [i for i, pc in enumerate(self.point_clouds) if len(pc) > 0]
        
        self.rolls = [self.rolls[i] for i in non_empty_indices]
        self.point_clouds = [self.point_clouds[i] for i in non_empty_indices]
        self.labels = [self.labels[i] for i in non_empty_indices]
        self.annotations = [self.annotations[i] for i in non_empty_indices]    

    def process(self, sample, debug=False):
        self.sample = sample
        points = sample['points']
        annotations = sample['annotations']
        
        counter = 0
        # Process each camera image separately
        for camera_name, camera_data in sample['images'].items():
            bboxes = self._detect_2d_boxes(camera_data)

            bboxes_boxes = bboxes['boxes']
            bboxes_labels = bboxes['labels']
            calib_intrinsic = sample['camera_calibrations'][camera_name]['intrinsic']
            calib_extrinsic = sample['camera_calibrations'][camera_name]['extrinsic']

            roll_set = self._bbox_to_rolls(camera_data, bboxes_boxes)

            # Allocate point clouds to the detected boxes
            allocated_point_clouds = self._allocate_point_clouds(bboxes_boxes, points, calib_intrinsic, calib_extrinsic)
            filtered_point_clouds = self._filter_point_cloud(allocated_point_clouds)

            if len(roll_set) != len(filtered_point_clouds) and len(roll_set) != len(bboxes_labels):
                print(f"Length mismatch for camera {camera_name}:")
                print(f"  Rolls: {len(roll_set)}")
                print(f"  Point Clouds: {len(filtered_point_clouds)}")
                print(f"  Labels: {len(bboxes_labels)}")

                raise ValueError("Mismatch in number of rolls, point clouds, and labels.")

            # Unpack and store all objects from all cameras in flat lists
            self.rolls.extend(roll_set)
            self.labels.extend(bboxes_labels)
            self.point_clouds.extend(filtered_point_clouds)

            if debug:
                self._visualize_point_assignment(camera_data, bboxes_boxes, allocated_point_clouds, bboxes_labels, points)
                break

            counter += len(roll_set)

        for _ in range(counter):
            self.annotations.append(annotations)


    def _visualize_point_assignment(self, image_tensor, boxes, point_clouds, labels, all_points):
        """
        Show the image with bounding boxes and legend, and visualize the point clouds using o3d make color of points 
        
        Args:
            image_tensor (torch.Tensor): The input RGB image tensor. [3, 388, 512]
            boxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
            point_clouds (list): List of point clouds corresponding to each bounding box.
            labels (list): List of labels for each bounding box.
            all_points (torch.Tensor): The complete point cloud tensor. [N, 3]
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import open3d as o3d
        import numpy as np
        
        # Define colors for different object classes (RGB format for Open3D)
        color_map = {
            'PEDESTRIAN': [1.0, 0.0, 0.0],    # Red
            'REGULAR_VEHICLE': [0.0, 0.0, 1.0],  # Blue
            'LARGE_VEHICLE': [0.0, 1.0, 0.0],    # Green
            'TRUCK': [1.0, 0.5, 0.0]             # Orange
        }
        
        # 1. Show image with bounding boxes
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image_tensor.permute(1, 2, 0).numpy())
        
        # Keep track of unique labels for legend
        unique_labels = set()
        
        for box, label in zip(boxes, labels):
            if isinstance(box, torch.Tensor):
                x1, y1, x2, y2 = box.cpu().numpy()
            else:
                x1, y1, x2, y2 = box
            
            width = x2 - x1
            height = y2 - y1
            color = color_map.get(label, [1.0, 0.0, 0.0])  # Default to red
            
            # Convert RGB to matplotlib color format
            matplotlib_color = (color[0], color[1], color[2])
            
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, 
                                   edgecolor=matplotlib_color, facecolor='none')
            ax.add_patch(rect)
            unique_labels.add(label)
        
        # Create legend
        legend_elements = [patches.Patch(facecolor=color_map.get(label, [1.0, 0.0, 0.0]), 
                                       edgecolor=color_map.get(label, [1.0, 0.0, 0.0]), 
                                       label=label) 
                         for label in unique_labels]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        ax.set_title('2D Bounding Boxes with Allocated Points')
        plt.tight_layout()
        plt.show()
        
        # 2. Visualize point clouds in 3D with Open3D
        if len(point_clouds) > 0 and any(len(pc) > 0 for pc in point_clouds):
            # Create Open3D geometries
            geometries = []
            
            # Add all points in grey as background
            if isinstance(all_points, torch.Tensor):
                all_points_np = all_points.cpu().numpy()
            else:
                all_points_np = all_points
            
            # Create point cloud for all points
            all_pcd = o3d.geometry.PointCloud()
            all_pcd.points = o3d.utility.Vector3dVector(all_points_np[:, :3])
            # Color all points grey
            grey_color = [0.7, 0.7, 0.7]  # Light grey
            all_colors = np.tile(grey_color, (len(all_points_np), 1))
            all_pcd.colors = o3d.utility.Vector3dVector(all_colors)
            geometries.append(all_pcd)
            
            # Add allocated points with specific colors (these will overlay the grey points)
            for i, (point_cloud, label) in enumerate(zip(point_clouds, labels)):
                if len(point_cloud) > 0:
                    # Convert tensor to numpy if needed
                    if isinstance(point_cloud, torch.Tensor):
                        points_np = point_cloud.cpu().numpy()
                    else:
                        points_np = point_cloud
                    
                    # Create Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])  # x, y, z coordinates
                    
                    # Color points based on label
                    color = color_map.get(label, [1.0, 0.0, 0.0])
                    colors = np.tile(color, (len(points_np), 1))
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    geometries.append(pcd)
            
            if geometries:
                # Create coordinate frame for reference
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
                geometries.append(coord_frame)
                
                o3d.visualization.draw_geometries(
                    geometries,
                    window_name="Point Clouds Allocated to Bounding Boxes",
                    width=1200,
                    height=800,
                    left=50,
                    top=50
                )
            else:
                print("No valid point clouds to visualize.")
        else:
            print("No point clouds allocated to any bounding boxes.")

    def _filter_point_cloud(self, allocated_point_clouds, min_points=100, max_points=400):
        """
        Loop through the allocated point clouds and filter each of them based on the number of points. If less than min_points replace with zero tensor, if more than max_points randomly sample points, if less than max_points but more than min_points, pad with zeros.
        
        Args:
            allocated_point_clouds (list): List of point clouds corresponding to each bounding box.
            min_points (int): Minimum number of points required in a point cloud.
            max_points (int): Maximum number of points allowed in a point cloud.
        
        Returns:
            list: Filtered list of point clouds that meet the criteria.
        """
        filtered_point_clouds = []

        for pc in allocated_point_clouds:
            if len(pc) < min_points:
                # Replace with zero tensor
                filtered_point_clouds.append(torch.zeros((0, pc.shape[1]), dtype=pc.dtype))
            elif len(pc) > max_points:
                # Randomly sample points
                indices = torch.randperm(len(pc))[:max_points]
                filtered_point_clouds.append(pc[indices])
            else:
                # Pad with zeros
                padding = torch.zeros((max_points - len(pc), pc.shape[1]), dtype=pc.dtype)
                filtered_point_clouds.append(torch.cat([pc, padding], dim=0))

        return filtered_point_clouds

    def _bbox_to_rolls(self, image_tensor, bboxes):
        """
        Convert bounding boxes to rolls format. Cropping the image and resizing it to 224x224 pixels.

        Args:
            image_tensor (torch.Tensor): The input RGB image tensor. [3, 388, 512]
            bboxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        Returns:
            list: A list of cropped and resized image tensors, each of shape [3, 224, 224].
        """        
        rolls = []
        
        if len(bboxes) == 0:
            return rolls
        
        # Get image dimensions
        _, height, width = image_tensor.shape  # [3, 388, 512]
        
        for bbox in bboxes:
            # Convert bbox to numpy if it's a tensor
            if isinstance(bbox, torch.Tensor):
                x1, y1, x2, y2 = bbox.cpu().numpy()
            else:
                x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))
            
            # Skip invalid boxes (too small or inverted)
            if x2 <= x1 or y2 <= y1:
                # Add a placeholder tensor for invalid boxes
                placeholder = torch.zeros(3, 224, 224, dtype=image_tensor.dtype)
                rolls.append(placeholder)
                continue
            
            # Crop the image region
            cropped = image_tensor[:, y1:y2, x1:x2]  # [3, crop_h, crop_w]
            
            # Resize to 224x224 using bilinear interpolation
            # TF.resize expects [C, H, W] format which we already have
            resized = TF.resize(cropped, size=(224, 224), interpolation=TF.InterpolationMode.BILINEAR)
            
            rolls.append(resized)
        
        return rolls

    def _detect_2d_boxes(self, image_tensor, threshold=0.9, debug=False):
        """
        Detect 2D bounding boxes in the RGB image using a pre-trained Faster R-CNN model.

        Args:
            torch.Tensor: The input RGB image tensor. [3, 388, 512]

        Returns:
            list: A list of detected bounding boxes, each represented as a dictionary with keys 'boxes', 'labels', and 'scores'.
        """
        # Ensure we are locating equivalent to Argoverse dataset labels in COCO format
        COCO_LABEL_MAP = {
        1: 'PEDESTRIAN',
        3: 'REGULAR_VEHICLE',
        6: 'LARGE_VEHICLE',
        8: 'TRUCK'
        }
        TARGET_LABELS = set(COCO_LABEL_MAP.keys())

        # Load model
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.eval()

        # Convert image to float
        image_tensor = convert_image_dtype(image_tensor, dtype=torch.float)

        # Inference
        with torch.no_grad():
            predictions = model([image_tensor])[0]

        # Filter by confidence threshold and label
        target_labels_tensor = torch.tensor(list(TARGET_LABELS))
        label_mask = torch.isin(predictions['labels'], target_labels_tensor)
        valid_indices = (predictions['scores'] > threshold) & label_mask

        # Apply filter
        boxes = predictions['boxes'][valid_indices]
        labels = predictions['labels'][valid_indices]

        # Map numeric labels to custom class names (optional)
        label_names = [COCO_LABEL_MAP[label.item()] for label in labels]

        detected_boxes = {
            'boxes': boxes,
            'labels': label_names
        }

        # Optionally, show image with bounding boxes
        def show_boxes(image, boxes, labels):
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            # Define colors for different object classes
            color_map = {
                'PEDESTRIAN': 'red',
                'REGULAR_VEHICLE': 'blue',
                'LARGE_VEHICLE': 'green',
                'TRUCK': 'orange'
            }

            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image.permute(1, 2, 0).numpy())

            # Keep track of unique labels for legend
            unique_labels = set()

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                color = color_map.get(label, 'red')  # Default to red if label not found
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                unique_labels.add(label)

            # Create legend
            legend_elements = [patches.Patch(facecolor=color_map.get(label, 'red'), 
                                           edgecolor=color_map.get(label, 'red'), 
                                           label=label) 
                             for label in unique_labels]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

            plt.tight_layout()
            plt.show()

        if debug:
            show_boxes(image_tensor, boxes, label_names)

        return detected_boxes
    
    def _allocate_point_clouds(self, boxes, points, calib_intrinsic, calib_extrinsic):
        """
        Extract 3D points from the LiDAR/point cloud that project into the detected 2D boxes using camera calibration.
        
        Args:
            boxes (torch.Tensor): Bounding boxes in the format [x1, y1, x2, y2].
            points (torch.Tensor): 3D points in the format [N, 3] (x, y, z).
            calib_intrinsic (torch.Tensor): Camera intrinsic matrix [3, 3].
            calib_extrinsic (torch.Tensor): Camera extrinsic matrix [4, 4].
        Returns:
            list: A list of point clouds corresponding to each bounding box.
            """
        allocated_point_clouds = []
        
        if len(boxes) == 0:
            return allocated_point_clouds
        
        # Convert to numpy for easier manipulation
        if isinstance(points, torch.Tensor):
            points_np = points.cpu().numpy()
        else:
            points_np = points
            
        if isinstance(calib_intrinsic, torch.Tensor):
            intrinsic = calib_intrinsic.cpu().numpy()
        else:
            intrinsic = calib_intrinsic
            
        if isinstance(calib_extrinsic, torch.Tensor):
            extrinsic = calib_extrinsic.cpu().numpy()
        else:
            extrinsic = calib_extrinsic
        
        # Step 2: Extract 3D coordinates from point cloud
        points_3d = points_np[:, :3]  # Extract x, y, z coordinates [N, 3]
        num_points = points_3d.shape[0]
        
        # Step 3: Convert to homogeneous coordinates and transform to camera coordinate system
        # Add ones column for homogeneous coordinates [N, 4]
        points_3d_homo = np.hstack([points_3d, np.ones((num_points, 1))])
        
        # Transform from LiDAR to camera coordinate system
        # points_camera = extrinsic @ points_lidar_homo.T -> [4, N] then transpose to [N, 4]
        points_camera = (extrinsic @ points_3d_homo.T).T

        # Step 4: Filter points by depth (remove points behind camera and beyond 40m)
        valid_depth_mask = (points_camera[:, 2] > 0) & (points_camera[:, 2] < 40.0)  # 0 < Z < 40m
        points_camera_valid = points_camera[valid_depth_mask]
        valid_indices = np.where(valid_depth_mask)[0]
        
        if len(points_camera_valid) == 0:
            # No valid points, return empty tensors for each box
            return [torch.empty(0, points_np.shape[1], dtype=torch.float32) for _ in range(len(boxes))]
        
        # Step 5: Project 3D camera coordinates to 2D image plane
        points_camera_3d = points_camera_valid[:, :3]  # [N_valid, 3] - drop homogeneous coordinate
        
        # Apply intrinsic matrix: points_2d_homo = intrinsic @ points_camera_3d.T
        points_2d_homo = (intrinsic @ points_camera_3d.T).T  # [N_valid, 3]
        
        # Convert from homogeneous to Cartesian coordinates
        # Divide x, y by z to get pixel coordinates
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # [N_valid, 2]
        
        # Step 6 & 7: For each bounding box, find and collect corresponding points
        for box in boxes:
            # Convert box to numpy if it's a tensor
            if isinstance(box, torch.Tensor):
                box_np = box.cpu().numpy()
            else:
                box_np = box
            
            x1, y1, x2, y2 = box_np
            
            # Step 6: Check which points fall within this bounding box
            inside_box_mask = (
                (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) &  # x within bounds
                (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)    # y within bounds
            )
            
            # Step 7: Extract original points that project into this box
            box_point_indices = valid_indices[inside_box_mask]
            
            if len(box_point_indices) > 0:
                # Get original points (with all attributes like intensity)
                box_points = torch.tensor(points_np[box_point_indices], dtype=torch.float32)
            else:
                # No points in this box, return empty tensor with correct shape
                box_points = torch.empty(0, points_np.shape[1], dtype=torch.float32)
            
            allocated_point_clouds.append(box_points)
        
        if len(allocated_point_clouds) != len(boxes):
            raise ValueError("Mismatch in number of allocated point clouds and bounding boxes.")
        
        return allocated_point_clouds
        

class CNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # (32, 112, 112)
            nn.BatchNorm2d(32),  # Add batch normalization
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# (128, 28, 28)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# (256, 14, 14)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),  # More flexible than fixed size
            nn.Dropout(0.5)  # Add regularization
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):  # x shape: (B, 3, 224, 224)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Output shape: (B, out_dim)
        return x

class PointNet(nn.Module):
    def __init__(self, point_dim=3, feat_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim)
        )
        # No batchnorm as per paper

    def forward(self, x):  # x shape: (N, 3)
        point_feats = self.mlp(x)            # shape: (N, feat_dim)
        global_feat = torch.max(point_feats, dim=0, keepdim=True)[0]  # (1, feat_dim)
        return point_feats, global_feat

class FuseNet(nn.Module):
    """
    Fuse the features from the CNN and PointNet. Predict 3d bounding boxes and classification scores.
    """
    def __init__(self, cnn_out_dim=512, point_feat_dim=1024, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(cnn_out_dim + point_feat_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        
        # Separate heads for classification and regression
        self.classification_head = nn.Linear(512, num_classes)
        self.regression_head = nn.Linear(512, 7)  # x, y, z, l, w, h, heading
        self.confidence_head = nn.Linear(512, 1)  # confidence score

    def forward(self, cnn_features, point_features):
        # Concatenate features
        x = torch.cat((cnn_features, point_features), dim=1)  # (B, cnn_out_dim + point_feat_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Separate outputs
        cls_scores = self.classification_head(x)  # (B, num_classes)
        bbox_pred = self.regression_head(x)      # (B, 7) - x,y,z,l,w,h,heading
        confidence = torch.sigmoid(self.confidence_head(x))  # (B, 1) - confidence [0,1]
        
        return {
            'cls_scores': cls_scores,
            'bbox_pred': bbox_pred,
            'confidence': confidence
        }
    
class PointFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.point_net = PointNet()
        self.fuse_net = FuseNet()

    def forward(self, image_batch, point_cloud_batch):
        """
        Forward pass of PointFusion model.
        
        Args:
            image_batch (torch.Tensor): Batch of cropped images [B, 3, 224, 224]
            point_cloud_batch (torch.Tensor): Batch of point clouds [B, N, 3]
        
        Returns:
            dict: Predictions containing:
                - 'cls_scores': Classification scores [B, num_classes]
                - 'bbox_pred': Bounding box predictions [B, 7]
                - 'confidence': Confidence scores [B, 1]
        """
        # Extract CNN features from images
        cnn_features = self.cnn(image_batch)  # [B, 512]
        
        # Extract PointNet features from point clouds
        batch_size = point_cloud_batch.shape[0]
        point_global_features = []
        
        for i in range(batch_size):
            points = point_cloud_batch[i]  # [N, 3]
            # Remove zero-padded points
            valid_mask = torch.any(points != 0, dim=1)
            valid_points = points[valid_mask]
            
            if len(valid_points) > 0:
                _, global_feat = self.point_net(valid_points)  # [1, 1024]
                point_global_features.append(global_feat.squeeze(0))  # [1024]
            else:
                # Handle empty point clouds
                point_global_features.append(torch.zeros(1024, device=points.device))
        
        point_features = torch.stack(point_global_features)  # [B, 1024]
        
        # Fuse features and make predictions
        predictions = self.fuse_net(cnn_features, point_features)
        
        return predictions


"""
Step 1: Detect 2D Bounding Boxes (Rols)
    - Use a pre-trained Faster R-CNN model to detect 2D bounding boxes in the RGB image.
    - The model outputs bounding boxes, labels, and scores for each detected object.
    - Resize the image to 224x224 pixels for processing.
"""

"""
Step 2: Crop RGB Image + Point Cloud
    - For each 2D box:
        - Crop the corresponding region from the image (resized to 224×224).
        - Extract 3D points from the LiDAR/point cloud that project into this box using camera calibration.
"""

"""
Step 3: Process Image Features (CNN)
    - Run the cropped image through a pretrained ResNet-50 (block 4).
    - Extract a 2048-D feature vector representing the object’s appearance and shape.
"""

"""
Step 4: Step 4: Process Point Cloud Features (Modified PointNet)
    - Apply a variant of PointNet to the cropped 3D points:
    - Point-wise features: per-point descriptors
    - Global feature: describes the overall point set
    - No batch norm and with input rotation normalization (R_c) to remove camera bias.
"""

if __name__ == "__main__":
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Example 1: Process new data and save it
    print("=== Processing new data ===")
    data = PointFusionloader(dataset_path, split='train')
    processor = Preprocessor()

    for sample in tqdm.tqdm(data):
        processor.process(sample)
        break # for testing only

    processor.remove_empty_point_clouds()
    processor.assign_gt()
    
    print("Number of rolls:", len(processor.rolls))
    print("Number of point clouds:", len(processor.point_clouds))
    print("Number of labels:", len(processor.labels))
    print("Number of annotations:", len(processor.annotations))
    
    # Save the processed data
    # saved_path = processor.save_processed_data(os.path.join(dataset_path, 'preprocessed_data.pkl'))

    

# python -m src.models.PointFusion