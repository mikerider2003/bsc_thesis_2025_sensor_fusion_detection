import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import convert_image_dtype

class PointFusion(nn.Module):
    def __init__(self, num_classes=4, point_feature_dim=256, img_feature_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.scale_factor = 1/4  # For 2 maxpool layers (each halves dimensions)
        
        # Point cloud encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, point_feature_dim)
        )
        
        # Image encoder (without final pooling)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, img_feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(point_feature_dim + img_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Prediction heads
        self.classifier = nn.Linear(256, num_classes)
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # For 3D box parameters
        )

    def forward(self, batch):
        predictions = {}
        
        for camera_name in batch['images'].keys():
            # Extract image features once per camera batch
            images = batch['images'][camera_name]
            img_features = self.img_encoder(images)
            
            camera_preds = []
            for batch_idx in range(len(batch['allocated_points'][camera_name])):
                batch_preds = []
                allocated = batch['allocated_points'][camera_name][batch_idx]
                
                for obj in allocated:
                    # Skip if no points
                    if len(obj['points']) == 0:
                        continue
                        
                    # Encode point cloud features
                    points = obj['points']
                    point_features = self.point_encoder(points)
                    point_features = point_features.mean(dim=0)  # Aggregate points
                    
                    # Extract image ROI features
                    img_feature = self._get_roi_features(
                        img_features[batch_idx], 
                        obj['box']
                    )
                    
                    # Skip if invalid ROI
                    if img_feature is None:
                        continue
                    
                    # Fuse features
                    combined = torch.cat([point_features, img_feature], dim=0)
                    fused = self.fusion(combined)
                    
                    # Predictions
                    cls_pred = self.classifier(fused)
                    reg_pred = self.regressor(fused)
                    
                    batch_preds.append({
                        'class': cls_pred,
                        'box_3d': reg_pred,
                        'score_2d': obj['score'],
                        'label_2d': obj['label'],
                        'points': points  # Add points for loss computation
                    })
                
                camera_preds.append(batch_preds)
            
            predictions[camera_name] = camera_preds
        
        return predictions

    def _get_roi_features(self, feature_map, box):
        """Extract ROI features from feature map using box coordinates"""
        # Scale box coordinates to feature map size
        x1, y1, x2, y2 = box * self.scale_factor
        
        # Convert to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get feature map dimensions
        _, H, W = feature_map.shape
        
        # Check for valid ROI
        if x1 >= W or y1 >= H or x2 <= x1 or y2 <= y1:
            return None
        
        # Clamp coordinates to feature map dimensions
        x1 = max(0, min(x1, W-1))
        y1 = max(0, min(y1, H-1))
        x2 = max(x1+1, min(x2, W))
        y2 = max(y1+1, min(y2, H))
        
        # Extract and pool ROI
        roi = feature_map[:, y1:y2, x1:x2]
        return F.adaptive_avg_pool2d(roi.unsqueeze(0), (1, 1)).flatten()

class MeanAveragePrecision3D:
    """
    Compute 3D Mean Average Precision for object detection
    """
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets"""
        self.predictions = []
        self.targets = []
    
    def add_batch(self, predictions, targets):
        """
        Add a batch of predictions and targets
        
        Args:
            predictions: Dict[camera_name] -> List[List[Dict]] - model predictions
            targets: List[Dict] - ground truth annotations for batch
        """
        try:
            # Convert predictions to a simpler format for mAP computation
            batch_preds = []
            batch_targets = []
            
            # Process each sample in the batch
            for batch_idx in range(len(targets)):
                sample_preds = []
                sample_target = targets[batch_idx]
                
                # Collect predictions from all cameras for this sample
                for camera_name, camera_preds in predictions.items():
                    if batch_idx < len(camera_preds):
                        for obj in camera_preds[batch_idx]:
                            if 'class' in obj and 'box_3d' in obj:
                                # Get predicted class
                                class_scores = torch.softmax(obj['class'], dim=0)
                                pred_class = torch.argmax(class_scores).item()
                                confidence = class_scores[pred_class].item()
                                
                                sample_preds.append({
                                    'class': pred_class,
                                    'confidence': confidence,
                                    'box_3d': obj['box_3d'].detach().cpu().numpy()
                                })
                
                batch_preds.append(sample_preds)
                batch_targets.append({
                    'boxes': sample_target['boxes'].cpu().numpy(),
                    'labels': sample_target['labels'].cpu().numpy()
                })
            
            self.predictions.extend(batch_preds)
            self.targets.extend(batch_targets)
            
        except Exception as e:
            print(f"Warning: Error in mAP computation: {e}")
    
    def compute(self):
        """
        Compute mean Average Precision
        Returns simplified mAP estimate
        """
        if len(self.predictions) == 0 or len(self.targets) == 0:
            return 0.0
        
        try:
            total_predictions = sum(len(preds) for preds in self.predictions)
            total_targets = sum(len(target['labels']) for target in self.targets)
            
            if total_predictions == 0 or total_targets == 0:
                return 0.0
            
            # Simplified mAP: ratio of predictions to targets (placeholder)
            # In a real implementation, you'd compute IoU-based matching
            return min(total_predictions / max(total_targets, 1), 1.0) * 0.5
            
        except Exception as e:
            print(f"Error computing mAP: {e}")
            return 0.0

class BoundingBoxExtractor(nn.Module):
    """
    Extracts bounding boxes from the image tensor using a pre-trained model.

    Args:
        Images: Tensor of shape (B, C, H, W) - batch of images
    
    Returns:
        List[Dict] of length B with keys 'boxes', 'scores', 'labels'
            'boxes': Tensor[N, 4] - bounding boxes in (x1, y1, x2, y2) format
            'scores': Tensor[N] - confidence scores for each box
            'labels': Tensor[N] - class labels for each box
    """
    def __init__(self, score_thresh=0.9):
        super().__init__()
        # Load pre-trained Faster R-CNN with new weights API
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()  # Set to eval mode
        self.score_thresh = score_thresh
        
    def forward(self, images):
        if images.dtype != torch.float32:
            images = convert_image_dtype(images, dtype=torch.float32)

        with torch.no_grad():
            preds = self.model(images)

        # Equivalent from COCO dataset classes to Argoverse classes
        TARGET_LABELS = {
            'PEDESTRIAN': 1,        # person
            'REGULAR_VEHICLE': 3,   # car
            'TRUCK': 8,             # truck
            'LARGE_VEHICLE': 6      # bus 
        }
        ALLOWED_LABEL_IDS = set(TARGET_LABELS.values())
        # Move tensor to same device as input
        allowed_tensor = torch.tensor(list(ALLOWED_LABEL_IDS), device=images.device)
        
        filtered_preds = []
        for pred in preds: 
            labels = pred['labels']
            scores = pred['scores']
            boxes = pred['boxes']
            
            keep = (scores > 0.5) & (torch.isin(labels, allowed_tensor))
            
            filtered_preds.append({
                'boxes': boxes[keep],
                'labels': labels[keep],
                'scores': scores[keep]
            })

        return filtered_preds

def visualize_extracted_boxes(images, detections):
    """
    Visualizes the bounding boxes extracted from the images.

    Args:
        images: Tensor of shape (B, C, H, W)
        detections: List[Dict] with keys 'boxes', 'scores', 'labels'
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

    # Color mapping for your target categories (COCO label IDs)
    LABEL_COLORS = {
        1: 'red',       # person -> PEDESTRIAN
        3: 'blue',      # car -> REGULAR_VEHICLE  
        6: 'orange',    # bus -> LARGE_VEHICLE
        8: 'green',     # truck -> TRUCK
    }
    
    # Label names mapping COCO to your categories
    LABEL_NAMES = {
        1: 'PEDESTRIAN',
        3: 'REGULAR_VEHICLE',
        6: 'LARGE_VEHICLE', 
        8: 'TRUCK',
    }

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(13, 3))
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]

    # Keep track of which labels are present for legend
    present_labels = set()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        
        # Draw bounding boxes with different colors
        for j, box in enumerate(detections[i]['boxes']):
            label = detections[i]['labels'][j].item()
            score = detections[i]['scores'][j].item()
            
            # Get color for this label (default to red if not found)
            color = LABEL_COLORS.get(label, 'red')
            label_name = LABEL_NAMES.get(label, f'class_{label}')
            
            # Add to present labels for legend
            present_labels.add((label, label_name, color))
            
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    # Create legend for present labels
    if present_labels:
        legend_elements = []
        for label_id, label_name, color in sorted(present_labels):
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=label_name))
        
        # Add legend to the right of the last subplot
        fig.legend(handles=legend_elements, loc='upper right')

    plt.suptitle(f'Extracted Bounding Boxes from Batch (size: {num_images})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print detection summary
    print("\nDetection Summary:")
    for i in range(num_images):
        print(f"Image {i+1}:")
        labels_count = {}
        for label in detections[i]['labels']:
            label_name = LABEL_NAMES.get(label.item(), f'class_{label.item()}')
            labels_count[label_name] = labels_count.get(label_name, 0) + 1
        
        for label_name, count in labels_count.items():
            print(f"  - {label_name}: {count}")
        if not labels_count:
            print("  - No detections")

class Allocate3dPoints(nn.Module):
    """
    Take 2d bounding boxes and allocate 3d points to them.

    Args:
        detections: Dict[camera_name] -> List[Dict] with detection results per camera
            Each detection dict has keys 'boxes', 'scores', 'labels'
                'boxes': Tensor[N, 4] - bounding boxes in (x1, y1, x2, y2) format
                'scores': Tensor[N] - confidence scores for each box
                'labels': Tensor[N] - class labels for each box
        points: Tensor[B, N, 3] - point cloud data (B batches, N points, 3 coordinates)
        camera_calibrations: List[Dict] - calibrations for each batch sample
            Each element: Dict[camera_name] -> Dict with 'intrinsic', 'extrinsic'
                'intrinsic': Tensor[3, 3] - camera intrinsic matrix  
                'extrinsic': Tensor[4, 4] - camera extrinsic matrix
    """
    def __init__(self, max_points_per_box=10000, min_points_per_box=10):
        super().__init__()
        self.max_points_per_box = max_points_per_box
        self.min_points_per_box = min_points_per_box

    def _points_in_boxes(self, points_2d, boxes):
        """
        Vectorized check for which points are inside which boxes.
        
        Args:
            points_2d: Tensor[N, 2] - 2D point coordinates
            boxes: Tensor[M, 4] - bounding boxes in (x1, y1, x2, y2) format
        
        Returns:
            Tensor[N, M] boolean mask where True indicates point is inside box
        """
        # Expand dimensions for broadcasting
        x = points_2d[:, 0].unsqueeze(1)  # [N, 1]
        y = points_2d[:, 1].unsqueeze(1)  # [N, 1]
        
        x1 = boxes[:, 0]  # [M]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Vectorized containment check
        inside = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)  # [N, M]
        return inside
    
    def project_3d_to_2d(self, points_3d, intrinsic_matrix, extrinsic_matrix):
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: Tensor[N, 3] - 3D points in world coordinates
            intrinsic_matrix: Tensor[3, 3] - camera intrinsic matrix
            extrinsic_matrix: Tensor[4, 4] - world to camera transformation
        
        Returns:
            points_2d: Tensor[N, 2] - 2D image coordinates
            valid_mask: Tensor[N] - mask for points in front of camera
        """
        # Convert to homogeneous coordinates
        N = points_3d.shape[0]
        ones = torch.ones(N, 1, device=points_3d.device)
        points_3d_hom = torch.cat([points_3d, ones], dim=1)  # [N, 4]
        
        # Transform to camera coordinates
        points_cam_hom = torch.matmul(points_3d_hom, extrinsic_matrix.T)  # [N, 4]
        points_cam_3d = points_cam_hom[:, :3]  # [N, 3]
        
        # Filter points behind camera
        valid_mask = points_cam_3d[:, 2] > 0.1  # z > 0.1 (in front of camera)
        
        # Project to image plane
        points_2d_hom = torch.matmul(points_cam_3d, intrinsic_matrix.T)  # [N, 3]
        
        # Convert from homogeneous to 2D coordinates
        points_2d = points_2d_hom[:, :2] / (points_2d_hom[:, 2:3] + 1e-8)  # [N, 2]
        
        return points_2d, valid_mask
    
    def forward(self, detections, points, camera_calibrations):
        """
        Args:
            detections: Dict[camera_name] -> List[Dict] (batch of detections per camera)
            points: Tensor[B, N, 3] where B is batch_size
            camera_calibrations: List[Dict] where each dict maps camera_name to calibration
        Returns:
            Dict of allocated points for each camera and batch.
        """
        allocated_points = {}
        
        for camera_name, camera_detections in detections.items():
            camera_allocated = []
            
            for batch_idx, detection in enumerate(camera_detections):
                # Initialize empty list for this batch
                batch_allocated = []
                
                if len(detection['boxes']) > 0:  # Only process if we have boxes
                    boxes = detection['boxes']
                    labels = detection['labels']
                    scores = detection['scores']
                    
                    batch_points = points[batch_idx]
                    
                    # Get calibration
                    batch_calibration = camera_calibrations[batch_idx][camera_name]
                    intrinsic_matrix = batch_calibration['intrinsic']
                    extrinsic_matrix = batch_calibration['extrinsic']

                    # Project points
                    points_2d, valid_mask = self.project_3d_to_2d(
                        batch_points, intrinsic_matrix, extrinsic_matrix
                    )
                    
                    valid_points_3d = batch_points[valid_mask]
                    valid_points_2d = points_2d[valid_mask]
                    
                    if len(valid_points_3d) > 0 and len(boxes) > 0:
                        inside_mask = self._points_in_boxes(valid_points_2d, boxes)
                        
                        for box_idx in range(len(boxes)):
                            box_points = valid_points_3d[inside_mask[:, box_idx]]
                            
                            # Only add if we have sufficient points
                            if len(box_points) >= self.min_points_per_box:
                                if len(box_points) > self.max_points_per_box:
                                    indices = torch.randperm(len(box_points))[:self.max_points_per_box]
                                    box_points = box_points[indices]
                                
                                batch_allocated.append({
                                    'points': box_points,
                                    'label': labels[box_idx],
                                    'score': scores[box_idx],
                                    'box': boxes[box_idx],
                                    'num_points': len(box_points)
                                })
                
                camera_allocated.append(batch_allocated)
            
            allocated_points[camera_name] = camera_allocated
            
        return allocated_points

def visualize_point_allocation(images, detections, allocated_points, camera_name):
    """
    Visualize 2D bounding boxes with the number of allocated 3D points
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    LABEL_COLORS = {1: 'red', 3: 'blue', 6: 'orange', 8: 'green'}
    LABEL_NAMES = {1: 'PEDESTRIAN', 3: 'REGULAR_VEHICLE', 6: 'LARGE_VEHICLE', 8: 'TRUCK'}
    
    batch_size = len(images)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
    if batch_size == 1:
        axes = [axes]
    
    for i in range(batch_size):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        
        camera_allocated = allocated_points[camera_name][i]
        
        for box_data in camera_allocated:
            box = box_data['box']
            label = box_data['label'].item()
            score = box_data['score'].item()
            num_points = box_data['num_points']
            
            color = LABEL_COLORS.get(label, 'red')
            label_name = LABEL_NAMES.get(label, f'class_{label}')
            
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Draw bounding box
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add text with point count
            ax.text(x1, y1-5, f'{label_name}\n{num_points} pts', 
                   color=color, fontsize=8, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.axis('off')
        ax.set_title(f'{camera_name} - Batch {i}: {len(camera_allocated)} detections')
    
    plt.tight_layout()
    plt.show()

def batch_inspection(batch):
    print(f"\nBatch structure (size: {batch['points'].shape[0]}):")
    print(f"Points shape: {batch['points'].shape}")
    for camera_name, image in batch['images'].items():
        print(f"Image {camera_name} shape: {image.shape}")
    print("\nAnnotations:")
    for i, annotations in enumerate(batch['annotations']):
        print(f"\tSample {i}: Boxes {annotations['boxes'].shape}, Labels {annotations['labels'].shape}")

if __name__ == "__main__":
    import os
    from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
    from torch.utils.data import DataLoader
    from dotenv import load_dotenv
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Test with sample data
    dataset = PointFusionloader(dataset_path, split='train')
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=custom_collate
        )
    
    batch = next(iter(dataloader))
    """
    batch: Dict containing 'images', 'points', and 'annotations'.
            'images': Dict with keys as camera names and values as image tensors
                e.g. {'ring_front_left': Tensor[B, C, H, W], ...
            'points': Tensor of shape (B, N, 3) - point cloud data
            'annotations': List[Dict] with keys 'boxes', 'labels'
                'boxes': Tensor[N, 7] - bounding boxes in (x, y, z, l, w, h, heading) format
                'labels': Tensor[N] - class labels for each box
    """
    # batch_inspection(batch)

    # Bounding box extraction test
    bbox_extractor = BoundingBoxExtractor(score_thresh=0.9)
    # images = batch['images']['ring_side_right']  # Example image tensor 

    # detections = bbox_extractor(images)
    # visualize_extracted_boxes(images, detections)

    all_camera_detections = {}
    for camera_name, image in batch['images'].items():
        detections = bbox_extractor(image)
        all_camera_detections[camera_name] = detections

    batch['detections'] = all_camera_detections

    alocator = Allocate3dPoints()
    allocated_points = alocator(batch['detections'], batch['points'], batch['camera_calibrations'])

    if allocated_points:
        if 'ring_front_center' in allocated_points:
            visualize_point_allocation(
                batch['images']['ring_front_center'], 
                batch['detections']['ring_front_center'], 
                allocated_points, 
                'ring_front_center'
            )
    
    batch['allocated_points'] = allocated_points
    
    # 3. Run PointFusion model
    model = PointFusion()
    outputs = model(batch)

    # Print sample predictions
    print("\nSample predictions:")
    for camera_name, preds in outputs.items():
        print(f"\nCamera: {camera_name}")
        for batch_idx, batch_preds in enumerate(preds):
            print(f"  Batch {batch_idx}: {len(batch_preds)} objects")
            for obj_idx, obj in enumerate(batch_preds[:3]):  # Print first 3 objects
                print(f"    Object {obj_idx}:")
                print(f"      2D label: {obj['label_2d'].item()}")
                print(f"      2D score: {obj['score_2d'].item():.2f}")
                print(f"      3D class: {torch.argmax(obj['class']).item()}")
                print(f"      3D box: {obj['box_3d'].detach().numpy()}")

# python -m src.models.PointFusion