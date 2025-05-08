import torch
import joblib
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

from src.models.PointPillars import PointPillarsModel
from src.loaders.loader_Point_Pillars import PointPillarsLoader


def calculate_iou_3d(box1, box2):
    """
    Calculate 3D IoU between two boxes.
    
    Args:
        box1: [x, y, z, l, w, h, theta]
        box2: [x, y, z, l, w, h, theta]
    """
    # For simplicity, approximate IoU with 2D IoU (bird's eye view)
    # In a real implementation, you would compute true 3D IoU

    # Extract coordinates and dimensions
    x1, y1, _, l1, w1, _, yaw1 = box1
    x2, y2, _, l2, w2, _, yaw2 = box2
    
    # For simplicity, we'll use axis-aligned boxes for IoU
    # This is a simplification - in practice you'd want to handle rotations
    
    # Calculate box extents
    x1_min, x1_max = x1 - l1/2, x1 + l1/2
    y1_min, y1_max = y1 - w1/2, y1 + w1/2
    
    x2_min, x2_max = x2 - l2/2, x2 + l2/2
    y2_min, y2_max = y2 - w2/2, y2 + w2/2
    
    # Calculate intersection area
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-7)  # Add small epsilon to avoid division by zero
    
    return iou

def convert_predictions_to_boxes(bbox_preds, cls_scores, dir_scores, voxel_size, x_range, y_range, score_threshold=0.1):
    """
    Convert model predictions to bounding boxes.
    
    Args:
        bbox_preds: Tensor of shape [B, num_classes*7, H, W]
        cls_scores: Tensor of shape [B, num_classes, H, W]
        dir_scores: Tensor of shape [B, 2, H, W]
        voxel_size: Tuple (x_size, y_size)
        x_range: Tuple (min_x, max_x)
        y_range: Tuple (min_y, max_y)
        
    Returns:
        List of boxes per batch item, each containing [x, y, z, l, w, h, theta, cls_id, score]
    """
    B, C7, H, W = bbox_preds.shape
    num_classes = cls_scores.shape[1]
    
    # Apply softmax to class scores
    cls_probs = torch.softmax(cls_scores, dim=1)
    
    # Apply softmax to direction scores
    dir_probs = torch.softmax(dir_scores, dim=1)
    
    batch_boxes = []
    
    for b in range(B):
        boxes = []
        
        # Get the highest class probability and its index for each voxel
        max_probs, cls_ids = torch.max(cls_probs[b], dim=0)
        
        # Get positions where class probability exceeds threshold
        above_threshold = max_probs > score_threshold
        
        if (above_threshold.sum() == 0):
            # No detections above threshold
            print(f"No detections above threshold {score_threshold}. Max probability: {max_probs.max().item()}")
            batch_boxes.append([])
            continue
        else:
            print(f"Found {above_threshold.sum().item()} detections above threshold {score_threshold}")
        
        y_indices, x_indices = torch.where(above_threshold)
        
        for i in range(len(y_indices)):
            y_idx, x_idx = y_indices[i], x_indices[i]
            cls_id = cls_ids[y_idx, x_idx].item()
            score = max_probs[y_idx, x_idx].item()
            
            # Skip background class (typically 0)
            if cls_id == 0:
                continue
                
            # Get bounding box parameters
            # Each class has its own set of 7 parameters
            box_param_index = cls_id * 7
            box_params = bbox_preds[b, box_param_index:box_param_index+7, y_idx, x_idx]
            
            # Convert voxel indices to world coordinates
            cx = x_range[0] + (x_idx + 0.5) * voxel_size[0] * (2**2)  # Account for 2 max pooling layers
            cy = y_range[0] + (y_idx + 0.5) * voxel_size[1] * (2**2)  # Account for 2 max pooling layers
            
            # Apply box deltas to anchor
            x = cx + box_params[0].item()
            y = cy + box_params[1].item()
            z = box_params[2].item()
            l = torch.exp(box_params[3]).item()  # length
            w = torch.exp(box_params[4]).item()  # width
            h = torch.exp(box_params[5]).item()  # height
            
            # Get direction from classification
            dir_idx = torch.argmax(dir_probs[b, :, y_idx, x_idx]).item()
            theta = box_params[6].item() + (np.pi if dir_idx == 1 else 0)
            
            # Normalize angle to [-pi, pi]
            while theta > np.pi:
                theta -= 2 * np.pi
            while theta < -np.pi:
                theta += 2 * np.pi
                
            boxes.append([x, y, z, l, w, h, theta, cls_id, score])
            
        batch_boxes.append(boxes)
        
    return batch_boxes

def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision using the 11-point interpolation.
    
    Args:
        recalls: List of recall values
        precisions: List of precision values
        
    Returns:
        AP value
    """
    ap = 0.0
    
    # 11-point interpolation
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
        
    return ap

def evaluate_map(model, dataloader, classes, voxel_size=(0.3, 0.3), 
                x_range=(-100, 100), y_range=(-100, 100), iou_threshold=0.5):
    """Evaluate mAP for the model."""
    model.eval()
    
    # Dictionary to store all ground truth and predictions
    all_gt_boxes = defaultdict(list)
    all_pred_boxes = defaultdict(list)
    
    frame_id = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # With batch_size=1, batch is a list with a single sample
            sample = batch[0]  
            
            if "lidar_processed" not in sample:
                pillars, coords = sample["lidar_file"]
            else:
                pillars, coords = sample["lidar_processed"]
            
            # Convert to torch tensors and add batch dimension
            pillars = torch.from_numpy(pillars).float().unsqueeze(0)  # Add batch dimension
            coords = torch.from_numpy(coords).int().unsqueeze(0)      # Add batch dimension
            
            # Extract ground truth boxes from annotations
            annotations = sample["annotations"]
            
            for _, anno in annotations.iterrows():
                cls_name = anno["category"]
                try:
                    cls_id = list(classes).index(cls_name) + 1  # +1 because 0 is background
                except ValueError:
                    # Skip if class is not in the list
                    continue
                
                # Extract position and dimensions
                x, y, z = anno["tx_m"], anno["ty_m"], anno["tz_m"]
                l, w, h = anno["length_m"], anno["width_m"], anno["height_m"]
                
                # Extract rotation
                qw, qx, qy, qz = anno["qw"], anno["qx"], anno["qy"], anno["qz"]
                yaw = 2 * np.arctan2(qz, qw)
                
                gt_box = [x, y, z, l, w, h, yaw]
                all_gt_boxes[cls_id].append([frame_id, gt_box])
            
            # Run inference
            bbox_preds, cls_scores, dir_scores = model(pillars, coords)
            
            # Convert predictions to boxes
            pred_boxes = convert_predictions_to_boxes(
                bbox_preds, cls_scores, dir_scores, 
                voxel_size, x_range, y_range
            )[0]  # Get first item since we have batch_size=1
            
            # Store predictions
            for box in pred_boxes:
                x, y, z, l, w, h, theta, cls_id, score = box
                pred_box = [x, y, z, l, w, h, theta]
                all_pred_boxes[cls_id].append([frame_id, pred_box, score])
            
            frame_id += 1
    
    # Calculate AP for each class
    aps = {}
    
    for cls_id in all_gt_boxes.keys():
        # Sort predictions by score
        predictions = sorted(all_pred_boxes[cls_id], key=lambda x: x[2], reverse=True)
        
        # Initialize counters
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        gt_count = len(all_gt_boxes[cls_id])
        
        # Create a dictionary for faster lookup
        gt_by_frame = defaultdict(list)
        for frame_id, gt_box in all_gt_boxes[cls_id]:
            gt_by_frame[frame_id].append(gt_box)
            
        # Mark all ground truth as not detected yet
        gt_detected = {frame_id: [False] * len(boxes) for frame_id, boxes in gt_by_frame.items()}
        
        # Evaluate each prediction
        for i, (frame_id, pred_box, _) in enumerate(predictions):
            if frame_id not in gt_by_frame:
                fp[i] = 1
                continue
                
            gt_boxes = gt_by_frame[frame_id]
            max_iou = 0
            max_idx = -1
            
            # Find the ground truth box with the highest IoU
            for j, gt_box in enumerate(gt_boxes):
                if gt_detected[frame_id][j]:
                    continue
                    
                iou = calculate_iou_3d(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            # Check if the detection is a true positive
            if max_iou >= iou_threshold:
                tp[i] = 1
                gt_detected[frame_id][max_idx] = True
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / gt_count if gt_count > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Add sentinel values for easier calculation
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
            
        # Calculate AP for this class
        ap = calculate_ap(recalls, precisions)
        aps[cls_id] = ap
        
    # Calculate mAP across all classes
    map_value = np.mean(list(aps.values())) if aps else 0.0
    
    return map_value, aps

def visualize_results(test_dataset, model, num_samples=5):
    """
    Visualize model predictions vs ground truth.
    """
    model.eval()
    
    for i in range(min(num_samples, len(test_dataset))):
        sample = test_dataset[i]
        
        # Get lidar data
        if "lidar_processed" not in sample:
            pillars, coords = test_dataset._process_lidar(sample["lidar_file"])
        else:
            pillars, coords = sample["lidar_processed"]
            
        # Convert to torch tensors and add batch dimension
        pillars = torch.from_numpy(pillars).float().unsqueeze(0)
        coords = torch.from_numpy(coords).int().unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            bbox_preds, cls_scores, dir_scores = model(pillars, coords)
            
        # Convert predictions to boxes
        pred_boxes = convert_predictions_to_boxes(
            bbox_preds, cls_scores, dir_scores, 
            voxel_size=(0.3, 0.3), 
            x_range=(-100, 100), 
            y_range=(-100, 100)
        )[0]
        
        # Extract ground truth boxes
        annotations = sample["annotations"]
        gt_boxes = []
        
        for _, anno in annotations.iterrows():
            cls_name = anno["category"]
            if cls_name not in test_dataset.classes:
                continue
                
            cls_id = list(test_dataset.classes).index(cls_name) + 1  # +1 because 0 is background
            
            # Extract position and dimensions
            x, y, z = anno["tx_m"], anno["ty_m"], anno["tz_m"]
            l, w, h = anno["length_m"], anno["width_m"], anno["height_m"]
            
            # Extract rotation (quaternion to Euler)
            qw, qx, qy, qz = anno["qw"], anno["qx"], anno["qy"], anno["qz"]
            yaw = 2 * np.arctan2(qz, qw)  # Simplified conversion
            
            gt_boxes.append([x, y, z, l, w, h, yaw, cls_id])
        
        # Visualize this sample (bird's eye view)
        plt.figure(figsize=(10, 10))
        
        # Plot LiDAR points (2D projection)
        points = np.zeros((pillars.shape[1], 2))
        valid_indices = np.where(coords[0, :, 1] > 0)[0]
        if len(valid_indices) > 0:
            for idx, idx_val in enumerate(valid_indices):
                x_idx, y_idx = coords[0, idx_val, 1], coords[0, idx_val, 2]
                points[idx] = [x_idx * 0.3 - 100, y_idx * 0.3 - 100]
            
            plt.scatter(points[:len(valid_indices), 0], points[:len(valid_indices), 1], s=0.1, color='gray', alpha=0.5)
        
        # Plot ground truth boxes in green
        for box in gt_boxes:
            x, y, z, l, w, h, yaw, cls_id = box
            
            # Get box corners (simplified 2D)
            corners = np.array([
                [l/2, w/2],
                [l/2, -w/2],
                [-l/2, -w/2],
                [-l/2, w/2]
            ])
            
            # Rotate corners
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            corners = corners @ R.T
            
            # Translate corners
            corners = corners + np.array([x, y])  # Use array addition instead of adding to each column
            
            # Plot box
            plt.plot(np.append(corners[:, 0], corners[0, 0]),
                     np.append(corners[:, 1], corners[0, 1]), 
                     color='g', linewidth=2)
            
        # Plot predicted boxes in red
        for box in pred_boxes:
            x, y, z, l, w, h, theta, cls_id, score = box
            
            # Get box corners (simplified 2D)
            corners = np.array([
                [l/2, w/2],
                [l/2, -w/2],
                [-l/2, -w/2],
                [-l/2, w/2]
            ])
            
            # Rotate corners
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            corners = corners @ R.T
            
            # Translate corners
            corners = corners + np.array([x, y])  # Use array addition instead of adding to each column
            
            # Plot box
            plt.plot(np.append(corners[:, 0], corners[0, 0]),
                     np.append(corners[:, 1], corners[0, 1]), 
                     color='r', linewidth=2)
            
            # Add class label and score
            cls_name = list(test_dataset.classes)[cls_id - 1] if cls_id > 0 and cls_id <= len(test_dataset.classes) else "unknown"
            plt.text(x, y, f"{cls_name}: {score:.2f}", color='r')
        
        plt.title(f"Sample {i}")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)
        plt.axis('equal')
        plt.savefig(f"outputs/visualization/sample_{i}.png")
        plt.close()


def custom_collate_fn(batch):
    """Custom collate function that handles pandas DataFrames."""
    # Just return the batch as is without any further processing
    # This is okay since we're using batch_size=1 anyway
    return batch

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    from torch.utils.data import DataLoader

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset and loader
    test_dataset = PointPillarsLoader(dataset_path, split='test')
    
    # Process samples
    processed_samples = test_dataset.process_all_samples(limit=10)  # You may want to test on full dataset
    
    # Create dataloader with batch_size=1 and custom collate function
    test_loader = DataLoader(
        processed_samples, 
        batch_size=1, 
        shuffle=False,
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    # Load model
    num_classes = len(test_dataset.classes) + 1  # +1 for the background class
    model = PointPillarsModel(num_classes=num_classes)
    print(f"Model created with {num_classes} classes (including background)")
    
    # Load checkpoint (final model weights)
    model_path = 'outputs/models/final_model.pth'
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded final model checkpoint from {model_path}, epoch {checkpoint['epoch']}")
        
        # Print some model statistics
        if 'loss' in checkpoint:
            print(f"Final training loss: {checkpoint['loss']:.4f}")
            
    except Exception as e:
        print(f"Error loading checkpoint from {model_path}: {e}")
        
        # Try backup model (best model)
        try:
            backup_model_path = 'outputs/models/best_model.pth'
            checkpoint = torch.load(backup_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model checkpoint from {backup_model_path}, epoch {checkpoint['epoch']}")
        except Exception as e:
            print(f"Error loading backup model: {e}")
            print("Evaluating with untrained model.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Ensure output directories exist
    os.makedirs('outputs/visualization', exist_ok=True)
    
    # Debug the class distribution in the test set
    class_counts = {}
    for sample in processed_samples:
        annotations = sample["annotations"]
        for _, anno in annotations.iterrows():
            cls_name = anno["category"]
            if cls_name not in class_counts:
                class_counts[cls_name] = 0
            class_counts[cls_name] += 1
    
    print("\nClass distribution in test set:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count}")
    
    # Evaluate mAP
    print("\nEvaluating model mAP...")
    map_value, class_aps = evaluate_map(model, test_loader, test_dataset.classes)
    
    print(f"mAP@{0.5}: {map_value:.4f}")
    print("Class APs:")
    for cls_id, ap in class_aps.items():
        cls_name = list(test_dataset.classes)[cls_id - 1]  # -1 because cls_id starts at 1
        print(f"  {cls_name}: {ap:.4f}")
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(test_dataset, model, num_samples=5)
    
    print("Evaluation complete.")










