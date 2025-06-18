import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from src.loaders.loader_Point_Pillars import PointPillarsLoader, collate_fn
from src.models.PointPillars import PointPillarsModel

def decode_box_predictions(predictions, anchors):
    """
    Decode model predictions into actual 3D boxes.
    
    Args:
        predictions (dict): Model output containing:
            'box_preds': Predicted box residuals [B, H, W, num_anchors, 10]
            'cls_preds': Class predictions [B, H, W, num_anchors, num_classes]
        anchors (torch.Tensor): Anchor boxes [total_anchors, 10]
        
    Returns:
        tuple: (decoded_boxes, class_scores, class_labels)
    """
    box_preds = predictions['box_preds']
    cls_preds = predictions['cls_preds']
    
    B, H, W, num_anchors, _ = box_preds.shape
    num_classes = cls_preds.shape[-1]
    total_anchors = H * W * num_anchors
    
    # Flatten predictions
    box_preds_flat = box_preds.view(B, total_anchors, -1)
    cls_preds_flat = cls_preds.view(B, total_anchors, num_classes)
    
    decoded_boxes = []
    class_scores = []
    class_labels = []
    
    for b in range(B):
        batch_boxes = []
        batch_scores = []
        batch_labels = []
        
        # Get anchors for current spatial locations
        batch_anchors = anchors.to(box_preds_flat.device)
        
        # Extract components
        pred_center = box_preds_flat[b, :, :3]
        pred_size = box_preds_flat[b, :, 3:6]
        pred_quat = box_preds_flat[b, :, 6:]
        
        anchor_center = batch_anchors[:, :3]
        anchor_size = batch_anchors[:, 3:6]
        anchor_quat = batch_anchors[:, 6:]
        
        # Decode centers
        decoded_center = pred_center * anchor_size + anchor_center
        
        # Decode sizes
        decoded_size = torch.exp(pred_size) * anchor_size
        
        # Decode orientation (simple addition for quaternion difference)
        decoded_quat = pred_quat + anchor_quat
        decoded_quat = F.normalize(decoded_quat, p=2, dim=1)
        
        # Combine into boxes [cx, cy, cz, l, w, h, qw, qx, qy, qz]
        decoded_box = torch.cat([decoded_center, decoded_size, decoded_quat], dim=1)
        batch_boxes.append(decoded_box)
        
        # Process class predictions
        cls_probs = F.softmax(cls_preds_flat[b], dim=1)
        scores, labels = torch.max(cls_probs, dim=1)
        
        batch_scores.append(scores)
        batch_labels.append(labels)
        
        decoded_boxes.append(torch.stack(batch_boxes))
        class_scores.append(torch.stack(batch_scores))
        class_labels.append(torch.stack(batch_labels))
    
    return decoded_boxes, class_scores, class_labels

def rotate_boxes_3d(boxes):
    """
    Convert boxes with quaternion rotation to boxes with yaw rotation.
    
    Args:
        boxes (torch.Tensor): [N, 10] - [cx, cy, cz, l, w, h, qw, qx, qy, qz]
        
    Returns:
        torch.Tensor: [N, 7] - [cx, cy, cz, l, w, h, yaw]
    """
    # Extract quaternion components
    qw, qx, qy, qz = boxes[:, 6], boxes[:, 7], boxes[:, 8], boxes[:, 9]
    
    # Convert to yaw (rotation around Z-axis)
    # Formula: yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 
                     1 - 2 * (qy * qy + qz * qz))
    
    return torch.stack([
        boxes[:, 0],  # cx
        boxes[:, 1],  # cy
        boxes[:, 2],  # cz
        boxes[:, 3],  # l
        boxes[:, 4],  # w
        boxes[:, 5],  # h
        yaw           # yaw
    ], dim=1)

def calculate_3d_iou(boxes1, boxes2):
    """
    Calculate 3D IoU between two sets of boxes.
    
    Args:
        boxes1 (torch.Tensor): [N, 7] - [cx, cy, cz, l, w, h, yaw]
        boxes2 (torch.Tensor): [M, 7] - [cx, cy, cz, l, w, h, yaw]
        
    Returns:
        torch.Tensor: [N, M] IoU matrix
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 7]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 7]
    
    # Calculate min and max coordinates
    min_x1 = boxes1[..., 0] - boxes1[..., 3] / 2
    max_x1 = boxes1[..., 0] + boxes1[..., 3] / 2
    min_y1 = boxes1[..., 1] - boxes1[..., 4] / 2
    max_y1 = boxes1[..., 1] + boxes1[..., 4] / 2
    min_z1 = boxes1[..., 2] - boxes1[..., 5] / 2
    max_z1 = boxes1[..., 2] + boxes1[..., 5] / 2
    
    min_x2 = boxes2[..., 0] - boxes2[..., 3] / 2
    max_x2 = boxes2[..., 0] + boxes2[..., 3] / 2
    min_y2 = boxes2[..., 1] - boxes2[..., 4] / 2
    max_y2 = boxes2[..., 1] + boxes2[..., 4] / 2
    min_z2 = boxes2[..., 2] - boxes2[..., 5] / 2
    max_z2 = boxes2[..., 2] + boxes2[..., 5] / 2
    
    # Calculate intersection coordinates
    inter_min_x = torch.max(min_x1, min_x2)
    inter_max_x = torch.min(max_x1, max_x2)
    inter_min_y = torch.max(min_y1, min_y2)
    inter_max_y = torch.min(max_y1, max_y2)
    inter_min_z = torch.max(min_z1, min_z2)
    inter_max_z = torch.min(max_z1, max_z2)
    
    # Calculate intersection volume
    inter_width = torch.clamp(inter_max_x - inter_min_x, min=0)
    inter_length = torch.clamp(inter_max_y - inter_min_y, min=0)
    inter_height = torch.clamp(inter_max_z - inter_min_z, min=0)
    inter_volume = inter_width * inter_length * inter_height
    
    # Calculate union volume
    vol1 = (max_x1 - min_x1) * (max_y1 - min_y1) * (max_z1 - min_z1)
    vol2 = (max_x2 - min_x2) * (max_y2 - min_y2) * (max_z2 - min_z2)
    union_volume = vol1 + vol2 - inter_volume
    
    # Calculate IoU
    iou = inter_volume / (union_volume + 1e-6)
    
    return iou

def non_max_suppression(boxes, scores, labels, iou_threshold=0.5):
    """
    Apply class-aware Non-Maximum Suppression (NMS) to 3D boxes.
    
    Args:
        boxes (torch.Tensor): [N, 7] boxes in [cx, cy, cz, l, w, h, yaw] format
        scores (torch.Tensor): [N] confidence scores
        labels (torch.Tensor): [N] class labels
        iou_threshold (float): IoU threshold for suppression
        
    Returns:
        tuple: (filtered_boxes, filtered_scores, filtered_labels)
    """
    unique_labels = torch.unique(labels)
    keep_boxes = []
    keep_scores = []
    keep_labels = []
    
    for cls in unique_labels:
        cls_mask = (labels == cls)
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_labels = labels[cls_mask]
        
        # Sort by score
        sorted_indices = torch.argsort(cls_scores, descending=True)
        sorted_boxes = cls_boxes[sorted_indices]
        sorted_scores = cls_scores[sorted_indices]
        sorted_labels = cls_labels[sorted_indices]
        
        keep = []
        while sorted_boxes.size(0) > 0:
            # Keep the highest scoring box
            keep.append(sorted_indices[0])
            
            if sorted_boxes.size(0) == 1:
                break
                
            # Calculate IoU with remaining boxes
            ious = calculate_3d_iou(
                sorted_boxes[0:1], 
                sorted_boxes[1:]
            ).squeeze(0)
            
            # Remove boxes with IoU > threshold
            mask = ious <= iou_threshold
            sorted_boxes = sorted_boxes[1:][mask]
            sorted_scores = sorted_scores[1:][mask]
            sorted_labels = sorted_labels[1:][mask]
            sorted_indices = sorted_indices[1:][mask]
        
        keep_boxes.append(cls_boxes[keep])
        keep_scores.append(cls_scores[keep])
        keep_labels.append(cls_labels[keep])
    
    if keep_boxes:
        keep_boxes = torch.cat(keep_boxes, dim=0)
        keep_scores = torch.cat(keep_scores, dim=0)
        keep_labels = torch.cat(keep_labels, dim=0)
        return keep_boxes, keep_scores, keep_labels
    
    return torch.empty(0, 7), torch.empty(0), torch.empty(0)

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    """
    Evaluate the model and compute mAP.
    
    Args:
        model (nn.Module): Trained PointPillars model
        data_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        iou_threshold (float): IoU threshold for true positive
        
    Returns:
        dict: Evaluation metrics including mAP
    """
    model.eval()
    results = []
    
    # Class mapping
    class_to_idx = {
        'PEDESTRIAN': 0,
        'TRUCK': 1, 
        'LARGE_VEHICLE': 2,
        'REGULAR_VEHICLE': 3
    }
    
    # Store all detections and ground truths
    all_detections = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Decode predictions
            decoded_boxes, class_scores, class_labels = decode_box_predictions(
                predictions, predictions['anchors']
            )
            
            # Process each sample in the batch
            for b in range(len(batch['annotations']['boxes'])):
                # Get decoded boxes for this sample
                sample_boxes = decoded_boxes[b]
                sample_scores = class_scores[b]
                sample_labels = class_labels[b]
                
                # Convert to yaw representation for NMS
                sample_boxes_yaw = rotate_boxes_3d(sample_boxes)
                
                # Apply NMS
                nms_boxes, nms_scores, nms_labels = non_max_suppression(
                    sample_boxes_yaw, sample_scores, sample_labels, iou_threshold=0.5
                )
                
                # Get ground truth boxes for this sample
                gt_boxes = batch['annotations']['boxes']
                gt_categories = batch['annotations']['categories']
                batch_mask = gt_boxes[:, 10] == b
                gt_boxes_batch = gt_boxes[batch_mask, :10]
                gt_categories_batch = [gt_categories[i] for i, mask in enumerate(batch_mask) if mask]
                
                # Convert GT boxes to yaw representation
                gt_boxes_yaw = rotate_boxes_3d(gt_boxes_batch[:, :10])
                
                # Convert GT categories to indices
                gt_labels = torch.tensor([
                    class_to_idx[cat] for cat in gt_categories_batch
                ], device=device)
                
                # Store results
                all_detections[b] = {
                    'boxes': nms_boxes,
                    'scores': nms_scores,
                    'labels': nms_labels
                }
                
                all_ground_truths[b] = {
                    'boxes': gt_boxes_yaw,
                    'labels': gt_labels
                }
    
    # Calculate mAP
    aps = []
    for cls_name, cls_idx in class_to_idx.items():
        # Collect all detections and ground truths for this class
        det_boxes = []
        det_scores = []
        det_matches = []
        n_gt = 0
        
        for sample_id, dets in all_detections.items():
            gts = all_ground_truths[sample_id]
            
            # Filter detections for this class
            cls_mask = (dets['labels'] == cls_idx)
            cls_boxes = dets['boxes'][cls_mask]
            cls_scores = dets['scores'][cls_mask]
            
            # Filter ground truths for this class
            gt_cls_mask = (gts['labels'] == cls_idx)
            gt_boxes = gts['boxes'][gt_cls_mask]
            
            # Sort detections by score
            sorted_indices = torch.argsort(cls_scores, descending=True)
            cls_boxes = cls_boxes[sorted_indices]
            cls_scores = cls_scores[sorted_indices]
            
            # Initialize matches
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            sample_matches = torch.zeros(len(cls_boxes), dtype=torch.bool)
            
            # Match detections to ground truths
            for i, det_box in enumerate(cls_boxes):
                if len(gt_boxes) == 0:
                    continue
                    
                # Calculate 3D IoU
                ious = calculate_3d_iou(det_box.unsqueeze(0), gt_boxes).squeeze()
                
                # Find best match
                best_iou = torch.max(ious) if ious.numel() > 0 else 0
                best_idx = torch.argmax(ious) if ious.numel() > 0 else -1
                
                if best_iou > iou_threshold and not gt_matched[best_idx]:
                    gt_matched[best_idx] = True
                    sample_matches[i] = True
            
            det_boxes.append(cls_boxes)
            det_scores.append(cls_scores)
            det_matches.append(sample_matches)
            n_gt += len(gt_boxes)
        
        # Flatten all detections
        all_scores = torch.cat(det_scores).cpu().numpy()
        all_matches = torch.cat(det_matches).cpu().numpy()
        
        # Sort by score
        sorted_indices = np.argsort(-all_scores)
        all_matches = all_matches[sorted_indices]
        
        # Calculate precision-recall curve
        tp = np.cumsum(all_matches)
        fp = np.cumsum(~all_matches)
        
        recall = tp / max(n_gt, 1e-6)
        precision = tp / (tp + fp + 1e-6)
        
        # Compute average precision
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recall >= t
            if np.any(mask):
                p = np.max(precision[mask])
            else:
                p = 0
            ap += p / 11
        
        aps.append(ap)
        print(f"Class {cls_name} AP: {ap:.4f}")
    
    # Calculate mAP
    mAP = np.mean(aps)
    print(f"mAP: {mAP:.4f}")
    
    return {
        'mAP': mAP,
        'class_AP': dict(zip(class_to_idx.keys(), aps))
    }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = PointPillarsLoader(dataset_path, split='test')
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load trained model
    model = PointPillarsModel(
        grid_size=(300, 200), 
        num_classes=4,
        grid_resolution=0.2
    ).to(device)
    
    # Load weights (update path to your trained model)
    checkpoint_path = "checkpoints/final_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(f"No model found at {checkpoint_path}")
        exit()
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"mAP: {metrics['mAP']:.4f}")
    for cls, ap in metrics['class_AP'].items():
        print(f"{cls} AP: {ap:.4f}")

# python -m src.evaluation.eval_Point_Pillars