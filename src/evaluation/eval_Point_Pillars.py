import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import torchvision.ops as tv_ops
import matplotlib.pyplot as plt
from src.loaders.loader_Point_Pillars import PointPillarsLoader, collate_fn
from src.models.PointPillars import PointPillarsModel

def decode_predictions(predictions, anchors, score_threshold=0.1):
    """
    Decode model predictions to absolute bounding boxes.
    
    Args:
        predictions (dict): Model predictions containing 'box_preds' and 'cls_preds'
        anchors (torch.Tensor): Anchor boxes [num_anchors, 10]
        score_threshold (float): Minimum confidence score threshold
        
    Returns:
        dict: Decoded predictions for each batch
    """
    box_preds = predictions['box_preds']  # [B, H, W, num_anchors, 10]
    cls_preds = predictions['cls_preds']  # [B, H, W, num_anchors, num_classes]
    
    B, H, W, num_anchors_per_location, _ = box_preds.shape
    num_classes = cls_preds.shape[-1] - 1  # Exclude background class
    
    # Flatten predictions
    box_preds_flat = box_preds.view(B, -1, 10)  # [B, H*W*num_anchors, 10]
    cls_preds_flat = cls_preds.view(B, -1, cls_preds.shape[-1])  # [B, H*W*num_anchors, num_classes]
    
    # Apply softmax to get probabilities
    cls_probs = F.softmax(cls_preds_flat, dim=-1)
    
    batch_predictions = {}
    
    for batch_idx in range(B):
        batch_boxes = box_preds_flat[batch_idx]  # [num_anchors, 10]
        batch_probs = cls_probs[batch_idx]  # [num_anchors, num_classes]
        
        # Decode boxes relative to anchors
        decoded_boxes = decode_boxes(batch_boxes, anchors)
        
        # Get predictions for each class (excluding background)
        class_predictions = []
        
        for class_idx in range(num_classes):
            class_scores = batch_probs[:, class_idx]
            
            # Filter by score threshold
            score_mask = class_scores > score_threshold
            
            if score_mask.sum() == 0:
                continue
                
            filtered_boxes = decoded_boxes[score_mask]
            filtered_scores = class_scores[score_mask]
            
            class_predictions.append({
                'boxes': filtered_boxes,
                'scores': filtered_scores,
                'class': class_idx
            })
        
        batch_predictions[batch_idx] = class_predictions
    
    return batch_predictions

def decode_boxes(encoded_boxes, anchors):
    """
    Decode encoded box predictions relative to anchors.
    
    Args:
        encoded_boxes (torch.Tensor): Encoded predictions [num_anchors, 10]
        anchors (torch.Tensor): Anchor boxes [num_anchors, 10]
        
    Returns:
        torch.Tensor: Decoded absolute boxes [num_anchors, 10]
    """
    # Extract components
    encoded_center = encoded_boxes[:, :3]
    encoded_size = encoded_boxes[:, 3:6]
    encoded_quat = encoded_boxes[:, 6:10]
    
    anchor_center = anchors[:, :3]
    anchor_size = anchors[:, 3:6]
    anchor_quat = anchors[:, 6:10]
    
    # Decode center: decoded = anchor + encoded * anchor_size
    decoded_center = anchor_center + encoded_center * anchor_size
    
    # Decode size: decoded = anchor * exp(encoded)
    decoded_size = anchor_size * torch.exp(encoded_size)
    
    # Decode quaternion: multiply relative rotation with anchor rotation
    decoded_quat = quaternion_multiply(encoded_quat, anchor_quat)
    decoded_quat = F.normalize(decoded_quat, p=2, dim=1)
    
    # Combine decoded components
    decoded_boxes = torch.cat([decoded_center, decoded_size, decoded_quat], dim=1)
    
    return decoded_boxes

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1.unbind(dim=1)
    w2, x2, y2, z2 = q2.unbind(dim=1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=1)

def apply_nms(predictions, iou_threshold=0.5, max_detections=100):
    """
    Apply Non-Maximum Suppression to predictions.
    
    Args:
        predictions (list): List of prediction dictionaries
        iou_threshold (float): IoU threshold for NMS
        max_detections (int): Maximum number of detections to keep
        
    Returns:
        list: Filtered predictions after NMS
    """
    if not predictions:
        return []
    
    nms_predictions = []
    
    for pred in predictions:
        boxes = pred['boxes']  # [N, 10]
        scores = pred['scores']  # [N]
        
        if len(boxes) == 0:
            continue
        
        # Convert to 2D boxes for NMS (use x, y, length, width)
        boxes_2d = torch.zeros(len(boxes), 4, device=boxes.device)
        boxes_2d[:, 0] = boxes[:, 0] - boxes[:, 3] / 2  # x1
        boxes_2d[:, 1] = boxes[:, 1] - boxes[:, 4] / 2  # y1
        boxes_2d[:, 2] = boxes[:, 0] + boxes[:, 3] / 2  # x2
        boxes_2d[:, 3] = boxes[:, 1] + boxes[:, 4] / 2  # y2
        
        # Apply NMS
        keep_indices = tv_ops.nms(boxes_2d, scores, iou_threshold)
        
        # Limit to max detections
        if len(keep_indices) > max_detections:
            keep_indices = keep_indices[:max_detections]
        
        if len(keep_indices) > 0:
            nms_predictions.append({
                'boxes': boxes[keep_indices],
                'scores': scores[keep_indices],
                'class': pred['class']
            })
    
    return nms_predictions

def calculate_iou_3d(boxes1, boxes2):
    """
    Calculate 3D IoU between two sets of boxes.
    Simplified to 2D IoU for now.
    
    Args:
        boxes1 (torch.Tensor): [N, 10] boxes
        boxes2 (torch.Tensor): [M, 10] boxes
        
    Returns:
        torch.Tensor: [N, M] IoU matrix
    """
    # Convert to 2D boxes [x1, y1, x2, y2]
    def boxes_to_2d(boxes):
        boxes_2d = torch.zeros(len(boxes), 4, device=boxes.device)
        boxes_2d[:, 0] = boxes[:, 0] - boxes[:, 3] / 2  # x1
        boxes_2d[:, 1] = boxes[:, 1] - boxes[:, 4] / 2  # y1
        boxes_2d[:, 2] = boxes[:, 0] + boxes[:, 3] / 2  # x2
        boxes_2d[:, 3] = boxes[:, 1] + boxes[:, 4] / 2  # y2
        return boxes_2d
    
    boxes1_2d = boxes_to_2d(boxes1)
    boxes2_2d = boxes_to_2d(boxes2)
    
    return tv_ops.box_iou(boxes1_2d, boxes2_2d)

def calculate_ap(predictions, ground_truth, iou_threshold=0.5, class_idx=0):
    """
    Calculate Average Precision for a single class.
    
    Args:
        predictions (list): List of predictions across all samples
        ground_truth (list): List of ground truth annotations across all samples
        iou_threshold (float): IoU threshold for positive detection
        class_idx (int): Class index to evaluate
        
    Returns:
        float: Average Precision
    """
    # Collect all predictions and GT for this class
    all_predictions = []
    all_gt = []
    
    for sample_idx, (pred_list, gt_list) in enumerate(zip(predictions, ground_truth)):
        # Filter predictions for this class
        class_preds = [p for p in pred_list if p['class'] == class_idx]
        
        for pred in class_preds:
            for i in range(len(pred['boxes'])):
                all_predictions.append({
                    'sample_idx': sample_idx,
                    'box': pred['boxes'][i],
                    'score': pred['scores'][i].item(),
                    'class': class_idx
                })
        
        # Filter GT for this class
        for gt in gt_list:
            if gt['class'] == class_idx:
                all_gt.append({
                    'sample_idx': sample_idx,
                    'box': gt['box'],
                    'class': class_idx,
                    'matched': False
                })
    
    if len(all_predictions) == 0 or len(all_gt) == 0:
        return 0.0
    
    # Sort predictions by confidence score (descending)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate precision and recall
    tp = np.zeros(len(all_predictions))
    fp = np.zeros(len(all_predictions))
    
    # Create GT lookup by sample
    gt_by_sample = defaultdict(list)
    for gt in all_gt:
        gt_by_sample[gt['sample_idx']].append(gt)
    
    for pred_idx, pred in enumerate(all_predictions):
        sample_idx = pred['sample_idx']
        pred_box = pred['box'].unsqueeze(0)  # [1, 10]
        
        # Find best matching GT in the same sample
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_by_sample[sample_idx]):
            if gt['matched']:
                continue
                
            gt_box = gt['box'].unsqueeze(0)  # [1, 10]
            iou = calculate_iou_3d(pred_box, gt_box)[0, 0].item()
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if prediction is positive
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_by_sample[sample_idx][best_gt_idx]['matched'] = True
        else:
            fp[pred_idx] = 1
    
    # Calculate cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(all_gt)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap

def evaluate_model(model, test_loader, device, score_threshold=0.1, iou_threshold=0.5, nms_threshold=0.5):
    """
    Evaluate PointPillars model using mAP metric.
    
    Args:
        model: Trained PointPillars model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        score_threshold (float): Minimum confidence score
        iou_threshold (float): IoU threshold for mAP calculation
        nms_threshold (float): IoU threshold for NMS
        
    Returns:
        dict: Evaluation results including mAP and per-class AP
    """
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    # Class mapping
    class_names = ['PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE']
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Forward pass
            predictions = model(batch)
            
            # Decode predictions
            decoded_preds = decode_predictions(
                predictions, 
                predictions['anchors'], 
                score_threshold=score_threshold
            )
            
            # Apply NMS
            for batch_idx_inner in decoded_preds:
                decoded_preds[batch_idx_inner] = apply_nms(
                    decoded_preds[batch_idx_inner], 
                    iou_threshold=nms_threshold
                )
            
            # Process ground truth
            gt_boxes = batch['annotations']['boxes']
            gt_categories = batch['annotations']['categories']
            
            # Group by batch
            batch_gt = defaultdict(list)
            for i, (box, cat) in enumerate(zip(gt_boxes, gt_categories)):
                batch_idx_val = int(box[10].item())  # Last element is batch index
                batch_gt[batch_idx_val].append({
                    'box': box[:10],  # Remove batch index
                    'class': cat.item()
                })
            
            # Store results
            batch_size = len(decoded_preds) if decoded_preds else max(batch_gt.keys()) + 1 if batch_gt else 1
            
            for batch_idx_inner in range(batch_size):
                all_predictions.append(decoded_preds.get(batch_idx_inner, []))
                all_ground_truth.append(batch_gt.get(batch_idx_inner, []))
    
    # Calculate mAP
    print("Calculating mAP...")
    class_aps = {}
    
    for class_idx, class_name in enumerate(class_names):
        ap = calculate_ap(
            all_predictions, 
            all_ground_truth, 
            iou_threshold=iou_threshold, 
            class_idx=class_idx
        )
        class_aps[class_name] = ap
        print(f"{class_name}: AP = {ap:.4f}")
    
    # Calculate mean AP
    mean_ap = np.mean(list(class_aps.values()))
    
    results = {
        'mAP': mean_ap,
        'class_APs': class_aps,
        'num_samples': len(all_predictions),
        'config': {
            'score_threshold': score_threshold,
            'iou_threshold': iou_threshold,
            'nms_threshold': nms_threshold
        }
    }
    
    return results

def plot_evaluation_results(results):
    """Plot evaluation results."""
    class_names = list(results['class_APs'].keys())
    ap_values = list(results['class_APs'].values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, ap_values, alpha=0.7)
    
    # Add mAP line
    plt.axhline(y=results['mAP'], color='red', linestyle='--', 
                label=f"mAP: {results['mAP']:.4f}")
    
    # Add value labels on bars
    for bar, ap in zip(bars, ap_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ap:.3f}', ha='center', va='bottom')
    
    plt.ylabel('Average Precision')
    plt.title('PointPillars Evaluation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = PointPillarsLoader(dataset_path, split='train')
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load trained model
    print("Loading model...")
    # Get grid size from first sample
    sample = test_dataset[0]
    grid_size = tuple(sample['grid_dims'].tolist())
    
    model = PointPillarsModel(
        grid_size=grid_size,
        num_classes=4,
        grid_resolution=0.2
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best.pth'
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e2:
                print(f"Failed to load checkpoint: {e2}")
                print("Using untrained model for evaluation")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using untrained model for evaluation")
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        score_threshold=0.1,
        iou_threshold=0.5,
        nms_threshold=0.5
    )
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Number of test samples: {results['num_samples']}")
    print("\nPer-class Average Precision:")
    for class_name, ap in results['class_APs'].items():
        print(f"  {class_name}: {ap:.4f}")
    
    # Plot results
    plot_evaluation_results(results)

# python -m src.evaluation.eval_Point_Pillars