# src/evaluation/eval_Point_Fusion.py
import os
import torch
import torch.nn.functional as F
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import json

from src.loaders.loader_Point_Fusion import PointFusionloader
from src.models.PointFusion import Preprocessor, PointFusion
from src.training.train_PointFusion import PointFusionDataset, collate_fn


def compute_iou_3d(box1, box2):
    """
    Compute 3D IoU between two bounding boxes.
    
    Args:
        box1, box2: [7] tensors representing [x, y, z, l, w, h, heading]
    
    Returns:
        iou: float, IoU value between 0 and 1
    """
    # Extract center coordinates and dimensions
    x1, y1, z1, l1, w1, h1, heading1 = box1
    x2, y2, z2, l2, w2, h2, heading2 = box2
    
    # For simplicity, we'll compute axis-aligned IoU (ignoring rotation for now)
    # This is a common approximation for evaluation
    
    # Calculate half extents
    half_l1, half_w1, half_h1 = l1/2, w1/2, h1/2
    half_l2, half_w2, half_h2 = l2/2, w2/2, h2/2
    
    # Calculate box corners (axis-aligned approximation)
    min_x1, max_x1 = x1 - half_l1, x1 + half_l1
    min_y1, max_y1 = y1 - half_w1, y1 + half_w1
    min_z1, max_z1 = z1 - half_h1, z1 + half_h1
    
    min_x2, max_x2 = x2 - half_l2, x2 + half_l2
    min_y2, max_y2 = y2 - half_w2, y2 + half_w2
    min_z2, max_z2 = z2 - half_h2, z2 + half_h2
    
    # Calculate intersection
    inter_min_x = max(min_x1, min_x2)
    inter_max_x = min(max_x1, max_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_y = min(max_y1, max_y2)
    inter_min_z = max(min_z1, min_z2)
    inter_max_z = min(max_z1, max_z2)
    
    # Check if there's any intersection
    if inter_min_x >= inter_max_x or inter_min_y >= inter_max_y or inter_min_z >= inter_max_z:
        return 0.0
    
    # Calculate intersection volume
    inter_volume = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y) * (inter_max_z - inter_min_z)
    
    # Calculate union volume
    volume1 = l1 * w1 * h1
    volume2 = l2 * w2 * h2
    union_volume = volume1 + volume2 - inter_volume
    
    # Calculate IoU
    iou = inter_volume / union_volume if union_volume > 0 else 0.0
    
    return float(iou)


def compute_ap(recalls, precisions):
    """
    Compute Average Precision (AP) from precision-recall curve using 11-point interpolation.
    
    Args:
        recalls: list of recall values
        precisions: list of precision values
    
    Returns:
        ap: Average Precision value
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    
    # Convert to numpy arrays
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # Compute AP using the 11-point interpolation method
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        # Find precisions for recalls >= t
        p = precisions[recalls >= t]
        if len(p) > 0:
            ap += np.max(p) / 11.0
    
    return ap


def compute_ap_interpolated(recalls, precisions):
    """
    Compute Average Precision (AP) using all-point interpolation (COCO style).
    
    Args:
        recalls: list of recall values
        precisions: list of precision values
    
    Returns:
        ap: Average Precision value
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    
    # Convert to numpy arrays
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    # Add sentinel values at beginning and end
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Look for points where recall changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum area under curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def evaluate_predictions(predictions, ground_truth, iou_threshold=0.5):
    """
    Evaluate predictions against ground truth using precision, recall, and AP.
    
    Args:
        predictions: list of dicts with 'boxes', 'scores', 'labels'
        ground_truth: list of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        metrics: dict with mAP and per-class AP values
    """
    # Class names
    class_names = ['PEDESTRIAN', 'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'TRUCK']
    class_aps = {}
    
    for class_idx, class_name in enumerate(class_names):
        # Collect all predictions and GT for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            # Get predictions for this class
            if 'labels' in pred and len(pred['labels']) > 0:
                pred_labels = pred['labels']
                if hasattr(pred_labels, 'cpu'):
                    pred_labels = pred_labels.cpu()
                
                class_mask = pred_labels == class_idx
                if class_mask.any():
                    pred_boxes = pred['boxes'][class_mask]
                    pred_scores = pred['scores'][class_mask]
                    
                    for box, score in zip(pred_boxes, pred_scores):
                        all_pred_boxes.append((i, box, score))  # (sample_idx, box, score)
                        all_pred_scores.append(score.item() if hasattr(score, 'item') else score)
            
            # Get ground truth for this class (already pre-assigned during preprocessing)
            if 'labels' in gt and len(gt['labels']) > 0:
                gt_labels = gt['labels']
                if hasattr(gt_labels, 'cpu'):
                    gt_labels = gt_labels.cpu()
                
                # Handle both tensor and numpy arrays
                if hasattr(gt_labels, 'numpy'):
                    gt_labels = gt_labels.numpy()
                elif isinstance(gt_labels, list):
                    gt_labels = np.array(gt_labels)
                
                # Check if any labels match the current class
                class_mask = gt_labels == class_idx
                if class_mask.any():
                    gt_boxes = gt['boxes']
                    if hasattr(gt_boxes, 'cpu'):
                        gt_boxes = gt_boxes.cpu()
                    
                    # Filter boxes by class
                    if len(gt_boxes.shape) > 1:  # Multiple boxes
                        gt_boxes_for_class = gt_boxes[class_mask]
                    else:  # Single box
                        gt_boxes_for_class = gt_boxes.unsqueeze(0)
                    
                    for box in gt_boxes_for_class:
                        all_gt_boxes.append((i, box))  # (sample_idx, box)
        
        if len(all_pred_boxes) == 0:
            class_aps[class_name] = 0.0
            continue
        
        # Sort predictions by confidence score (descending)
        if len(all_pred_scores) > 0:
            sorted_indices = np.argsort(all_pred_scores)[::-1]
        else:
            sorted_indices = []
        
        # Track which GT boxes have been matched
        gt_matched = set()
        
        # Calculate precision and recall at each prediction
        tp = []  # True positives
        fp = []  # False positives
        
        for idx in sorted_indices:
            sample_idx, pred_box, score = all_pred_boxes[idx]
            
            # Find the best matching GT box in the same sample
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, (gt_sample_idx, gt_box) in enumerate(all_gt_boxes):
                if gt_sample_idx == sample_idx:  # Same sample
                    iou = compute_iou_3d(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                tp.append(1)
                fp.append(0)
                gt_matched.add(best_gt_idx)
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        num_gt = len(all_gt_boxes)
        
        if num_gt == 0:
            class_aps[class_name] = 0.0
            continue
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        recalls = tp_cumsum / num_gt
        
        # Compute AP using the more accurate method
        ap = compute_ap_interpolated(recalls, precisions)
        class_aps[class_name] = ap
    
    # Calculate mAP
    valid_aps = [ap for ap in class_aps.values() if ap > 0]
    map_score = np.mean(list(class_aps.values())) if class_aps else 0.0
    
    return {
        'mAP': map_score,
        'class_APs': class_aps,
        'num_valid_classes': len(valid_aps)
    }


def compute_detection_metrics(predictions, ground_truth, iou_threshold=0.5, confidence_threshold=0.5):
    """
    Compute additional detection metrics: precision, recall, F1-score.
    
    Args:
        predictions: list of prediction dicts
        ground_truth: list of ground truth dicts
        iou_threshold: IoU threshold for positive detection
        confidence_threshold: confidence threshold for predictions
    
    Returns:
        dict with precision, recall, F1-score per class and overall
    """
    class_names = ['PEDESTRIAN', 'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'TRUCK']
    class_metrics = {}
    
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    
    for class_idx, class_name in enumerate(class_names):
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives
        
        for pred, gt in zip(predictions, ground_truth):
            # Count ground truth instances for this class
            gt_labels = gt['labels']
            if hasattr(gt_labels, 'cpu'):
                gt_labels = gt_labels.cpu().numpy()
            elif isinstance(gt_labels, list):
                gt_labels = np.array(gt_labels)
            
            gt_count_for_class = np.sum(gt_labels == class_idx)
            
            # Count predictions for this class above confidence threshold
            if 'labels' in pred and len(pred['labels']) > 0:
                pred_labels = pred['labels']
                pred_scores = pred['scores']
                
                if hasattr(pred_labels, 'cpu'):
                    pred_labels = pred_labels.cpu()
                    pred_scores = pred_scores.cpu()
                
                # Filter by confidence and class
                class_mask = pred_labels == class_idx
                conf_mask = pred_scores >= confidence_threshold
                valid_preds = class_mask & conf_mask
                
                if valid_preds.any():
                    pred_boxes = pred['boxes'][valid_preds]
                    
                    # Check IoU with ground truth
                    matched_gt = set()
                    for pred_box in pred_boxes:
                        best_iou = 0.0
                        for gt_idx, gt_box in enumerate(gt['boxes']):
                            if gt_labels[gt_idx] == class_idx:
                                iou = compute_iou_3d(pred_box, gt_box)
                                if iou > best_iou and gt_idx not in matched_gt:
                                    best_iou = iou
                                    if best_iou >= iou_threshold:
                                        tp += 1
                                        matched_gt.add(gt_idx)
                                        break
                        else:
                            fp += 1  # No matching ground truth found
                else:
                    # No valid predictions for this class
                    pass
            
            # Count false negatives (unmatched ground truth)
            if gt_count_for_class > 0:
                # This is simplified - in practice, you'd track which GT boxes were matched
                fn += max(0, gt_count_for_class - tp)
        
        # Compute metrics for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
    
    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    return {
        'class_metrics': class_metrics,
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn
        }
    }


def run_inference(model, dataloader, device, confidence_threshold=0.1):
    """
    Run inference on the test dataset.
    
    Args:
        model: Trained PointFusion model
        dataloader: Test data loader
        device: torch device
        confidence_threshold: minimum confidence for detections
    
    Returns:
        predictions: list of prediction dicts
        ground_truth: list of ground truth dicts
    """
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Move to device
            images = batch['images'].to(device)
            point_clouds = batch['point_clouds'].to(device)
            annotations = batch['annotations']
            
            # Forward pass
            outputs = model(images, point_clouds)
            
            # Process outputs for each sample in batch
            batch_size = len(images)
            for i in range(batch_size):
                # Extract predictions for this sample
                cls_scores = outputs['cls_scores'][i]  # [num_classes]
                bbox_pred = outputs['bbox_pred'][i]    # [7]
                confidence = outputs['confidence'][i]  # [1]
                
                # Apply softmax to get class probabilities
                class_probs = F.softmax(cls_scores, dim=0)
                predicted_class = torch.argmax(class_probs)
                class_score = class_probs[predicted_class]
                
                # Combine class score with confidence
                final_score = class_score * confidence.squeeze()
                
                # Only keep detections above confidence threshold
                if final_score.item() >= confidence_threshold:
                    # Store prediction
                    pred_dict = {
                        'boxes': bbox_pred.cpu().unsqueeze(0),  # [1, 7]
                        'scores': final_score.cpu().unsqueeze(0),  # [1]
                        'labels': predicted_class.cpu().unsqueeze(0)  # [1]
                    }
                else:
                    # Empty prediction if below threshold
                    pred_dict = {
                        'boxes': torch.empty(0, 7),
                        'scores': torch.empty(0),
                        'labels': torch.empty(0, dtype=torch.long)
                    }
                
                predictions.append(pred_dict)
                
                # Store ground truth (already pre-assigned during preprocessing)
                gt_dict = annotations[i]
                ground_truth.append(gt_dict)
    
    return predictions, ground_truth


def load_model(checkpoint_path, device):
    """
    Load trained PointFusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: torch device
    
    Returns:
        model: Loaded PointFusion model
    """
    model = PointFusion().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def plot_results(metrics, save_path):
    """
    Plot evaluation results.
    
    Args:
        metrics: dict with evaluation metrics
        save_path: path to save the plot
    """
    class_names = list(metrics['class_APs'].keys())
    ap_values = list(metrics['class_APs'].values())
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot of per-class AP
    plt.subplot(1, 2, 1)
    bars = plt.bar(class_names, ap_values)
    plt.title('Average Precision per Class')
    plt.ylabel('AP')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, ap_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # mAP display
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f'mAP@0.5\n{metrics["mAP"]:.3f}', 
             ha='center', va='center', fontsize=24, fontweight='bold',
             transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Mean Average Precision')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(metrics, predictions, ground_truth, save_dir, all_metrics=None, detection_metrics=None):
    """
    Save evaluation results to files.
    
    Args:
        metrics: dict with evaluation metrics for main IoU threshold
        predictions: list of predictions
        ground_truth: list of ground truth
        save_dir: directory to save results
        all_metrics: dict with metrics for all IoU thresholds
        detection_metrics: dict with precision, recall, F1-score metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare results dictionary
    results_dict = {
        'mAP@0.5': metrics['mAP'],
        'class_APs@0.5': metrics['class_APs'],
        'num_samples': len(predictions),
        'num_valid_classes': metrics.get('num_valid_classes', len(metrics['class_APs']))
    }
    
    # Add multi-threshold metrics if available
    if all_metrics:
        for key, metric_data in all_metrics.items():
            if key.startswith('mAP@'):
                results_dict[key] = metric_data['mAP']
                if key != 'mAP@0.5:0.95':
                    results_dict[f'class_APs_{key}'] = metric_data['class_APs']
    
    # Add detection metrics if available
    if detection_metrics:
        results_dict['detection_metrics'] = detection_metrics
    
    # Convert all data to JSON-serializable format
    results_dict = convert_to_json_serializable(results_dict)
    
    # Save metrics as JSON
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Save detailed results as text
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("PointFusion Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of test samples: {len(predictions)}\n")
        f.write(f"Number of valid classes: {results_dict['num_valid_classes']}\n\n")
        
        # Main metrics
        f.write(f"Mean Average Precision (mAP@0.5): {metrics['mAP']:.4f}\n")
        if all_metrics and 'mAP@0.5:0.95' in all_metrics:
            f.write(f"Mean Average Precision (mAP@0.5:0.95): {all_metrics['mAP@0.5:0.95']['mAP']:.4f}\n")
        
        # Detection metrics
        if detection_metrics:
            f.write(f"\nOverall Detection Metrics (IoU@0.5, Conf@0.1):\n")
            f.write(f"Precision: {detection_metrics['overall']['precision']:.4f}\n")
            f.write(f"Recall: {detection_metrics['overall']['recall']:.4f}\n")
            f.write(f"F1-Score: {detection_metrics['overall']['f1_score']:.4f}\n")
        
        f.write("\nPer-class Average Precision (IoU@0.5):\n")
        for class_name, ap in metrics['class_APs'].items():
            f.write(f"  {class_name}: {ap:.4f}\n")
        
        # Per-class detection metrics
        if detection_metrics:
            f.write("\nPer-class Detection Metrics:\n")
            for class_name, class_metrics in detection_metrics['class_metrics'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1_score']:.4f}\n")
        
        # Additional IoU thresholds
        if all_metrics:
            for key, metric_data in all_metrics.items():
                if key.startswith('mAP@') and key not in ['mAP@0.5', 'mAP@0.5:0.95']:
                    f.write(f"\nPer-class Average Precision ({key}):\n")
                    for class_name, ap in metric_data['class_APs'].items():
                        f.write(f"  {class_name}: {ap:.4f}\n")
    
    print(f"Results saved to {save_dir}")
    print(f"  - evaluation_results.json")
    print(f"  - evaluation_results.txt")


def convert_to_json_serializable(obj):
    """
    Convert numpy/torch objects to JSON-serializable Python objects.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):  # torch tensors
        return obj.item()
    elif hasattr(obj, 'cpu'):  # torch tensors
        return convert_to_json_serializable(obj.cpu().numpy())
    else:
        return obj


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print("Loading test dataset...")
    test_data_path = os.path.join(dataset_path, 'processed_test_data.pkl')
    test_data_path_improved = os.path.join(dataset_path, 'processed_test_data_improved.pkl')

    if os.path.exists(test_data_path_improved):
        print("Loading existing preprocessed test data (improved)...")
        test_preprocessor = Preprocessor(load_path=test_data_path_improved)
    elif os.path.exists(test_data_path):
        print("Loading existing preprocessed test data...")
        test_preprocessor = Preprocessor(load_path=test_data_path)
        test_preprocessor.assign_gt()
    else:
        print("Creating new preprocessed test data...")
        # Load and preprocess test data
        test_dataset_raw = PointFusionloader(dataset_path, split='test')
        test_preprocessor = Preprocessor()
        
        for sample in tqdm(test_dataset_raw, desc="Processing test samples"):
            test_preprocessor.process(sample)
        
        test_preprocessor.remove_empty_point_clouds()
        test_preprocessor.save_processed_data(save_path=test_data_path)
    
    # Create test dataset and dataloader
    test_dataset = PointFusionDataset(test_preprocessor)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load trained model
    checkpoint_path = os.path.join('checkpoints', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    print("Loading trained model...")
    model = load_model(checkpoint_path, device)
    
    # Run inference
    print("Running inference on test set...")
    predictions, ground_truth = run_inference(model, test_loader, device, confidence_threshold=0.1)
    
    # Evaluate predictions at different IoU thresholds
    print("Evaluating predictions...")
    iou_thresholds = [0.3, 0.5, 0.7]
    all_metrics = {}
    
    for iou_thresh in iou_thresholds:
        print(f"Computing mAP at IoU threshold {iou_thresh}...")
        metrics = evaluate_predictions(predictions, ground_truth, iou_threshold=iou_thresh)
        all_metrics[f'mAP@{iou_thresh}'] = metrics
    
    # Compute mAP@0.5:0.95 (average over multiple IoU thresholds)
    print("Computing mAP@0.5:0.95...")
    iou_range = np.arange(0.5, 1.0, 0.05)
    map_values = []
    for iou_thresh in iou_range:
        metrics = evaluate_predictions(predictions, ground_truth, iou_threshold=iou_thresh)
        map_values.append(metrics['mAP'])
    
    map_0_5_0_95 = np.mean(map_values)
    all_metrics['mAP@0.5:0.95'] = {'mAP': map_0_5_0_95}
    
    # Compute additional detection metrics
    print("Computing precision, recall, and F1-score...")
    detection_metrics = compute_detection_metrics(predictions, ground_truth, 
                                                iou_threshold=0.5, 
                                                confidence_threshold=0.1)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of test samples: {len(predictions)}")
    print(f"Mean Average Precision (mAP@0.5): {all_metrics['mAP@0.5']['mAP']:.4f}")
    print(f"Mean Average Precision (mAP@0.5:0.95): {map_0_5_0_95:.4f}")
    
    print(f"\nOverall Detection Metrics (IoU@0.5, Conf@0.1):")
    print(f"Precision: {detection_metrics['overall']['precision']:.4f}")
    print(f"Recall: {detection_metrics['overall']['recall']:.4f}")
    print(f"F1-Score: {detection_metrics['overall']['f1_score']:.4f}")
    
    print("\nDetailed Results:")
    for iou_thresh in iou_thresholds:
        metrics = all_metrics[f'mAP@{iou_thresh}']
        print(f"\nmAP@{iou_thresh}: {metrics['mAP']:.4f}")
        print("Per-class Average Precision:")
        for class_name, ap in metrics['class_APs'].items():
            print(f"  {class_name}: {ap:.4f}")
    
    print("\nPer-class Detection Metrics:")
    for class_name, metrics in detection_metrics['class_metrics'].items():
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # Save results
    save_dir = 'checkpoints'
    save_results(all_metrics['mAP@0.5'], predictions, ground_truth, save_dir, all_metrics, detection_metrics)
    
    # Plot results
    plot_path = os.path.join(save_dir, 'evaluation_plot.png')
    plot_results(all_metrics['mAP@0.5'], plot_path)
    print(f"Evaluation plot saved to {plot_path}")
    
    print("\nEvaluation completed!")


# python -m src.evaluation.eval_Point_Fusion