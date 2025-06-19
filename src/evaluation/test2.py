import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.loaders.loader_Point_Pillars import PointPillarsLoader, collate_fn
from src.models.PointPillars import PointPillarsModel


def visualize_bev(points, gt_boxes, pred_boxes, pred_scores, pred_classes, class_names):
    """
    Visualize Bird's Eye View with ground truth and predicted bounding boxes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Colors for different classes
    class_colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot 1: Ground Truth
    if points.shape[0] > 0:
        ax1.scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.3, c='lightgray', label='Points')
    
    # Draw ground truth boxes with class colors
    gt_legend_added = set()
    for i, box in enumerate(gt_boxes):
        if len(box) >= 7:  # Ensure we have class info
            class_idx = int(box[6]) if len(box) > 6 else 0
            color = class_colors[class_idx % len(class_colors)]
            class_name = class_names[class_idx] if class_idx < len(class_names) else f'Class_{class_idx}'
            
            label = f'GT {class_name}' if class_name not in gt_legend_added else ''
            if label:
                gt_legend_added.add(class_name)
            
            draw_2d_box(ax1, box, color=color, linewidth=2, label=label)
    
    ax1.set_title(f'Ground Truth (BEV) - {len(gt_boxes)} boxes')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Predictions
    if points.shape[0] > 0:
        ax2.scatter(points[:, 0], points[:, 1], s=0.1, alpha=0.3, c='lightgray', label='Points')
    
    # Draw ground truth boxes (for reference)
    for box in gt_boxes:
        if len(box) >= 6:
            draw_2d_box(ax2, box, color='black', linewidth=1, linestyle='--', alpha=0.3)
    
    # Draw predicted boxes
    pred_legend_added = set()
    prediction_stats = {}
    
    for i, (box, score, cls) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
        color = class_colors[int(cls) % len(class_colors)]
        class_name = class_names[int(cls)] if int(cls) < len(class_names) else f'Class_{int(cls)}'
        
        # Track prediction statistics
        if class_name not in prediction_stats:
            prediction_stats[class_name] = {'count': 0, 'max_score': 0, 'scores': []}
        prediction_stats[class_name]['count'] += 1
        prediction_stats[class_name]['max_score'] = max(prediction_stats[class_name]['max_score'], score)
        prediction_stats[class_name]['scores'].append(score)
        
        label = f'Pred {class_name}' if class_name not in pred_legend_added else ''
        if label:
            pred_legend_added.add(class_name)
        
        # Vary line width based on confidence
        line_width = 1 + 2 * score / max(pred_scores) if len(pred_scores) > 0 else 1
        draw_2d_box(ax2, box, color=color, linewidth=line_width, alpha=0.8, label=label)
        
        # Add score text for top predictions
        if score > np.percentile(pred_scores, 90):  # Top 10% of predictions
            ax2.text(box[0], box[1], f'{score:.3f}', fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    # Add prediction statistics to title
    stats_text = []
    for class_name, stats in prediction_stats.items():
        avg_score = np.mean(stats['scores'])
        stats_text.append(f"{class_name}: {stats['count']} (avg: {avg_score:.3f}, max: {stats['max_score']:.3f})")
    
    title = f'Predictions (BEV) - {len(pred_boxes)} boxes\n' + '\n'.join(stats_text)
    ax2.set_title(title, fontsize=10)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def draw_2d_box(ax, box, color='red', linewidth=1, linestyle='-', alpha=1.0, label=''):
    """Draw a 2D bounding box in bird's eye view."""
    x, y, _, l, w, _ = box[:6]
    
    # Create rectangle corners
    corners = np.array([
        [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
    ])
    
    # Translate to box center
    corners += np.array([x, y])
    
    ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=linewidth, 
            linestyle=linestyle, alpha=alpha, label=label)


def decode_simple_predictions(predictions, score_threshold=0.1):
    """
    Simple prediction decoding for visualization.
    """
    print(f"Available prediction keys: {predictions.keys()}")
    
    if 'box_preds' not in predictions or 'cls_preds' not in predictions:
        print("Missing box_preds or cls_preds in predictions")
        return [], [], []
    
    box_preds = predictions['box_preds']
    cls_preds = predictions['cls_preds']
    
    print(f"Box preds shape: {box_preds.shape}")
    print(f"Cls preds shape: {cls_preds.shape}")
    
    # Flatten predictions
    if box_preds.dim() > 2:
        box_preds = box_preds.view(-1, box_preds.shape[-1])
    if cls_preds.dim() > 2:
        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
    
    print(f"Flattened box preds shape: {box_preds.shape}")
    print(f"Flattened cls preds shape: {cls_preds.shape}")
    
    # Apply softmax to get probabilities
    cls_probs = F.softmax(cls_preds, dim=-1)
    
    # Check if we need to exclude background class
    if cls_probs.shape[-1] > 4:  # More than 4 classes means background is included
        max_scores, pred_classes = torch.max(cls_probs[:, :-1], dim=-1)  # Exclude background
        print("Excluding background class")
    else:
        max_scores, pred_classes = torch.max(cls_probs, dim=-1)  # Use all classes
        print("Using all classes")
    
    # Print detailed statistics
    print(f"Score statistics:")
    print(f"  Min: {max_scores.min():.6f}")
    print(f"  Max: {max_scores.max():.6f}")
    print(f"  Mean: {max_scores.mean():.6f}")
    print(f"  Std: {max_scores.std():.6f}")
    print(f"  Median: {max_scores.median():.6f}")
    
    # Print percentiles
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(max_scores, p/100.0)
        print(f"  {p}th percentile: {val:.6f}")
    
    # Adaptive threshold selection
    if max_scores.max() < 0.1:
        # If all scores are very low, use a percentile-based threshold
        adaptive_threshold = torch.quantile(max_scores, 0.95).item()  # Top 5%
        print(f"Using adaptive threshold (95th percentile): {adaptive_threshold:.6f}")
        score_threshold = adaptive_threshold
    
    print(f"Final score threshold: {score_threshold}")
    
    # Filter by score threshold
    score_mask = max_scores > score_threshold
    print(f"Number passing threshold: {score_mask.sum()}")
    
    # Class distribution analysis
    print("\nClass distribution in predictions:")
    for class_idx in range(4):  # Assuming 4 classes
        class_mask = pred_classes == class_idx
        class_count = class_mask.sum()
        if class_count > 0:
            class_scores = max_scores[class_mask]
            print(f"  Class {class_idx}: {class_count} predictions, "
                  f"score range: {class_scores.min():.6f} - {class_scores.max():.6f}")
    
    if score_mask.sum() == 0:
        print("No predictions above threshold")
        return [], [], []
    
    filtered_boxes = box_preds[score_mask]
    filtered_scores = max_scores[score_mask]
    filtered_classes = pred_classes[score_mask]
    
    print(f"Returning {len(filtered_boxes)} predictions")
    
    return filtered_boxes.cpu().numpy(), filtered_scores.cpu().numpy(), filtered_classes.cpu().numpy()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    data_path = os.getenv('DATA_PATH', default='src/data/')

    # Create dataset and dataloader
    dataset = PointPillarsLoader(data_path, split='train')  # Changed to 'train' to ensure data exists
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get grid size from first sample
    sample = dataset[0]
    grid_size = tuple(sample['grid_dims'].tolist())
    print(f"Grid size: {grid_size}")

    # Initialize model
    model = PointPillarsModel(
        grid_size=grid_size,
        num_classes=4,
        grid_resolution=0.2
    ).to(device)
    
    # Load checkpoint if available
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
                print(f"Could not load checkpoint: {e2}")
                print("Using untrained model")
    else:
        print("No checkpoint found, using untrained model")
    
    model.eval()

    # Class names
    class_names = ['PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE']

    # Process just one sample for debugging
    num_samples = 1
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            print(f"\nProcessing sample {batch_idx + 1}/{num_samples}")
            print(f"Batch keys: {batch.keys()}")
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Forward pass
            print("Running forward pass...")
            outputs = model(batch)
            print(f"Model output keys: {outputs.keys()}")
            
            # Decode predictions with adaptive threshold
            pred_boxes, pred_scores, pred_classes = decode_simple_predictions(outputs, score_threshold=0.01)
            
            print(f"Number of predictions: {len(pred_boxes)}")
            if len(pred_scores) > 0:
                print(f"Score range: {pred_scores.min():.3f} - {pred_scores.max():.3f}")
                print(f"Class distribution in filtered predictions:")
                for class_idx in range(4):
                    class_count = np.sum(pred_classes == class_idx)
                    if class_count > 0:
                        class_scores = pred_scores[pred_classes == class_idx]
                        print(f"  {class_names[class_idx]}: {class_count} predictions, "
                              f"avg score: {np.mean(class_scores):.3f}")
            
            # Get ground truth
            if 'annotations' in batch:
                gt_boxes = batch['annotations']['boxes']
                gt_categories = batch['annotations']['categories']
                
                # Filter GT for this batch (batch index 0)
                gt_mask = gt_boxes[:, 10] == 0  # Last element is batch index
                batch_gt_boxes = gt_boxes[gt_mask][:, :10]  # Remove batch index
                batch_gt_categories = gt_categories[gt_mask]
                
                print(f"\nGround Truth Analysis:")
                print(f"Number of GT boxes: {len(batch_gt_boxes)}")
                if len(batch_gt_categories) > 0:
                    gt_class_names = [class_names[int(c)] for c in batch_gt_categories.cpu().numpy()]
                    print(f"GT categories: {gt_class_names}")
                    
                    # GT class distribution
                    print("GT class distribution:")
                    for class_idx in range(4):
                        class_count = torch.sum(batch_gt_categories == class_idx).item()
                        if class_count > 0:
                            print(f"  {class_names[class_idx]}: {class_count} boxes")
                    
                    # Add category information to GT boxes for visualization
                    gt_boxes_with_class = torch.cat([batch_gt_boxes, batch_gt_categories.unsqueeze(1).float()], dim=1)
                else:
                    gt_boxes_with_class = batch_gt_boxes
            else:
                batch_gt_boxes = torch.empty(0, 10)
                gt_boxes_with_class = torch.empty(0, 11)
                print("No ground truth annotations found")
            
            # Get point cloud for visualization
            if 'pillars' in batch:
                # Extract points from pillars for visualization
                pillars = batch['pillars'][0]  # First batch
                
                # Convert pillars back to points (simplified)
                points = pillars.view(-1, pillars.shape[-1])
                # Remove padding (points with all zeros)
                valid_mask = (points != 0).any(dim=1)
                points = points[valid_mask]
                points = points.cpu().numpy()
            else:
                points = np.empty((0, 4))
            
            print(f"Number of points for visualization: {len(points)}")
            
            # Visualize BEV
            if len(points) > 0 or len(batch_gt_boxes) > 0 or len(pred_boxes) > 0:
                visualize_bev(
                    points, 
                    gt_boxes_with_class.cpu().numpy(), 
                    pred_boxes, 
                    pred_scores,
                    pred_classes,
                    class_names
                )
            else:
                print("No data to visualize")

# python -m src.evaluation.test2
