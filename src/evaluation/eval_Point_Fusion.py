# src/evaluation/eval_map_3d.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from scipy.optimize import linear_sum_assignment
from dotenv import load_dotenv

# Import project modules
from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
from src.models.PointFusion import PointFusion3D, PointFusionLoss, HungarianMatcher

class BBoxEvaluator:
    def __init__(self, num_classes=4, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
        
    def reset(self):
        self.pred_boxes = {i: [] for i in range(self.num_classes)}
        self.gt_boxes = {i: [] for i in range(self.num_classes)}
        self.confidences = {i: [] for i in range(self.num_classes)}

    def add_batch(self, pred_corners, pred_labels, pred_scores, gt_corners, gt_labels):
        """Add batch of predictions and ground truths for processing"""
        for cls_idx in range(self.num_classes):
            cls_pred_mask = pred_labels == cls_idx
            cls_gt_mask = gt_labels == cls_idx
            
            # Store predictions
            self.pred_boxes[cls_idx].extend(pred_corners[cls_pred_mask].cpu().numpy())
            self.confidences[cls_idx].extend(pred_scores[cls_pred_mask].cpu().numpy())
            
            # Store ground truths
            self.gt_boxes[cls_idx].extend(gt_corners[cls_gt_mask].cpu().numpy())

    def _compute_3d_iou(self, box1, box2):
        """Compute oriented 3D IoU using separating axis theorem (SAT)"""
        # Convert to numpy arrays and reshape
        box1 = box1.reshape(8, 3)
        box2 = box2.reshape(8, 3)
        
        # Calculate intersection volume using convex hull approximation
        # This is simplified version - consider using trimesh for exact calculation
        min1, max1 = np.min(box1, axis=0), np.max(box1, axis=0)
        min2, max2 = np.min(box2, axis=0), np.max(box2, axis=0)
        
        # Intersection coordinates
        intersect_min = np.maximum(min1, min2)
        intersect_max = np.minimum(max1, max2)
        intersect_dims = np.clip(intersect_max - intersect_min, a_min=0, a_max=None)
        
        intersection = np.prod(intersect_dims)
        
        # Union volume
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
        union = vol1 + vol2 - intersection
        
        return intersection / union if union > 0 else 0

    def _calculate_ap(self, precisions, recalls):
        """Compute average precision from precision-recall curve"""
        # Append sentinel values
        recalls = np.concatenate([[0], recalls, [1]])
        precisions = np.concatenate([[0], precisions, [0]])

        # Smooth precision curve
        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        # Find integral under curve
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum(
            (recalls[indices] - recalls[indices-1]) * 
            precisions[indices]
        )
        return ap

    def evaluate(self):
        """Calculate mAP across all classes"""
        aps = []
        class_names = ["PEDESTRIAN", "REGULAR_VEHICLE", "LARGE_VEHICLE", "TRUCK"]
        
        for cls_idx in range(self.num_classes):
            # Skip classes with no ground truth
            if len(self.gt_boxes[cls_idx]) == 0:
                print(f"Class {class_names[cls_idx]} has no ground truth boxes")
                continue
                
            # Sort predictions by confidence
            sorted_indices = np.argsort(-np.array(self.confidences[cls_idx]))
            sorted_boxes = np.array(self.pred_boxes[cls_idx])[sorted_indices]
            sorted_scores = np.array(self.confidences[cls_idx])[sorted_indices]
            
            # Initialize TP/FP arrays
            n_preds = len(sorted_boxes)
            n_gts = len(self.gt_boxes[cls_idx])
            tps = np.zeros(n_preds)
            fps = np.zeros(n_preds)
            gt_matched = np.zeros(n_gts, dtype=bool)
            
            # Match predictions to ground truths
            for pred_idx in range(n_preds):
                best_iou = 0
                best_gt = -1
                
                # Find best matching GT
                for gt_idx in range(n_gts):
                    if gt_matched[gt_idx]:
                        continue
                        
                    iou = self._compute_3d_iou(sorted_boxes[pred_idx], 
                                             self.gt_boxes[cls_idx][gt_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx
                        
                # Assign TP/FP
                if best_iou >= self.iou_threshold and best_gt != -1:
                    tps[pred_idx] = 1
                    gt_matched[best_gt] = True
                else:
                    fps[pred_idx] = 1
            
            # Compute precision-recall
            cum_tps = np.cumsum(tps)
            cum_fps = np.cumsum(fps)
            
            recalls = cum_tps / n_gts
            precisions = cum_tps / (cum_tps + cum_fps + 1e-6)
            
            # Calculate AP
            ap = self._calculate_ap(precisions, recalls)
            aps.append(ap)
            print(f"Class {class_names[cls_idx]} AP: {ap:.4f}")
            
        mAP = np.mean(aps) if aps else 0
        print(f"\nMean Average Precision (mAP@{self.iou_threshold}): {mAP:.4f}")
        return mAP

def evaluate_model(model, test_loader, device, confidence_thresh=0.3):
    """Run model evaluation on test set"""
    evaluator = BBoxEvaluator(iou_threshold=0.5)
    model.eval()
    
    # Initialize matcher and loss function for box conversion
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5)
    loss_fn = PointFusionLoss(matcher).to(device)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            points = batch['points'].to(device)
            images = {k: v.to(device) for k, v in batch['images'].items()}
            
            # Get predictions
            outputs = model({'points': points, 'images': images})
            
            # Convert outputs to boxes
            scores = torch.sigmoid(outputs['scores'])  # [B, max_pred]
            class_logits = outputs['class_logits']     # [B, max_pred, 4]
            pred_classes = torch.argmax(class_logits, dim=-1)
            pred_corners = outputs['corners']          # [B, max_pred, 8, 3]
            
            # Process batch
            batch_size = points.size(0)
            for b in range(batch_size):
                # Filter predictions by confidence
                mask = scores[b] > confidence_thresh
                batch_corners = pred_corners[b][mask]
                batch_classes = pred_classes[b][mask]
                batch_scores = scores[b][mask]
                
                # Get ground truths
                gt_annots = batch['annotations'][b]
                gt_boxes = gt_annots['boxes'].to(device)  # [N, 7]
                gt_labels = gt_annots['labels'].to(device)
                
                # Convert GT parameters to corners
                gt_corners = loss_fn.box_params_to_corners(gt_boxes)  # Now called on instantiated loss_fn
                
                evaluator.add_batch(
                    batch_corners.cpu(), 
                    batch_classes.cpu(),
                    batch_scores.cpu(),
                    gt_corners.cpu(),
                    gt_labels.cpu()
                )
    
    return evaluator.evaluate()

import open3d as o3d
import numpy as np

def plot_3d_boxes(points, pred_corners, gt_corners, title="3D Visualization"):
    """Visualize LiDAR points with predicted and ground truth boxes using Open3D"""
    # Create Open3D visualization window with title containing box counts
    window_title = f"{title} | GT Boxes: {len(gt_corners)} | Pred Boxes: {len(pred_corners)}"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1200, height=800)
    
    # Configure rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 1.0
    opt.line_width = 5.0  # Make lines thicker for better visibility
    
    # Create point cloud from LiDAR points
    # Sample points for better performance
    num_points = min(10000, len(points))
    step = max(1, len(points) // num_points)
    sampled_points = points[::step][:num_points]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([0.8, 0.8, 0.8]), (len(sampled_points), 1)))
    vis.add_geometry(pcd)
    
    # Add ground truth boxes (green)
    for box in gt_corners:
        lines = create_bounding_box_lines(box)
        vis.add_geometry(lines)
        
    # Add predicted boxes (red)
    for box in pred_corners:
        lines = create_bounding_box_lines(box, color=[1.0, 0.0, 0.0])  # Red
        vis.add_geometry(lines)
    
    # Remove the problematic line that tried to access get_window()
    # vis.get_window().set_title(f"{title} | GT Boxes: {len(gt_corners)} | Pred Boxes: {len(pred_corners)}")
    
    # Set view control
    vc = vis.get_view_control()
    vc.set_lookat([0, 0, 0])
    vc.set_front([-0.5, -0.5, -0.5])  # Similar to matplotlib view
    vc.set_up([0, 0, 1])
    vc.set_zoom(0.3)
    
    # Run the visualization
    vis.run()
    vis.destroy_window()

def create_bounding_box_lines(corners, color=[0.0, 1.0, 0.0]):
    """Create line set for a 3D bounding box"""
    # Define edges of the cube
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Top
        [4, 5], [5, 7], [7, 6], [6, 4],  # Bottom
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]
    
    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    
    # Set line color
    colors = [color for _ in range(len(edges))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def visualize_first_sample(model, test_loader, device, confidence_thresh=0.3):
    """Visualize predictions and ground truth for first sample"""
    model.eval()
    
    # Get first batch
    batch = next(iter(test_loader))
    
    with torch.no_grad():
        # Move data to device
        points = batch['points'].to(device)
        images = {k: v.to(device) for k, v in batch['images'].items()}
        
        # Get predictions
        outputs = model({'points': points, 'images': images})
        
        # Process first sample in batch
        first_sample = {
            'points': points[0].cpu().numpy(),
            'images': {k: v[0].cpu() for k, v in images.items()},
            'pred_corners': outputs['corners'][0].cpu().numpy(),
            'pred_scores': outputs['scores'][0].sigmoid().cpu().numpy(),  # Apply sigmoid to scores
            'gt_boxes': batch['annotations'][0]['boxes'].cpu().numpy(),
            'gt_labels': batch['annotations'][0]['labels'].cpu().numpy()
        }

    # Convert GT boxes to corners
    matcher = HungarianMatcher()
    loss_fn = PointFusionLoss(matcher)
    gt_corners = loss_fn.box_params_to_corners(
        torch.tensor(first_sample['gt_boxes'])
    ).numpy()

    # Filter predictions
    mask = first_sample['pred_scores'] > confidence_thresh
    pred_corners = first_sample['pred_corners'][mask]

    # Plot 3D visualization using Open3D
    plot_3d_boxes(
        first_sample['points'],
        pred_corners,
        gt_corners,
        "3D Detection Results"
    )

def main():
    # Config
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.getenv('DATA_PATH', default='src/data/')
    model_path = "outputs/best_model_point_fusion.pth"
    
    # Load model
    model = PointFusion3D(max_predictions=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from {model_path}")
    
    # Load test dataset
    test_dataset = PointFusionloader(data_path, split='test')

    # TODO: Remove this line
    indicies = list(range(10))
    test_dataset = Subset(test_dataset, indicies)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=4, 
        collate_fn=custom_collate,
        shuffle=False
    )
    print(f"Loaded test set with {len(test_dataset)} samples")
    
    # Run evaluation
    # evaluate_model(model, test_loader, device)
    visualize_first_sample(model, test_loader, device)


if __name__ == "__main__":
    main()

# python -m src.evaluation.eval_Point_Fusion