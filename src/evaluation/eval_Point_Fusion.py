# src/evaluation/eval_map_3d.py
import os
import torch
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import project modules
from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
from src.models.PointFusion import PointFusion, MeanAveragePrecision3D, BoundingBoxExtractor, Allocate3dPoints

class EnhancedMeanAveragePrecision3D:
    """Enhanced mAP calculator that provides per-class metrics"""
    def __init__(self, num_classes=4, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.class_names = {
            0: "PEDESTRIAN",
            1: "REGULAR_VEHICLE",
            2: "LARGE_VEHICLE", 
            3: "TRUCK"
        }
        self.reset()
    
    def reset(self):
        # For each class, we'll store predictions with their confidence and match status
        self.predictions = {i: [] for i in range(self.num_classes)}
        # For each class, we'll store ground truth boxes with their match status
        self.ground_truths = {i: [] for i in range(self.num_classes)}
    
    def compute_3d_iou(self, pred_box, gt_box):
        """
        Compute IoU between 3D bounding boxes
        box format: [x, y, z, l, w, h, heading]
        """
        # Extract centers and dimensions
        pred_center, pred_dims = pred_box[:3], pred_box[3:6]
        gt_center, gt_dims = gt_box[:3], gt_box[3:6]
        
        # Calculate min/max corners (simplified - ignoring rotation)
        pred_min = pred_center - pred_dims/2
        pred_max = pred_center + pred_dims/2
        gt_min = gt_center - gt_dims/2
        gt_max = gt_center + gt_dims/2
        
        # Calculate intersection
        intersection_min = np.maximum(pred_min, gt_min)
        intersection_max = np.minimum(pred_max, gt_max)
        
        if np.any(intersection_max < intersection_min):
            return 0.0
        
        intersection_dims = intersection_max - intersection_min
        intersection_volume = intersection_dims[0] * intersection_dims[1] * intersection_dims[2]
        
        pred_volume = pred_dims[0] * pred_dims[1] * pred_dims[2]
        gt_volume = gt_dims[0] * gt_dims[1] * gt_dims[2]
        union_volume = pred_volume + gt_volume - intersection_volume
        
        if union_volume < 1e-6:
            return 0.0
        
        iou = intersection_volume / union_volume
        return iou
    
    def add_batch(self, predictions, targets):
        """Add a batch of predictions and targets"""
        # Process each sample in batch
        for batch_idx in range(len(targets)):
            target = targets[batch_idx]
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            
            # Add ground truths to respective class
            for i in range(len(gt_labels)):
                class_id = gt_labels[i]
                if class_id < self.num_classes:
                    self.ground_truths[class_id].append({
                        'box': gt_boxes[i],
                        'matched': False  # Track whether this gt has been matched
                    })
            
            # Process predictions from all cameras
            for camera_name, camera_preds in predictions.items():
                if batch_idx < len(camera_preds):
                    # Process each object detection
                    for pred_obj in camera_preds[batch_idx]:
                        if 'class' in pred_obj and 'box_3d' in pred_obj:
                            # Get predicted class and confidence
                            logits = pred_obj['class']
                            scores = torch.softmax(logits, dim=0)
                            pred_class = torch.argmax(scores).item()
                            confidence = scores[pred_class].item()
                            
                            # Add to predictions list for this class
                            if pred_class < self.num_classes:
                                self.predictions[pred_class].append({
                                    'box': pred_obj['box_3d'].detach().cpu().numpy(),
                                    'confidence': confidence,
                                    'matched': False  # Track whether this prediction has been matched
                                })
    
    def compute(self):
        """Compute mAP overall and per class"""
        class_ap = {}
        total_predictions = sum(len(preds) for preds in self.predictions.values())
        total_gt = sum(len(gts) for gts in self.ground_truths.values())
        
        print(f"Total predictions: {total_predictions}, Total ground truths: {total_gt}")
        
        # Calculate AP for each class
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            predictions = self.predictions[class_id]
            ground_truths = self.ground_truths[class_id]
            
            print(f"Class {class_name}: {len(predictions)} predictions, {len(ground_truths)} ground truths")
            
            # If no predictions or ground truths, AP = 0
            if len(predictions) == 0 or len(ground_truths) == 0:
                class_ap[class_name] = 0.0
                continue
            
            # Sort predictions by confidence (descending)
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Initialize arrays for precision/recall calculation
            tp = np.zeros(len(predictions))
            fp = np.zeros(len(predictions))
            
            # For each prediction, determine if it's a TP or FP
            for i, pred in enumerate(predictions):
                # Find ground truth with highest IoU
                max_iou = -1
                max_idx = -1
                
                for j, gt in enumerate(ground_truths):
                    if gt['matched']:  # Skip already matched ground truths
                        continue
                    
                    # Calculate IoU
                    iou = self.compute_3d_iou(pred['box'], gt['box'])
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
                
                # Check if the match is valid
                if max_iou >= self.iou_threshold:
                    tp[i] = 1  # True positive
                    ground_truths[max_idx]['matched'] = True
                else:
                    fp[i] = 1  # False positive
            
            # Compute cumulative TP and FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            # Compute precision and recall
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-10)
            recall = cum_tp / max(len(ground_truths), 1e-10)
            
            # Compute average precision using 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0.0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11.0
            
            class_ap[class_name] = float(ap)
        
        # Compute mAP
        mAP = np.mean(list(class_ap.values())) if class_ap else 0.0
        
        return {
            "mAP": mAP,
            "class_ap": class_ap
        }

def main():
    # Config
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.getenv('DATA_PATH', default='src/data/')
    print(f"Using device: {device}")

    # Load model
    model = PointFusion().to(device)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/best_model.pth"
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load test dataset
    test_dataset = PointFusionloader(data_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=custom_collate, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples")

    # Prepare bounding box extractor and point allocator
    bbox_extractor = BoundingBoxExtractor(score_thresh=0.5).to(device)  # Lower threshold for better recall
    point_allocator = Allocate3dPoints()

    model.eval()
    # Initialize mAP metric
    metric = EnhancedMeanAveragePrecision3D(num_classes=4, iou_threshold=0.5)
    metric.reset()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch["points"] = batch["points"].to(device)
            for cam in batch["images"]:
                batch["images"][cam] = batch["images"][cam].to(device)
            for i in range(len(batch["camera_calibrations"])):
                for cam in batch["camera_calibrations"][i]:
                    batch["camera_calibrations"][i][cam]["intrinsic"] = \
                        batch["camera_calibrations"][i][cam]["intrinsic"].to(device)
                    batch["camera_calibrations"][i][cam]["extrinsic"] = \
                        batch["camera_calibrations"][i][cam]["extrinsic"].to(device)

            # Extract bounding boxes
            all_camera_detections = {}
            for camera_name, img_tensor in batch['images'].items():
                detections = bbox_extractor(img_tensor)
                all_camera_detections[camera_name] = detections
            batch["detections"] = all_camera_detections

            # Allocate 3D points
            allocated_points = point_allocator(batch['detections'], batch['points'], batch['camera_calibrations'])
            batch["allocated_points"] = allocated_points

            # Run model
            predictions = model(batch)

            # Update evaluation metric
            metric.add_batch(predictions, batch["annotations"])

    # Compute metrics
    results = metric.compute()
    
    # Print formatted results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Mean Average Precision (3D): {results['mAP']:.4f}")
    print("\nPer-Class Results:")
    print("-"*50)
    print(f"{'Class':<20} {'Average Precision':<20}")
    print("-"*50)
    for class_name, ap in results['class_ap'].items():
        print(f"{class_name:<20} {ap:.4f}")

if __name__ == "__main__":
    main()

# python -m src.evaluation.eval_Point_Fusion