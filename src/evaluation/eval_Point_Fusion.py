# src/evaluation/eval_map_3d.py
import os
import torch
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm  # Add import for progress bar

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
        # Track per-class statistics
        self.class_predictions = {i: [] for i in range(self.num_classes)}
        self.class_targets = {i: [] for i in range(self.num_classes)}
        
        # Track all predictions and targets
        self.all_predictions = []
        self.all_targets = []
    
    def compute_3d_iou(self, box1, box2):
        """
        Compute IoU between 3D bounding boxes
        Simplified version that uses axis-aligned boxes
        box format: [x, y, z, l, w, h, heading]
        """
        # Extract center and dimensions
        center1, dims1 = box1[:3], box1[3:6]
        center2, dims2 = box2[:3], box2[3:6]
        
        # Calculate min/max bounds (ignore rotation for simplicity)
        min1 = center1 - dims1/2
        max1 = center1 + dims1/2
        min2 = center2 - dims2/2
        max2 = center2 + dims2/2
        
        # Intersection bounds
        intersect_min = np.maximum(min1, min2)
        intersect_max = np.minimum(max1, max2)
        
        # Check if boxes intersect
        if np.any(intersect_max < intersect_min):
            return 0.0
        
        # Calculate volumes
        dims_intersect = intersect_max - intersect_min
        vol_intersect = np.prod(dims_intersect)
        vol1 = np.prod(dims1)
        vol2 = np.prod(dims2)
        vol_union = vol1 + vol2 - vol_intersect
        
        return vol_intersect / vol_union
    
    def add_batch(self, predictions, targets):
        """
        Add a batch of predictions and targets
        """
        # Process each sample in batch
        for batch_idx in range(len(targets)):
            sample_preds = []
            sample_target = targets[batch_idx]
            
            # Get all predictions for this sample from all cameras
            for camera_name, camera_preds in predictions.items():
                if batch_idx < len(camera_preds):
                    for obj in camera_preds[batch_idx]:
                        if 'class' in obj and 'box_3d' in obj:
                            # Get predicted class
                            class_scores = torch.softmax(obj['class'], dim=0)
                            pred_class = torch.argmax(class_scores).item()
                            confidence = class_scores[pred_class].item()
                            
                            # Add to predictions
                            sample_preds.append({
                                'class': pred_class,
                                'confidence': confidence,
                                'box_3d': obj['box_3d'].detach().cpu().numpy()
                            })
                            
                            # Add to class-specific predictions
                            self.class_predictions[pred_class].append({
                                'confidence': confidence,
                                'box_3d': obj['box_3d'].detach().cpu().numpy(),
                                'matched': False  # For tracking TP/FP
                            })
            
            # Add all predictions for this sample
            self.all_predictions.extend(sample_preds)
            
            # Add ground truths
            gt_boxes = sample_target['boxes'].cpu().numpy()
            gt_labels = sample_target['labels'].cpu().numpy()
            
            # Add to all targets
            self.all_targets.append({
                'boxes': gt_boxes,
                'labels': gt_labels
            })
            
            # Add to class-specific targets
            for gt_idx in range(len(gt_labels)):
                gt_class = gt_labels[gt_idx]
                gt_box = gt_boxes[gt_idx]
                
                self.class_targets[gt_class].append({
                    'box_3d': gt_box,
                    'matched': False  # For tracking TP/FP
                })
    
    def compute(self):
        """
        Compute mAP overall and per class
        """
        if len(self.all_predictions) == 0 or len(self.all_targets) == 0:
            return {"mAP": 0.0, "class_ap": {name: 0.0 for _, name in self.class_names.items()}}
        
        # Per-class AP calculation
        class_ap = {}
        
        for class_id in range(self.num_classes):
            # Get class name
            class_name = self.class_names[class_id]
            
            # Get predictions for this class
            preds = self.class_predictions[class_id]
            # Get targets for this class
            targets = self.class_targets[class_id]
            
            # Sort predictions by confidence
            preds.sort(key=lambda x: x['confidence'], reverse=True)
            
            # If no predictions or targets, AP = 0
            if len(preds) == 0 or len(targets) == 0:
                class_ap[class_name] = 0.0
                continue
            
            # Calculate TP and FP
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            
            # Match predictions to ground truths
            for pred_idx, pred in enumerate(preds):
                best_iou = 0
                best_target_idx = -1
                
                # Find best matching ground truth
                for target_idx, target in enumerate(targets):
                    if target['matched']:
                        continue
                        
                    # Compute IoU
                    iou = self.compute_3d_iou(pred['box_3d'], target['box_3d'])
                    
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_target_idx = target_idx
                
                # Assign TP or FP
                if best_target_idx >= 0:
                    tp[pred_idx] = 1
                    targets[best_target_idx]['matched'] = True
                else:
                    fp[pred_idx] = 1
            
            # Calculate precision and recall
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            
            precision = cum_tp / (cum_tp + cum_fp + 1e-8)
            recall = cum_tp / (len(targets) + 1e-8)
            
            # Calculate AP using all recall points
            ap = 0
            
            # 11-point interpolation
            for t in np.arange(0, 1.1, 0.1):
                p_interp = 0
                for i in range(len(precision)):
                    if recall[i] >= t:
                        p_interp = max(p_interp, precision[i])
                        
                ap += p_interp / 11
                
            class_ap[class_name] = ap
        
        # Calculate overall mAP
        mAP = sum(class_ap.values()) / len(class_ap)
        
        return {"mAP": mAP, "class_ap": class_ap}

def main():
    # Config
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.getenv('DATA_PATH', default='src/data/')

    # Load model
    model = PointFusion().to(device)
    
    # Load checkpoint if available
    checkpoint_path = "checkpoints/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load test dataset
    test_dataset = PointFusionloader(data_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=custom_collate, shuffle=False)

    # Prepare bounding box extractor and point allocator
    bbox_extractor = BoundingBoxExtractor(score_thresh=0.9).to(device)
    point_allocator = Allocate3dPoints()

    model.eval()
    # Use our enhanced mAP metric
    metric = EnhancedMeanAveragePrecision3D(num_classes=4)
    metric.reset()

    with torch.no_grad():
        # Wrap test_loader with tqdm for progress
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move items to device
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

            # Update mAP
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