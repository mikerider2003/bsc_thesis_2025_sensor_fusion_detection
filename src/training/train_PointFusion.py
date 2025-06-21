# src/training/train_PointFusion.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from dotenv import load_dotenv
import torch.nn.functional as F
import wandb
from datetime import datetime

from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
from src.models.PointFusion import PointFusion, MeanAveragePrecision3D, BoundingBoxExtractor, Allocate3dPoints

class PointFusionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.model = PointFusion(
            num_classes=config['num_classes'],
            point_feature_dim=config['point_feature_dim'],
            img_feature_dim=config['img_feature_dim']
        ).to(self.device)
        
        # Fix: Move BoundingBoxExtractor to device
        self.bbox_extractor = BoundingBoxExtractor(score_thresh=config['bbox_score_thresh']).to(self.device)
        
        # Fix: Move Allocate3dPoints to device  
        self.point_allocator = Allocate3dPoints(
            max_points_per_box=config['max_points_per_box'],
            min_points_per_box=config.get('min_points_per_box', 10)
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['scheduler_step'],
            gamma=config['scheduler_gamma']
        )
        
        # Initialize metrics
        self.train_map = MeanAveragePrecision3D(config['num_classes'], iou_threshold=0.5)
        self.val_map = MeanAveragePrecision3D(config['num_classes'], iou_threshold=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        
    def compute_iou_3d_simple(self, pred_box, gt_box):
        """
        Simplified 3D IoU calculation for box matching
        Both boxes in format [x, y, z, l, w, h, heading]
        """
        # Extract centers and dimensions
        pred_center = pred_box[:3]
        pred_dims = pred_box[3:6]
        
        gt_center = gt_box[:3]
        gt_dims = gt_box[3:6]
        
        # Calculate axis-aligned bounding boxes (ignore rotation for simplicity)
        pred_min = pred_center - pred_dims / 2
        pred_max = pred_center + pred_dims / 2
        
        gt_min = gt_center - gt_dims / 2
        gt_max = gt_center + gt_dims / 2
        
        # Intersection
        inter_min = torch.maximum(pred_min, gt_min)
        inter_max = torch.minimum(pred_max, gt_max)
        inter_dims = torch.clamp(inter_max - inter_min, min=0)
        inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]
        
        # Union
        pred_vol = pred_dims[0] * pred_dims[1] * pred_dims[2]
        gt_vol = gt_dims[0] * gt_dims[1] * gt_dims[2]
        union_vol = pred_vol + gt_vol - inter_vol
        
        return inter_vol / (union_vol + 1e-8)

    def find_best_gt_match(self, obj, gt_boxes, gt_labels):
        """
        Find the best ground truth match for a prediction using multiple criteria
        """
        if len(gt_boxes) == 0:
            return None, None, 0.0
        
        points = obj['points']
        if len(points) == 0:
            return None, None, 0.0
        
        # Method 1: Centroid distance
        center_pred = torch.mean(points, dim=0)
        gt_centers = gt_boxes[:, :3]
        distances = torch.norm(gt_centers - center_pred.unsqueeze(0), dim=1)
        
        # Method 2: IoU with predicted box (if we have a reasonable prediction)
        pred_box = obj['box_3d']
        ious = torch.zeros(len(gt_boxes), device=self.device)
        
        for i, gt_box in enumerate(gt_boxes):
            try:
                iou = self.compute_iou_3d_simple(pred_box, gt_box)
                ious[i] = iou
            except:
                ious[i] = 0.0
        
        # Combine criteria: prioritize IoU if > threshold, otherwise use distance
        iou_threshold = 0.1
        distance_threshold = self.config.get('matching_distance_threshold', 5.0)
        
        # Find best IoU match
        best_iou, best_iou_idx = torch.max(ious, dim=0)
        
        # Find best distance match
        min_distance, best_dist_idx = torch.min(distances, dim=0)
        
        if best_iou > iou_threshold:
            # Use IoU-based matching if we have a good overlap
            return gt_boxes[best_iou_idx], gt_labels[best_iou_idx], best_iou.item()
        elif min_distance < distance_threshold:
            # Fall back to distance-based matching
            return gt_boxes[best_dist_idx], gt_labels[best_dist_idx], min_distance.item()
        else:
            # No good match found
            return None, None, min_distance.item()

    def compute_loss(self, predictions, ground_truths):
        """
        Compute combined classification and regression loss with improved GT matching
        """
        total_loss = 0.0
        cls_loss_total = 0.0
        reg_loss_total = 0.0
        num_objects = 0
        num_matched = 0
        num_unmatched = 0
        
        matching_stats = {'iou_matches': 0, 'distance_matches': 0, 'no_matches': 0}
        
        # Process each camera's predictions
        for camera_name, camera_preds in predictions.items():
            for batch_idx, batch_preds in enumerate(camera_preds):
                if len(batch_preds) == 0:
                    continue
                
                # Ensure we have valid ground truth for this batch
                if batch_idx >= len(ground_truths):
                    continue
                    
                # Get corresponding ground truth
                gt = ground_truths[batch_idx]
                gt_boxes = gt['boxes'].to(self.device)
                gt_labels = gt['labels'].to(self.device)
                
                if len(gt_boxes) == 0:
                    continue
                
                for obj in batch_preds:
                    if 'class' not in obj or 'box_3d' not in obj or 'points' not in obj:
                        continue
                    
                    # Check for valid tensors
                    if torch.isnan(obj['class']).any() or torch.isnan(obj['box_3d']).any():
                        continue
                    
                    try:
                        # Find best ground truth match
                        target_box, target_label, match_quality = self.find_best_gt_match(
                            obj, gt_boxes, gt_labels
                        )
                        
                        if target_box is not None:
                            # Valid match found
                            cls_pred = obj['class']
                            cls_loss = F.cross_entropy(cls_pred.unsqueeze(0), target_label.unsqueeze(0))
                            
                            if not (torch.isnan(cls_loss) or torch.isinf(cls_loss)):
                                cls_loss_total += cls_loss
                                
                                # Regression loss with matched target
                                reg_pred = obj['box_3d']
                                if reg_pred.shape[0] == target_box.shape[0]:
                                    reg_loss = F.mse_loss(reg_pred, target_box)
                                    
                                    if not (torch.isnan(reg_loss) or torch.isinf(reg_loss)):
                                        reg_loss_total += reg_loss
                                        num_matched += 1
                                        
                                        # Track matching method
                                        if match_quality > 0.1:  # IoU-based
                                            matching_stats['iou_matches'] += 1
                                        else:  # Distance-based
                                            matching_stats['distance_matches'] += 1
                                
                                num_objects += 1
                        else:
                            # No valid match
                            matching_stats['no_matches'] += 1
                            num_unmatched += 1
                            
                    except Exception as e:
                        print(f"Error in loss computation for object: {e}")
                        continue
        
        if num_objects > 0:
            cls_loss_avg = cls_loss_total / num_objects
            reg_loss_avg = reg_loss_total / max(num_matched, 1)
            total_loss = cls_loss_avg + self.config['reg_loss_weight'] * reg_loss_avg
        else:
            # Create tensors on the correct device
            total_loss = torch.tensor(0.001, requires_grad=True, device=self.device)
            cls_loss_avg = torch.tensor(0.0, device=self.device)
            reg_loss_avg = torch.tensor(0.0, device=self.device)
        
        # Print detailed matching statistics
        if hasattr(self, '_loss_call_count'):
            self._loss_call_count += 1
        else:
            self._loss_call_count = 1
            
        if self._loss_call_count % 100 == 0:
            print(f"Matching stats: IoU={matching_stats['iou_matches']}, "
                  f"Distance={matching_stats['distance_matches']}, "
                  f"No match={matching_stats['no_matches']}, "
                  f"Total predictions={num_objects + num_unmatched}")
            
        return total_loss, cls_loss_avg, reg_loss_avg

    def _map_2d_to_3d_label(self, label_2d):
        """Map 2D detection labels to 3D class indices"""
        mapping = {
            1: 0,  # person -> PEDESTRIAN
            3: 1,  # car -> REGULAR_VEHICLE
            6: 2,  # bus -> LARGE_VEHICLE
            8: 3   # truck -> TRUCK
        }
        return mapping.get(label_2d, 0)  # Default to PEDESTRIAN
    
    def process_batch(self, batch):
        """Process a batch through the entire pipeline"""
        # Move data to device
        batch['points'] = batch['points'].to(self.device)
        for camera_name in batch['images'].keys():
            batch['images'][camera_name] = batch['images'][camera_name].to(self.device)
        
        # Move camera calibrations to device
        for batch_idx in range(len(batch['camera_calibrations'])):
            for camera_name in batch['camera_calibrations'][batch_idx].keys():
                batch['camera_calibrations'][batch_idx][camera_name]['intrinsic'] = \
                    batch['camera_calibrations'][batch_idx][camera_name]['intrinsic'].to(self.device)
                batch['camera_calibrations'][batch_idx][camera_name]['extrinsic'] = \
                    batch['camera_calibrations'][batch_idx][camera_name]['extrinsic'].to(self.device)
        
        # Extract 2D bounding boxes (now both model and data are on same device)
        all_camera_detections = {}
        for camera_name, images in batch['images'].items():
            with torch.no_grad():
                detections = self.bbox_extractor(images)
            all_camera_detections[camera_name] = detections
        
        batch['detections'] = all_camera_detections
        
        # Allocate 3D points to 2D boxes (now all on same device)
        with torch.no_grad():
            allocated_points = self.point_allocator(
                batch['detections'], 
                batch['points'], 
                batch['camera_calibrations']
            )
        
        batch['allocated_points'] = allocated_points
        
        # Forward pass through PointFusion
        predictions = self.model(batch)
        
        return predictions
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0
        successful_batches = 0
        
        # Reset metrics
        self.train_map.reset()
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            try:
                self.optimizer.zero_grad()
                
                # Process batch
                predictions = self.process_batch(batch)
                
                # Check if we have any predictions
                total_predictions = sum(len(camera_preds) for camera_preds in predictions.values() 
                                      for camera_preds in camera_preds)
                
                if total_predictions == 0:
                    print("No predictions in batch, skipping...")
                    continue
                
                # Compute loss
                loss, cls_loss, reg_loss = self.compute_loss(predictions, batch['annotations'])
                
                # Backward pass
                if loss.requires_grad and loss.item() > 0:
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    successful_batches += 1
                
                # Update metrics (only if we have valid predictions)
                try:
                    self.train_map.add_batch(predictions, batch['annotations'])
                except Exception as e:
                    print(f"Error updating metrics: {e}")
                
                # Accumulate losses
                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item() if hasattr(cls_loss, 'item') else float(cls_loss)
                epoch_reg_loss += reg_loss.item() if hasattr(reg_loss, 'item') else float(reg_loss)
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Cls': f"{cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss:.4f}",
                    'Reg': f"{reg_loss.item() if hasattr(reg_loss, 'item') else reg_loss:.4f}",
                    'Success': f"{successful_batches}/{num_batches}"
                })
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute epoch metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_cls_loss = epoch_cls_loss / max(num_batches, 1)
        avg_reg_loss = epoch_reg_loss / max(num_batches, 1)
        
        try:
            train_map_score = self.train_map.compute()
        except Exception as e:
            print(f"Error computing training mAP: {e}")
            train_map_score = 0.0
        
        print(f"Successful batches: {successful_batches}/{num_batches}")
        return avg_loss, avg_cls_loss, avg_reg_loss, train_map_score
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0
        
        # Reset metrics
        self.val_map.reset()
        
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    # Process batch
                    predictions = self.process_batch(batch)
                    
                    # Compute loss
                    loss, cls_loss, reg_loss = self.compute_loss(predictions, batch['annotations'])
                    
                    # Update metrics
                    self.val_map.add_batch(predictions, batch['annotations'])
                    
                    # Accumulate losses
                    epoch_loss += loss.item()
                    epoch_cls_loss += cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss
                    epoch_reg_loss += reg_loss.item() if hasattr(reg_loss, 'item') else reg_loss
                    num_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'Val Loss': f"{loss.item():.4f}",
                        'Cls': f"{cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss:.4f}",
                        'Reg': f"{reg_loss.item() if hasattr(reg_loss, 'item') else reg_loss:.4f}"
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        # Compute epoch metrics
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_cls_loss = epoch_cls_loss / max(num_batches, 1)
        avg_reg_loss = epoch_reg_loss / max(num_batches, 1)
        val_map_score = self.val_map.compute()
        
        return avg_loss, avg_cls_loss, avg_reg_loss, val_map_score
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_maps': self.val_maps
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with mAP: {self.val_maps[-1]:.4f}")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        best_val_map = 0.0
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Training
            train_loss, train_cls_loss, train_reg_loss, train_map = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_cls_loss, val_reg_loss, val_map = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_maps.append(val_map)
            
            # Print epoch results
            print(f"Train - Loss: {train_loss:.4f}, Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f}, mAP: {train_map:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Cls: {val_cls_loss:.4f}, Reg: {val_reg_loss:.4f}, mAP: {val_map:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log to wandb if available
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_cls_loss': train_cls_loss,
                    'train_reg_loss': train_reg_loss,
                    'train_map': train_map,
                    'val_loss': val_loss,
                    'val_cls_loss': val_cls_loss,
                    'val_reg_loss': val_reg_loss,
                    'val_map': val_map,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_map > best_val_map
            if is_best:
                best_val_map = val_map
            
            if (epoch + 1) % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        print(f"\nTraining completed! Best validation mAP: {best_val_map:.4f}")
        return self.train_losses, self.val_losses, self.val_maps

if __name__ == "__main__":
    # Configuration
    config = {
        'num_classes': 4,
        'point_feature_dim': 256,
        'img_feature_dim': 256,
        'bbox_score_thresh': 0.5,
        'max_points_per_box': 1000,
        'min_points_per_box': 10,
        'matching_distance_threshold': 5.0,  # Distance threshold for GT matching in meters
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'reg_loss_weight': 1.0,
        'scheduler_step': 10,
        'scheduler_gamma': 0.5,
        'epochs': 10,
        'batch_size': 4,
        'save_every': 5,
        'save_dir': 'checkpoints',
        'use_wandb': False
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize wandb if requested
    if config['use_wandb']:
        wandb.init(
            project="pointfusion-training",
            config=config,
            name=f"pointfusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Load dataset
    load_dotenv()
    data_path = os.getenv('DATA_PATH', default='src/data/')
    full_dataset = PointFusionloader(data_path, split='train')
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=42
    )
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_set, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate
    )
    
    print(f"Dataset loaded: {len(train_set)} train, {len(val_set)} val samples")
    
    # Initialize trainer
    trainer = PointFusionTrainer(config)
    
    # Start training
    try:
        train_losses, val_losses, val_maps = trainer.train(train_loader, val_loader)
        
        # Plot training curves
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(val_maps, label='Val mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.title('Validation mAP')
        
        plt.subplot(1, 3, 3)
        plt.plot([trainer.optimizer.param_groups[0]['lr']] * len(train_losses))
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
        plt.show()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if config['use_wandb']:
            wandb.finish()

# python -m src.training.train_PointFusion