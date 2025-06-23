# src/training/train_PointFusion.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import numpy as np
from dotenv import load_dotenv
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt

from src.loaders.loader_Point_Fusion import PointFusionloader
from src.models.PointFusion import Preprocessor, PointFusion


class PointFusionDataset(Dataset):
    """Dataset wrapper for preprocessed PointFusion data"""
    def __init__(self, preprocessor):
        self.rolls = preprocessor.rolls
        self.point_clouds = preprocessor.point_clouds
        self.labels = preprocessor.labels
        self.annotations = preprocessor.annotations
        
        # Create label mapping
        self.label_to_idx = {
            'PEDESTRIAN': 0,
            'REGULAR_VEHICLE': 1,
            'LARGE_VEHICLE': 2,
            'TRUCK': 3
        }
        
    def __len__(self):
        return len(self.rolls)
    
    def __getitem__(self, idx):
        # Get data
        roll = self.rolls[idx]  # [3, 224, 224]
        point_cloud = self.point_clouds[idx]  # [N, 3] or [max_points, 3]
        label = self.labels[idx]  # string
        annotation = self.annotations[idx]  # dict with 'boxes' and 'labels'
        
        # Convert label to index
        label_idx = self.label_to_idx.get(label, 0)
        
        # Ensure point cloud has consistent shape (pad or truncate to 400 points)
        max_points = 400
        if len(point_cloud) == 0:
            # Empty point cloud - fill with zeros
            point_cloud = torch.zeros(max_points, 3)
        elif len(point_cloud) < max_points:
            # Pad with zeros if too few points
            padding = torch.zeros(max_points - len(point_cloud), 3)
            point_cloud = torch.cat([point_cloud, padding], dim=0)
        elif len(point_cloud) > max_points:
            # Randomly sample if too many points
            indices = torch.randperm(len(point_cloud))[:max_points]
            point_cloud = point_cloud[indices]
        
        return {
            'image': roll.float(),
            'point_cloud': point_cloud.float(),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'annotation': annotation
        }


class PointFusionLoss(nn.Module):
    """
    Loss function for PointFusion model. 
    Since GT assignment is now done during preprocessing, each sample has exactly one matched GT box and label.
    """
    def __init__(self, alpha=1.0, beta=10.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta   # Weight for regression loss  
        self.gamma = gamma # Weight for confidence loss
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.SmoothL1Loss()
        self.confidence_loss = nn.BCELoss()
        
    def forward(self, predictions, labels, annotations):
        """
        Compute loss for PointFusion predictions
        Args:
            predictions: dict with 'cls_scores', 'bbox_pred', 'confidence'
            labels: [B] tensor of ground truth class labels (already matched during preprocessing)
            annotations: list of annotation dicts (each contains single matched box and label)
        Returns:
            total_loss: scalar loss value
            loss_dict: dict with individual loss components
        """
        # Extract predictions
        cls_scores = predictions['cls_scores']  # [B, num_classes]
        bbox_pred = predictions['bbox_pred']    # [B, 7]
        confidence = predictions['confidence']  # [B, 1]
        
        batch_size = cls_scores.shape[0]
        device = cls_scores.device
        
        # Extract ground truth boxes from annotations (already matched during preprocessing)
        gt_boxes = []
        valid_samples = []
        
        for i, annotation in enumerate(annotations):
            if annotation['boxes'].shape[0] > 0:
                # Each annotation should have exactly one box after preprocessing
                gt_box = annotation['boxes'][0]  # [7] - single matched box
                if isinstance(gt_box, torch.Tensor):
                    gt_boxes.append(gt_box.to(device))
                else:
                    gt_boxes.append(torch.tensor(gt_box, device=device, dtype=torch.float32))
                valid_samples.append(i)
            else:
                # Skip samples with no valid ground truth
                continue
        
        if len(valid_samples) == 0:
            # No valid samples - return zero loss
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return total_loss, {'cls_loss': 0.0, 'reg_loss': 0.0, 'conf_loss': 0.0}
        
        # Convert to tensors and filter valid samples
        valid_indices = torch.tensor(valid_samples, device=device)
        gt_boxes_tensor = torch.stack(gt_boxes)  # [N_valid, 7]
        
        # Filter predictions and labels to only valid samples
        valid_cls_scores = cls_scores[valid_indices]  # [N_valid, num_classes]
        valid_bbox_pred = bbox_pred[valid_indices]    # [N_valid, 7]
        valid_confidence = confidence[valid_indices]  # [N_valid, 1]
        valid_labels = labels[valid_indices]          # [N_valid]
        
        # Classification loss
        cls_loss = self.classification_loss(valid_cls_scores, valid_labels)
        
        # Regression loss (for box parameters)
        reg_loss = self.regression_loss(valid_bbox_pred, gt_boxes_tensor)
        
        # Confidence loss (high confidence for all valid samples since GT is already matched)
        target_confidence = torch.ones_like(valid_confidence)  # High confidence for all matched predictions
        conf_loss = self.confidence_loss(valid_confidence, target_confidence)
        
        # Combine losses
        total_loss = (self.alpha * cls_loss + 
                     self.beta * reg_loss + 
                     self.gamma * conf_loss)
        
        loss_dict = {
            'cls_loss': cls_loss.item(),
            'reg_loss': reg_loss.item(), 
            'conf_loss': conf_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


def collate_fn(batch):
    """Custom collate function for PointFusion data"""
    images = torch.stack([item['image'] for item in batch])
    point_clouds = torch.stack([item['point_cloud'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    annotations = [item['annotation'] for item in batch]
    
    return {
        'images': images,
        'point_clouds': point_clouds,
        'labels': labels,
        'annotations': annotations
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_components = {'cls_loss': 0.0, 'reg_loss': 0.0, 'conf_loss': 0.0}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        images = batch['images'].to(device)
        point_clouds = batch['point_clouds'].to(device)
        labels = batch['labels'].to(device)
        annotations = batch['annotations']
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images, point_clouds)
        
        # Compute loss (now passing labels directly since GT is assigned during preprocessing)
        loss, loss_dict = criterion(predictions, labels, annotations)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        batch_size = len(images)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key] * batch_size
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'cls': f"{loss_dict.get('cls_loss', 0):.4f}",
            'reg': f"{loss_dict.get('reg_loss', 0):.4f}",
            'conf': f"{loss_dict.get('conf_loss', 0):.4f}"
        })
    
    # Average losses
    avg_loss = total_loss / total_samples
    for key in loss_components:
        loss_components[key] /= total_samples
    
    return avg_loss, loss_components


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_components = {'cls_loss': 0.0, 'reg_loss': 0.0, 'conf_loss': 0.0}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            # Move to device
            images = batch['images'].to(device)
            point_clouds = batch['point_clouds'].to(device)
            labels = batch['labels'].to(device)
            annotations = batch['annotations']
            
            # Forward pass
            predictions = model(images, point_clouds)
            
            # Compute loss (now passing labels directly since GT is assigned during preprocessing)
            loss, loss_dict = criterion(predictions, labels, annotations)
            
            # Accumulate metrics
            batch_size = len(images)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key] * batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{loss_dict.get('cls_loss', 0):.4f}",
                'reg': f"{loss_dict.get('reg_loss', 0):.4f}",
                'conf': f"{loss_dict.get('conf_loss', 0):.4f}"
            })
    
    # Average losses
    avg_loss = total_loss / total_samples
    for key in loss_components:
        loss_components[key] /= total_samples
    
    return avg_loss, loss_components


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot and save training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Total loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [loss['total_loss'] for loss in train_losses], 'b-', label='Train')
    plt.plot(epochs, [loss['total_loss'] for loss in val_losses], 'r-', label='Val')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Classification loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [loss['cls_loss'] for loss in train_losses], 'b-', label='Train')
    plt.plot(epochs, [loss['cls_loss'] for loss in val_losses], 'r-', label='Val')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Regression loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [loss['reg_loss'] for loss in train_losses], 'b-', label='Train')
    plt.plot(epochs, [loss['reg_loss'] for loss in val_losses], 'r-', label='Val')
    plt.title('Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Confidence loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [loss['conf_loss'] for loss in train_losses], 'b-', label='Train')
    plt.plot(epochs, [loss['conf_loss'] for loss in val_losses], 'r-', label='Val')
    plt.title('Confidence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Check if preprocessed data exists
    train_data_path = os.path.join(dataset_path, 'processed_train_data.pkl')
    val_data_path = os.path.join(dataset_path, 'processed_val_data.pkl')
    train_data_path_improved = os.path.join(dataset_path, 'processed_train_data_improved.pkl')
    val_data_path_improved = os.path.join(dataset_path, 'processed_val_data_improved.pkl')


    if os.path.exists(train_data_path_improved) and os.path.exists(val_data_path_improved):
        print("Loading existing preprocessed data...")
        train_preprocessor = Preprocessor(load_path=train_data_path_improved)
        val_preprocessor = Preprocessor(load_path=val_data_path_improved)
    elif os.path.exists(train_data_path) and os.path.exists(val_data_path):
        print("Loading existing preprocessed data...")
        train_preprocessor = Preprocessor(load_path=train_data_path)
        train_preprocessor.assign_gt()
        train_preprocessor.save_processed_data(save_path=train_data_path_improved)

        val_preprocessor = Preprocessor(load_path=val_data_path)
        val_preprocessor.assign_gt()
        val_preprocessor.save_processed_data(save_path=val_data_path_improved)

    else:
        print("Creating new preprocessed data...")
        # Load and split dataset
        dataset = PointFusionloader(dataset_path, split='train')
        train_indices, val_indices = train_test_split(
            list(range(len(dataset))), test_size=0.2, random_state=42
        )
        
        # Preprocess training data
        print("Preprocessing training data...")
        train_preprocessor = Preprocessor()
        for i in tqdm(train_indices, desc="Processing train samples"):
            train_preprocessor.process(dataset[i])
        train_preprocessor.remove_empty_point_clouds()
        train_preprocessor.save_processed_data(save_path=train_data_path)
        
        # Preprocess validation data
        print("Preprocessing validation data...")
        val_preprocessor = Preprocessor()
        for i in tqdm(val_indices, desc="Processing val samples"):
            val_preprocessor.process(dataset[i])
        val_preprocessor.remove_empty_point_clouds()
        val_preprocessor.save_processed_data(save_path=val_data_path)
    
    # Create datasets
    train_dataset = PointFusionDataset(train_preprocessor)
    val_dataset = PointFusionDataset(val_preprocessor)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = PointFusion().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss function and optimizer
    criterion = PointFusionLoss(alpha=1.0, beta=10.0, gamma=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training parameters
    num_epochs = 20
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_components = train_epoch(model, train_loader, criterion, optimizer, device)
        train_components['total_loss'] = train_loss
        train_losses.append(train_components)
        
        # Validate
        val_loss, val_components = validate_epoch(model, val_loader, criterion, device)
        val_components['total_loss'] = val_loss
        val_losses.append(val_components)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Components: cls={train_components['cls_loss']:.4f}, "
              f"reg={train_components['reg_loss']:.4f}, conf={train_components['conf_loss']:.4f}")
        print(f"Val Components: cls={val_components['cls_loss']:.4f}, "
              f"reg={val_components['reg_loss']:.4f}, conf={val_components['conf_loss']:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save model checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"New best model saved! Val loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Plot training curves
        if (epoch + 1) % 5 == 0:
            plot_training_curves(
                train_losses, 
                val_losses, 
                os.path.join(save_dir, 'training_curves.png')
            )
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {save_dir}")


# python -m src.training.train_PointFusion