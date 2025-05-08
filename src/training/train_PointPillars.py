import os
import torch
import numpy as np
import time
import argparse
import joblib
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dotenv import load_dotenv

from src.models.PointPillars import PointPillarsModel, PointPillarsLoss
from src.loaders.loader_Point_Pillars import PointPillarsLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train PointPillars model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--metrics_dir', type=str, default='outputs/metrics', help='Directory to save training metrics')
    parser.add_argument('--model_dir', type=str, default='outputs/models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(val_ratio=0.2, batch_size=1, seed=42, target_classes=None):
    """Create train and validation data loaders with sklearn's train_test_split"""
    # Load environment variables
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset
    if target_classes is not None:
        train_dataset = PointPillarsLoader(dataset_path, split='train', target_classes=target_classes)
    else:
        raise Exception("Target classes must be provided for the dataset.")
    
    # Process all samples (or a subset for faster development)
    processed_samples = train_dataset.process_all_samples(limit=16)
    
    # Get indices for train/validation split
    indices = list(range(len(processed_samples)))
    train_indices, val_indices = train_test_split(indices, test_size=val_ratio, random_state=seed)
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for the point pillars data loader"""
    pillars_list = []
    coords_list = []
    targets_list = []  # Keep targets as a list of lists

    for i, sample in enumerate(batch):
        pillar_data, coords_data = sample["lidar_processed"]

        # Convert to tensors
        pillars = torch.from_numpy(pillar_data).float()
        coords = torch.from_numpy(coords_data).int()

        # Add batch index to coords
        batch_size = coords.shape[0]
        batch_index = torch.full((batch_size, 1), i, dtype=torch.int)
        coords = torch.cat([batch_index, coords], dim=1)

        pillars_list.append(pillars)
        coords_list.append(coords)
        targets_list.append(sample["annotations"])  # Keep targets as is

    # Pad to the maximum number of pillars
    max_pillars = max(pillar.shape[0] for pillar in pillars_list)

    padded_pillars = []
    padded_coords = []

    for pillars, coords in zip(pillars_list, coords_list):
        num_padding = max_pillars - pillars.shape[0]

        # Pad pillars and coords
        padded_pillars.append(torch.cat([pillars, torch.zeros((num_padding, pillars.shape[1], pillars.shape[2]))], dim=0))
        padded_coords.append(torch.cat([coords, torch.zeros((num_padding, coords.shape[1]), dtype=torch.int)], dim=0))

    # Stack the padded tensors
    pillars_stacked = torch.stack(padded_pillars)
    coords_stacked = torch.stack(padded_coords)

    return {
        "pillars": pillars_stacked,
        "coords": coords_stacked,
        "targets": targets_list  # Return targets as a list of lists
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        pillars = batch["pillars"].to(device)
        coords = batch["coords"].to(device)
        targets = batch["targets"]  # List of targets for each sample
        
        # Forward pass
        bbox_preds, cls_scores, dir_scores = model(pillars, coords)
        
        # TODO: Convert targets to expected format for loss calculation
        # This depends on how you've structured your targets and loss function
        pred_boxes = ...  # Extract from model outputs
        gt_boxes = ...  # Extract from targets
        pred_cls = ...
        gt_cls = ...
        pred_dir = ...
        gt_dir = ...
        pos_mask = ...  # Positive anchor mask
        
        # Calculate loss
        loss = criterion(pred_boxes, gt_boxes, pred_cls, gt_cls, pred_dir, gt_dir, pos_mask)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({"batch_loss": loss.item()})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    progress_bar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            pillars = batch["pillars"].to(device)
            coords = batch["coords"].to(device)
            targets = batch["targets"]
            
            # Forward pass
            bbox_preds, cls_scores, dir_scores = model(pillars, coords)
            
            # TODO: Convert targets to expected format for loss calculation
            # Similar to the training function
            pred_boxes = ...
            gt_boxes = ...
            pred_cls = ...
            gt_cls = ...
            pred_dir = ...
            gt_dir = ...
            pos_mask = ...
            
            # Calculate loss
            loss = criterion(pred_boxes, gt_boxes, pred_cls, gt_cls, pred_dir, gt_dir, pos_mask)
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": loss.item()})
    
    return total_loss / len(val_loader)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Initialize metrics dictionary to store training history
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'epochs': [],
        'best_val_loss': float('inf'),
        'best_epoch': -1
    }
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        val_ratio=args.val_ratio, 
        batch_size=args.batch_size,
        seed=args.seed,
        target_classes={'PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE'}
    )

    # for batch_idx, batch in enumerate(train_loader):
    #     print("\nInspecting batch:", batch_idx)
    #     print(f"Pillars shape: {batch['pillars'].shape}")
    #     print(f"Coords shape: {batch['coords'].shape}")
    #     print(f"Number of targets: {len(batch['targets'])}")

    # for batch_idx, batch in enumerate(val_loader):
    #     print("\nInspecting validation batch:", batch_idx)
    #     print(f"Pillars shape: {batch['pillars'].shape}")
    #     print(f"Coords shape: {batch['coords'].shape}")
    #     print(f"Number of targets: {len(batch['targets'])}")
    
    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.dataset.classes) + 1  # +1 for background
    
    # Initialize model and loss
    model = PointPillarsModel(num_classes=num_classes).to(device)
    criterion = PointPillarsLoss().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log metrics
        print(f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Update metrics dictionary
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['learning_rate'].append(current_lr)
        metrics['epochs'].append(epoch + 1)
        
        # Save metrics to disk
        joblib.dump(metrics, os.path.join(args.metrics_dir, 'training_metrics.joblib'))
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            metrics['best_val_loss'] = best_val_loss
            metrics['best_epoch'] = epoch + 1
            
            # Save the best model
            print(f"New best validation loss: {val_loss:.6f}, saving model...")
            model_path = os.path.join(args.model_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'num_classes': num_classes
            }, model_path)
    
    # Save final metrics
    joblib.dump(metrics, os.path.join(args.metrics_dir, 'final_metrics.joblib'))
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'num_classes': num_classes
    }, final_model_path)
    
    print("Training complete!")
    print(f"Best model achieved validation loss of {metrics['best_val_loss']:.6f} at epoch {metrics['best_epoch']}")
    print(f"Best model saved to {os.path.join(args.model_dir, 'best_model.pth')}")
    print(f"Final model saved to {os.path.join(args.model_dir, 'final_model.pth')}")


if __name__ == "__main__":
    main()

# python -m src.training.train_PointPillars