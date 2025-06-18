import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.loaders.loader_Point_Pillars import PointPillarsLoader, collate_fn
from src.models.PointPillars import PointPillarsModel, PointPillarsLoss

class LossTracker:
    """Helper class to track and visualize losses during training."""
    
    def __init__(self):
        self.losses = {
            'total_loss': [],
            'cls_loss': [],
            'box_loss': [],
            'num_positives': []
        }
        self.epoch_losses = {
            'total_loss': [],
            'cls_loss': [],
            'box_loss': [],
            'num_positives': []
        }
    
    def update(self, loss_dict):
        """Update loss history."""
        for key, value in loss_dict.items():
            if key in self.losses:
                if isinstance(value, torch.Tensor):
                    self.losses[key].append(value.item())
                else:
                    self.losses[key].append(value)
    
    def end_epoch(self):
        """Calculate epoch averages and reset batch losses."""
        for key in self.epoch_losses.keys():
            if len(self.losses[key]) > 0:
                self.epoch_losses[key].append(np.mean(self.losses[key]))
                self.losses[key] = []  # Reset for next epoch
    
    def get_latest_epoch_avg(self):
        """Get latest epoch averages."""
        latest = {}
        for key, values in self.epoch_losses.items():
            latest[key] = values[-1] if values else 0.0
        return latest
    
    def plot_losses(self):
        """Plot training losses."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.epoch_losses['total_loss']) + 1)
        
        # Total Loss
        axes[0, 0].plot(epochs, self.epoch_losses['total_loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification Loss
        axes[0, 1].plot(epochs, self.epoch_losses['cls_loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box Regression Loss
        axes[1, 0].plot(epochs, self.epoch_losses['box_loss'], 'g-', linewidth=2)
        axes[1, 0].set_title('Box Regression Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of Positive Samples
        axes[1, 1].plot(epochs, self.epoch_losses['num_positives'], 'm-', linewidth=2)
        axes[1, 1].set_title('Positive Samples per Batch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'pointpillars_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded: {checkpoint_path}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0

def validate_model(model, val_loader, criterion, device):
    """Validate the model on validation set."""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Get anchor assignments
            batch_assignments = model.anchor_generator.assign(batch['annotations'])
            
            # Compute loss
            loss_dict = criterion(predictions, batch_assignments, batch['annotations'])
            val_losses.append(loss_dict['total_loss'].item())
    
    model.train()
    return np.mean(val_losses)

def main():
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Training configuration
    config = {
        'batch_size': 2,  # Reduced for memory efficiency
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grid_size': (300, 200),
        'grid_resolution': 0.2,
        'num_classes': 4,
        'save_every': 5,  # Save checkpoint every 5 epochs
        'validate_every': 5,  # Validate every 5 epochs
        'checkpoint_dir': 'checkpoints/',
        'use_wandb': False,  # Set to True to use Weights & Biases logging
    }
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PointPillarsLoader(dataset_path, split='train')
    val_dataset = PointPillarsLoader(dataset_path, split='val')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize model, loss function, and optimizer
    print("Initializing model...")
    model = PointPillarsModel(
        grid_size=config['grid_size'], 
        num_classes=config['num_classes'],
        grid_resolution=config['grid_resolution']
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = PointPillarsLoss(
        cls_weight=1.0,
        box_weight=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Loss tracker
    loss_tracker = LossTracker()
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer, 
                                 os.path.join(config['checkpoint_dir'], 'latest.pth'))
    
    # Training loop
    print("\nStarting training...")
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")
        
        # Training phase
        model.train()
        epoch_start_time = time.time()
        
        # Progress bar for batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                predictions = model(batch)
                
                # Get anchor assignments
                batch_assignments = model.anchor_generator.assign(batch['annotations'])
                
                # Compute loss
                loss_dict = criterion(predictions, batch_assignments, batch['annotations'])
                total_loss = loss_dict['total_loss']
                
                # Check for NaN loss
                if torch.isnan(total_loss):
                    print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_idx}")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                # Update weights
                optimizer.step()
                
                # Track losses
                loss_tracker.update(loss_dict)
                
                # Update progress bar
                pbar.set_postfix({
                    'Total': f"{loss_dict['total_loss'].item():.4f}",
                    'Cls': f"{loss_dict['cls_loss'].item():.4f}",
                    'Box': f"{loss_dict['box_loss'].item():.4f}",
                    'Pos': f"{loss_dict['num_positives']}"
                })
                
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # End of epoch processing
        loss_tracker.end_epoch()
        epoch_time = time.time() - epoch_start_time
        
        # Get epoch averages
        epoch_losses = loss_tracker.get_latest_epoch_avg()
        
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average losses - Total: {epoch_losses['total_loss']:.4f}, "
              f"Cls: {epoch_losses['cls_loss']:.4f}, "
              f"Box: {epoch_losses['box_loss']:.4f}, "
              f"Pos: {epoch_losses['num_positives']:.1f}")
        
        # Update learning rate
        scheduler.step()
        
        # Validation
        if (epoch + 1) % config['validate_every'] == 0:
            print("Running validation...")
            val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, 
                               os.path.join(config['checkpoint_dir'], 'best.pth'))
                print(f"New best validation loss: {val_loss:.4f}")
            
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, epoch_losses['total_loss'], 
                           config['checkpoint_dir'])
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, epoch, epoch_losses['total_loss'], 
                       os.path.join(config['checkpoint_dir'], 'latest.pth'))
    
    print("\nTraining completed!")
    
    # Plot training losses
    print("Plotting training losses...")
    loss_tracker.plot_losses()
    
    # Final model save
    final_checkpoint = {
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss_history': loss_tracker.epoch_losses
    }
    
    final_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    print(f"Final model saved: {final_path}")
    
if __name__ == "__main__":
    main()

# python -m src.training.train_PointPillars