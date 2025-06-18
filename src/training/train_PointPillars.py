import os
import torch
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gc
from dotenv import load_dotenv
from torch.utils.data import random_split

# Import your custom modules
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
        self.val_losses = {
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
    
    def update_val(self, val_dict):
        """Update validation loss history."""
        for key, value in val_dict.items():
            if key in self.val_losses:
                self.val_losses[key].append(value)
    
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
    
    def plot_losses(self, save_path=None):
        """Plot training and validation losses."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(self.epoch_losses['total_loss']) + 1)
        
        # Total Loss
        axes[0, 0].plot(epochs, self.epoch_losses['total_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses['total_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification Loss
        axes[0, 1].plot(epochs, self.epoch_losses['cls_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_losses['cls_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box Regression Loss
        axes[1, 0].plot(epochs, self.epoch_losses['box_loss'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.val_losses['box_loss'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('Box Regression Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of Positive Samples
        axes[1, 1].plot(epochs, self.epoch_losses['num_positives'], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs, self.val_losses['num_positives'], 'r-', label='Validation', linewidth=2)
        axes[1, 1].set_title('Positive Samples per Batch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
        plt.close()

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, checkpoint_path):
    """Save model checkpoint."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path, device):
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Checkpoint loaded: {checkpoint_path}, resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0

def validate_model(model, val_loader, criterion, device):
    """Validate the model on validation set."""
    model.eval()
    val_losses = {
        'total_loss': [],
        'cls_loss': [],
        'box_loss': [],
        'num_positives': []
    }
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for batch in pbar:
            # Move batch to device with non-blocking
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Generate anchor assignments
            batch_assignments = model.anchor_generator.assign(batch['annotations'])
            
            # Forward pass
            predictions = model(batch)
            
            # Compute loss with correct arguments
            loss_dict = criterion(predictions, batch_assignments, batch['annotations'])
            
            # Track losses
            for key in val_losses:
                if key in loss_dict:
                    val_losses[key].append(loss_dict[key].item() if isinstance(
                        loss_dict[key], torch.Tensor) else loss_dict[key])
    
    # Calculate average losses
    avg_losses = {key: np.mean(values) for key, values in val_losses.items()}
    return avg_losses

def main():
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Validate dataset paths
    train_path = os.path.join(dataset_path, 'train')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    # Training configuration
    config = {
        'batch_size': 2,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'grid_size': (300, 200),
        'grid_resolution': 0.2,
        'num_classes': 4,
        'save_every': 5,
        'validate_every': 1,  # Validate every epoch
        'checkpoint_dir': 'checkpoints/',
        'use_wandb': False,
        'use_amp': True,  # Enable Automatic Mixed Precision
        'grad_clip': 10.0,
        'num_workers': min(4, os.cpu_count() // 2),  # Adaptive workers
        'validation_split': 0.2,  # 20% of training data for validation
        'random_seed': 42,  # For reproducible train/val split
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create full training dataset
    print("Creating datasets...")
    full_train_dataset = PointPillarsLoader(dataset_path, split='train')
    
    # Split into training and validation sets
    val_size = int(len(full_train_dataset) * config['validation_split'])
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['random_seed'])
    )
    
    print(f"Full training dataset size: {len(full_train_dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Initialize model
    print("Initializing model...")
    model = PointPillarsModel(
        grid_size=config['grid_size'], 
        num_classes=config['num_classes'],
        grid_resolution=config['grid_resolution']
    ).to(device)
    
    # Handle anchors: Move to device and ensure they're tensors
    anchors = model.anchor_generator.anchors
    if isinstance(anchors, torch.Tensor):
        model.anchor_generator.anchors = anchors.to(device)
    else:
        # Convert to tensor if needed
        model.anchor_generator.anchors = torch.tensor(
            anchors, device=device, dtype=torch.float32
        )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize loss, optimizer, and AMP scaler
    criterion = PointPillarsLoss(
        cls_weight=1.0,
        box_weight=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] and device.type == 'cuda' else None
    
    # Loss tracker
    loss_tracker = LossTracker()
    
    # Checkpoint paths
    best_checkpoint = os.path.join(config['checkpoint_dir'], 'best.pth')
    latest_checkpoint = os.path.join(config['checkpoint_dir'], 'latest.pth')
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(
        model, optimizer, scheduler, scaler, 
        latest_checkpoint, device
    )
    
    # Initialize Weights & Biases if enabled
    if config['use_wandb']:
        import wandb
        wandb.init(project="pointpillars")
        wandb.config.update(config)
        wandb.watch(model, log="all")
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")
        model.train()
        epoch_start_time = time.time()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        batch_losses = []
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device with non-blocking
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            try:
                # Generate anchor assignments
                batch_assignments = model.anchor_generator.assign(batch['annotations'])
                
                with torch.amp.autocast(device_type=device.type, enabled=config['use_amp'] and device.type == 'cuda'):
                    predictions = model(batch)
                    loss_dict = criterion(predictions, batch_assignments, batch['annotations'])
                    total_loss = loss_dict['total_loss']
                
                # Skip batch if NaN loss
                if torch.isnan(total_loss).any().item():
                    print(f"Warning: NaN loss detected at batch {batch_idx} - skipping")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Backward pass with AMP
                if scaler:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config['grad_clip']
                )
                
                # Update weights
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Track losses
                loss_tracker.update(loss_dict)
                batch_losses.append(total_loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Cls': f"{loss_dict['cls_loss'].item():.4f}",
                    'Box': f"{loss_dict['box_loss'].item():.4f}",
                })
                
                # Periodically clear memory
                if batch_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad(set_to_none=True)
                continue
        
        # End of epoch processing
        epoch_time = time.time() - epoch_start_time
        loss_tracker.end_epoch()
        epoch_losses = loss_tracker.get_latest_epoch_avg()
        
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Avg Loss: {epoch_losses['total_loss']:.4f}, "
              f"Cls: {epoch_losses['cls_loss']:.4f}, "
              f"Box: {epoch_losses['box_loss']:.4f}, "
              f"Pos: {epoch_losses['num_positives']:.1f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Validation
        if (epoch + 1) % config['validate_every'] == 0:
            print("\nRunning validation...")
            val_losses = validate_model(model, val_loader, criterion, device)
            val_loss = val_losses['total_loss']
            
            print(f"Validation Loss: {val_loss:.4f}, "
                  f"Cls: {val_losses['cls_loss']:.4f}, "
                  f"Box: {val_losses['box_loss']:.4f}, "
                  f"Pos: {val_losses['num_positives']:.1f}")
            
            # Track validation losses
            loss_tracker.update_val(val_losses)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, scaler, 
                    epoch + 1, val_loss, best_checkpoint
                )
                print(f"New best validation loss: {val_loss:.4f}")
            
            # Log to wandb
            if config['use_wandb']:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_total_loss": epoch_losses['total_loss'],
                    "train_cls_loss": epoch_losses['cls_loss'],
                    "train_box_loss": epoch_losses['box_loss'],
                    "val_total_loss": val_loss,
                    "val_cls_loss": val_losses['cls_loss'],
                    "val_box_loss": val_losses['box_loss'],
                    "lr": current_lr
                })
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0 or epoch == config['epochs'] - 1:
            epoch_checkpoint = os.path.join(
                config['checkpoint_dir'], 
                f'pointpillars_epoch_{epoch+1}.pth'
            )
            save_checkpoint(
                model, optimizer, scheduler, scaler, 
                epoch + 1, epoch_losses['total_loss'], epoch_checkpoint
            )
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler, 
            epoch + 1, epoch_losses['total_loss'], latest_checkpoint
        )
    
    print("\nTraining completed!")
    
    # Plot and save training losses
    plot_path = os.path.join(config['checkpoint_dir'], 'training_losses.png')
    loss_tracker.plot_losses(save_path=plot_path)
    
    # Final model save
    final_checkpoint = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    save_checkpoint(
        model, optimizer, scheduler, scaler, 
        config['epochs'], loss_tracker.epoch_losses, final_checkpoint
    )
    print(f"Final model saved: {final_checkpoint}")
    
    # Save loss history
    loss_history_path = os.path.join(config['checkpoint_dir'], 'loss_history.npy')
    np.save(loss_history_path, {
        'train': loss_tracker.epoch_losses,
        'val': loss_tracker.val_losses
    })
    print(f"Loss history saved: {loss_history_path}")
    
    # Finish wandb
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()
    
# python -m src.training.train_PointPillars