# src/training/train_PointFusion.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from dotenv import load_dotenv

from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
from src.models.PointFusion import PointFusion3D, PointFusionLoss, HungarianMatcher

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    lr = 1e-4
    max_predictions = 128  # Should match your dataset's maximum annotations + buffer
    
    # Initialize model and loss
    model = PointFusion3D(max_predictions=max_predictions).to(device)
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_heading=2)  # New
    criterion = PointFusionLoss(matcher).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Load dataset
    load_dotenv()
    data_path = os.getenv('DATA_PATH', default='src/data/')
    full_dataset = PointFusionloader(data_path, split='train')
    
    # # TODO: Temporary subset for development
    # indices = list(range(100))  # Remove this for full training
    # full_dataset = Subset(full_dataset, indices)
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=42
    )
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=custom_collate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in progress_bar:
            # Move data to device
            points = batch['points'].to(device)
            images = {k: v.to(device) for k, v in batch['images'].items()}
            annotations = [
                {
                    'boxes': a['boxes'].to(device),
                    'labels': a['labels'].to(device)
                } for a in batch['annotations']
            ]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model({'points': points, 'images': images})

            # Calculate loss
            loss = criterion(outputs, annotations)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                points = batch['points'].to(device)
                images = {k: v.to(device) for k, v in batch['images'].items()}
                annotations = [
                    {
                        'boxes': a['boxes'].to(device),
                        'labels': a['labels'].to(device)
                    } for a in batch['annotations']
                ]

                outputs = model({'points': points, 'images': images})
                loss = criterion(outputs, annotations)
                val_loss += loss.item()

        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "outputs/best_model_point_fusion.pth")
            print(f"Saved new best model with val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()

# python -m src.training.train_PointFusion