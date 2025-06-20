# src/training/train_PointFusion.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from dotenv import load_dotenv

from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
from src.models.PointFusion import PointFusion, MeanAveragePrecision3D, BoundingBoxExtractor, Allocate3dPoints

if __name__ == "__main__":
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
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, collate_fn=custom_collate)

# python -m src.training.train_PointFusion