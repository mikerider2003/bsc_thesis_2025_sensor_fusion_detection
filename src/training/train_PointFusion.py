
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate



def main():
    pass

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    data_path = os.getenv('DATA_PATH', default='src/data/')

    train_set = PointFusionloader(data_path, split='train')
    
    # TODO: REMOVE BEFORE FINAL TRAINING
    # Limit the number of samples to process
    indices = list(range(100))
    train_set = Subset(train_set, indices)

    # Use sklearn to split the dataset into train and validation sets
    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=custom_collate)

    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")

# python -m src.training.train_PointFusion

