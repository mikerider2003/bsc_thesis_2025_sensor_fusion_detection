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


class AnchorGenerator:
    """Generates anchors for PointPillars detection"""
    
    def __init__(self, x_range, y_range, z_range, voxel_size, anchor_sizes, rotations=(0, np.pi/2)):
        """
        Args:
            x_range: (min, max) in meters
            y_range: (min, max) in meters
            z_range: (min, max) in meters
            voxel_size: (x_size, y_size) in meters
            anchor_sizes: list of (length, width, height) for each class
            rotations: list of rotations to apply to anchors
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.voxel_size = voxel_size
        self.anchor_sizes = anchor_sizes
        self.rotations = rotations
        
        # Calculate grid dimensions
        self.nx = int((x_range[1] - x_range[0]) / voxel_size[0] / 4)  # /4 due to backbone downsampling
        self.ny = int((y_range[1] - y_range[0]) / voxel_size[1] / 4)  # /4 due to backbone downsampling
        
        # Generate anchors
        self.anchors = self._generate_anchors()
        
    def _generate_anchors(self):
        """Generate anchor boxes for all positions in the feature map"""
        anchors = []
        x_centers = np.linspace(self.x_range[0], self.x_range[1], self.nx)
        y_centers = np.linspace(self.y_range[0], self.y_range[1], self.ny)
        
        # Use mean of z_range as anchor height
        z_center = (self.z_range[0] + self.z_range[1]) / 2
        
        # Generate anchors for each position
        for x in x_centers:
            for y in y_centers:
                for size in self.anchor_sizes:
                    length, width, height = size
                    for rotation in self.rotations:
                        anchors.append([x, y, z_center, length, width, height, rotation])
        
        return np.array(anchors)
    
    def encode_targets(self, batch_annotations):
        """
        Encode ground truth boxes to match anchor format
        
        Returns:
            gt_boxes: tensor of shape [batch_size, num_anchors, 7]
            gt_classes: tensor of shape [batch_size, num_anchors]
            gt_directions: tensor of shape [batch_size, num_anchors]
            pos_mask: tensor of shape [batch_size, num_anchors]
        """
        batch_size = len(batch_annotations)
        num_anchors = len(self.anchors)
        
        gt_boxes = torch.zeros((batch_size, num_anchors, 7), dtype=torch.float32)
        gt_classes = torch.zeros((batch_size, num_anchors), dtype=torch.long)
        gt_directions = torch.zeros((batch_size, num_anchors), dtype=torch.long)
        pos_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool)
        
        # For each batch item
        for b, annotations in enumerate(batch_annotations):
            # Skip if no annotations
            if annotations is None or len(annotations) == 0:
                continue
                
            # Convert annotations to tensor format [N, 7] - (x, y, z, l, w, h, yaw)
            num_gt = len(annotations)
            gt_boxes_raw = torch.zeros((num_gt, 7), dtype=torch.float32)
            
            # Extract values from annotations
            for i, anno in enumerate(annotations.iterrows()):
                anno = anno[1]  # Get the pandas Series from the tuple
                
                # Get box center and dimensions
                x, y, z = anno['tx_m'], anno['ty_m'], anno['tz_m']
                l, w, h = anno['length_m'], anno['width_m'], anno['height_m']
                
                # Convert quaternion to yaw (rotation around z-axis)
                qw, qx, qy, qz = anno['qw'], anno['qx'], anno['qy'], anno['qz']
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
                
                # Store in tensor
                gt_boxes_raw[i] = torch.tensor([x, y, z, l, w, h, yaw])
                
                # Get category ID
                category = anno['category']
                category_id = self._get_category_id(category) + 1  # Add 1 to skip background (0)
                
                # Find matching anchors (IoU > threshold)
                # For simplicity, assign to closest anchor
                ious = self._calculate_iou_2d(self.anchors, gt_boxes_raw[i].numpy())
                max_iou_idx = np.argmax(ious)
                
                # Assign ground truth to this anchor
                gt_boxes[b, max_iou_idx] = gt_boxes_raw[i]
                gt_classes[b, max_iou_idx] = category_id
                
                # Determine direction class (0: forward, 1: backward)
                # This is a simplified approach; you may need to adjust based on your dataset
                gt_directions[b, max_iou_idx] = 0 if yaw > 0 else 1
                
                # Mark this anchor as positive
                pos_mask[b, max_iou_idx] = True
            
        return gt_boxes, gt_classes, gt_directions, pos_mask
    
    def _calculate_iou_2d(self, anchors, gt_box):
        """Vectorized IoU calculation between anchors and a ground truth box (2D)."""
        anchors = np.array(anchors)
        gt_box = np.array(gt_box)

        # Anchor corners
        x1_anchors = anchors[:, 0] - anchors[:, 3] / 2
        y1_anchors = anchors[:, 1] - anchors[:, 4] / 2
        x2_anchors = anchors[:, 0] + anchors[:, 3] / 2
        y2_anchors = anchors[:, 1] + anchors[:, 4] / 2

        # Ground truth box corners
        x1_gt = gt_box[0] - gt_box[3] / 2
        y1_gt = gt_box[1] - gt_box[4] / 2
        x2_gt = gt_box[0] + gt_box[3] / 2
        y2_gt = gt_box[1] + gt_box[4] / 2

        # Intersection
        x1_inter = np.maximum(x1_anchors, x1_gt)
        y1_inter = np.maximum(y1_anchors, y1_gt)
        x2_inter = np.minimum(x2_anchors, x2_gt)
        y2_inter = np.minimum(y2_anchors, y2_gt)

        inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)

        # Union
        anchor_area = (x2_anchors - x1_anchors) * (y2_anchors - y1_anchors)
        gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = anchor_area + gt_area - inter_area

        # IoU
        iou = inter_area / np.maximum(union_area, 1e-6)
        return iou
        
    def _get_category_id(self, category):
        """Convert category string to ID based on the full dataset classes"""
        categories = {
            'REGULAR_VEHICLE': 0,
            'PEDESTRIAN': 1, 
            'BICYCLE': 2,
            'BICYCLIST': 3,
            'MOTORCYCLE': 4,
            'MOTORCYCLIST': 5,
            'BOX_TRUCK': 6,
            'TRUCK': 7,
            'TRUCK_CAB': 8,
            'LARGE_VEHICLE': 9,
            'BUS': 10,
            'ARTICULATED_BUS': 11,
            'VEHICULAR_TRAILER': 12,
            'CONSTRUCTION_CONE': 13,
            'CONSTRUCTION_BARREL': 14,
            'SIGN': 15,
            'STOP_SIGN': 16,
            'BOLLARD': 17,
            'OFFICIAL_SIGNALER': 18,
            'STROLLER': 19,
            'DOG': 20,
            'WHEELED_DEVICE': 21
        }
        
        # Return the category ID or a default value (could use -1 to flag unknown classes)
        return categories.get(category, 0)  # Default to REGULAR_VEHICLE if unknown


def parse_args():
    parser = argparse.ArgumentParser(description='Train PointPillars model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
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


def get_data_loaders(val_ratio=0.2, batch_size=4, seed=42):
    """Create train and validation data loaders with sklearn's train_test_split"""
    # Load environment variables
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset
    train_dataset = PointPillarsLoader(dataset_path, split='train')
    
    # Process all samples (or a subset for faster development)
    processed_samples = train_dataset.process_all_samples()
    
    # Get indices for train/validation split
    indices = list(range(len(processed_samples)))
    train_indices, val_indices = train_test_split(indices, test_size=val_ratio, random_state=seed)
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Set num_workers=0 for MacOS MPS compatibility
    num_workers = 0  # Better for MPS backend
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Don't use pin_memory on MPS
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Don't use pin_memory on MPS
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


def train_one_epoch(model, train_loader, criterion, optimizer, device, anchor_generator=None):
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
        
        # Process targets using anchor generator
        if anchor_generator is None:
            # Create default anchor generator if not provided
            x_range = (-100, 100)
            y_range = (-100, 100)
            z_range = (-3, 1)
            voxel_size = (0.3, 0.3)
            
            # Default anchor sizes for common classes (l, w, h)
            anchor_sizes = [
                # Small road users
                (0.8, 0.8, 1.7),  # PEDESTRIAN
                (0.6, 0.6, 1.2),  # STROLLER
                (0.8, 0.4, 0.6),  # DOG
                
                # Bicycles and motorcycles
                (1.8, 0.6, 1.2),  # BICYCLE
                (2.0, 0.8, 1.7),  # BICYCLIST
                (2.2, 0.9, 1.4),  # MOTORCYCLE
                (2.2, 0.9, 1.8),  # MOTORCYCLIST
                (1.0, 0.6, 1.0),  # WHEELED_DEVICE
                
                # Standard vehicles
                (4.5, 2.0, 1.6),  # REGULAR_VEHICLE
                
                # Large vehicles
                (6.5, 2.3, 2.3),  # TRUCK
                (8.0, 2.5, 2.5),  # LARGE_VEHICLE
                (6.0, 2.3, 3.0),  # BUS
                (12.0, 2.5, 3.2),  # ARTICULATED_BUS
                (5.5, 2.5, 2.5),  # BOX_TRUCK
                (4.0, 2.3, 2.5),  # TRUCK_CAB
                (6.0, 2.3, 2.0),  # VEHICULAR_TRAILER
                
                # Static objects
                (1.0, 1.0, 2.0),  # SIGN
                (0.5, 0.5, 2.0),  # STOP_SIGN
                (0.4, 0.4, 1.0),  # BOLLARD
                (1.2, 1.0, 1.8),  # OFFICIAL_SIGNALER
                (0.5, 0.5, 0.8),  # CONSTRUCTION_CONE
                (0.6, 0.6, 1.0),  # CONSTRUCTION_BARREL
            ]
            
            anchor_generator = AnchorGenerator(
                x_range, y_range, z_range, voxel_size, anchor_sizes
            )
        
        # Encode targets
        gt_boxes, gt_cls, gt_dir, pos_mask = anchor_generator.encode_targets(targets)
        
        # Move tensors to device
        gt_boxes = gt_boxes.to(device)
        gt_cls = gt_cls.to(device)
        gt_dir = gt_dir.to(device)
        pos_mask = pos_mask.to(device)
        
        # Reshape network outputs to match target format
        B, C7, H, W = bbox_preds.shape
        num_classes = C7 // 7
        
        # Use reshape instead of view to handle non-contiguous tensors
        # Reshape bbox_preds: [B, C*7, H, W] -> [B, H*W*num_classes, 7]
        pred_boxes = bbox_preds.reshape(B, num_classes, 7, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 7)
        
        # Reshape cls_scores: [B, C, H, W] -> [B, H*W*num_classes]
        pred_cls = cls_scores.permute(0, 2, 3, 1).reshape(B, -1)
        
        # Reshape dir_scores: [B, 2, H, W] -> [B, H*W*num_classes, 2]
        pred_dir = dir_scores.permute(0, 2, 3, 1).reshape(B, -1, 2)
        
        # Calculate loss
        loss, loc_loss, cls_loss, dir_loss = criterion(
            pred_boxes, gt_boxes, pred_cls, gt_cls, pred_dir, gt_dir, pos_mask
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({
            "batch_loss": loss.item(), 
            "loc_loss": loc_loss.item(),
            "cls_loss": cls_loss.item(),
            "dir_loss": dir_loss.item()
        })
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, anchor_generator=None):
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
            
            # Process targets using anchor generator
            if anchor_generator is None:
                # Create default anchor generator if not provided
                x_range = (-100, 100)
                y_range = (-100, 100)
                z_range = (-3, 1)
                voxel_size = (0.3, 0.3)
                
                # Default anchor sizes for common classes (l, w, h)
                anchor_sizes = [
                    # Small road users
                    (0.8, 0.8, 1.7),  # PEDESTRIAN
                    (0.6, 0.6, 1.2),  # STROLLER
                    (0.8, 0.4, 0.6),  # DOG
                    
                    # Bicycles and motorcycles
                    (1.8, 0.6, 1.2),  # BICYCLE
                    (2.0, 0.8, 1.7),  # BICYCLIST
                    (2.2, 0.9, 1.4),  # MOTORCYCLE
                    (2.2, 0.9, 1.8),  # MOTORCYCLIST
                    (1.0, 0.6, 1.0),  # WHEELED_DEVICE
                    
                    # Standard vehicles
                    (4.5, 2.0, 1.6),  # REGULAR_VEHICLE
                    
                    # Large vehicles
                    (6.5, 2.3, 2.3),  # TRUCK
                    (8.0, 2.5, 2.5),  # LARGE_VEHICLE
                    (6.0, 2.3, 3.0),  # BUS
                    (12.0, 2.5, 3.2),  # ARTICULATED_BUS
                    (5.5, 2.5, 2.5),  # BOX_TRUCK
                    (4.0, 2.3, 2.5),  # TRUCK_CAB
                    (6.0, 2.3, 2.0),  # VEHICULAR_TRAILER
                    
                    # Static objects
                    (1.0, 1.0, 2.0),  # SIGN
                    (0.5, 0.5, 2.0),  # STOP_SIGN
                    (0.4, 0.4, 1.0),  # BOLLARD
                    (1.2, 1.0, 1.8),  # OFFICIAL_SIGNALER
                    (0.5, 0.5, 0.8),  # CONSTRUCTION_CONE
                    (0.6, 0.6, 1.0),  # CONSTRUCTION_BARREL
                ]
                
                anchor_generator = AnchorGenerator(
                    x_range, y_range, z_range, voxel_size, anchor_sizes
                )
            
            # Encode targets
            gt_boxes, gt_cls, gt_dir, pos_mask = anchor_generator.encode_targets(targets)
            
            # Move tensors to device
            gt_boxes = gt_boxes.to(device)
            gt_cls = gt_cls.to(device)
            gt_dir = gt_dir.to(device)
            pos_mask = pos_mask.to(device)
            
            # Reshape network outputs to match target format
            B, C7, H, W = bbox_preds.shape
            num_classes = C7 // 7
            
            # Use reshape instead of view to handle non-contiguous tensors
            # Reshape bbox_preds: [B, C*7, H, W] -> [B, H*W*num_classes, 7]
            pred_boxes = bbox_preds.reshape(B, num_classes, 7, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 7)
            
            # Reshape cls_scores: [B, C, H, W] -> [B, H*W*num_classes]
            pred_cls = cls_scores.permute(0, 2, 3, 1).reshape(B, -1)
            
            # Reshape dir_scores: [B, 2, H, W] -> [B, H*W*num_classes, 2]
            pred_dir = dir_scores.permute(0, 2, 3, 1).reshape(B, -1, 2)
            
            # Calculate loss
            loss, loc_loss, cls_loss, dir_loss = criterion(
                pred_boxes, gt_boxes, pred_cls, gt_cls, pred_dir, gt_dir, pos_mask
            )
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({
                "batch_loss": loss.item(),
                "loc_loss": loc_loss.item(),
                "cls_loss": cls_loss.item(),
                "dir_loss": dir_loss.item()
            })
    
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
        seed=args.seed
    )

    # Debug: Inspect a few batches to verify data format
    # print("\n=== Inspecting Data Batches ===")
    # for batch_idx, batch in enumerate(train_loader):
    #     if batch_idx >= 2:  # Only inspect first two batches
    #         break
    #     print("\nInspecting train batch:", batch_idx)
    #     print(f"Pillars shape: {batch['pillars'].shape}")
    #     print(f"Coords shape: {batch['coords'].shape}")
    #     print(f"Number of targets: {len(batch['targets'])}")

    # for batch_idx, batch in enumerate(val_loader):
    #     if batch_idx >= 2:  # Only inspect first two batches
    #         break
    #     print("\nInspecting validation batch:", batch_idx)
    #     print(f"Pillars shape: {batch['pillars'].shape}")
    #     print(f"Coords shape: {batch['coords'].shape}")
    #     print(f"Number of targets: {len(batch['targets'])}")
    
    print("\n=== Starting Training ===")
    
    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.dataset.classes) + 1  # +1 for background
    print(f"Training with {num_classes} classes")
    
    # Initialize model and loss
    model = PointPillarsModel(num_classes=num_classes).to(device)
    criterion = PointPillarsLoss().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Create anchor generator for target encoding
    x_range = (-100, 100)
    y_range = (-100, 100)
    z_range = (-3, 1)
    voxel_size = (0.3, 0.3)
    
    # Default anchor sizes for common classes (l, w, h)
    anchor_sizes = [
        # Small road users
        (0.8, 0.8, 1.7),  # PEDESTRIAN
        (0.6, 0.6, 1.2),  # STROLLER
        (0.8, 0.4, 0.6),  # DOG
        
        # Bicycles and motorcycles
        (1.8, 0.6, 1.2),  # BICYCLE
        (2.0, 0.8, 1.7),  # BICYCLIST
        (2.2, 0.9, 1.4),  # MOTORCYCLE
        (2.2, 0.9, 1.8),  # MOTORCYCLIST
        (1.0, 0.6, 1.0),  # WHEELED_DEVICE
        
        # Standard vehicles
        (4.5, 2.0, 1.6),  # REGULAR_VEHICLE
        
        # Large vehicles
        (6.5, 2.3, 2.3),  # TRUCK
        (8.0, 2.5, 2.5),  # LARGE_VEHICLE
        (6.0, 2.3, 3.0),  # BUS
        (12.0, 2.5, 3.2),  # ARTICULATED_BUS
        (5.5, 2.5, 2.5),  # BOX_TRUCK
        (4.0, 2.3, 2.5),  # TRUCK_CAB
        (6.0, 2.3, 2.0),  # VEHICULAR_TRAILER
        
        # Static objects
        (1.0, 1.0, 2.0),  # SIGN
        (0.5, 0.5, 2.0),  # STOP_SIGN
        (0.4, 0.4, 1.0),  # BOLLARD
        (1.2, 1.0, 1.8),  # OFFICIAL_SIGNALER
        (0.5, 0.5, 0.8),  # CONSTRUCTION_CONE
        (0.6, 0.6, 1.0),  # CONSTRUCTION_BARREL
    ]
    
    anchor_generator = AnchorGenerator(
        x_range, y_range, z_range, voxel_size, anchor_sizes
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, anchor_generator)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, anchor_generator)
        
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