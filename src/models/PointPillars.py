import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loaders.loader_Point_Pillars import PointPillarsLoader

class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=64):
        super(PillarFeatureNet, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):  # x: [B, P, N, 9]
        B, P, N, D = x.shape
        # Reshape for linear layer
        x = x.view(B * P * N, D)
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        # Reshape back to [B, P, N, C]
        x = x.view(B, P, N, -1)
        # Max pooling over points in each pillar
        x = torch.max(x, dim=2)[0]  # [B, P, C]
        return x

class PseudoImageScatter(nn.Module):
    def __init__(self, output_shape, num_features=64):
        """
        Scatter pillar features to a 2D pseudo-image
        
        Args:
            output_shape: Tuple (H, W) defining the output pseudo-image dimensions
            num_features: Number of features per pillar
        """
        super(PseudoImageScatter, self).__init__()
        self.output_shape = output_shape
        self.num_features = num_features
        
    def forward(self, pillar_features, coords):
        """
        Args:
            pillar_features: Tensor of shape [B, P, C] with features
            coords: Tensor of shape [B, P, 4] with indices (batch_idx, x_idx, y_idx, z_idx)
        
        Returns:
            pseudo_image: Tensor of shape [B, C, H, W]
        """
        B, P, C = pillar_features.shape
        H, W = self.output_shape
        
        # Create empty pseudo-image tensor with batch_size B
        pseudo_image = torch.zeros(
            (B, self.num_features, H, W),
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )
        
        # Process each batch separately
        for b in range(B):
            batch_features = pillar_features[b]  # [P, C]
            batch_coords = coords[b]  # [P, 4]
            
            # Only use non-empty pillars (assume all are valid for now)
            if batch_features.shape[0] > 0:
                # Get indices from coords
                y_idx = batch_coords[:, 1].long()  # y is at index 1
                x_idx = batch_coords[:, 2].long()  # x is at index 2
                
                # Filter out invalid coordinates
                valid_mask = (
                    (x_idx >= 0) & (x_idx < W) & 
                    (y_idx >= 0) & (y_idx < H)
                )
                
                if valid_mask.any():
                    y_idx_valid = y_idx[valid_mask]
                    x_idx_valid = x_idx[valid_mask]
                    features_valid = batch_features[valid_mask]
                    
                    # Scatter features to pseudo image
                    for i in range(len(y_idx_valid)):
                        pseudo_image[b, :, y_idx_valid[i], x_idx_valid[i]] = features_valid[i]
        
        return pseudo_image

class CNN_BackBone(nn.Module):
    def __init__(self, in_channels=64):
        super(CNN_BackBone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

# SSD Detection Head for predicting bounding boxes
class SSDDetectionHead(nn.Module):
    def __init__(self, num_classes, in_channels=128):
        super(SSDDetectionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        # For each class, predict 7 values: (x, y, z, length, width, height, θ)
        self.conv2 = nn.Conv2d(256, num_classes * 7, kernel_size=3, padding=1)
        
        # Class prediction
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
        
        # Direction prediction (binary: forward/backward)
        self.conv_dir = nn.Conv2d(256, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        bbox_regression = self.conv2(x)
        class_scores = self.conv3(x)
        direction_scores = self.conv_dir(x)
        return bbox_regression, class_scores, direction_scores

class PointPillarsLoss(nn.Module):
    def __init__(self, beta_loc=2.0, beta_cls=1.0, beta_dir=0.2,
                 alpha=0.25, gamma=2.0):
        super(PointPillarsLoss, self).__init__()
        self.beta_loc = beta_loc
        self.beta_cls = beta_cls
        self.beta_dir = beta_dir
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # for direction classification

    def forward(self, pred_boxes, gt_boxes, pred_cls, gt_cls,
                pred_dir, gt_dir, pos_mask):
        """
        pred_boxes: [N, 7] predicted box residuals (x, y, z, w, l, h, θ)
        gt_boxes:   [N, 7] ground truth residuals
        pred_cls:   [N, C] class probabilities (after sigmoid or softmax)
        gt_cls:     [N] ground truth class indices (0 = background)
        pred_dir:   [N, 2] direction class logits
        gt_dir:     [N] direction class indices (e.g., 0 or 1)
        pos_mask:   [N] boolean mask for positive anchors
        """
        # Get shapes of all inputs
        n_box_preds = pred_boxes.size(0)
        n_cls_preds = pred_cls.size(0)
        n_dir_preds = pred_dir.size(0)
        n_targets = pos_mask.size(0)
        
        # Store original sizes for debugging
        original_sizes = {
            'pred_boxes': n_box_preds,
            'pred_cls': n_cls_preds,
            'pred_dir': n_dir_preds,
            'pos_mask': n_targets
        }
        
        # Create a consistent mask size that works with all predictions
        # We'll use the smallest size among all prediction tensors
        min_size = min(n_box_preds, n_cls_preds, n_dir_preds)
        
        # Adjust all inputs to have the same first dimension
        if min_size < n_targets:
            # If predictions are smaller than targets, truncate targets
            pos_mask = pos_mask[:min_size]
            gt_boxes = gt_boxes[:min_size]
            gt_cls = gt_cls[:min_size]
            gt_dir = gt_dir[:min_size]
            
            # Also truncate any prediction tensors that are too large
            if n_box_preds > min_size:
                pred_boxes = pred_boxes[:min_size]
            if n_cls_preds > min_size:
                pred_cls = pred_cls[:min_size]
            if n_dir_preds > min_size:
                pred_dir = pred_dir[:min_size]
        else:
            # If targets are smaller than predictions, pad targets
            pad_size = min_size - n_targets
            if pad_size > 0:
                # Pad pos_mask with False values
                pos_mask = torch.cat([pos_mask, torch.zeros(pad_size, dtype=torch.bool, device=pos_mask.device)], dim=0)
                
                # Pad gt_boxes with zeros
                padding = torch.zeros(pad_size, 7, dtype=gt_boxes.dtype, device=gt_boxes.device)
                gt_boxes = torch.cat([gt_boxes, padding], dim=0)
                
                # Pad gt_cls with zeros (background class)
                gt_cls = torch.cat([gt_cls, torch.zeros(pad_size, dtype=gt_cls.dtype, device=gt_cls.device)], dim=0)
                
                # Pad gt_dir with zeros
                gt_dir = torch.cat([gt_dir, torch.zeros(pad_size, dtype=gt_dir.dtype, device=gt_dir.device)], dim=0)
            
            # Also truncate any prediction tensors that are larger than min_size
            if n_box_preds > min_size:
                pred_boxes = pred_boxes[:min_size]
            if n_cls_preds > min_size:
                pred_cls = pred_cls[:min_size]
            if n_dir_preds > min_size:
                pred_dir = pred_dir[:min_size]
            
        # Final sanity check - ensure all shapes match
        assert pred_boxes.size(0) == pos_mask.size(0)
        assert pred_cls.size(0) == pos_mask.size(0)
        assert pred_dir.size(0) == pos_mask.size(0)
        assert gt_boxes.size(0) == pos_mask.size(0)
        assert gt_cls.size(0) == pos_mask.size(0)
        assert gt_dir.size(0) == pos_mask.size(0)

        # Number of positive anchors
        N_pos = pos_mask.sum().clamp(min=1).float()

        # ----- Localization Loss -----
        loc_loss = self.smooth_l1(pred_boxes[pos_mask], gt_boxes[pos_mask])
        loc_loss = loc_loss.sum() / N_pos

        # ----- Direction Classification Loss -----
        dir_loss = self.ce_loss(pred_dir[pos_mask], gt_dir[pos_mask])
        dir_loss = dir_loss.sum() / N_pos

        # ----- Classification Loss (Focal Loss) -----
        cls_loss = self.focal_loss(pred_cls, gt_cls)
        cls_loss = cls_loss.sum() / N_pos

        # ----- Total Loss -----
        total_loss = (self.beta_loc * loc_loss +
                      self.beta_cls * cls_loss +
                      self.beta_dir * dir_loss)

        return total_loss, loc_loss, cls_loss, dir_loss

    def focal_loss(self, inputs, targets):
        """
        Focal loss for classification.
        inputs: [N, C] logits (before softmax or sigmoid)
        targets: [N] ground truth class indices
        """
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        probs = F.softmax(inputs, dim=1)
        pt = probs * targets_one_hot
        pt = pt.sum(dim=1)  # [N]

        log_pt = torch.log(pt + 1e-6)
        focal = -self.alpha * (1 - pt) ** self.gamma * log_pt
        return focal

class PointPillarsModel(nn.Module):
    def __init__(self, num_classes, voxel_size=(0.3, 0.3), x_range=(-100, 100), y_range=(-100, 100)):
        super(PointPillarsModel, self).__init__()
        
        # Calculate grid dimensions
        self.nx = int(np.floor((x_range[1] - x_range[0]) / voxel_size[0]))
        self.ny = int(np.floor((y_range[1] - y_range[0]) / voxel_size[1]))
        
        # Model components
        self.pfn = PillarFeatureNet(in_channels=9, out_channels=64)
        self.scatter = PseudoImageScatter(output_shape=(self.ny, self.nx), num_features=64)
        self.backbone = CNN_BackBone(in_channels=64)
        self.head = SSDDetectionHead(num_classes=num_classes, in_channels=128)
        
    def forward(self, pillars, coords):
        # Pillar feature encoding - handle batched input
        pillar_features = self.pfn(pillars)  # [B, P, C]
        
        # Scatter to pseudo image
        pseudo_image = self.scatter(pillar_features, coords)  # [B, C, H, W]
        
        # CNN backbone
        cnn_features = self.backbone(pseudo_image)  # [B, 128, H/4, W/4]
        
        # SSD head
        bbox_preds, cls_scores, dir_scores = self.head(cnn_features)
        
        return bbox_preds, cls_scores, dir_scores

# For testing/debugging
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')

    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    # Create dataset and loader
    target_classes = {'PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE'}
    train_dataset = PointPillarsLoader(dataset_path, split='train', target_classes=target_classes)

    # Create a small sample for testing
    processed_samples = train_dataset.process_all_samples(limit=3)
    sample = processed_samples[0]
    pillars, coords = sample["lidar_processed"]

    # Convert numpy arrays to torch tensors
    pillars_tensor = torch.from_numpy(pillars).float().unsqueeze(0)  # Add batch dimension
    coords_tensor = torch.from_numpy(coords).int().unsqueeze(0)      # Add batch dimension

    print(f"Input pillar shape: {pillars_tensor.shape}")
    print(f"Coords shape: {coords_tensor.shape}")

    # Initialize the model
    model = PointPillarsModel(num_classes=len(train_dataset.classes) + 1)  # +1 for background class

    # Forward pass
    bbox_preds, cls_scores, dir_scores = model(pillars_tensor, coords_tensor)
    print(f"Bounding box predictions shape: {bbox_preds.shape}")
    print(f"Class scores shape: {cls_scores.shape}")
    print(f"Direction scores shape: {dir_scores.shape}")


    # python -m src.models.PointPillars



