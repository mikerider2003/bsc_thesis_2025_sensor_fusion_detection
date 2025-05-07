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
        x = x.view(B * P * N, D)
        x = self.linear(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(B, P, N, -1)
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
            pillar_features: Tensor of shape [B, num_pillars, num_features]
            coords: Tensor of shape [B, num_pillars, 4] with indices (batch_idx, x_idx, y_idx)
        
        Returns:
            pseudo_image: Tensor of shape [B, num_features, H, W]
        """
        H, W = self.output_shape
        B = pillar_features.shape[0]

        # Create empty pseudo-image tensor
        pseudo_image = torch.zeros(
            (B, self.num_features, H, W),
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )

        # Get indices from coords
        batch_indices = coords[:, :, 0].long()  # [B, num_pillars]
        y_indices = coords[:, :, 1].long()      # [B, num_pillars]
        x_indices = coords[:, :, 2].long()      # [B, num_pillars]

        # Scatter pillar features to the pseudo-image
        for b in range(B):
            pillar_features_sample = pillar_features[b]  # [num_pillars, num_features]
            batch_indices_sample = batch_indices[b]      # [num_pillars]
            y_indices_sample = y_indices[b]              # [num_pillars]
            x_indices_sample = x_indices[b]              # [num_pillars]

            # Create a mask to filter out padding
            mask = (x_indices_sample >= 0) & (x_indices_sample < W) & \
                   (y_indices_sample >= 0) & (y_indices_sample < H)

            # Apply the mask
            pillar_features_valid = pillar_features_sample[mask]
            y_indices_valid = y_indices_sample[mask]
            x_indices_valid = x_indices_sample[mask]

            # Scatter the valid features
            pseudo_image[b, :, y_indices_valid, x_indices_valid] = pillar_features_valid.T

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
        pred_boxes: [B, num_anchors, 7] or [B, H*W*num_classes, 7] predicted box residuals
        gt_boxes:   [B, num_anchors, 7] ground truth residuals
        pred_cls:   [B, num_anchors] or [B, H*W*num_classes] class probabilities
        gt_cls:     [B, num_anchors] ground truth class indices
        pred_dir:   [B, num_anchors, 2] or [B, H*W*num_classes, 2] direction class logits
        gt_dir:     [B, num_anchors] direction class indices (e.g., 0 or 1)
        pos_mask:   [B, num_anchors] boolean mask for positive anchors
        """
        # Ensure shapes are compatible
        B = pred_boxes.shape[0]
        
        # Handle shape mismatches by reshaping the predictions if needed
        if pred_boxes.shape[1] != gt_boxes.shape[1]:
            # Resize pred_boxes to match target size
            pred_boxes = self._resize_predictions(pred_boxes, gt_boxes.shape[1])
            
        if pred_cls.shape[1] != gt_cls.shape[1]:
            # Resize pred_cls to match target size
            pred_cls = self._resize_predictions(pred_cls, gt_cls.shape[1])
            
        if pred_dir.shape[1] != gt_dir.shape[1]:
            # Resize pred_dir to match target size while preserving the direction dimension
            pred_dir = self._resize_predictions(pred_dir, gt_dir.shape[1], keep_last_dim=True)
        
        # Flatten all tensors for simpler calculation - use reshape instead of view
        # to handle non-contiguous tensors
        pred_boxes = pred_boxes.reshape(-1, 7)
        gt_boxes = gt_boxes.reshape(-1, 7)
        pred_cls = pred_cls.reshape(-1)
        gt_cls = gt_cls.reshape(-1)
        pred_dir = pred_dir.reshape(-1, 2)
        gt_dir = gt_dir.reshape(-1)
        pos_mask = pos_mask.reshape(-1)

        # Number of positive anchors
        N_pos = pos_mask.sum().clamp(min=1).float()

        # ----- Localization Loss -----
        if pos_mask.sum() > 0:
            loc_loss = self.smooth_l1(pred_boxes[pos_mask], gt_boxes[pos_mask])
            loc_loss = loc_loss.sum() / N_pos
        else:
            loc_loss = torch.tensor(0.0, device=pred_boxes.device)

        # ----- Direction Classification Loss -----
        if pos_mask.sum() > 0:
            dir_loss = self.ce_loss(pred_dir[pos_mask], gt_dir[pos_mask])
            dir_loss = dir_loss.sum() / N_pos
        else:
            dir_loss = torch.tensor(0.0, device=pred_dir.device)

        # ----- Classification Loss (Focal Loss) -----
        cls_loss = self.focal_loss(pred_cls, gt_cls)
        cls_loss = cls_loss.sum() / N_pos

        # ----- Total Loss -----
        total_loss = (self.beta_loc * loc_loss +
                      self.beta_cls * cls_loss +
                      self.beta_dir * dir_loss)

        return total_loss, loc_loss, cls_loss, dir_loss

    def _resize_predictions(self, preds, target_size, keep_last_dim=False):
        """Resize prediction tensor to match target size."""
        B = preds.shape[0]
        
        if keep_last_dim:
            # For tensors with last dimension (e.g., direction scores [B, N, 2])
            last_dim = preds.shape[-1]
            # Use interpolate to resize
            preds_resized = F.interpolate(
                preds.reshape(B, -1, last_dim).permute(0, 2, 1), 
                size=target_size, 
                mode='linear'
            ).permute(0, 2, 1)
            return preds_resized
        else:
            # For tensors without last dimension (e.g., class scores [B, N])
            if len(preds.shape) > 2:
                # Handle box predictions [B, N, 7]
                resized = torch.zeros((B, target_size, preds.shape[2]), 
                                    device=preds.device,
                                    dtype=preds.dtype)
                # Just take first N predictions or repeat if needed
                if target_size > preds.shape[1]:
                    # If target is larger, repeat predictions
                    repeat_factor = (target_size + preds.shape[1] - 1) // preds.shape[1]  # Ceiling division
                    resized[:, :preds.shape[1]*repeat_factor, :] = preds.repeat(1, repeat_factor, 1)[:, :target_size, :]
                else:
                    # If target is smaller, take first N
                    resized = preds[:, :target_size, :]
            else:
                # Handle 2D tensors like class scores [B, N]
                resized = torch.zeros((B, target_size), 
                                    device=preds.device,
                                    dtype=preds.dtype)
                if target_size > preds.shape[1]:
                    repeat_factor = (target_size + preds.shape[1] - 1) // preds.shape[1]  # Ceiling division
                    resized[:, :preds.shape[1]*repeat_factor] = preds.repeat(1, repeat_factor)[:, :target_size]
                else:
                    resized = preds[:, :target_size]
                    
            return resized

    def focal_loss(self, inputs, targets):
        """
        Focal loss for classification.
        inputs: [N] logits (before softmax or sigmoid)
        targets: [N] ground truth class indices
        """
        # Handle the case where inputs are not multiclass
        if len(inputs.shape) == 1 or inputs.shape[-1] == 1:
            # Binary classification case
            probs = torch.sigmoid(inputs)
            pt = torch.where(targets > 0, probs, 1-probs)
        else:
            # Multiclass classification case
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
        # Pillar feature encoding
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
    train_dataset = PointPillarsLoader(dataset_path, split='train')

    # Create a small sample for testing
    processed_samples = train_dataset.process_all_samples(limit=3)
    sample = processed_samples[0]
    pillars, coords = sample["lidar_processed"]

    # Convert numpy arrays to torch tensors
    pillars_tensor = torch.from_numpy(pillars).float()
    coords_tensor = torch.from_numpy(coords).int()

    # Add batch dimension
    pillars_tensor = pillars_tensor.unsqueeze(0)  # [1, P, N, 9]
    coords_tensor = coords_tensor.unsqueeze(0)    # [1, P, 4]

    print(f"Input Pillars tensor shape: {pillars_tensor.shape}")
    print(f"Input Coords tensor shape: {coords_tensor.shape}")

    # Initialize the model
    model = PointPillarsModel(num_classes=len(train_dataset.classes) + 1)  # +1 for background class

    # Forward pass
    bbox_preds, cls_scores, dir_scores = model(pillars_tensor, coords_tensor)
    print(f"Bounding box predictions shape: {bbox_preds.shape}")
    print(f"Class scores shape: {cls_scores.shape}")
    print(f"Direction scores shape: {dir_scores.shape}")



