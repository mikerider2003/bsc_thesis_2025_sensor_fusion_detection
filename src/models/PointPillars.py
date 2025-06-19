import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import torchvision.ops as tv_ops

from src.loaders.loader_Point_Pillars import PointPillarsLoader, collate_fn

class PillarFeatureNet(nn.Module):
    def __init__(self, num_input_features: int = 9, num_output_features: int = 64):
        """
        Pillar Feature Network (PFN).
        Transforms raw point features within each pillar into a pillar-level feature representation.

        Args:
            num_input_features (int): Number of features for each point within a pillar.
                                      Typically 9 for PointPillars (x,y,z,r, xc,yc,zc, xp,yp).
                                      This corresponds to F in the input shape (∑P, N, F).
            num_output_features (int): Dimensionality of the learned pillar features.
        """
        super().__init__()
        self.num_output_features = num_output_features
        self.num_input_features = num_input_features

        # Simplified PointNet-like structure: Linear -> BatchNorm -> ReLU
        self.fc = nn.Linear(self.num_input_features, self.num_output_features, bias=False)
        self.norm = nn.BatchNorm1d(self.num_output_features, eps=1e-3, momentum=0.01)

    def forward(self, pillar_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PillarFeatureNet.

        Args:
            pillar_features (torch.Tensor): Tensor of point features for each pillar.
                Shape: (P, N, D_in), where:
                    P = total number of non-empty pillars in the batch (∑P).
                    N = maximum number of points per pillar.
                    D_in = num_input_features (F, features per point).

        Returns:
            torch.Tensor: Learned feature representation for each pillar.
                Shape: (P, D_out), where D_out = num_output_features.
        """
        P, N, D_in = pillar_features.shape
        if D_in != self.num_input_features:
            raise ValueError(
                f"Input feature dimension ({D_in}) does not match "
                f"PillarFeatureNet's num_input_features ({self.num_input_features})"
            )

        # Reshape for applying linear layer to all points: (P * N, D_in)
        x = pillar_features.view(P * N, D_in)

        # Apply Linear -> BatchNorm -> ReLU
        x = self.fc(x)
        x = self.norm(x)
        x = F.relu(x)

        x = x.view(P, N, self.num_output_features)

        # mask_shape: (P, N). True for valid points, False for padded points.
        # Padded points are those points (x,y,z) == 0
        is_padded_point = torch.all(pillar_features[:, :, :3] == 0, dim=2)
        mask = ~is_padded_point # True for valid points, shape (P, N)

        x_masked = torch.where(
            mask.unsqueeze(-1),  
            x,                   
            torch.tensor(float('-inf'), device=x.device, dtype=x.dtype) 
        )

        # Max pooling over the N (points per pillar) dimension
        encoded_pillars = torch.max(x_masked, dim=1).values

        return encoded_pillars


class PsuedoScatter(nn.Module):
    def __init__(self, num_input_features: int, grid_size_xy: tuple[int, int]):
        """
        Pillar Scatter operation.
        Scatters learned pillar features into a 2D pseudo-image.

        Args:
            num_input_features (int): Number of features for each encoded pillar.
                                      This is D_out from PillarFeatureNet.
            grid_size_xy (tuple[int, int]): The H, W dimensions of the 2D pseudo-image.
                                            (e.g., (grid_height, grid_width))
        """
        super().__init__()
        self.num_input_features = num_input_features
        self.grid_height = grid_size_xy[0]
        self.grid_width = grid_size_xy[1]

    def forward(self, encoded_pillars: torch.Tensor, pillar_coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PillarScatter.

        Args:
            encoded_pillars (torch.Tensor): Learned pillar features from PFN.
                                            Shape: (P, D_out), where P is the total number
                                            of pillars in the batch.
            pillar_coords (torch.Tensor): Coordinates for each pillar.
                                          Shape: (P, 3)
                                          Expected format: (x_idx, y_idx, batch_idx).
                                          Ensure y_idx and x_idx are within grid_height and grid_width.

        Returns:
            torch.Tensor: Batched 2D pseudo-image.
                          Shape: (B, D_out, grid_height, grid_width), where B is batch size.
        """
        batch_size = int(torch.max(pillar_coords[:, 2]).item() + 1)
        
        # Initialize an empty canvas for the pseudo-image
        # Shape: (B, D_out, H, W)
        canvas = torch.zeros(
            batch_size,
            self.num_input_features,
            self.grid_height,
            self.grid_width,
            dtype=encoded_pillars.dtype,
            device=encoded_pillars.device
        )

        # Scatter pillar features onto the canvas
        y_indices = pillar_coords[:, 0].long()
        x_indices = pillar_coords[:, 1].long()
        batch_indices = pillar_coords[:, 2].long()

        # Ensure indices are within bounds (optional, but good practice if not guaranteed by preprocessing)
        y_indices = torch.clamp(y_indices, 0, self.grid_height - 1)
        x_indices = torch.clamp(x_indices, 0, self.grid_width - 1)
        
        canvas[batch_indices, :, y_indices, x_indices] = encoded_pillars
        
        return canvas

def visualize_pseudo_image(canvas: torch.Tensor, title: str = "Pseudo Image"):
    """
    Visualizes the 2D pseudo-image for all batch samples as subplots.

    Args:
        canvas (torch.Tensor): The pseudo-image tensor to visualize.
                               Shape: (B, D_out, H, W).
        title (str): Title for the plot.
    """
    if canvas.is_cuda:
        canvas = canvas.cpu()
    canvas = canvas.detach()

    B, D_out, H, W = canvas.shape

    # Calculate subplot grid size
    ncols = min(B, 4)
    nrows = math.ceil(B / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = axs.flatten() if B > 1 else [axs]

    for b in range(B):
        img = canvas[b].sum(dim=0)
        ax = axs[b]
        im = ax.imshow(img, cmap='viridis')
        ax.set_xlabel("X (grid width)")
        ax.set_ylabel("Y (grid height)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for i in range(B, len(axs)):
        axs[i].axis('off')

    plt.suptitle(f'{title} (Batch size = {B})', fontsize=16)
    plt.tight_layout()
    plt.show()

class Backbone(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        # Block 1: Downsample 2x
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Block 2: Downsample 2x
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Block 3: Maintain resolution
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Block 4: Maintain resolution
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)  # 1/2
        x = self.block2(x)  # 1/4
        x = self.block3(x)  # 1/4
        x = self.block4(x)  # 1/4
        return x

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=4, num_anchors_per_location=8, box_code_size=10):
        """
        Args:
            in_channels (int): Number of channels from the backbone output.
            num_classes (int): Number of object classes (excluding background).
            num_anchors_per_location (int): Number of anchors per spatial location.
            box_code_size (int): Number of box regression parameters [cx,cy,cz,l,w,h,qw,qx,qy,qz, batch_idx].
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location
        
        # Regression head: predicts residuals for each anchor
        self.box_head = nn.Conv2d(
            in_channels, 
            num_anchors_per_location * box_code_size, 
            kernel_size=1
        )
        
        # Classification head: predicts class scores for each anchor
        self.cls_head = nn.Conv2d(
            in_channels, 
            num_anchors_per_location * num_classes, 
            kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Backbone feature map, shape [B, in_channels, H, W]
        Returns:
            box_preds (torch.Tensor): [B, H, W, num_anchors_per_location, box_code_size]
            cls_preds (torch.Tensor): [B, H, W, num_anchors_per_location, num_classes]
        """
        B, _, H, W = x.shape
        
        # Raw predictions
        box_preds = self.box_head(x)  # [B, num_anchors*box_code_size, H, W]
        cls_preds = self.cls_head(x)  # [B, num_anchors*num_classes, H, W]
        
        # Reshape to separate anchors and features
        box_preds = box_preds.view(B, self.num_anchors_per_location, -1, H, W)
        box_preds = box_preds.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, num_anchors, box_code_size]
        
        cls_preds = cls_preds.view(B, self.num_anchors_per_location, -1, H, W)
        cls_preds = cls_preds.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, num_anchors, num_classes]
        
        return box_preds, cls_preds

class PointPillarsModel(nn.Module):
    def __init__(self, grid_size, num_classes=4, grid_resolution=0.2):
        """
        Complete PointPillars model.
        
        Args:
            grid_size (tuple): Size of the grid in cells (height, width)
            num_classes (int): Number of object classes
            grid_resolution (float): Meters per grid cell
        """
        super().__init__()
        # 4x downsampling
        self.downsample_factor = 4
        self.downsampled_grid_size = (
            grid_size[0] // self.downsample_factor,
            grid_size[1] // self.downsample_factor
        )
        
        # Components
        self.pfn = PillarFeatureNet()
        self.scatter = PsuedoScatter(num_input_features=64, grid_size_xy=grid_size)
        self.backbone = Backbone()  # 4x downsampling
        
        # Anchor generator for downsampled grid
        self.anchor_generator = Anchor(
            grid_size=self.downsampled_grid_size,
            grid_resolution=grid_resolution * self.downsample_factor
        )
        
        # Detection head
        self.detection_head = DetectionHead(
            in_channels=256,
            num_classes=num_classes + 1,  
            num_anchors_per_location=len(self.anchor_generator.anchor_sizes) * 
                                     len(self.anchor_generator.anchor_rotations),
            box_code_size=10    # [cx,cy,cz,l,w,h,qw,qx,qy,qz, batch_idx]
        )
        
    def forward(self, batch):
        """
        Forward pass of PointPillars model.
        
        Args:
            batch (dict): Batch data containing:
                - 'features': pillar features [P, N, 9]
                - 'pillar_coords': pillar coordinates [P, 3]
                
        Returns:
            dict: Predictions containing:
                - 'box_preds': Box regression predictions
                - 'cls_preds': Classification predictions
                - 'anchors': Generated anchors
        """
        # Extract pillar features
        pillar_features = batch['features']
        pillar_coords = batch['pillar_coords']
        
        # 1. Pillar Feature Network
        learned_features = self.pfn(pillar_features)
        
        # 2. Pseudo Image Creation
        canvas = self.scatter(learned_features, pillar_coords)
        
        # 3. Backbone
        backbone_features = self.backbone(canvas)
        
        # 4. Detection Head
        box_preds, cls_preds = self.detection_head(backbone_features)
        
        return {
            'box_preds': box_preds,
            'cls_preds': cls_preds,
            'anchors': self.anchor_generator.anchors,
            'anchor_classes': self.anchor_generator.anchor_classes
        }
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointPillarsLoss(nn.Module):
    def __init__(self, 
                 cls_weight=1.0, 
                 box_weight=2.0, 
                 pos_cls_weight=1.0, 
                 neg_cls_weight=1.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        """
        PointPillars Loss Function combining classification and regression losses.
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.pos_cls_weight = pos_cls_weight
        self.neg_cls_weight = neg_cls_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, predictions, batch_assignments, annotations):
        """
        Compute the total loss for PointPillars.
        """
        box_preds = predictions['box_preds']
        cls_preds = predictions['cls_preds']
        anchors = predictions['anchors']
        
        B, H, W, num_anchors_per_location, _ = box_preds.shape
        num_classes = cls_preds.shape[-1]
        
        # Flatten predictions for easier processing
        box_preds_flat = box_preds.view(-1, 10)  # [B*H*W*num_anchors, 10]
        cls_preds_flat = cls_preds.view(-1, num_classes)  # [B*H*W*num_anchors, num_classes]
        
        total_cls_loss = 0.0
        total_box_loss = 0.0
        total_positive_samples = 0
        
        # Process each batch
        for batch_idx in range(B):
            if batch_idx not in batch_assignments:
                continue
                
            matched_gt_indices, matched_gt_boxes = batch_assignments[batch_idx]
            
            # Get predictions for current batch
            start_idx = batch_idx * H * W * num_anchors_per_location
            end_idx = start_idx + H * W * num_anchors_per_location
            
            batch_box_preds = box_preds_flat[start_idx:end_idx]
            batch_cls_preds = cls_preds_flat[start_idx:end_idx]
            
            # Classification loss
            cls_loss = self._compute_classification_loss(
                batch_cls_preds, matched_gt_indices, annotations, batch_idx
            )
            
            # Box regression loss (only for positive samples)
            box_loss, num_positives = self._compute_box_regression_loss(
                batch_box_preds, matched_gt_indices, matched_gt_boxes, anchors
            )
            
            total_cls_loss += cls_loss
            total_box_loss += box_loss
            total_positive_samples += num_positives
        
        # Normalize losses
        if total_positive_samples > 0:
            total_box_loss = total_box_loss / total_positive_samples
        else:
            total_box_loss = torch.tensor(0.0, device=box_preds.device)
        
        total_cls_loss = total_cls_loss / B
        
        # Check for NaN values and handle them
        if torch.isnan(total_cls_loss):
            print("Warning: Classification loss is NaN, setting to 0")
            total_cls_loss = torch.tensor(0.0, device=box_preds.device)
            
        if torch.isnan(total_box_loss):
            print("Warning: Box loss is NaN, setting to 0")
            total_box_loss = torch.tensor(0.0, device=box_preds.device)
        
        # Weighted total loss
        total_loss = self.cls_weight * total_cls_loss + self.box_weight * total_box_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': total_cls_loss,
            'box_loss': total_box_loss,
            'num_positives': total_positive_samples
        }
    
    def _compute_classification_loss(self, cls_preds, matched_gt_indices, annotations, batch_idx):
        num_anchors = cls_preds.shape[0]
        num_classes = cls_preds.shape[1]  # Now includes background
        
        # Initialize targets to BACKGROUND CLASS (last index)
        cls_targets = torch.full(
            (num_anchors,), 
            num_classes - 1,  # Background class index
            dtype=torch.long, 
            device=cls_preds.device
        )
        
        # Get GT boxes for current batch
        boxes = annotations['boxes']
        categories = annotations['categories']
        batch_mask = boxes[:, 10] == batch_idx
        batch_categories = [categories[i] for i, mask in enumerate(batch_mask) if mask]
        
        # Class name to index mapping (background is last)
        class_to_idx = {
            'PEDESTRIAN': 0,
            'TRUCK': 1, 
            'LARGE_VEHICLE': 2,
            'REGULAR_VEHICLE': 3
        }
        # Background is automatically num_classes-1
        
        # Set positive sample targets
        positive_mask = matched_gt_indices >= 0
        if positive_mask.sum() > 0:
            gt_indices = matched_gt_indices[positive_mask]
            for i, gt_idx in enumerate(gt_indices):
                anchor_idx = torch.where(positive_mask)[0][i]
                if gt_idx < len(batch_categories):
                    cls_name = batch_categories[gt_idx]
                    # Only set if valid class
                    if cls_name in class_to_idx:
                        cls_targets[anchor_idx] = class_to_idx[cls_name]
        
        # Compute focal loss
        cls_loss = self._focal_loss(cls_preds, cls_targets, positive_mask)
        
        return cls_loss
    
    def _focal_loss(self, predictions, targets, positive_mask):
        """
        Compute multi-class focal loss with explicit background class.
        Uses softmax instead of sigmoid.
        """
        # Move positive_mask to the same device as predictions
        positive_mask = positive_mask.to(predictions.device)
        
        num_classes = predictions.shape[1]
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Compute softmax probabilities
        log_softmax = F.log_softmax(predictions, dim=1)
        softmax = torch.exp(log_softmax)
        
        # Get probability of true class
        p_t = torch.sum(softmax * targets_one_hot, dim=1)
        
        # Compute cross entropy
        ce_loss = -torch.sum(log_softmax * targets_one_hot, dim=1)
        
        # Focal factor
        focal_factor = (1 - p_t) ** self.focal_gamma
        
        # Apply focal weighting
        focal_loss = focal_factor * ce_loss
        
        # Weight positive and negative samples differently
        pos_weight = positive_mask.float() * self.pos_cls_weight
        neg_weight = (~positive_mask).float() * self.neg_cls_weight
        sample_weights = pos_weight + neg_weight
        
        # Apply sample weighting
        weighted_loss = focal_loss * sample_weights
        
        # Return mean loss
        total_weight = sample_weights.sum()
        if total_weight > 0:
            return weighted_loss.sum() / total_weight
        else:
            return torch.tensor(0.0, device=predictions.device)
    
    def _compute_box_regression_loss(self, box_preds, matched_gt_indices, matched_gt_boxes, anchors):
        """
        Compute smooth L1 loss for box regression (only for positive samples).
        """
        positive_mask = matched_gt_indices >= 0
        num_positives = positive_mask.sum().item()
        
        if num_positives == 0:
            return torch.tensor(0.0, device=box_preds.device), 0
        
        # Get positive predictions and targets
        pos_box_preds = box_preds[positive_mask]
        pos_matched_gt = matched_gt_boxes[positive_mask]
        pos_anchors = anchors[positive_mask]
        
        # Encode GT boxes relative to anchors
        encoded_gt = self._encode_boxes(pos_matched_gt, pos_anchors)
        
        # Check for NaN in encoded targets
        if torch.isnan(encoded_gt).any():
            print("Warning: NaN found in encoded GT boxes")
            # Replace NaN with zeros
            encoded_gt = torch.where(torch.isnan(encoded_gt), torch.zeros_like(encoded_gt), encoded_gt)
        
        # Compute smooth L1 loss
        box_loss = self.smooth_l1_loss(pos_box_preds, encoded_gt)
        
        # Weight different box parameters differently
        box_weights = torch.tensor([
            1.0, 1.0, 1.0,  # cx, cy, cz
            2.0, 2.0, 2.0,  # l, w, h (size more important)
            1.0, 1.0, 1.0, 1.0  # quaternion components
        ], device=box_loss.device)
        
        weighted_box_loss = box_loss * box_weights.unsqueeze(0)
        
        # Check for NaN in loss
        if torch.isnan(weighted_box_loss).any():
            print("Warning: NaN found in box loss")
            return torch.tensor(0.0, device=box_preds.device), num_positives
        
        return weighted_box_loss.sum(), num_positives
    
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1.unbind(dim=1)
        w2, x2, y2, z2 = q2.unbind(dim=1)
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=1)

    
    def _encode_boxes(self, gt_boxes, anchors):
        """
        Encode ground truth boxes relative to anchor boxes with correct quaternion rotation.
        """
        # Extract components
        gt_center = gt_boxes[:, :3]  # cx, cy, cz
        gt_size = gt_boxes[:, 3:6]   # l, w, h
        gt_quat = gt_boxes[:, 6:10]  # qw, qx, qy, qz
        
        anchor_center = anchors[:, :3]
        anchor_size = anchors[:, 3:6]
        anchor_quat = anchors[:, 6:10]
        
        # Clamp sizes to avoid division by zero
        anchor_size = torch.clamp(anchor_size, min=1e-6)
        gt_size = torch.clamp(gt_size, min=1e-6)
        
        # Encode centers as residuals normalized by anchor size
        encoded_center = (gt_center - anchor_center) / anchor_size
        
        # Encode sizes as log ratios
        size_ratios = gt_size / anchor_size
        encoded_size = torch.log(size_ratios)
        
        # ------ FIXED QUATERNION HANDLING ------
        # Normalize quaternions
        gt_quat_norm = F.normalize(gt_quat, p=2, dim=1)
        anchor_quat_norm = F.normalize(anchor_quat, p=2, dim=1)
        
        # Compute relative rotation: q_rel = gt_quat * anchor_quat_inv
        # Create inverse anchor quaternions (conjugate for unit quaternions)
        anchor_quat_inv = torch.stack([
            anchor_quat_norm[:, 0],   # w component stays positive
            -anchor_quat_norm[:, 1],  # x component negated
            -anchor_quat_norm[:, 2],  # y component negated
            -anchor_quat_norm[:, 3]   # z component negated
        ], dim=1)
        
        # Quaternion multiplication: q_rel = gt_quat_norm * anchor_quat_inv
        w1, x1, y1, z1 = gt_quat_norm.unbind(dim=1)
        w2, x2, y2, z2 = anchor_quat_inv.unbind(dim=1)
        
        encoded_quat = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w component
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x component
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y component
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z component
        ], dim=1)
        
        # Normalize relative quaternion
        encoded_quat = F.normalize(encoded_quat, p=2, dim=1)
        # ------ END FIXED QUATERNION HANDLING ------
        
        # Combine all encoded components
        encoded_boxes = torch.cat([encoded_center, encoded_size, encoded_quat], dim=1)
        
        # Final NaN safety check
        if torch.isnan(encoded_boxes).any():
            print("Warning: NaN detected in box encoding")
            encoded_boxes = torch.where(
                torch.isnan(encoded_boxes),
                torch.zeros_like(encoded_boxes),
                encoded_boxes
            )
        
        return encoded_boxes

class Anchor():
    def __init__(self, grid_size, grid_resolution=0.2, anchor_sizes=None, anchor_rotations=None):
        """
        Initialize the Anchor object.
        Args:
            grid_size (tuple): Size of the grid in cells (height, width)
            grid_resolution (float): Meters per grid cell (default: 0.2m)
            anchor_sizes (dict): Anchor dimensions for each class in meters (l, w, h)
            anchor_rotations (list): Anchor rotations in degrees
        """
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution  # meters per grid cell

        if anchor_sizes is None:
            self.anchor_sizes = {
                'PEDESTRIAN':      (0.8, 0.6, 1.7),
                'TRUCK':           (15.0, 3.0, 3.5),
                'LARGE_VEHICLE':   (8.0, 2.8, 3.0),
                'REGULAR_VEHICLE': (4.5, 1.8, 1.6),
            } 
        else:
            self.anchor_sizes = anchor_sizes

        if anchor_rotations is None:
            self.anchor_rotations = [i for i in range(0, 180, 30)]
            self.anchor_rotations = [0, 90]
        else:
            self.anchor_rotations = anchor_rotations
        
        self.anchors, self.anchor_classes = self.generate()

    def generate(self):
        """
        Generate anchors for the entire BEV grid.
        Returns:
            anchors (torch.Tensor): [num_anchors, 10] (cx, cy, cz, l, w, h, qw, qx, qy, qz)
            anchor_classes (list): List of class names for each anchor
        """
        H, W = self.grid_size
        anchors = []
        anchor_classes = []
        
        # FIXED: Match the coordinate system used in pillarization
        max_range = (30, 20, 4)  # Should match your pillarization range
        
        # Convert grid centers to meters matching your pillar coordinate system
        x_centers = (torch.arange(W, dtype=torch.float32) + 0.5) * self.grid_resolution - max_range[0]
        y_centers = (torch.arange(H, dtype=torch.float32) + 0.5) * self.grid_resolution - max_range[1]

        for class_name, (l, w, h) in self.anchor_sizes.items():
            cz = h / 2
            for rot_deg in self.anchor_rotations:
                # Convert rotation to radians and compute quaternion
                rot_rad = torch.deg2rad(torch.tensor(rot_deg, dtype=torch.float32))
                qw = torch.cos(rot_rad / 2)
                qz = torch.sin(rot_rad / 2)  # Rotation around Z-axis (yaw)
                
                # Create grid for centers
                grid_y, grid_x = torch.meshgrid(y_centers, x_centers, indexing='ij')
                num_anchors = grid_x.numel()
                
                # Create anchor tensor block: [H*W, 10]
                block_anchors = torch.zeros((num_anchors, 10), dtype=torch.float32)
                
                # Fill anchor components
                block_anchors[:, 0] = grid_x.reshape(-1)  # cx (meters, now in [-30, 30])
                block_anchors[:, 1] = grid_y.reshape(-1)  # cy (meters, now in [-20, 20])
                block_anchors[:, 2] = cz                  # cz (meters)
                block_anchors[:, 3] = l                   # length (meters)
                block_anchors[:, 4] = w                   # width (meters)
                block_anchors[:, 5] = h                   # height (meters)
                block_anchors[:, 6] = qw                  # qw
                block_anchors[:, 7] = 0.0                 # qx (always 0 for BEV)
                block_anchors[:, 8] = 0.0                 # qy (always 0 for BEV)
                block_anchors[:, 9] = qz                  # qz
                
                anchors.append(block_anchors)
                anchor_classes.extend([class_name] * num_anchors)
    
        anchors = torch.cat(anchors, dim=0)
        return anchors, anchor_classes
    
    def plot_anchors(self, sample_stride=50, max_anchors=100):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.transforms as mtrans
        import numpy as np

        anchors, anchor_classes = self.anchors, self.anchor_classes
        anchors = anchors.numpy()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Anchor Boxes - Bird\'s Eye View')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        
        # Calculate plot limits based on anchor positions
        cx_min, cx_max = np.min(anchors[:, 0]), np.max(anchors[:, 0])
        cy_min, cy_max = np.min(anchors[:, 1]), np.max(anchors[:, 1])
        
        # Add margin based on largest anchor dimension
        max_dim = max(np.max(anchors[:, 3]), np.max(anchors[:, 4]))
        margin = max_dim * 1.5;
        
        ax.set_xlim(cx_min - margin, cx_max + margin)
        ax.set_ylim(cy_min - margin, cy_max + margin)
        
        class_colors = {
            'PEDESTRIAN': 'red',
            'TRUCK': 'blue',
            'LARGE_VEHICLE': 'green',
            'REGULAR_VEHICLE': 'purple'
        }
        
        legend_handles = []
        for cls, color in class_colors.items():
            legend_handles.append(patches.Patch(color=color, label=cls))
        
        num_anchors = anchors.shape[0]
        sample_indices = range(0, num_anchors, sample_stride)
        if len(sample_indices) > max_anchors:
            sample_indices = np.random.choice(num_anchors, max_anchors, replace=False)
        
        # Plot center points first
        for idx in sample_indices:
            cx, cy, cz, l, w, h, qw, qx, qy, qz = anchors[idx]
            class_name = anchor_classes[idx]
            ax.plot(cx, cy, 'o', markersize=2, color=class_colors.get(class_name, 'gray'))
        
        # Then add rectangles
        for idx in sample_indices:
            cx, cy, cz, l, w, h, qw, qx, qy, qz = anchors[idx]
            class_name = anchor_classes[idx]
            
            # Calculate yaw angle from quaternion
            yaw = 2 * np.arctan2(qz, qw)
            
            # Create transformation for rotation around CENTER
            t = mtrans.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
            
            # Create rectangle
            rect = patches.Rectangle(
                (cx - l/2, cy - w/2),  # Bottom-left corner
                l, w,                   # Length and width
                transform=t,            # Apply center-based rotation
                color=class_colors.get(class_name, 'gray'),
                alpha=0.4
            )
            ax.add_patch(rect)  # Add directly to axes
        
        ax.legend(handles=legend_handles, loc='upper right')
        
        H, W = self.grid_size
        plt.title(f"Anchor Boxes (Grid: {H}x{W} cells, {self.grid_resolution}m/cell)\n"
                f"Showing {len(sample_indices)} of {num_anchors} anchors")
        
        plt.tight_layout()
        plt.show()
        
    def assign(self, annotations):
        """
        Assigns ground truth annotations to anchors using 3D IoU matching.
        Returns matched indices for each batch separately.
        
        Args:
            annotations (Dict): 
                'boxes': Tensor of shape (N, 11) - [cx,cy,cz,l,w,h,qw,qx,qy,qz,batch_idx]
                'categories': List of class names for each box
        
        Returns:
            Dict: Dictionary with batch indices as keys and tuples (matched_gt_indices, matched_gt_boxes) as values
                matched_gt_indices: (num_anchors,) index of matched GT (-1 = background)
                matched_gt_boxes: (num_anchors, 10) matched GT boxes
        """
        boxes = annotations['boxes']
        class_names = annotations['categories']
        
        num_anchors = self.anchors.shape[0]
        batch_indices = boxes[:, 10].unique()
        
        batch_results = {}
        
        for batch_idx in batch_indices:
            # Initialize for current batch
            matched_gt_indices = torch.full((num_anchors,), -1, dtype=torch.long)
            matched_gt_boxes = torch.zeros((num_anchors, 10), dtype=torch.float32)
            
            # Filter boxes for current batch
            batch_mask = boxes[:, 10] == batch_idx
            gt_boxes = boxes[batch_mask, :10]
            
            # Skip if no GT boxes in this batch
            if gt_boxes.shape[0] == 0:
                batch_results[int(batch_idx.item())] = (matched_gt_indices, matched_gt_boxes)
                continue
            
            # Use 2D IoU instead of 3D
            iou_matrix = self.calculate_2d_iou(gt_boxes)  # (num_anchors, num_gt)
            
            # 1. Assign best anchor for each GT
            best_anchor_per_gt = iou_matrix.argmax(dim=0)
            best_iou_per_gt = iou_matrix.max(dim=0).values
            
            for gt_idx, (anchor_idx, iou) in enumerate(zip(best_anchor_per_gt, best_iou_per_gt)):
                if iou > 0.05:  # Minimum IoU threshold
                    matched_gt_indices[anchor_idx] = gt_idx
                    matched_gt_boxes[anchor_idx] = gt_boxes[gt_idx]
            
            # 2. Assign high IoU anchors
            max_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=1)
            high_iou_mask = (max_iou_per_anchor > 0.75) & (matched_gt_indices == -1)
            
            matched_gt_indices[high_iou_mask] = best_gt_per_anchor[high_iou_mask]
            matched_gt_boxes[high_iou_mask] = gt_boxes[best_gt_per_anchor[high_iou_mask]]
            
            # 3. Mark low IoU anchors as background
            low_iou_mask = max_iou_per_anchor < 0.45
            matched_gt_indices[low_iou_mask] = -1  # Background
            
            # Store results for this batch
            batch_results[int(batch_idx.item())] = (matched_gt_indices, matched_gt_boxes)

            num_of_gt_boxes = gt_boxes.shape[0]
            num_of_matched_anchors = (matched_gt_indices >= 0).sum().item()
            # print(f'Out of {num_of_gt_boxes} GT boxes, {num_of_matched_anchors} anchors matched in batch.')
        
        return batch_results
    
    def calculate_2d_iou(self, gt_boxes):
        """
        Calculate 2D IoU between anchors and ground truth boxes.
        Works with batched operations for efficiency.
        
        Args:
            gt_boxes (torch.Tensor): GT boxes of shape (N, 10) - [cx,cy,cz,l,w,h,qw,qx,qy,qz]
        
        Returns:
            torch.Tensor: IoU matrix of shape (num_anchors, N)
        """
        num_anchors = self.anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        
        # Convert anchors and GT boxes to [x1, y1, x2, y2] format for IoU calculation
        anchor_boxes = torch.zeros((num_anchors, 4), device=gt_boxes.device)
        gt_boxes_2d = torch.zeros((num_gt, 4), device=gt_boxes.device)
        
        # For anchors: [cx, cy, l, w] -> [x1, y1, x2, y2]
        anchor_boxes[:, 0] = self.anchors[:, 0] - self.anchors[:, 3] / 2  # x1 = cx - l/2
        anchor_boxes[:, 1] = self.anchors[:, 1] - self.anchors[:, 4] / 2  # y1 = cy - w/2
        anchor_boxes[:, 2] = self.anchors[:, 0] + self.anchors[:, 3] / 2  # x2 = cx + l/2
        anchor_boxes[:, 3] = self.anchors[:, 1] + self.anchors[:, 4] / 2  # y2 = cy + w/2
        
        # For GT boxes: [cx, cy, l, w] -> [x1, y1, x2, y2]
        gt_boxes_2d[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 3] / 2  # x1 = cx - l/2
        gt_boxes_2d[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 4] / 2  # y1 = cy - w/2
        gt_boxes_2d[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 3] / 2  # x2 = cx + l/2
        gt_boxes_2d[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 4] / 2  # y2 = cy + w/2
        
        # Calculate IoU using torchvision's box_iou
        # Returns tensor of shape (num_anchors, num_gt)
        iou_matrix = tv_ops.box_iou(anchor_boxes, gt_boxes_2d)
        
        return iou_matrix
    
    def visualize_matched_anchors(self, batch_results, annotations):
        """
        Visualize the matched anchors and their corresponding ground truth boxes for each batch.
        Visualizes all positive anchors and their matched ground truth boxes.
        
        Args:
            batch_results (Dict): Dictionary with batch indices as keys and tuples 
                                (matched_gt_indices, matched_gt_boxes) as values
            annotations (Dict): 
                'boxes': Tensor of shape (N, 11) - [cx,cy,cz,l,w,h,qw,qx,qy,qz,batch_idx]
                'categories': List of class names for each box
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.transforms as mtrans
        import numpy as np
        
        boxes = annotations['boxes']
        categories = annotations['categories']
        
        # Get number of batches
        num_batches = len(batch_results)
        
        # Calculate subplot grid
        ncols = min(num_batches, 4)
        nrows = math.ceil(num_batches / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5))
        if num_batches == 1:
            axes = [axes]
        elif nrows == 1 and num_batches > 1:
            axes = axes.reshape(1, -1)
        if num_batches > 1:
            axes = axes.flatten()
    
        # Color mapping for different classes
        class_colors = {
            'PEDESTRIAN': 'red',
            'TRUCK': 'blue', 
            'LARGE_VEHICLE': 'green',
            'REGULAR_VEHICLE': 'purple'
        }
        
        for i, (batch_idx, (matched_gt_indices, matched_gt_boxes)) in enumerate(batch_results.items()):
            ax = axes[i] if num_batches > 1 else axes[0]
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_aspect('equal')
            
            # Get positive anchors (matched to GT)
            positive_mask = matched_gt_indices >= 0
            positive_anchors = self.anchors[positive_mask]
            positive_gt_indices = matched_gt_indices[positive_mask]
            
            # Filter GT boxes and categories for current batch
            batch_mask = boxes[:, 10] == batch_idx
            gt_boxes_batch = boxes[batch_mask, :10]
            
            # Get categories for this batch
            batch_gt_indices = torch.where(batch_mask)[0]
            gt_categories_batch = [categories[idx] for idx in batch_gt_indices]
            
            # Plot matched anchors
            for anchor_idx, gt_idx in enumerate(positive_gt_indices):
                anchor = positive_anchors[anchor_idx]
                cx, cy, cz, l, w, h, qw, qx, qy, qz = anchor
                
                # Get anchor class
                original_anchor_idx = torch.where(positive_mask)[0][anchor_idx]
                anchor_class = self.anchor_classes[original_anchor_idx]
                color = class_colors.get(anchor_class, 'gray')
                
                # Calculate yaw angle from quaternion - fix for numpy 2.0 warning
                yaw = 2 * torch.atan2(qz, qw).item()
                
                # Create transformation for rotation around center
                t = mtrans.Affine2D().rotate_around(cx.item(), cy.item(), yaw) + ax.transData
                
                # Anchor box as filled rectangle - fix color warning
                anchor_rect = patches.Rectangle(
                    (cx.item() - l.item()/2, cy.item() - w.item()/2),
                    l.item(), w.item(),
                    transform=t,
                    facecolor=color,
                    alpha=0.3,
                    edgecolor=color,
                    linewidth=1,
                    label=f'Anchor ({anchor_class})' if anchor_idx == 0 else ""
                )
                ax.add_patch(anchor_rect)
                
                # Add anchor center point
                ax.plot(cx.item(), cy.item(), 's', color=color, markersize=4, alpha=0.7)
            
            # Plot GT boxes in thick lines
            for gt_idx, (gt_box, gt_category) in enumerate(zip(gt_boxes_batch, gt_categories_batch)):
                cx, cy, cz, l, w, h, qw, qx, qy, qz = gt_box
                
                # Calculate yaw angle from quaternion - fix for numpy 2.0 warning
                yaw = 2 * torch.atan2(qz, qw).item()
                
                # Create transformation for rotation around center
                t = mtrans.Affine2D().rotate_around(cx.item(), cy.item(), yaw) + ax.transData
                
                # GT box as thick outline
                gt_rect = patches.Rectangle(
                    (cx.item() - l.item()/2, cy.item() - w.item()/2),
                    l.item(), w.item(),
                    transform=t,
                    fill=False,
                    edgecolor='black',
                    linewidth=1,
                    label='Ground Truth' if gt_idx == 0 else ""
                )
                ax.add_patch(gt_rect)
            
            # Set axis limits to cover 60x40 meter area
            ax.set_ylim(0, 60)  # 60 meters in X direction
            ax.set_xlim(0, 40)  # 40 meters in Y direction

            # Add legend (only for first subplot to avoid clutter)
            if i == 0:
                # Create legend elements for anchor classes
                legend_elements = []
                
                # Add GT boxes legend
                legend_elements.append(
                    patches.Patch(facecolor='none', edgecolor='black', linewidth=3, label='Ground Truth')
                )
                
                # Add anchor class colors
                for class_name, color in class_colors.items():
                    legend_elements.append(
                        patches.Patch(facecolor=color, alpha=0.3, edgecolor=color, 
                                    label=f'Anchor ({class_name})')
                    )
                
                ax.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(1.0, 1.0), fontsize=8)
            
        
        # Hide unused subplots
        for i in range(num_batches, len(axes) if num_batches > 1 else 1):
            if num_batches > 1:
                axes[i].axis('off')
        
        plt.suptitle('Anchor Assignment Visualization', fontsize=16)
        plt.tight_layout()
        plt.show()

def compute_loss_example():
    """Example of how to use the loss function."""
    
    # Mock predictions (normally from your model)
    B, H, W, num_anchors, num_classes = 2, 15, 10, 8, 4
    
    # Use more realistic values to avoid NaN
    predictions = {
        'box_preds': torch.randn(B, H, W, num_anchors, 10) * 0.1,  # Smaller variance
        'cls_preds': torch.randn(B, H, W, num_anchors, num_classes) * 0.5,
        'anchors': torch.abs(torch.randn(H * W * num_anchors, 10)) + 0.1  # Positive values
    }
    
    # Mock batch assignments (normally from anchor.assign())
    total_anchors = H * W * num_anchors
    batch_assignments = {
        0: (torch.randint(-1, 5, (total_anchors,)), 
            torch.abs(torch.randn(total_anchors, 10)) + 0.1),  # Positive GT boxes
        1: (torch.randint(-1, 3, (total_anchors,)), 
            torch.abs(torch.randn(total_anchors, 10)) + 0.1)
    }
    
    # Mock annotations with proper batch indices
    annotations = {
        'boxes': torch.cat([
            torch.cat([torch.abs(torch.randn(4, 10)) + 0.1, torch.zeros(4, 1)], dim=1),  # batch 0
            torch.cat([torch.abs(torch.randn(4, 10)) + 0.1, torch.ones(4, 1)], dim=1)    # batch 1
        ], dim=0),
        'categories': ['PEDESTRIAN', 'TRUCK', 'REGULAR_VEHICLE', 'LARGE_VEHICLE'] * 2
    }
    
    # Compute loss
    loss_fn = PointPillarsLoss()
    loss_dict = loss_fn(predictions, batch_assignments, annotations)
    
    print("Loss computation example:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value}")

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

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,  # Adjust batch size as needed
        shuffle=True,
        collate_fn=collate_fn
    )

    print("\n=== Testing Model Components ===")

    sample = next(iter(data_loader))
    grid_size = (sample['grid_dims'][0], sample['grid_dims'][1])
    
    # # Test Pillar Feature Network
    # pfn = PillarFeatureNet(num_input_features=9, num_output_features=64)
    # pillar_features = sample['features']  
    # learned_features = pfn(pillar_features)
    # print(f"PillarFeatureNet output shape: {learned_features.shape}")

    # # Test PsuedoScatter
    # scatter = PsuedoScatter(num_input_features=64, grid_size_xy=grid_size)
    # canvas = scatter(learned_features, sample['pillar_coords'])
    # print(f"PsuedoScatter output shape: {canvas.shape}")
    # # visualize_pseudo_image(canvas)

    # # Test backbone
    # backbone = Backbone()
    # backbone_output = backbone(canvas)
    # print(f"PointPillarsBackbone output shape: {backbone_output.shape}")

    # # Test detection head
    # dh = DetectionHead(in_channels=256, num_classes=4, num_anchors_per_location=8, box_code_size=10)
    # box_preds, cls_preds = dh(backbone_output)
    # print(f"DetectionHead box_preds shape: {box_preds.shape}, cls_preds shape: {cls_preds.shape}")

    # # Test Anchor
    # anchor = Anchor(grid_size=grid_size)
    # # anchor.plot_anchors()
    # # batch_results = anchor.assign(sample['annotations'])
    # # anchor.visualize_matched_anchors(batch_results, sample['annotations'])


    # Test whole PointPillars model
    print("\n=== Testing PointPillarsModel ===")
    model = PointPillarsModel(grid_size=grid_size, num_classes=4, grid_resolution=0.2)
    model_output = model(sample)
    print(f"PointPillarsModel output: {model_output.keys()}")
    print(f"Box predictions shape: {model_output['box_preds'].shape}")
    print(f"Class predictions shape: {model_output['cls_preds'].shape}")
    print(f"Anchors shape: {model_output['anchors'].shape}")

    compute_loss_example()

    # python -m src.models.PointPillars



