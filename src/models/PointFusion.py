# src/models/PointFusion.py
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from scipy.optimize import linear_sum_assignment

class ImageBackbone(nn.Module):
    """Processes multi-camera input using shared ResNet-50"""
    def __init__(self, num_cameras=7):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_reduction = nn.Sequential(
            nn.Linear(2048 * num_cameras, 2048),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, images):
        features = []
        for cam_name, img in images.items():
            feat = self.resnet(img)
            feat = self.pool(feat).flatten(1)
            features.append(feat)
        return self.feature_reduction(torch.cat(features, 1))

class ModifiedPointNet(nn.Module):
    """Processes LiDAR point clouds with adaptive pooling"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1024), nn.ReLU())
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, points):
        point_feats = self.mlp(points)
        return self.global_pool(point_feats.transpose(1,2)).squeeze(-1)

class FusionNetwork(nn.Module):
    """Fuses features and makes multiple predictions"""
    def __init__(self, max_predictions=128):
        super().__init__()
        self.max_pred = max_predictions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2048+1024, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU())
        
        # Prediction heads
        self.offset_head = nn.Linear(512, 24*max_predictions)
        self.score_head = nn.Linear(512, max_predictions)
        self.class_head = nn.Linear(512, 4*max_predictions)
        self.anchor_selector = nn.Linear(1024, max_predictions)

    def forward(self, img_feats, point_feats):
        fused = torch.cat([img_feats, point_feats], 1)
        fused = self.fusion_mlp(fused)
        
        return {
            'offsets': self.offset_head(fused).view(-1, self.max_pred, 24),
            'scores': self.score_head(fused),
            'class_logits': self.class_head(fused).view(-1, self.max_pred, 4),
            'anchor_scores': self.anchor_selector(point_feats)
        }

class HungarianMatcher(nn.Module):
    """Matches predictions to ground truth boxes"""
    def __init__(self, cost_class=1, cost_bbox=5):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size = outputs['class_logits'].shape[0]
        indices = []
        
        for b in range(batch_size):
            # Calculate cost matrix
            pred_logits = outputs['class_logits'][b].softmax(-1)
            tgt_labels = targets[b]['labels']
            
            class_cost = -pred_logits[:, tgt_labels]
            box_cost = torch.cdist(outputs['corners'][b].mean(1), 
                          targets[b]['boxes'][:, :3])
            
            # Hungarian matching
            C = self.cost_class*class_cost + self.cost_bbox*box_cost
            indices.append(linear_sum_assignment(C.cpu()))
            
        return indices

class PointFusionLoss(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.box_loss = nn.SmoothL1Loss(reduction='none')
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        total_loss = 0
        
        for batch_idx, (pred_indices, tgt_indices) in enumerate(indices):
            # Convert target parameters to corners
            tgt_boxes = targets[batch_idx]['boxes'][tgt_indices]
            tgt_corners = self.box_params_to_corners(tgt_boxes)
            
            # Get matched predictions
            pred_corners = outputs['corners'][batch_idx, pred_indices]
            pred_logits = outputs['class_logits'][batch_idx, pred_indices]
            
            # Calculate losses
            box_loss = self.box_loss(pred_corners, tgt_corners).mean()
            cls_loss = self.cls_loss(pred_logits, targets[batch_idx]['labels'][tgt_indices])
            
            total_loss += box_loss + cls_loss
            
        return total_loss / len(indices)
    
    def box_params_to_corners(self, boxes):
        """
        Convert 3D bounding box parameters to 8 corner points.
        
        Args:
            boxes: Tensor of shape [N, 7] containing (x, y, z, l, w, h, heading)
            
        Returns:
            corners: Tensor of shape [N, 8, 3] with 3D coordinates of box corners
        """
        device = boxes.device
        N = boxes.shape[0]
        
        # Split into components
        x = boxes[:, 0]
        y = boxes[:, 1]
        z = boxes[:, 2]
        l = boxes[:, 3]
        w = boxes[:, 4]
        h = boxes[:, 5]
        heading = boxes[:, 6]
        
        # Calculate half dimensions
        half_l = l / 2
        half_w = w / 2
        half_h = h / 2

        # Create template for 8 corners (local coordinates before rotation)
        corner_signs = torch.tensor([
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ], dtype=boxes.dtype, device=device)
        
        # Expand dimensions for broadcasting
        local_corners = torch.zeros((N, 8, 3), device=device)
        local_corners[:, :, 0] = half_l.unsqueeze(-1) * corner_signs[:, 0]
        local_corners[:, :, 1] = half_w.unsqueeze(-1) * corner_signs[:, 1]
        local_corners[:, :, 2] = half_h.unsqueeze(-1) * corner_signs[:, 2]

        # Create rotation matrices (z-axis rotation)
        cos_theta = torch.cos(heading)
        sin_theta = torch.sin(heading)
        ones = torch.ones_like(cos_theta)
        zeros = torch.zeros_like(cos_theta)
        
        # Rotation matrix [N, 3, 3]
        rot_matrix = torch.stack([
            cos_theta, -sin_theta, zeros,
            sin_theta, cos_theta, zeros,
            zeros, zeros, ones
        ], dim=1).view(N, 3, 3)

        # Autonomous driving standard
        rot_matrix = torch.stack([
            cos_theta, sin_theta, zeros,    
            -sin_theta, cos_theta, zeros,   
            zeros, zeros, ones
        ], dim=1).view(N, 3, 3)

        # Rotate local corners
        rotated_corners = torch.bmm(local_corners, rot_matrix)
        
        # Translate to global coordinates
        center = torch.stack([x, y, z], dim=1).unsqueeze(1)
        global_corners = rotated_corners + center
        
        return global_corners

class PointFusion3D(nn.Module):
    """Complete 3D detection model with multi-camera support"""
    def __init__(self, num_cameras=7, max_predictions=128):
        super().__init__()
        self.max_pred = max_predictions
        self.img_backbone = ImageBackbone(num_cameras)
        self.pointnet = ModifiedPointNet()
        self.fusion = FusionNetwork(max_predictions)
        
    def forward(self, batch):
        # Feature extraction
        img_feats = self.img_backbone(batch['images'])
        point_feats = self.pointnet(batch['points'])
        
        # Get predictions
        outputs = self.fusion(img_feats, point_feats)
        
        # Select anchors and convert to absolute coordinates
        _, anchor_idx = torch.topk(outputs['anchor_scores'], self.max_pred, dim=1)
        outputs['corners'] = self.offsets_to_absolute(
            batch['points'],
            outputs['offsets'],
            anchor_idx
        )
        
        return outputs
    
    def offsets_to_absolute(self, points, offsets, anchor_idx):
        """Convert relative offsets to absolute coordinates"""
        batch_size = points.size(0)
        anchors = torch.gather(
            points[:, :, :3],
            1,
            anchor_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, K, 3]
        
        return anchors.unsqueeze(2) + offsets.view(batch_size, self.max_pred, 8, 3)

def batch_inspection(batch):
    # Check the batch structure
    print()
    print(f"Batch structure (size: {batch['points'].shape[0]}):")
    print(f"Points shape: {batch['points'].shape}\n")  # [B, N, 3]
    for camera_name, image in batch['images'].items():
        print(f"Image {camera_name} shape: {image.shape}")  # [B, C, H, W]
    print()

    print("Annotations (for each sample in batch):")
    for i, annotations in enumerate(batch['annotations']):
        boxes = annotations['boxes']  # [N_i, 7] tensor for sample i
        labels = annotations['labels']

        print(f"\tSample {i}: \n{3*'\t'}Boxes shape: {boxes.shape}, \n{3*'\t'}Labels shape: {labels.shape}")
    

# Example Usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')


    from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
    from torch.utils.data import DataLoader
    
    # Test with sample data
    dataset = PointFusionloader(dataset_path, split='train')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)
    batch = next(iter(dataloader))

    batch_inspection(batch)
    
    model = PointFusion3D()
    outputs = model(batch)
    
    print("Model outputs:")
    print(f"Corners shape: {outputs['corners'].shape}")  # [2, 128, 8, 3]
    print(f"Scores shape: {outputs['scores'].shape}")     # [2, 128]
    print(f"Class logits shape: {outputs['class_logits'].shape}")  # [2, 128, 4]


    # Test box_params_to_corners function
    test = PointFusionLoss(HungarianMatcher())

    test_box = torch.tensor([[0, 0, 0, 4, 2, 1, torch.pi/2]], dtype=torch.float32)  # 90-degree rotation
    print(test_box.shape)
    corners = test.box_params_to_corners(test_box)

    # Expected corners after 90-degree rotation:
    # Original front (x=+2) becomes y=+2
    # Original right (y=+1) becomes x=-1
    print("Rotated Corners:")
    print(corners)
    # Expected output:
    # tensor([[[-1.0000,  2.0000,  0.5000],
    #      [-1.0000,  2.0000, -0.5000],
    #      [ 1.0000,  2.0000,  0.5000],
    #      [ 1.0000,  2.0000, -0.5000],
    #      [-1.0000, -2.0000,  0.5000],
    #      [-1.0000, -2.0000, -0.5000],
    #      [ 1.0000, -2.0000,  0.5000],
    #      [ 1.0000, -2.0000, -0.5000]]])
