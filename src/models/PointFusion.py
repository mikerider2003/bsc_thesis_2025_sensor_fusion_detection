# src/models/PointFusion.py
import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.models import ResNet50_Weights

class ImageBackbone(nn.Module):
    """Processes multi-camera input using shared ResNet-50"""
    def __init__(self, num_cameras=7):
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_reduction = nn.Sequential(
            nn.Linear(2048 * num_cameras, 2048),
            nn.ReLU(inplace=True))
        
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
    """Fuses features and makes parameter predictions"""
    def __init__(self, max_predictions=128):
        super().__init__()
        self.max_pred = max_predictions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2048+1024, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU())
        
        # Prediction heads
        self.param_head = nn.Linear(512, 7*max_predictions)  # Direct parameters
        self.score_head = nn.Linear(512, max_predictions)
        self.class_head = nn.Linear(512, 4*max_predictions)

    def forward(self, img_feats, point_feats):
        fused = torch.cat([img_feats, point_feats], 1)
        fused = self.fusion_mlp(fused)
        
        return {
            'params': self.param_head(fused).view(-1, self.max_pred, 7),
            'scores': self.score_head(fused),
            'class_logits': self.class_head(fused).view(-1, self.max_pred, 4)
        }

class HungarianMatcher(nn.Module):
    """Matches predictions using parameter distances"""
    def __init__(self, cost_class=1, cost_bbox=5, cost_heading=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_heading = cost_heading

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size = outputs['params'].shape[0]
        indices = []
        
        for b in range(batch_size):
            # Get predictions and targets
            pred_params = outputs['params'][b]  # [K, 7]
            tgt_params = targets[b]['boxes']    # [M, 7]
            tgt_labels = targets[b]['labels']   # [M]
            
            # Position cost (x,y,z) - pairwise [K, M]
            pos_cost = torch.cdist(pred_params[:, :3], tgt_params[:, :3])
            
            # Dimension cost (l,w,h) - pairwise [K, M]
            dim_cost = torch.cdist(pred_params[:, 3:6], tgt_params[:, 3:6])
            
            # Heading cost - pairwise [K, M]
            pred_heading = pred_params[:, 6].unsqueeze(1)  # [K, 1]
            tgt_heading = tgt_params[:, 6].unsqueeze(0)    # [1, M]
            heading_diff = torch.abs(pred_heading - tgt_heading)
            heading_cost = torch.min(heading_diff, 2*np.pi - heading_diff)
            
            # Class cost - pairwise [K, M]
            pred_logits = outputs['class_logits'][b].softmax(-1)  # [K, 4]
            class_cost = -pred_logits[:, tgt_labels]  # [K, M]
            
            # Total cost [K, M]
            C = (self.cost_bbox * (pos_cost + dim_cost) + 
                 self.cost_heading * heading_cost +
                 self.cost_class * class_cost)
            
            # Hungarian matching
            indices.append(linear_sum_assignment(C.cpu()))
            
        return indices

class PointFusionLoss(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.box_loss = nn.SmoothL1Loss(reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')
        self.heading_loss = nn.L1Loss(reduction='mean')
        
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        total_loss = 0
        
        for batch_idx, (pred_indices, tgt_indices) in enumerate(indices):
            # Position/dimension loss
            pred_params = outputs['params'][batch_idx, pred_indices]
            tgt_params = targets[batch_idx]['boxes'][tgt_indices]
            
            box_loss = self.box_loss(pred_params[:, :6], tgt_params[:, :6])
            
            # Heading loss (circular)
            heading_diff = torch.abs(pred_params[:, 6] - tgt_params[:, 6])
            heading_loss = torch.mean(torch.min(heading_diff, 2*np.pi - heading_diff))
            
            # Class loss
            cls_loss = self.cls_loss(
                outputs['class_logits'][batch_idx, pred_indices],
                targets[batch_idx]['labels'][tgt_indices]
            )
            
            total_loss += box_loss + heading_loss + cls_loss
            
        return total_loss / len(indices)

class PointFusion3D(nn.Module):
    """Simplified model predicting direct parameters"""
    def __init__(self, num_cameras=7, max_predictions=128):
        super().__init__()
        self.img_backbone = ImageBackbone(num_cameras)
        self.pointnet = ModifiedPointNet()
        self.fusion = FusionNetwork(max_predictions)
        
    def forward(self, batch):
        img_feats = self.img_backbone(batch['images'])
        point_feats = self.pointnet(batch['points'])
        return self.fusion(img_feats, point_feats)

def batch_inspection(batch):
    print(f"\nBatch structure (size: {batch['points'].shape[0]}):")
    print(f"Points shape: {batch['points'].shape}")
    for camera_name, image in batch['images'].items():
        print(f"Image {camera_name} shape: {image.shape}")
    print("\nAnnotations:")
    for i, annotations in enumerate(batch['annotations']):
        print(f"\tSample {i}: Boxes {annotations['boxes'].shape}, Labels {annotations['labels'].shape}")

if __name__ == "__main__":
    from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
    from torch.utils.data import DataLoader
    from dotenv import load_dotenv
    load_dotenv()
    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Test with sample data
    dataset = PointFusionloader(dataset_path, split='train')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)
    batch = next(iter(dataloader))

    batch_inspection(batch)
    
    model = PointFusion3D()
    outputs = model(batch)
    
    print("\nModel outputs:")
    print(f"Parameters shape: {outputs['params'].shape}")  # [2, 128, 7]
    print(f"Scores shape: {outputs['scores'].shape}")       # [2, 128]
    print(f"Class logits shape: {outputs['class_logits'].shape}")  # [2, 128, 4]

# python -m src.models.PointFusion