import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ImageBackbone(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1024), nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1), nn.Softmax(dim=1))
        
    def forward(self, points):
        point_feats = self.mlp(points)
        attn_weights = self.attention(point_feats)
        return torch.sum(point_feats * attn_weights, dim=1)

class FusionNetwork(nn.Module):
    def __init__(self, max_predictions=128):
        super().__init__()
        self.max_pred = max_predictions
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2048 + 1024 + 3, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU())
        
        self.param_head = nn.Linear(512, 7*max_predictions)
        self.score_head = nn.Linear(512, max_predictions)
        self.class_head = nn.Linear(512, 4*max_predictions)

        nn.init.kaiming_normal_(self.param_head.weight, mode='fan_out')
        nn.init.constant_(self.param_head.bias, 0.0)
        
    def forward(self, img_feats, point_feats, spatial_priors):
        fused = self.fusion_mlp(torch.cat([img_feats, point_feats, spatial_priors], 1))
        raw_params = self.param_head(fused).view(-1, self.max_pred, 7)
        
        params = torch.clone(raw_params)
        params[..., 3:6] = torch.sigmoid(raw_params[..., 3:6]) * 10
        params[..., 6] = torch.remainder(raw_params[..., 6], 2 * np.pi)
        
        return {
            'params': params,
            'scores': torch.sigmoid(self.score_head(fused)),
            'class_logits': self.class_head(fused).view(-1, self.max_pred, 4)
        }

class HungarianMatcher(nn.Module):
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
            pred_params = outputs['params'][b]
            tgt_boxes = targets[b]['boxes']
            tgt_labels = targets[b]['labels']

            if tgt_boxes.shape[0] == 0:
                indices.append((np.array([]), np.array([])))
                continue

            K, M = pred_params.shape[0], tgt_boxes.shape[0]
            
            pos_cost = torch.cdist(pred_params[:, :3], tgt_boxes[:, :3])
            dim_cost = torch.cdist(pred_params[:, 3:6], tgt_boxes[:, 3:6])
            
            pred_heading = pred_params[:, 6].unsqueeze(1)
            tgt_heading = tgt_boxes[:, 6].unsqueeze(0)
            heading_diff = torch.abs(pred_heading - tgt_heading)
            heading_cost = torch.min(heading_diff, 2*np.pi - heading_diff)
            
            pred_probs = outputs['class_logits'][b].softmax(-1)
            class_cost = 1 - pred_probs[:, tgt_labels]

            C = (
                self.cost_bbox * (pos_cost + 1.5*dim_cost) +
                self.cost_heading * heading_cost +
                self.cost_class * class_cost
            )

            pred_idx, tgt_idx = linear_sum_assignment(C.cpu())
            indices.append((pred_idx, tgt_idx))
            
        return indices

class PointFusionLoss(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher
        self.box_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.heading_loss = nn.L1Loss()
        
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        total_loss = torch.tensor(0.0, device=outputs['params'].device)
        valid_batches = 0
        
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0 or len(pred_idx) == 0:
                continue

            pred_idx = torch.as_tensor(pred_idx, dtype=torch.long, device=outputs['params'].device)
            tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.long, device=outputs['params'].device)

            matched_preds = outputs['params'][batch_idx][pred_idx]
            matched_targets = targets[batch_idx]['boxes'][tgt_idx]
            matched_labels = targets[batch_idx]['labels'][tgt_idx]

            box_loss = self.box_loss(matched_preds[:, :6], matched_targets[:, :6])
            heading_diff = torch.abs(matched_preds[:, 6] - matched_targets[:, 6])
            heading_loss = torch.mean(torch.min(heading_diff, 2*np.pi - heading_diff))
            cls_loss = self.cls_loss(outputs['class_logits'][batch_idx, pred_idx], matched_labels)

            total_loss += box_loss + heading_loss + cls_loss
            valid_batches += 1

        if valid_batches > 0:
            pred_positions = outputs['params'][..., :3]
            spread_loss = torch.mean(torch.cdist(pred_positions, pred_positions))
            total_loss += 0.1 * spread_loss

        return total_loss / valid_batches if valid_batches > 0 else total_loss

class PointFusion3D(nn.Module):
    def __init__(self, num_cameras=7, max_predictions=128):
        super().__init__()
        self.img_backbone = ImageBackbone(num_cameras)
        self.pointnet = ModifiedPointNet()
        self.fusion = FusionNetwork(max_predictions)
        
    def forward(self, batch):
        img_feats = self.img_backbone(batch['images'])
        point_feats = self.pointnet(batch['points'])
        spatial_priors = batch['points'].mean(dim=1)
        return self.fusion(img_feats, point_feats, spatial_priors)

def batch_inspection(batch):
    print(f"\nBatch structure (size: {batch['points'].shape[0]}):")
    print(f"Points shape: {batch['points'].shape}")
    for camera_name, image in batch['images'].items():
        print(f"Image {camera_name} shape: {image.shape}")
    print("\nAnnotations:")
    for i, annotations in enumerate(batch['annotations']):
        print(f"\tSample {i}: Boxes {annotations['boxes'].shape}, Labels {annotations['labels'].shape}")

if __name__ == "__main__":
    import os
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