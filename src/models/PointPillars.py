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
        # Example backbone: 3 convolutional blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x  # [B, 256, H/8, W/8]

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=3, box_code_size=9):
        """
        Args:
            in_channels (int): Number of channels from the backbone output.
            num_classes (int): Number of object classes (excluding background).
            box_code_size (int): Number of box regression parameters (default 9: cx, cy, cz, l, w, h, qw, qx, qy).
        """
        super().__init__()
        self.box_head = nn.Conv2d(in_channels, box_code_size, kernel_size=1)
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Backbone feature map, shape [B, in_channels, H, W]
        Returns:
            box_preds (torch.Tensor): [B, box_code_size, H, W]
            cls_preds (torch.Tensor): [B, num_classes, H, W]
        """
        box_preds = self.box_head(x)
        cls_preds = self.cls_head(x)
        return box_preds, cls_preds

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
        
        # Convert grid centers to meters (0.2m per cell)
        x_centers = (torch.arange(W, dtype=torch.float32) + 0.5) * self.grid_resolution
        y_centers = (torch.arange(H, dtype=torch.float32) + 0.5) * self.grid_resolution

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
                block_anchors[:, 0] = grid_x.reshape(-1)  # cx (meters)
                block_anchors[:, 1] = grid_y.reshape(-1)  # cy (meters)
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
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5))
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
            
        
        # Hide unused subplots
        for i in range(num_batches, len(axes) if num_batches > 1 else 1):
            if num_batches > 1:
                axes[i].axis('off')
        
        plt.suptitle('Anchor Assignment Visualization', fontsize=16)
        plt.show()
        

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
    
    # Test Pillar Feature Network
    pfn = PillarFeatureNet(num_input_features=9, num_output_features=64)
    pillar_features = sample['features']  
    learned_features = pfn(pillar_features)
    print(f"PillarFeatureNet output shape: {learned_features.shape}")

    # Test PsuedoScatter
    scatter = PsuedoScatter(num_input_features=64, grid_size_xy=grid_size)
    canvas = scatter(learned_features, sample['pillar_coords'])
    print(f"PsuedoScatter output shape: {canvas.shape}")
    # visualize_pseudo_image(canvas)

    # Test backbone
    backbone = Backbone()
    backbone_output = backbone(canvas)
    print(f"PointPillarsBackbone output shape: {backbone_output.shape}")

    # Test detection head
    detection_head = DetectionHead(in_channels=256, num_classes=train_dataset.num_classes, box_code_size=9)
    box_preds, cls_preds = detection_head(backbone_output)
    print(f"DetectionHead box_preds shape: {box_preds.shape}, cls_preds shape: {cls_preds.shape}")

    # Test Anchor
    anchor = Anchor(grid_size=grid_size)
    # anchor.plot_anchors()
    batch_results = anchor.assign(sample['annotations'])
    anchor.visualize_matched_anchors(batch_results, sample['annotations'])



    # small_anchor = Anchor(grid_size=(2, 2))
    # small_anchor.plot_anchors()  

    # python -m src.models.PointPillars



