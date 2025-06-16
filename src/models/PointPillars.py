import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

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
                'TRUCK':           (12.0, 2.5, 3.5),
                'LARGE_VEHICLE':   (8.0, 2.8, 3.0),
                'REGULAR_VEHICLE': (4.0, 1.8, 1.6),
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
            cz = h 
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
    
    def plot_anchors(self, sample_stride=50, max_anchors=500):
        """
        Plot a sample of generated anchors in Bird's Eye View (BEV).
        
        Args:
            sample_stride (int): Stride for sampling anchors to plot
            max_anchors (int): Maximum number of anchors to display
        """

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from matplotlib.collections import PatchCollection

        anchors, anchor_classes = self.generate()

        # Convert to numpy for easier handling
        anchors = anchors.numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Anchor Boxes - Bird\'s Eye View')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_aspect('equal')
        
        # Determine plot limits
        max_extent = max(np.max(anchors[:, 3]), np.max(anchors[:, 4])) * 1.2
        cx_min, cx_max = np.min(anchors[:, 0]), np.max(anchors[:, 0])
        cy_min, cy_max = np.min(anchors[:, 1]), np.max(anchors[:, 1])
        
        ax.set_xlim(cx_min - max_extent, cx_max + max_extent)
        ax.set_ylim(cy_min - max_extent, cy_max + max_extent)
        
        # Create color mapping for classes
        class_colors = {
            'PEDESTRIAN': 'red',
            'TRUCK': 'blue',
            'LARGE_VEHICLE': 'green',
            'REGULAR_VEHICLE': 'purple'
        }
        
        # Create legend handles
        legend_handles = []
        for cls, color in class_colors.items():
            legend_handles.append(patches.Patch(color=color, label=cls))
        
        # Create patches for anchors
        all_patches = []
        
        # Sample anchors to plot
        num_anchors = anchors.shape[0]
        sample_indices = range(0, num_anchors, sample_stride)
        if len(sample_indices) > max_anchors:
            sample_indices = np.random.choice(num_anchors, max_anchors, replace=False)
        
        for idx in sample_indices:
            cx, cy, cz, l, w, h, qw, qx, qy, qz = anchors[idx]
            class_name = anchor_classes[idx]
            
            # Calculate yaw angle from quaternion
            yaw = 2 * np.arctan2(qz, qw)
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (cx - l/2, cy - w/2),  # bottom left corner
                l, w,                   # length and width
                angle=np.degrees(yaw),   # rotation in degrees
                color=class_colors.get(class_name, 'gray'),
                alpha=0.4
            )
            all_patches.append(rect)
            
            # Add center point
            ax.plot(cx, cy, 'o', markersize=2, color=class_colors.get(class_name, 'gray'))
        
        # Add all patches to the plot
        collection = PatchCollection(all_patches, match_original=True)
        ax.add_collection(collection)
        
        # Add legend
        ax.legend(handles=legend_handles, loc='upper right')
        
        # Add grid information to title
        H, W = self.grid_size
        plt.title(f"Anchor Boxes (Grid: {H}x{W} cells, {self.grid_resolution}m/cell)\n"
                  f"Showing {len(all_patches)} of {num_anchors} anchors")
        
        plt.tight_layout()
        plt.show()
        
    def assign(self, annotations_df):
        pass
    

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
    visualize_pseudo_image(canvas)

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
    achors, anchor_classes = anchor.generate()
    anchor.plot_anchors()
    print(f"Generated {len(achors)} anchors with classes: {len(anchor_classes)}")



    # small_anchor = Anchor(grid_size=(2, 2))
    # small_anchor.plot_anchors()  

    # python -m src.models.PointPillars



