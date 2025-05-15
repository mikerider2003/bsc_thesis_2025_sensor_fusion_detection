import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.models.PointFusion import PointFusion3D
from dotenv import load_dotenv
from src.loaders.loader_Point_Fusion import PointFusionloader, custom_collate
from sklearn.model_selection import train_test_split

def visualize_first_sample(model, test_loader, device, confidence_thresh=0.3):
    """Visualize raw predictions and ground truth without normalization"""
    import open3d as o3d
    
    model.eval()
    
    # Get first sample
    batch = next(iter(test_loader))

    points = batch['points'][0].numpy()
    images = {k: v[0] for k, v in batch['images'].items()}
    annotations = batch['annotations'][0]
    
    # Get raw predictions (no denormalization)
    with torch.no_grad():
        outputs = model({
            'points': batch['points'].to(device),
            'images': {k: v.to(device) for k, v in batch['images'].items()}
        })
    
    # Extract raw outputs
    raw_params = outputs['params'][0].cpu().numpy()      # [K, 7] (x,y,z,l,w,h,heading)
    raw_scores = outputs['scores'][0].cpu().numpy()      # [K]
    raw_classes = torch.argmax(outputs['class_logits'][0], dim=1).cpu().numpy()  # [K]

    # Filter predictions
    mask = raw_scores >= confidence_thresh
    filtered_params = raw_params[mask]
    filtered_scores = raw_scores[mask]
    filtered_classes = raw_classes[mask]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey points
    
    # Ground truth boxes (green)
    gt_boxes = []
    for box in annotations['boxes'].numpy():
        x, y, z, l, w, h, heading = box
        center = (x, y, z)
        extent = np.array([l, w, h])
        
        R = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])
        
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        lines.paint_uniform_color([0, 1, 0])  # Green for GT
        gt_boxes.append(lines)

    # Predicted boxes (red) with raw parameters
    pred_boxes = []
    for box in filtered_params:
        x, y, z, l, w, h, heading = box
        center = (x, y, z)
        extent = np.array([l, w, h])
        
        R = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])
        
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        lines.paint_uniform_color([1, 0, 0])  # Red for predictions
        pred_boxes.append(lines)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Add boxes with coordinate labels
    for box in gt_boxes + pred_boxes:
        vis.add_geometry(box)
    
    # Print raw values for debugging
    print("\nRaw Predictions:")
    print(f"Position (x,y,z): {filtered_params[:, :3]}")
    print(f"Dimensions (l,w,h): {filtered_params[:, 3:6]}")
    print(f"Heading (rad): {filtered_params[:, 6]}")
    print(f"Confidence: {filtered_scores}")
    print(f"Class IDs: {filtered_classes}")

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_front([0.4, -0.2, 0.2])
    ctr.set_lookat([0, 0, 1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.03)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.getenv('DATA_PATH', default='src/data/')
    
    # Load dataset
    full_dataset = PointFusionloader(data_path, split='train')
    indicies = list(range(10))  # Temporary subset for development
    full_dataset = Subset(full_dataset, indicies)
    
    # Create dataloader
    test_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    
    # Initialize model
    model = PointFusion3D(max_predictions=128).to(device)
    
    # Visualize first sample
    visualize_first_sample(model, test_loader, device)


# python -m src.evaluation.test