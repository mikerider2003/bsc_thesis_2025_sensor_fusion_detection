import os
import pandas as pd
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from dotenv import load_dotenv
from pyarrow import feather
from PIL import Image

from src.loaders.loader import ArgoDataset

def visualize_point_cloud(lidar_file, annotations, save_path=None):
    # Load the point cloud data from the feather file
    point_cloud_data = feather.read_table(lidar_file).to_pandas()
    # Point cloud data is pd.DataFrame with columns [x, y, z, intensity]
    # Annotations are pd.DataFrame with columns [category  length_m   width_m  height_m        qw   qx   qy        qz        tx_m       ty_m      tz_m  ]
    
    # Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[['x', 'y', 'z']].values)
    
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Grey points

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720, visible=save_path is None)
    vis.add_geometry(pcd)
    
    # Adjust point size (smaller than default)
    opt = vis.get_render_option()
    opt.point_size = 2.0  # Smaller point size (adjust as needed)
    
    # Add bounding boxes for each annotation
    target_classes = {'PEDESTRIAN', 'TRUCK', 'LARGE_VEHICLE', 'REGULAR_VEHICLE'}
    
    if annotations is not None and not annotations.empty:
        for _, ann in annotations.iterrows():
            # Only visualize target classes
            if ann['category'] in target_classes:
                # Create bounding box
                box = create_bounding_box(
                    length=ann['length_m'],
                    width=ann['width_m'], 
                    height=ann['height_m'],
                    quaternion=[ann['qw'], ann['qx'], ann['qy'], ann['qz']],
                    center=[ann['tx_m'], ann['ty_m'], ann['tz_m']],
                    category=ann['category']
                )
                vis.add_geometry(box)
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_front([0.4, -0.2, 0.2])
    ctr.set_lookat([0, 0, 1])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.03)
    
    # Update geometry and render
    vis.poll_events()
    vis.update_renderer()
    
    # Save to file if path is provided
    if save_path is not None:
        vis.capture_screen_image(save_path, do_render=True)
        print(f"Visualization saved to {save_path}")
        vis.destroy_window()
    else:
        # Interactive visualization
        vis.run()
        vis.destroy_window()

def create_bounding_box(length, width, height, quaternion, center, category):
    """
    Create an Open3D LineSet for visualizing a 3D bounding box.
    
    Args:
        length (float): Length of the bounding box
        width (float): Width of the bounding box
        height (float): Height of the bounding box
        quaternion (list): Rotation quaternion [qw, qx, qy, qz]
        center (list): Center position [x, y, z]
        category (str): Object category for color assignment
    
    Returns:
        o3d.geometry.LineSet: The 3D bounding box as a line set
    """
    # Create a box centered at origin with dimensions length x width x height
    box_points = [
        [-length/2, -width/2, -height/2],
        [length/2, -width/2, -height/2],
        [length/2, width/2, -height/2],
        [-length/2, width/2, -height/2],
        [-length/2, -width/2, height/2],
        [length/2, -width/2, height/2],
        [length/2, width/2, height/2],
        [-length/2, width/2, height/2]
    ]
    box_points = np.array(box_points)
    
    # Create rotation matrix from quaternion
    r = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # scipy uses [x,y,z,w]
    rotation_matrix = r.as_matrix()
    
    # Rotate the box
    box_points = np.dot(box_points, rotation_matrix.T)
    
    # Translate the box
    box_points = box_points + np.array(center)
    
    # Create edges: connect each corner with its adjacent corners
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]
    
    # Choose color based on category
    color_map = {
        'REGULAR_VEHICLE': [0, 1, 0],     # Green
        'PEDESTRIAN': [1, 0, 0],          # Red
        'TRUCK': [0, 0, 1],               # Blue
        'LARGE_VEHICLE': [0.7, 0, 1]      # Purple
    }
    color = color_map.get(category, [1, 1, 0])  # Default to yellow if unknown
    
    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

def visualize_camera_frames(camera_frames, seqence_path, save_path=None):
    images = []
    
    # Load the camera frames
    for camera_name, image_id in camera_frames.items():
        # Load the image
        img_path = os.path.join(seqence_path, f"sensors/cameras/{camera_name}/{image_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        images.append(image)

    # Concat images next to each other
    if not images:
        print("No images to visualize")
        return
        
    # Resize all images to the same height
    heights = [img.height for img in images]
    min_height = min(heights)  # Use smallest height to avoid excessive memory usage
    
    # Resize images maintaining aspect ratio
    resized_images = []
    for img in images:
        aspect_ratio = img.width / img.height
        new_width = int(min_height * aspect_ratio)
        resized_images.append(img.resize((new_width, min_height), Image.LANCZOS))
    
    # Calculate total width
    total_width = sum(img.width for img in resized_images)
    
    # Create a new image with the combined width and height
    combined_image = Image.new('RGB', (total_width, min_height))
    
    # Paste images side by side
    current_width = 0
    for img in resized_images:
        combined_image.paste(img, (current_width, 0))
        current_width += img.width
    
    # Save or display the combined image
    if save_path is not None:
        combined_image.save(save_path)
        print(f"Camera visualization saved to {save_path}")
    else:
        # Display the image
        combined_image.show()
    
    return combined_image

def main():
    # Load the dataset
    load_dotenv()

    dataset_path = os.getenv('DATA_PATH', default='src/data/')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    
    training_set = ArgoDataset(root_dir=dataset_path,
                               split='train')
    
    # select first sample
    sample = training_set[0]
    print("Sample consists of:")
    for key, value in sample.items():
        print(f"\t{key}")

    lidar_file = sample['lidar_file']
    annotations = sample['annotations']
    # visualize_point_cloud(lidar_file=lidar_file, 
    #                       annotations=annotations,
    #                       save_path="src/utils/visualization/figures/example_lidar.png")

    camera_frames = sample['camera_frames']
    seqence_path = sample['sequence_path']
    visualize_camera_frames(camera_frames=camera_frames, 
                            seqence_path=seqence_path,
                            save_path="src/utils/visualization/figures/example_camera.png")

if __name__ == "__main__":
    main()

# python -m src.utils.visualization.vis_sample