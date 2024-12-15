import os
import open3d as o3d
from datetime import datetime

def create_output_directory(base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_path, f"process_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def save_geometry(save_dir, name, geometry):
    """Save point clouds or meshes with proper type checking"""
    if isinstance(geometry, o3d.geometry.PointCloud):
        path = os.path.join(save_dir, f"{name}.ply")
        o3d.io.write_point_cloud(path, geometry)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        path = os.path.join(save_dir, f"{name}.ply")
        o3d.io.write_triangle_mesh(path, geometry)
    return path

def load_point_cloud(file_path):
    """Load point cloud with error handling"""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError("Point cloud is empty")
        return pcd
    except Exception as e:
        raise ValueError(f"Failed to load point cloud from {file_path}: {str(e)}")