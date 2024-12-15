import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime
import open3d as o3d
from IPython.display import display

class Visualizer:
    @staticmethod
    def plot_point_cloud(points, colors=None, title="Point Cloud"):
        if colors is None:
            colors = np.zeros_like(points)
            colors[:, 2] = 1
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors[:, 2],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=800,
            height=800
        )
        
        return fig

    @staticmethod
    def plot_mesh(mesh, title="Reconstructed Surface"):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        colors = (vertices - vertices.min(axis=0)) / (vertices.max(axis=0) - vertices.min(axis=0))
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                vertexcolor=colors,
                opacity=0.8
            )
        ])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=800,
            height=800
        )
        
        return fig

    @staticmethod
    def save_visualizations(save_dir, geometries=None, figures=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"visualization_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        
        if geometries:
            geometry_path = os.path.join(save_path, "geometry")
            os.makedirs(geometry_path, exist_ok=True)
            
            for name, geometry in geometries.items():
                if isinstance(geometry, o3d.geometry.PointCloud):
                    path = os.path.join(geometry_path, f"{name}.ply")
                    o3d.io.write_point_cloud(path, geometry)
                elif isinstance(geometry, o3d.geometry.TriangleMesh):
                    path = os.path.join(geometry_path, f"{name}.ply")
                    o3d.io.write_triangle_mesh(path, geometry)
        
        if figures:
            viz_path = os.path.join(save_path, "visualizations")
            os.makedirs(viz_path, exist_ok=True)
            
            for name, fig in figures.items():
                path = os.path.join(viz_path, f"{name}.html")
                fig.write_html(path)
        
        return save_path
    @staticmethod
    def display_visualizations(figures):
        """
        Display multiple visualizations in a notebook environment
        
        Args:
            figures (dict): Dictionary of plotly figures with format {name: figure}
        """
        for name, fig in figures.items():
            print(f"\n{name}:")
            display(fig)