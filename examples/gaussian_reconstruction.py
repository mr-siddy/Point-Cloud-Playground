import argparse
import os
from point_cloud_processor.core.processor import PointCloudProcessor
from point_cloud_processor.visualization.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Reconstruct surface using Gaussian Splatting")
    parser.add_argument("input_file", help="Path to input point cloud file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--num-gaussians", type=int, default=1000, help="Number of Gaussians")
    parser.add_argument("--iterations", type=int, default=100, help="Number of optimization iterations")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # Create processor
    processor = PointCloudProcessor(file_path=args.input_file)

    # Align floor
    aligned_pcd = processor.detect_floor()

    # Reconstruct using Gaussian Splatting
    mesh = processor.reconstruct_surface(
        method='gaussian',
        num_gaussians=args.num_gaussians,
        num_iterations=args.iterations,
        learning_rate=args.learning_rate
    )

    # Create visualizations
    visualizer = Visualizer()
    aligned_fig = visualizer.plot_point_cloud(aligned_pcd.points)
    mesh_fig = visualizer.plot_mesh(mesh)

    # Save results
    save_path = visualizer.save_visualizations(
        args.output_dir,
        geometries={
            'aligned_cloud': aligned_pcd,
            'gaussian_mesh': mesh
        },
        figures={
            'aligned_viz': aligned_fig,
            'gaussian_viz': mesh_fig
        }
    )

    print(f"Results saved to: {save_path}")

if __name__ == "__main__":
    main()