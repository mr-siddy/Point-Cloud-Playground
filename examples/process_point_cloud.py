import argparse
from point_cloud_processor.core.processor import PointCloudProcessor
from point_cloud_processor.visualization.visualizer import Visualizer
from point_cloud_processor.utils.io_utils import create_output_directory

def main():
    parser = argparse.ArgumentParser(description="Process point cloud data")
    parser.add_argument("input_file", help="Path to input point cloud file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--method", choices=['poisson', 'bpa'], default='poisson',
                      help="Surface reconstruction method")
    args = parser.parse_args()

    # Create processor
    processor = PointCloudProcessor(file_path=args.input_file)

    # Process steps
    aligned_pcd = processor.detect_floor()
    reconstructed_mesh = processor.reconstruct_surface(method=args.method)
    test_results = processor.test_floor_alignment()

    # Create output directory
    save_dir = create_output_directory(args.output_dir)

    # Create visualizations
    visualizer = Visualizer()
    cloud_fig = visualizer.plot_point_cloud(aligned_pcd.points)
    mesh_fig = visualizer.plot_mesh(reconstructed_mesh)

    # Save results and visualizations
    visualizer.save_visualizations(save_dir, {
        'aligned_cloud': cloud_fig,
        'reconstructed_mesh': mesh_fig
    })

    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    main()