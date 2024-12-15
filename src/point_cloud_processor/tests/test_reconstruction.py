import unittest
import numpy as np
import open3d as o3d
from ..reconstruction import GaussianSurfaceReconstruction

class TestGaussianReconstruction(unittest.TestCase):
    def setUp(self):
        # Create test point cloud (sphere)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        self.pcd = mesh.sample_points_uniformly(number_of_points=1000)
        self.points = np.asarray(self.pcd.points)
        
    def test_initialization(self):
        reconstructor = GaussianSurfaceReconstruction(num_gaussians=100)
        self.assertIsNotNone(reconstructor)
        
    def test_reconstruction(self):
        reconstructor = GaussianSurfaceReconstruction(
            num_gaussians=100,
            num_iterations=10  # Reduced for testing
        )
        params = reconstructor.fit(self.points)
        mesh = reconstructor.generate_mesh()
        
        self.assertIsNotNone(mesh)
        self.assertTrue(len(np.asarray(mesh.vertices)) > 0)
        self.assertTrue(len(np.asarray(mesh.triangles)) > 0)
        
    def test_optimization(self):
        reconstructor = GaussianSurfaceReconstruction(
            num_gaussians=100,
            num_iterations=20
        )
        params = reconstructor.fit(self.points)
        
        # Check if loss decreases
        losses = params['losses']
        self.assertTrue(losses[-1] < losses[0])

if __name__ == '__main__':
    unittest.main()