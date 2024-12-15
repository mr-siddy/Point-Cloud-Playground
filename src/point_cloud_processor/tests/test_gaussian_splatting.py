import unittest
import numpy as np
import open3d as o3d
from ..reconstruction.gaussian_splatting import GaussianSplat, reconstruct_surface_gaussian

class TestGaussianSplatting(unittest.TestCase):
    def setUp(self):
        # Create a simple sphere point cloud for testing
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        self.test_pcd = sphere.sample_points_uniformly(number_of_points=1000)
        
    def test_gaussian_initialization(self):
        """Test if Gaussian parameters are properly initialized"""
        points = np.asarray(self.test_pcd.points)
        splatter = GaussianSplat(num_gaussians=100)
        positions, scales, rotations = splatter.initialize_gaussians(points)
        
        self.assertEqual(positions.shape, (100, 3))
        self.assertEqual(scales.shape, (100, 3))
        self.assertEqual(rotations.shape, (100, 4))
        
    def test_reconstruction(self):
        """Test if reconstruction produces valid mesh"""
        mesh, params = reconstruct_surface_gaussian(
            self.test_pcd,
            num_gaussians=100,
            num_iterations=10  # Reduced for testing
        )
        
        # Check if mesh is valid
        self.assertTrue(isinstance(mesh, o3d.geometry.TriangleMesh))
        self.assertTrue(len(mesh.vertices) > 0)
        self.assertTrue(len(mesh.triangles) > 0)
        
    def test_optimization(self):
        """Test if optimization reduces loss"""
        points = np.asarray(self.test_pcd.points)
        splatter = GaussianSplat(num_gaussians=100, num_iterations=20)
        params = splatter.fit(points)
        
        # Check if loss decreases
        losses = params['losses']
        self.assertTrue(losses[-1] < losses[0])
        
if __name__ == '__main__':
    unittest.main()