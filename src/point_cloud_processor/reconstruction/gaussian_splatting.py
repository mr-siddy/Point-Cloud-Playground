import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

class GaussianSplat:
    def __init__(self, num_gaussians=1000, learning_rate=0.01, num_iterations=100):
        self.num_gaussians = num_gaussians
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize_gaussians(self, points):
        """Initialize Gaussian parameters from point cloud"""
        # Sample points for Gaussian centers
        indices = np.random.choice(points.shape[0], self.num_gaussians, replace=False)
        positions = torch.from_numpy(points[indices]).float().to(self.device)
        
        # Initialize scales (covariance matrices)
        kdtree = KDTree(points)
        avg_dist = np.mean([kdtree.query(points[i].reshape(1, -1), k=10)[0] for i in indices])
        scales = torch.ones(self.num_gaussians, 3).to(self.device) * avg_dist
        
        # Initialize rotations as quaternions
        rotations = torch.zeros(self.num_gaussians, 4).to(self.device)
        rotations[:, 0] = 1.0  # Initialize as identity quaternions
        
        # Make parameters learnable
        self.positions = nn.Parameter(positions)
        self.scales = nn.Parameter(torch.log(scales))  # Use log-space for numerical stability
        self.rotations = nn.Parameter(rotations)
        
        return self.positions, self.scales, self.rotations
    
    def quaternion_to_rotation_matrix(self, quaternions):
        """Convert quaternions to rotation matrices"""
        batch_size = quaternions.size(0)
        
        # Normalize quaternions
        quaternions = F.normalize(quaternions, p=2, dim=1)
        
        # Extract components
        w, x, y, z = quaternions.unbind(1)
        
        # Compute rotation matrix elements
        rot_matrix = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=1).reshape(batch_size, 3, 3)
        
        return rot_matrix
    
    def compute_gaussian_covariance(self, scales, rotations):
        """Compute covariance matrices for all Gaussians"""
        # Convert scales from log-space
        scales = torch.exp(scales)
        
        # Get rotation matrices
        rot_matrices = self.quaternion_to_rotation_matrix(rotations)
        
        # Create scale matrices
        scale_matrices = torch.diag_embed(scales)
        
        # Compute covariance: R * S * R^T
        covariances = torch.bmm(torch.bmm(rot_matrices, scale_matrices), rot_matrices.transpose(1, 2))
        
        return covariances
    
    def compute_coverage_loss(self, target_points):
        """Compute how well Gaussians cover the target points"""
        # Convert target points to tensor
        target_points = torch.from_numpy(target_points).float().to(self.device)
        
        # Compute distances between Gaussians and target points
        dists = torch.cdist(self.positions, target_points)
        min_dists = torch.min(dists, dim=0)[0]
        
        # Coverage loss
        coverage_loss = torch.mean(min_dists)
        
        return coverage_loss
    
    def compute_overlap_loss(self, covariances):
        """Compute overlap between Gaussians to encourage better spacing"""
        # Compute pairwise distances between Gaussian centers
        dists = torch.cdist(self.positions, self.positions)
        
        # Compute approximate overlap using distance and scales
        scales_sum = torch.sum(torch.exp(self.scales), dim=1)
        overlap = torch.exp(-dists / (scales_sum.unsqueeze(0) + scales_sum.unsqueeze(1)))
        
        # Remove self-overlap from diagonal
        overlap = overlap * (1 - torch.eye(self.num_gaussians, device=self.device))
        
        return torch.mean(overlap)
    
    def fit(self, points):
        """Fit Gaussians to point cloud"""
        # Initialize parameters
        self.initialize_gaussians(points)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([self.positions, self.scales, self.rotations], 
                                   lr=self.learning_rate)
        
        losses = []
        for iteration in range(self.num_iterations):
            optimizer.zero_grad()
            
            # Compute covariance matrices
            covariances = self.compute_gaussian_covariance(self.scales, self.rotations)
            
            # Compute losses
            coverage_loss = self.compute_coverage_loss(points)
            overlap_loss = self.compute_overlap_loss(covariances)
            smoothness_loss = torch.mean(torch.abs(self.scales)) + torch.mean(torch.abs(self.rotations[:, 1:]))
            
            # Total loss
            loss = coverage_loss + 0.1 * overlap_loss + 0.01 * smoothness_loss
            losses.append(loss.item())
            
            # Optimize
            loss.backward()
            optimizer.step()
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        
        return {
            'positions': self.positions.detach().cpu().numpy(),
            'scales': torch.exp(self.scales).detach().cpu().numpy(),
            'rotations': self.rotations.detach().cpu().numpy(),
            'losses': losses
        }
    
    def to_mesh(self, params, resolution=32):
        """Convert Gaussian splats to mesh for visualization"""
        positions = params['positions']
        scales = params['scales']
        rotations = params['rotations']
        
        # Create mesh vertices for each Gaussian
        vertices = []
        triangles = []
        vertex_offset = 0
        
        for i in range(self.num_gaussians):
            # Create sphere vertices
            phi = np.linspace(0, 2*np.pi, resolution)
            theta = np.linspace(0, np.pi, resolution)
            phi, theta = np.meshgrid(phi, theta)
            
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            # Scale vertices
            scaled_vertices = np.stack([x, y, z], axis=2) * scales[i]
            
            # Rotate vertices
            rot_matrix = self.quaternion_to_rotation_matrix(torch.tensor(rotations[i:i+1]).float()).cpu().numpy()[0]
            rotated_vertices = np.dot(scaled_vertices.reshape(-1, 3), rot_matrix.T)
            
            # Translate vertices
            translated_vertices = rotated_vertices + positions[i]
            
            # Create triangles
            for j in range(resolution-1):
                for k in range(resolution-1):
                    v00 = j * resolution + k + vertex_offset
                    v01 = j * resolution + k + 1 + vertex_offset
                    v10 = (j + 1) * resolution + k + vertex_offset
                    v11 = (j + 1) * resolution + k + 1 + vertex_offset
                    
                    triangles.extend([
                        [v00, v01, v11],
                        [v00, v11, v10]
                    ])
            
            vertices.extend(translated_vertices)
            vertex_offset += resolution * resolution
        
        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        return mesh

def reconstruct_surface_gaussian(pcd, num_gaussians=1000, learning_rate=0.01, num_iterations=100):
    """Main function to perform Gaussian Splatting reconstruction"""
    points = np.asarray(pcd.points)
    
    # Create and fit Gaussian splats
    gaussian_splatter = GaussianSplat(num_gaussians, learning_rate, num_iterations)
    params = gaussian_splatter.fit(points)
    
    # Convert to mesh for visualization
    mesh = gaussian_splatter.to_mesh(params)
    
    return mesh, params