class PointCloudProcessor:
    def __init__(self, file_path=None, pcd=None):
        if pcd is not None:
            self.pcd = pcd
        elif file_path is not None:
            try:
                self.pcd = o3d.io.read_point_cloud(file_path)
                if not self.pcd.has_points():
                    raise ValueError("Point cloud is empty")
            except Exception as e:
                raise ValueError(f"Failed to load point cloud from {file_path}: {str(e)}")
        else:
            raise ValueError("Either file_path or pcd must be provided")
            
        self.points = np.asarray(self.pcd.points)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initialized PointCloudProcessor with {len(self.points)} points")

    def detect_floor(self):
        """
        Detect floor plane using RANSAC and align it with YZ plane
        """
        print("Starting floor detection...")
        try:
            plane_model, inliers = self.pcd.segment_plane(
                distance_threshold=0.01,
                ransac_n=3,
                num_iterations=1000
            )
            print(f"Found floor plane with {len(inliers)} inlier points")
            
            floor_points = self.points[inliers]
            floor_center = np.mean(floor_points, axis=0)
            
            # Get plane normal
            a, b, c, d = plane_model
            plane_normal = np.array([a, b, c])
            
            # Calculate rotation to align with Y=0 plane
            target_normal = np.array([0, 1, 0])
            rotation_axis = np.cross(plane_normal, target_normal)
            if np.all(rotation_axis == 0):
                rotation_axis = np.array([1, 0, 0])
            else:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(plane_normal, target_normal))
            
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * rotation_angle
            )
            
            self.pcd.rotate(R, center=floor_center)
            self.pcd.translate(-floor_center)
            
            self.points = np.asarray(self.pcd.points)
            print("Floor alignment completed successfully")
            return self.pcd
            
        except Exception as e:
            print(f"Error during floor detection: {str(e)}")
            raise

    # def reconstruct_surface(self, method='poisson', **kwargs):
        """
        Convert point cloud to mesh using surface reconstruction
        """
        print(f"Starting surface reconstruction using {method} method...")
        try:
            if not self.pcd.has_normals():
                print("Estimating normals...")
                self.pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30))
                self.pcd.orient_normals_consistent_tangent_plane(100)
            
            if method == 'poisson':
                print("Performing Poisson surface reconstruction...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    self.pcd,
                    depth=kwargs.get('depth', 9),
                    width=kwargs.get('width', 0),
                    scale=kwargs.get('scale', 1.1),
                    linear_fit=kwargs.get('linear_fit', False)
                )
                
                # Remove low-density vertices
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
            elif method == 'bpa':
                print("Performing Ball-Pivoting surface reconstruction...")
                radii = kwargs.get('radii', [0.005, 0.01, 0.02, 0.04])
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    self.pcd,
                    o3d.utility.DoubleVector(radii)
                )
            
            # Post-process mesh
            print("Post-processing mesh...")
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.compute_vertex_normals()
            
            return mesh
            
        except Exception as e:
            print(f"Error during surface reconstruction: {str(e)}")
            raise
    def reconstruct_surface(self, method='poisson', **kwargs):
        """
        Convert point cloud to mesh using surface reconstruction
        """
        print(f"Starting surface reconstruction using {method} method...")
        try:
            if method not in ['poisson', 'bpa', 'gaussian']:
                raise ValueError(f"Unknown reconstruction method: {method}")
                
            if not self.pcd.has_normals() and method != 'gaussian':
                print("Estimating normals...")
                self.pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30))
                self.pcd.orient_normals_consistent_tangent_plane(100)
            
            if method == 'poisson':
                print("Performing Poisson surface reconstruction...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    self.pcd,
                    depth=kwargs.get('depth', 9),
                    width=kwargs.get('width', 0),
                    scale=kwargs.get('scale', 1.1),
                    linear_fit=kwargs.get('linear_fit', False)
                )
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
            elif method == 'bpa':
                print("Performing Ball-Pivoting surface reconstruction...")
                radii = kwargs.get('radii', [0.005, 0.01, 0.02, 0.04])
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    self.pcd,
                    o3d.utility.DoubleVector(radii)
                )
                
            elif method == 'gaussian':
                print("Performing Gaussian Splatting reconstruction...")
                num_gaussians = kwargs.get('num_gaussians', 1000)
                learning_rate = kwargs.get('learning_rate', 0.01)
                num_iterations = kwargs.get('num_iterations', 100)
                
                mesh, params = reconstruct_surface_gaussian(
                    self.pcd,
                    num_gaussians=num_gaussians,
                    learning_rate=learning_rate,
                    num_iterations=num_iterations
                )
                
            # Post-process mesh
            print("Post-processing mesh...")
            mesh.remove_duplicated_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.compute_vertex_normals()
            
            return mesh
            
        except Exception as e:
            print(f"Error during surface reconstruction: {str(e)}")
            raise
        
    def test_floor_alignment(self, num_tests=5):
        """
        Unit test to verify floor alignment algorithm by:
        1. Applying random transformations to input point cloud
        2. Running our alignment algorithm
        3. Verifying if the floor is correctly aligned to Y=0
        """
        print(f"\nRunning {num_tests} unit tests...")
        results = []
        
        try:
            for i in range(num_tests):
                print(f"\nTest {i+1}/{num_tests}")
                
                # 1. Create a copy of original point cloud
                test_pcd = o3d.geometry.PointCloud(self.pcd)
                
                # 2. Apply random transformations
                # Random rotation
                angle = np.random.uniform(0, 2 * np.pi)
                axis = np.random.rand(3)
                axis = axis / np.linalg.norm(axis)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                test_pcd.rotate(R)
                
                # Random translation
                translation = np.random.rand(3) * 10  # Random translation up to 10 units
                test_pcd.translate(translation)
                
                # Random scaling
                scale = np.random.uniform(0.5, 2.0)
                test_pcd.scale(scale, center=test_pcd.get_center())
                
                print(f"Applied transformations:")
                print(f"- Rotation: angle={angle:.2f}, axis={axis}")
                print(f"- Translation: {translation}")
                print(f"- Scale: {scale}")
                
                # 3. Save the transformed point cloud for verification
                transformed_points = np.asarray(test_pcd.points)
                
                # 4. Run our alignment algorithm
                processor = PointCloudProcessor(pcd=test_pcd)
                aligned_pcd = processor.detect_floor()
                aligned_points = np.asarray(aligned_pcd.points)
                
                # 5. Verify alignment
                # Get floor plane after alignment
                plane_model, inliers = aligned_pcd.segment_plane(
                    distance_threshold=0.01,
                    ransac_n=3,
                    num_iterations=1000
                )
                
                # Calculate errors
                floor_points = aligned_points[inliers]
                
                # Check if floor points are close to Y=0
                height_deviation = np.abs(floor_points[:, 1]).mean()
                
                # Check if floor normal is aligned with Y-axis
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                normal = normal / np.linalg.norm(normal)
                normal_alignment = np.abs(np.dot(normal, [0, 1, 0]))
                
                # Success criteria
                height_threshold = 0.05  # Maximum allowed average deviation from Y=0
                normal_threshold = 0.95  # Minimum allowed alignment with Y-axis
                
                success = (height_deviation < height_threshold and 
                          normal_alignment > normal_threshold)
                
                # Store results
                test_result = {
                    'test_id': i,
                    'success': success,
                    'height_deviation': height_deviation,
                    'normal_alignment': normal_alignment,
                    'transformations': {
                        'rotation_angle': angle,
                        'rotation_axis': axis,
                        'translation': translation,
                        'scale': scale
                    }
                }
                results.append(test_result)
                
                # Print detailed results
                print(f"\nTest {i+1} Results:")
                print(f"- Height deviation from Y=0: {height_deviation:.4f} (threshold: {height_threshold})")
                print(f"- Normal alignment with Y-axis: {normal_alignment:.4f} (threshold: {normal_threshold})")
                print(f"- Overall result: {'Success' if success else 'Failure'}")
                
                # Optionally save transformed and aligned point clouds for visual verification
                if not success:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    o3d.io.write_point_cloud(
                        f"test_{i}_transformed_{timestamp}.ply",
                        test_pcd
                    )
                    o3d.io.write_point_cloud(
                        f"test_{i}_aligned_{timestamp}.ply",
                        aligned_pcd
                    )
                    print(f"Saved point clouds for failed test {i}")
            
            # Print summary
            successes = sum(1 for r in results if r['success'])
            print(f"\nTest Summary:")
            print(f"Total tests: {num_tests}")
            print(f"Successful: {successes}")
            print(f"Failed: {num_tests - successes}")
            
            return results
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise