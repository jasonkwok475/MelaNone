import cv2
import numpy as np
import os
from pathlib import Path
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Open3D is required. Install with: uv add open3d")

class ObjectReconstructor:
    def __init__(self, image_folder, angle_step):
        """
        Initialize the 3D object reconstructor.
        
        Args:
            image_folder: Path to folder containing images
            angle_step: Degrees between each horizontal image
        """
        self.image_folder = Path(image_folder)
        self.angle_step = angle_step
        self.images = []
        self.camera_poses = []
        self.point_cloud = None
        
    def load_images(self, pattern="*.jpg"):
        """Load all images from the folder."""
        image_paths = sorted(self.image_folder.glob(pattern))
        self.images = [cv2.imread(str(p)) for p in image_paths]
        print(f"Loaded {len(self.images)} images")
        return self.images
    
    def reconstruct_from_images(self):
        """
        Reconstruct 3D object using Structure from Motion (SfM).
        This uses feature matching and triangulation to find actual 3D points.
        """
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for reconstruction")
        
        # Step 1: Extract features from all images
        print("Extracting features...")
        all_keypoints, all_descriptors = self._extract_all_features()
        
        # Step 2: Match features between consecutive images
        print("Matching features between images...")
        matches_dict = self._match_consecutive_pairs(all_keypoints, all_descriptors)
        
        # Step 3: Estimate camera poses and triangulate points
        print("Estimating camera poses and reconstructing 3D points...")
        points_3d, colors_3d = self._triangulate_points(
            all_keypoints, matches_dict
        )
        
        # Step 4: Create point cloud
        print(f"Creating point cloud with {len(points_3d)} points...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors_3d)
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        self.point_cloud = pcd
        return pcd
    
    def _extract_all_features(self):
        """Extract SIFT features from all images."""
        sift = cv2.SIFT_create(nfeatures=5000)  # type: ignore
        all_keypoints = []
        all_descriptors = []
        
        for i, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            print(f"  Image {i}: {len(kp)} keypoints")
        
        return all_keypoints, all_descriptors
    
    def _match_consecutive_pairs(self, all_keypoints, all_descriptors):
        """Match features between consecutive image pairs."""
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches_dict = {}
        
        for i in range(len(self.images) - 1):
            matches = bf.knnMatch(all_descriptors[i], all_descriptors[i+1], k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            matches_dict[(i, i+1)] = good_matches
            print(f"  Images {i}-{i+1}: {len(good_matches)} good matches")
        
        return matches_dict
    
    def _triangulate_points(self, all_keypoints, matches_dict):
        """
        Triangulate 3D points from matched features.
        Uses essential matrix to recover camera pose and triangulate.
        """
        # Camera intrinsic parameters (estimate based on image size)
        h, w = self.images[0].shape[:2]
        focal_length = max(w, h)
        cx, cy = w / 2, h / 2
        
        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        all_points_3d = []
        all_colors = []
        
        # Process each consecutive pair
        for (i, j), matches in matches_dict.items():
            if len(matches) < 8:
                continue
            
            # Get matched points
            pts1 = np.float32([all_keypoints[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([all_keypoints[j][m.trainIdx].pt for m in matches])
            
            # Find essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, 
                                          prob=0.999, threshold=1.0)
            
            if E is None:
                continue
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
            
            # Projection matrices
            P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = K @ np.hstack((R, t))
            
            # Triangulate points
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Filter points (remove those too far or behind camera)
            valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 1000)
            points_3d = points_3d[valid_mask]
            
            # Get colors from first image
            colors = []
            for idx, m in enumerate(matches):
                if valid_mask[idx]:
                    x, y = all_keypoints[i][m.queryIdx].pt
                    x, y = int(x), int(y)
                    if 0 <= y < h and 0 <= x < w:
                        color = self.images[i][y, x]
                        colors.append(color[[2, 1, 0]] / 255.0)  # BGR to RGB
            
            all_points_3d.extend(points_3d)
            all_colors.extend(colors[:len(points_3d)])
        
        return np.array(all_points_3d), np.array(all_colors)
    
    def create_mesh_from_point_cloud(self, pcd=None, depth=9):
        """
        Create a mesh from the point cloud using Poisson reconstruction.
        
        Args:
            pcd: Point cloud (uses self.point_cloud if None)
            depth: Octree depth for Poisson reconstruction (higher = more detail)
        """
        if pcd is None:
            pcd = self.point_cloud
        
        if pcd is None:
            raise ValueError("No point cloud available. Run reconstruct_from_images first.")
        
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        )
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        print(f"Creating mesh using Poisson reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=True
        )
        
        # Remove low-density vertices (noise)
        print("Cleaning mesh...")
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Remove disconnected components
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Keep only the largest cluster
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        
        print(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        return mesh
    
    def generate_uv_map(self, mesh, output_path="uv_map.png", texture_size=2048):
        """
        Generate UV coordinates and create a flattened texture map.
        Uses smart unwrapping for arbitrary geometry.
        """
        vertices = np.asarray(mesh.vertices)
        
        # Use cylindrical unwrapping as a reasonable default
        # For better results, consider using proper UV unwrapping tools
        uv_coords = []
        
        # Find the central axis (assume Z is up)
        center = vertices.mean(axis=0)
        
        for vertex in vertices:
            # Translate to origin
            v = vertex - center
            x, y, z = v
            
            # U coordinate: angle around Z axis
            u = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
            
            # V coordinate: height (Z)
            z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
            v_coord = (z - z_min) / (z_max - z_min) if z_max != z_min else 0.5
            
            uv_coords.append([u, v_coord])
        
        uv_coords = np.array(uv_coords)
        
        # Create texture image
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        
        # Map vertex colors to texture
        if mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            
            # Ensure we don't exceed array bounds
            num_colors = len(colors)
            
            for i, (u, v) in enumerate(uv_coords):
                # Check if we have a color for this vertex
                if i >= num_colors:
                    break
                
                # Clamp UV coordinates to valid range [0, 1]
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                
                # Convert to texture coordinates
                tex_x = min(int(u * (texture_size - 1)), texture_size - 1)
                tex_y = min(int((1 - v) * (texture_size - 1)), texture_size - 1)
                
                # Ensure indices are within bounds
                if 0 <= tex_y < texture_size and 0 <= tex_x < texture_size:
                    color = (colors[i] * 255).astype(np.uint8)
                    texture[tex_y, tex_x] = color[::-1]  # RGB to BGR
        else:
            # If no vertex colors, create a simple gradient texture
            print("Warning: Mesh has no vertex colors, creating gradient texture")
            for i, (u, v) in enumerate(uv_coords):
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                tex_x = min(int(u * (texture_size - 1)), texture_size - 1)
                tex_y = min(int((1 - v) * (texture_size - 1)), texture_size - 1)
                if 0 <= tex_y < texture_size and 0 <= tex_x < texture_size:
                    texture[tex_y, tex_x] = [int(u*255), int(v*255), 128]
        
        # Apply inpainting to fill gaps instead of blur
        gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
        mask = (gray == 0).astype(np.uint8) * 255
        if mask.sum() > 0:
            texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
        
        cv2.imwrite(output_path, texture)
        print(f"UV map saved to {output_path}")
        
        return uv_coords, texture
    
    def save_mesh_with_uv(self, mesh, uv_coords, obj_path="mesh.obj", 
                          mtl_path="mesh.mtl", texture_path="uv_map.png"):
        """Save mesh with UV coordinates and material to OBJ file."""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Write MTL file
        with open(mtl_path, 'w') as f:
            f.write(f"newmtl material0\n")
            f.write(f"Ka 1.0 1.0 1.0\n")
            f.write(f"Kd 1.0 1.0 1.0\n")
            f.write(f"Ks 0.0 0.0 0.0\n")
            f.write(f"map_Kd {texture_path}\n")
        
        # Write OBJ file
        with open(obj_path, 'w') as f:
            f.write(f"mtllib {mtl_path}\n")
            f.write(f"usemtl material0\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # Write UV coordinates
            for uv in uv_coords:
                f.write(f"vt {uv[0]} {uv[1]}\n")
            
            # Write faces with texture coordinates
            for tri in triangles:
                f.write(f"f {tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}\n")
        
        print(f"Mesh saved to {obj_path}")


def main():
    """
    Example usage - UPDATE THESE PARAMETERS FOR YOUR SETUP
    """
    import sys
    
    # Configuration - CHANGE THESE VALUES
    IMAGE_FOLDER = os.getcwd() + "/images" # UPDATE THIS PATH
    ANGLE_STEP = 10  # Degrees between each image horizontally
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' does not exist!")
        print("\nPlease update the IMAGE_FOLDER path in the main() function.")
        print("Example: IMAGE_FOLDER = 'C:/Users/YourName/images'")
        sys.exit(1)
    
    reconstructor = ObjectReconstructor(
        image_folder=IMAGE_FOLDER,
        angle_step=ANGLE_STEP
    )
    
    # Load images
    images = reconstructor.load_images()
    
    if len(images) == 0:
        print(f"Error: No images found in '{IMAGE_FOLDER}'")
        print("Make sure your images are in .jpg format")
        sys.exit(1)
    
    # Reconstruct 3D object from images
    print("\n=== Step 1: Reconstructing 3D geometry ===")
    pcd = reconstructor.reconstruct_from_images()
    
    # Visualize point cloud
    print("\nVisualizing point cloud... (close window to continue)")
    o3d.visualization.draw_geometries([pcd],  # type: ignore
                                     window_name="Reconstructed Point Cloud",
                                     width=1024, height=768)
    
    # Create mesh from point cloud
    print("\n=== Step 2: Creating mesh ===")
    mesh = reconstructor.create_mesh_from_point_cloud(pcd, depth=9)
    
    # Visualize mesh
    print("\nVisualizing mesh... (close window to continue)")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh],  # type: ignore
                                     window_name="Reconstructed Mesh",
                                     width=1024, height=768)
    
    # Generate UV map
    print("\n=== Step 3: Generating UV map ===")
    uv_coords, texture = reconstructor.generate_uv_map(mesh)
    
    # Save mesh with UV coordinates
    reconstructor.save_mesh_with_uv(mesh, uv_coords)
    
    print("\n✓ Processing complete!")
    print(f"  - Mesh saved to: mesh.obj")
    print(f"  - Material saved to: mesh.mtl")
    print(f"  - UV map saved to: uv_map.png")


if __name__ == "__main__":
    main()