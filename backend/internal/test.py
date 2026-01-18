# test_open3d.py
import sys
print(f"Python version: {sys.version}")

try:
    import open3d as o3d
    print(f"Open3D version: {o3d.__version__}")
    print(f"Open3D location: {o3d.__file__}")
    
    # Test creating a point cloud
    import numpy as np
    points = np.random.rand(100, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print("✓ Open3D is working correctly!")
    
except ImportError as e:
    print(f"✗ Open3D import failed: {e}")
except AttributeError as e:
    print(f"✗ Open3D attribute error: {e}")
    print("This may be due to incomplete installation or version mismatch")