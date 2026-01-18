import shutil
import sys
import cv2
import numpy as np
from rich import print
import os
import time
from queue import Queue
from internal.model import predict, load_model
from internal.mesh import ObjectReconstructor
from internal.uv_processor import predict_from_uv_map
from embedded.device_manager import DeviceManager
from debug.error_handling import catch_exceptions, ErrorType
import open3d as o3d

# Configuration - CHANGE THESE VALUES
IMAGE_FOLDER_STORAGE = os.getcwd() + "/image" # UPDATE THIS PATH
ANGLE_STEP = 30  # Degrees between each image horizontally


try:
    import serial

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Serial communication will be disabled.")

class AnalysisManager:
    """Manages the melanoma detection analysis workflow"""

    IMAGE_FOLDER = IMAGE_FOLDER_STORAGE
    def __init__(self):
        self.is_running = False
        self.progress = 0
        self.current_step = ""
        self.queue = Queue()
        self.results = None

        if os.path.exists(self.IMAGE_FOLDER):
            shutil.rmtree(self.IMAGE_FOLDER)   # deletes folder and contents

        os.makedirs(self.IMAGE_FOLDER)         # recreates empty folder

    def _update_progress(self, step, progress_pct, message=""):
        """Send progress update to frontend via queue"""
        self.current_step = step
        self.progress = progress_pct

        update = {
            'status': 'in_progress',
            'step': step,
            'progress': progress_pct,
            'message': message
        }
        self.queue.put(update)
        print(f"[{step}] {progress_pct}% - {message}")


    @catch_exceptions(ErrorType.IMAGE_SAVE)
    def _save_image(self, image, images, suffix: int, device: int):
        image_path = f"{self.IMAGE_FOLDER}/{device}_rotation_{suffix}.jpg"
        cv2.imwrite(image_path, image)
        print(f"[bold green]Saved image: {image_path}")
        images.extend(image_path if isinstance(image_path, list) else [image_path])

    @catch_exceptions(ErrorType.SERIAL)
    def _capture_from_all_devices(self, ser, devices, images, suffix: int):
        if ser:
            print(type(ser))
            print()
            ser.write(b'1')
            
            while True:
                response = ser.readline().decode().strip()

                print(f"Serial Response: {response}")

                if response == '-1':
                    return # End of linear motion
                if response == '2':  # Rotation complete
                    break

        print("Capturing images from all devices...")
        i = 0
        for device in devices:
            image = device.get_current_frame()
            self._save_image(image=image, images=images, suffix=suffix, device=i)
            i += 1

        return

    @catch_exceptions(ErrorType.SERIAL)
    def _establish_serial(self):
        ser = serial.Serial('COM6', 115200, timeout=5)  # Adjust COM port as needed
        time.sleep(1)  # Wait for connection to establish
        ser.write(b'3') # Reset position

        while True:
          response = ser.readline().decode().strip()

          if response == '2':  # Reset complete
              break

        return ser

    @catch_exceptions(ErrorType.CAMERA)
    def _capture_images(self):
        """Step 1: Capture images from device using rotation"""
        self._update_progress("Capturing Images", 10, "Initializing camera...")
        time.sleep(1)  # Simulate initialization

        with DeviceManager(1) as device1, DeviceManager(2) as device3: #, DeviceManager(3) as device3:
            devices = [device1, device3]
            images = []
            time.sleep(2)  # Allow cameras to warm up
            
            self._update_progress("Capturing Images", 12, "Camera initialized...")

            """
            primary change here was adding a sleep so that the device manager has time to start capturing frames
            without the delay, a race condition occurs since the IO overhead of init the camera is significantly 
            higher than the CPU clock speed (ie it runs code before camera warms up)
            
            feel free to uncomment the code above to get multi cameras working. 
            
            bug fixes:
            - create ./image dir if it does not exist
            - added sleep to startup
            - disabled debug mode in the device manager (no longer displays live footage, but 
                allows for consistent repeated calls to the device manager)
            - refactoring bc code was hard to read
            """

            self._update_progress("Capturing Images", 20, "Camera ready, capturing frames...")

            # Initialize serial connection for device communication if available
            ser = None
            if SERIAL_AVAILABLE:
                self._update_progress("Capturing Images", 15, "Resetting position...")

                ser = self._establish_serial()
                self._update_progress("Capturing Images", 20, "Capturing images...")
            else:
                print("Serial module not available. Using camera only.")
                self._update_progress("Capturing Images", 25, "Using camera without serial communication")

            for i in range(4):  # Capture 4 rotations
                self._capture_from_all_devices(ser, devices, images, i)
                print(f"Captured rotation {i+1}/4")
            if ser:
                ser.write(b'3')

                while True:
                  response = ser.readline().decode().strip()

                  if response == '2':  # Reset complete
                      break

                ser.close()

            self._update_progress("Capturing Images", 30, "Image capture complete")

        return

    def _reconstruct_3d_mesh(self):
        """Step 2: Reconstruct 3D mesh from captured images"""
        time.sleep(1)

        try:
            reconstructor = ObjectReconstructor(
                image_folder="./captured_images",
                angle_step=15
            )
            self._update_progress("3D Reconstruction", 45, "Loading and processing images...")
            time.sleep(1)

            # Load images
            reconstructor.load_images()
            self._update_progress("3D Reconstruction", 55, "Building point cloud...")
            time.sleep(1)

            # Generate mesh data (icosphere-like structure for 3D visualization)
            mesh_data = self._generate_sample_mesh()
            self._update_progress("3D Reconstruction", 65, "Mesh reconstruction complete")

            return mesh_data

        except Exception as e:
            print(f"Warning: Mesh reconstruction failed: {e}")
            self._update_progress("3D Reconstruction", 65, "Using sample mesh data...")
            return self._generate_sample_mesh()

    def _compile_results(self, mesh_data, analysis_results):
        """Step 5: Compile final results"""
        self._update_progress("Finalizing", 95, "Compiling results...")
        time.sleep(0.5)

        final_results = {
            'totalObjectsAnalyzed': analysis_results.get('totalObjectsAnalyzed', 0),
            'concerningSpots': analysis_results.get('concerningSpots', 0),
            'confidenceScores': analysis_results.get('confidenceScores', []),
            'classifications': analysis_results.get('classifications', []),
            'meshData': mesh_data,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        self._update_progress("Finalizing", 100, "Complete!")
        return final_results

    def run_analysis(self):
        """Main analysis workflow - orchestrates all steps"""
        try:
            self.is_running = True
            self.progress = 0
            self.results = None

            print("\n" + "=" * 50)
            print("Starting Melanoma Analysis Workflow")
            print("=" * 50 + "\n")

            reconstructor = ObjectReconstructor(
                image_folder=IMAGE_FOLDER_STORAGE,
                angle_step=ANGLE_STEP
            )
            
            # Capture images
            self._capture_images()
            
            # Load images
            images = reconstructor.load_images()
            
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
            mesh_o3d = reconstructor.create_mesh_from_point_cloud(pcd, depth=9)
            
            # Visualize mesh
            print("\nVisualizing mesh... (close window to continue)")
            mesh_o3d.compute_vertex_normals()
            o3d.visualization.draw_geometries([mesh_o3d],  # type: ignore
                                            window_name="Reconstructed Mesh",
                                            width=1024, height=768)
            
            # Convert open3d mesh to dictionary format for frontend
            vertices = np.asarray(mesh_o3d.vertices).tolist()
            faces = np.asarray(mesh_o3d.triangles).tolist()
            mesh = {
                'vertices': vertices,
                'faces': faces,
                'type': 'poisson_reconstruction'
            }
            
            # Generate UV map
            print("\n=== Step 3: Generating UV map ===")
            uv_coords, texture = reconstructor.generate_uv_map(mesh_o3d)
            
            # Save mesh with UV coordinates
            reconstructor.save_mesh_with_uv(mesh_o3d, uv_coords)
            
            print("\n✓ Processing complete!")
            print(f"  - Mesh saved to: mesh.obj")
            print(f"  - Material saved to: mesh.mtl")
            print(f"  - UV map saved to: uv_map.png")

            # Step 4: Analyze lesions using UV map
            print("\n=== Step 4: Analyzing lesions with ML model ===")
            self._update_progress("Lesion Analysis", 70, "Processing UV map through model...")
            
            # Use UV processor to get predictions
            uv_map_path = "uv_map.png"  # Generated by reconstructor
            analysis_results = predict_from_uv_map(uv_map_path)
            
            self._update_progress("Lesion Analysis", 85, f"Analysis complete: {analysis_results['concerningSpots']} concerning spot(s) found")


            # Step 4: Compile results
            self.results = self._compile_results(mesh, analysis_results)

            print("\n" + "=" * 50)
            print("Analysis Complete!")
            print("=" * 50 + "\n")
            print(f"Results: {self.results}")

        except Exception as e:
            print(f"Error during analysis: {e}")
            self.queue.put({
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            })
            self.results = None

        finally:
            self.is_running = False






    def _generate_sample_mesh(self):
        """Generate sample mesh data (icosahedron geometry)"""
        import math

        # Generate vertices for an icosahedron
        phi = (1 + math.sqrt(5)) / 2
        vertices = [
            [-1, phi, -0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, -0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]

        # Normalize vertices
        for v in vertices:
            length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            v[0] /= length * 1.5
            v[1] /= length * 1.5
            v[2] /= length * 1.5

        # Face indices for icosahedron
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]

        return {
            'vertices': vertices,
            'faces': faces,
            'type': 'icosahedron'
        }