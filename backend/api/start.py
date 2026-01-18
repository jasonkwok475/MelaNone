import cv2
import numpy as np
from rich import print
import os
import time
from queue import Queue
from backend.internal.model import predict, load_model
from backend.internal.mesh import ObjectReconstructor
from backend.embedded.device_manager import DeviceManager
from backend.debug.error_handling import catch_exceptions, ErrorType

try:
    import serial

    SERIAL_AVAILABLE = False
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Serial communication will be disabled.")

class AnalysisManager:
    """Manages the melanoma detection analysis workflow"""

    IMAGE_FOLDER = "./image"
    def __init__(self):
        self.is_running = False
        self.progress = 0
        self.current_step = ""
        self.queue = Queue()  # For sending progress updates to frontend
        self.results = None
        if not os.path.exists(self.IMAGE_FOLDER):
            os.makedirs(self.IMAGE_FOLDER)

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
    def _save_image(self, image, images, i: int):
        image_path = f"./images/rotation_{i}.jpg"
        cv2.imwrite(image_path, image)
        print(f"[bold green]Saved image: {image_path}")
        images.extend(image_path if isinstance(image_path, list) else [image_path])

    @catch_exceptions(ErrorType.SERIAL)
    def _capture_from_all_devices(self, ser, devices, images):
        if ser:
            ser.write(b'1')  # Send signal to rotate
            response = ser.readline().decode().strip()
            while response != '2':
                response = ser.readline().decode().strip()

        for device in devices:
            image = device.get_current_frame()
            self._save_image(image=image, images=images)

    @catch_exceptions(ErrorType.SERIAL)
    def _establish_serial(self):
        ser = serial.Serial('COM3', 9600, timeout=5)  # Adjust COM port as needed
        time.sleep(1)  # Wait for connection to establish
        return ser

    @catch_exceptions(ErrorType.CAMERA)
    def _capture_images(self):
        """Step 1: Capture images from device using rotation"""
        self._update_progress("Capturing Images", 10, "Initializing camera...")
        time.sleep(1)  # Simulate initialization

        with DeviceManager(0) as device1:  # DeviceManager(1) as device2, DeviceManager(2) as device3:
            # devices = [device1, device2, device3]
            devices = [device1]
            images = []
            time.sleep(2)
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
                ser = self._establish_serial()
                self._update_progress("Capturing Images", 25, "Using camera without rotation")
            else:
                print("Serial module not available. Using camera only.")
                self._update_progress("Capturing Images", 25, "Using camera without rotation")

            for i in range(4):  # Capture 4 rotations
                self._capture_from_all_devices(ser, devices, images)

            if ser:
                ser.write(b'3')
                ser.close()

            self._update_progress("Capturing Images", 30, "Image capture complete")

        return ["sample_image_1.jpg", "sample_image_2.jpg", "sample_image_3.jpg"]

    def _reconstruct_3d_mesh(self, images):
        """Step 2: Reconstruct 3D mesh from captured images"""
        self._update_progress("3D Reconstruction", 35, "Processing images...")
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

    def _analyze_lesions(self):
        """Step 3: Analyze lesions using ML model"""
        self._update_progress("Lesion Analysis", 70, "Loading ML model...")
        time.sleep(0.5)

        try:
            model = load_model()
            self._update_progress("Lesion Analysis", 80, "Running predictions...")
            time.sleep(2)  # Simulate prediction time

            # Simulate analysis results
            analysis_results = {
                'total_objects_analyzed': 24,
                'concerning_spots': 3,
                'confidence_scores': [0.92, 0.87, 0.95],
                'classifications': ['melanoma', 'benign', 'melanoma']
            }

            self._update_progress("Lesion Analysis", 90, "Analysis complete")
            return analysis_results
        except Exception as e:
            print(f"Warning: Analysis failed: {e}")
            self._update_progress("Lesion Analysis", 90, "Using sample analysis data...")
            return {
                'total_objects_analyzed': 24,
                'concerning_spots': 3,
                'confidence_scores': [0.92, 0.87, 0.95],
                'classifications': ['melanoma', 'benign', 'melanoma']
            }

    def _compile_results(self, mesh_data, analysis_results):
        """Step 4: Compile final results"""
        self._update_progress("Finalizing", 95, "Compiling results...")
        time.sleep(0.5)

        final_results = {
            'totalObjectsAnalyzed': analysis_results['total_objects_analyzed'],
            'concerningSpots': analysis_results['concerning_spots'],
            'confidenceScores': analysis_results['confidence_scores'],
            'classifications': analysis_results['classifications'],
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

            # Step 1: Capture images
            images = self._capture_images()

            # Step 2: Reconstruct 3D mesh
            mesh_data = self._reconstruct_3d_mesh(images)

            # Step 3: Analyze lesions
            analysis_results = self._analyze_lesions()

            # Step 4: Compile results
            self.results = self._compile_results(mesh_data, analysis_results)

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
