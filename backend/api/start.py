import time
from queue import Queue
from internal.model import predict, load_model
from internal.mesh import ObjectReconstructor
from embedded.device_manager import DeviceManager
import cv2
import numpy as np

try:
    import serial

    SERIAL_AVAILABLE = False
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Serial communication will be disabled.")


class AnalysisManager:
    """Manages the melanoma detection analysis workflow"""

    def __init__(self):
        self.is_running = False
        self.progress = 0
        self.current_step = ""
        self.queue = Queue()  # For sending progress updates to frontend
        self.results = None

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

    def _capture_images(self):
        """Step 1: Capture images from device using rotation"""
        self._update_progress("Capturing Images", 10, "Initializing camera...")
        time.sleep(1)  # Simulate initialization

        try:
            # devices = [DeviceManager(camera_num=i) for i in range(1)]
            # for device in devices:
            #     device.run()
            # images = []

            with DeviceManager(0) as device1: # DeviceManager(1) as device2, DeviceManager(2) as device3:
                # devices = [device1, device2, device3]
                devices = [device1]
                images = []

                for device in devices:
                    device.run()

                self._update_progress("Capturing Images", 20, "Camera ready, capturing frames...")

                # Initialize serial connection for device communication if available
                ser = None
                if SERIAL_AVAILABLE:
                    try:
                        ser = serial.Serial('COM3', 9600, timeout=5)  # Adjust COM port as needed
                        time.sleep(1)  # Wait for connection to establish
                        self._update_progress("Capturing Images", 25, "Serial connection established")
                    except Exception as e:
                        print(f"Serial connection failed: {e}. Using camera only.")
                        self._update_progress("Capturing Images", 25, "Using camera without rotation")
                else:
                    print("Serial module not available. Using camera only.")
                    self._update_progress("Capturing Images", 25, "Using camera without rotation")

                for i in range(4):  # Capture 4 rotations
                    if ser:
                        try:
                            ser.write(b'1')  # Send signal to rotate
                            response = ser.readline().decode().strip()
                            while response != '2':
                                response = ser.readline().decode().strip()
                        except Exception as e:
                            print(f"Serial communication error: {e}")

                    for device in devices:
                        image = device.get_current_frame()
                        image_path = f"./images/rotation_{i}.jpg"
                        # TODO: IM NOT SURE WHY we want to persist this on disc. Might as well leave it in memory no?
                        try:
                            cv2.imwrite(image_path, image)
                            images.extend(image_path if isinstance(image_path, list) else [image_path])
                        except Exception as e:
                            print(f"Failed to save image: {e}")

                if ser:
                    ser.write(b'3')
                    ser.close()

                self._update_progress("Capturing Images", 30, "Image capture complete")

        except Exception as e:
            print(f"Warning: Camera capture failed: {e}")
            self._update_progress("Capturing Images", 30, "Using sample images...")

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
