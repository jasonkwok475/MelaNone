import cv2
import threading
import time
from rich import print

class DeviceManager:
    """
    Defines common interface for external devices

    Note: Setting DEFAULT_CAMERA = 0 uses the default camera on your machine
    """
    DEBUG = False
    MAIN_WINDOW = "Webcam Feed - Press SPACE to capture, ESC to exit"
    DEFAULT_CAMERA = 0
    def __init__(self, camera_num: int = 1, port: int = 5005):
        self.port = port
        self.camera = cv2.VideoCapture(camera_num)
        self.frame = None
        self.frame_ret_code = False
        self.running_frame_fetch = False

        # Create the image fetch thread
        self.run_img_fetch_thread = threading.Thread(
            target=self.grab_image, daemon=True
        )

        # Debug information
        self.last_frame_time = 0
        self.current_fps = None

    def grab_image(self):
        """
        Procedure to fetch new frames
        """
        print("Grabbing new frames...")
        while self.running_frame_fetch:
            # Read a frame from the webcam
            ret, frame = self.camera.read()
            if not ret:
                print("failed to grab frame")
                break
            self.frame = frame
            # Calculate FPS
            current_time = time.time()
            # FPS = 1 / time_taken_for_one_frame
            self.current_fps = 1 / (current_time - self.last_frame_time)
            self.last_frame_time = current_time

    def debug_display_img(self, frame):
        if not self.DEBUG:
            print("[bold magenta]Warning: NOT in debug mode, display disabled[/bold magenta]")
            return

        # Display FPS on the image
        if self.current_fps is not None:
            cv2.putText(frame, f"FPS: {int(self.current_fps)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(self.MAIN_WINDOW, frame)

    def __enter__(self):
        s = "DEBUG MODE ENABLED" if self.DEBUG else ""
        print(f"[bold blue]{s}")

        self.running_frame_fetch = True
        self.run_img_fetch_thread.start()

        # Initialize the webcam capture object (0 indicates the default camera)
        if self.DEBUG:
            cv2.namedWindow(self.MAIN_WINDOW)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running_frame_fetch = False
        self.run_img_fetch_thread.join()
        self.camera.release()
        cv2.destroyAllWindows()
        print("[bold red]Device Manager Finished[/]")

    def run(self):
        # executes main thread required procedures
        if self.frame is not None:
            self.debug_display_img(self.frame)

    def get_current_frame(self) -> cv2.typing.MatLike | None:
        """
        Requests the current frame. Returns None if no frame is available.
        """
        return self.frame