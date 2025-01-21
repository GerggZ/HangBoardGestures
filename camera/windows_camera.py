# camera/windows_camera.py
import cv2
from numpy.typing import NDArray
from camera.base_camera import BaseCamera


class WindowsCamera(BaseCamera):
    """
    Module for Windows webcam implementation using CV2
    """
    def __init__(self, camera_source: int = 0, width: int = 640, height: int = 480, verbose:bool = False) -> None:
        """
        Initializes the windows camera module

        Args:
            - camera_source (int) : The camera index
            - width (int): Desired width of camera feed
            - height (int): Desired height of camera feed
            - verbose (bool): Print debugging information
        """
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.verbose = verbose

        self.cap: cv2.VideoCapture | None = None

    def open_camera(self) -> None:
        """
        Creates the VideoCapture object and sets desired resolution
        """
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.verbose:
            print(f"Camera opened successfully (Source: {self.camera_source}, Resolution: {self.width}x{self.height})")

    def read_frame(self) -> NDArray:
        """
        Reads a frame from the camera
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Error: WindowsCamera is not open or not initialized.")

        # Retrieve frame
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: Failed to successfully capture frame from WindowsCamera")

        return frame

    def release(self) -> None:
        """
        Releases camera resource
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

            if self.verbose:
                print("Camera released successfully :)")

