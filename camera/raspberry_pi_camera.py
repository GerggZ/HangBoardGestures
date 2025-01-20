from picamera2 import Picamera2
import numpy as np
from numpy.typing import NDArray
from camera.base_camera import BaseCamera


class PiCamera(BaseCamera):
    """
    Module for Raspberry Pi Camera implementation using Picamera2
    """

    def __init__(self, width: int = 640, height: int = 480, verbose: bool = False) -> None:
        """
        Initializes the Raspberry Pi camera module

        Args:
            - width (int): Desired width of camera feed
            - height (int): Desired height of camera feed
            - verbose (bool): Print debugging information
        """
        self.width = width
        self.height = height
        self.verbose = verbose

        self.picam2 = None

    def open_camera(self) -> None:
        """
        Creates the Picamera2 object and sets desired resolution
        """
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(main={"size": (self.width, self.height)})
        self.picam2.configure(camera_config)
        self.picam2.start()

        if self.verbose:
            print(f"PiCamera opened successfully (Resolution: {self.width}x{self.height})")

    def read_frame(self) -> NDArray:
        """
        Reads a frame from the camera
        """
        if self.picam2 is None:
            raise RuntimeError("Error: PiCamera is not open or not initialized.")

        # Retrieve frame as numpy array
        frame = self.picam2.capture_array()

        if frame is None:
            raise RuntimeError("Error: Failed to successfully capture frame from PiCamera.")

        return frame

    def release(self) -> None:
        """
        Releases camera resource
        """
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None

            if self.verbose:
                print("PiCamera released successfully")
