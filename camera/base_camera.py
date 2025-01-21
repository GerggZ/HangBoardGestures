# camera/base_camera.py

from abc import ABC, abstractmethod
from numpy.typing import NDArray


class BaseCamera(ABC):
    """
    Base class for camera module
    Defines the functions which must be implemented in order to use camera
    """
    def open_camera(self) -> None:
        """
        Opens or initializes the camera resource

        Raises:
            - RuntimeError: If the camera cannot be opened
        """
        pass

    @abstractmethod
    def read_frame(self) -> NDArray:
        """
        Reads a frame from the camera.

        Returns:
            numpy.ndarray: The captured frame as a NumPy array (BGR format).
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Releases the camera resource.
        Should not return anything.
        """
        pass
