import random
import numpy as np
from numpy.typing import NDArray
from utils.scaling import scale_xy_data  # Uses your existing scaling function


class HandsAugmentor:
    """
    A class for applying hand gesture augmentations, including flipping and full 3D rotation.
    After augmentation, the Z-axis is removed, and the data is rescaled using `scaling_xy`.

    Args:
        flip_horizontal_prob (float): Probability of applying a horizontal flip (mirror along X-axis).
        flip_vertical_prob (float): Probability of applying a vertical flip (mirror along Y-axis).
        rotation_prob (float): Probability of applying a rotation.
        max_yaw (float): Maximum yaw rotation (in degrees) around the Y-axis (left/right rotation).
        max_pitch (float): Maximum pitch rotation (in degrees) around the X-axis (up/down rotation).
        max_roll (float): Maximum roll rotation (in degrees) around the Z-axis (in-plane rotation).
        wrist_index (int): Index of the wrist landmark (default: 0).
    """

    def __init__(
        self,
        flip_horizontal_prob=0.5,
        flip_vertical_prob=0.5,
        rotation_prob=0.5,
        max_yaw=30,
        max_pitch=30,
        max_roll=30,
        wrist_index=0
    ):
        self.flip_horizontal_prob = flip_horizontal_prob
        self.flip_vertical_prob = flip_vertical_prob
        self.rotation_prob = rotation_prob
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch
        self.max_roll = max_roll
        self.wrist_index = wrist_index  # Configurable index to rotate around

    def _flip_horizontally(self, XY: NDArray) -> NDArray:
        """Flips hand data horizontally around the wrist."""
        wrist_x = XY[self.wrist_index, 0]  # Wrist X-coordinate
        XY[:, 0] = 2 * wrist_x - XY[:, 0]  # Flip horizontally around wrist
        return XY

    def _flip_vertically(self, XY: NDArray) -> NDArray:
        """Flips hand data vertically around the wrist."""
        wrist_y = XY[self.wrist_index, 1]  # Wrist Y-coordinate
        XY[:, 1] = 2 * wrist_y - XY[:, 1]  # Flip vertically around wrist
        return XY

    def _apply_rotation(self, XYZ: NDArray) -> NDArray:
        """
        Applies a full 3D rotation to the hand data using Pitch, Yaw, and Roll.
        - Pitch: Rotation around X-axis (up/down head movement)
        - Yaw: Rotation around Y-axis (left/right head movement)
        - Roll: Rotation around Z-axis (in-plane rotation, equivalent to 2D rotation)
        """
        wrist = XYZ[self.wrist_index, :].reshape(1, 3)

        # Generate random rotation angles within the specified max ranges
        pitch = np.radians(random.uniform(-self.max_pitch, self.max_pitch))  # X rotation
        yaw = np.radians(random.uniform(-self.max_yaw, self.max_yaw))  # Y rotation
        roll = np.radians(random.uniform(-self.max_roll, self.max_roll))  # Z rotation

        # Define rotation matrices
        R_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        R_y = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R_z = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

        # Full 3D rotation matrix (applies Roll → Pitch → Yaw)
        R = R_z @ R_y @ R_x

        # Shift so wrist is at origin, rotate, then shift back
        XYZ = XYZ.squeeze(-1)  # Remove extra dimension (shape becomes (21, 3))
        XYZ -= wrist
        XYZ = XYZ @ R.T  # Apply rotation

        return XYZ[..., np.newaxis]  # Reshape back to (21, 3, 1)

    def __call__(self, XYZ_batch: NDArray) -> NDArray:
        """
        Applies the chosen augmentations to a batch of hand data.
        Expects shape (batch_size, 21, 3), returns shape (batch_size, 21, 2) after dropping Z.
        """
        batch_size = XYZ_batch.shape[0]

        for i in range(batch_size):
            if random.random() < self.flip_horizontal_prob:
                XYZ_batch[i][:, :2] = self._flip_horizontally(XYZ_batch[i][:, :2])  # Flip only X, Y

            if random.random() < self.flip_vertical_prob:
                XYZ_batch[i][:, :2] = self._flip_vertically(XYZ_batch[i][:, :2])  # Flip only X, Y

            if random.random() < self.rotation_prob:
                XYZ_batch[i] = self._apply_rotation(XYZ_batch[i])  # Apply full 3D rotation

        # Drop Z and rescale using the existing scaling function
        XY_batch = XYZ_batch[:, :, :2]  # Keep only X and Y
        XY_batch = scale_xy_data(XY_batch)  # Apply rescaling

        return  XY_batch[..., np.newaxis] # Returns (batch_size, 42, 1) for training
