# utils/scaling.py

import numpy as np
from numpy.typing import NDArray


def convert_landmarks_to_numpy(multi_hand_landmarks) -> NDArray | None:
    """
    Converts Mediapipe's multi_hand_landmarks results into a NumPy array
    of shape (num_hands, 21 x 3), containing [x, y, z] coordinates]
    """
    hand_coords_list = []
    for hand_landmarks in multi_hand_landmarks:
        # For each of the 21 landmarks, grap (x, y, z)
        coords = []
        for landmark in hand_landmarks.landmark:
            coords.append([landmark.x, landmark.y, landmark.z])
        coords = np.array(coords)
        hand_coords_list.append(coords)

    # Stack into shape (num_hands, 21, 3)
    hand_coords_array = np.stack(hand_coords_list, axis=0)
    return hand_coords_array

def scale_xy_data(landmarks_XY: NDArray) -> NDArray:
    """
    Scales X and Y values between 0 and 1 per row (gesture instance)
    """
    num_samples = landmarks_XY.shape[0]
    landmarks_XY = landmarks_XY.reshape(num_samples, 21, 2) # Reshape to (num_samples, 21, 2) → 21 landmarks (X, Y) per sample

    for i in range(num_samples):
        x_values = landmarks_XY[i, :, 0]
        y_values = landmarks_XY[i, :, 1]

        x_min, x_max = np.min(x_values), np.max(x_values)
        y_min, y_max = np.min(y_values), np.max(y_values)

        # Avoid division by zero
        x_range = x_max - x_min if x_max - x_min > 0 else 1
        y_range = y_max - y_min if y_max - y_min > 0 else 1

        # Normalize X and Y separately
        landmarks_XY[i, :, 0] = (x_values - x_min) / x_range
        landmarks_XY[i, :, 1] = (y_values - y_min) / y_range

    # Reshape back to (num_samples, 42)
    return landmarks_XY.reshape(num_samples, 42)
