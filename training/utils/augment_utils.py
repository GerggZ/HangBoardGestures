# training/augment_utils.py

import random
import numpy as np
from numpy.typing import NDArray


def augment_data(XY_batch: NDArray) -> NDArray:
    """
    Applies random rotation (-90° to 90°) and horizontal flipping per sample.
    Expects XY_batch shape: (batch_size, 42, 1)
        => We'll reshape to (batch_size, 21, 2) internally.
    Returns the augmented batch with the same shape.
    """
    batch_size = XY_batch.shape[0]
    XY_batch = XY_batch.reshape(batch_size, 21, 2)  # from (42,1) -> (21,2)

    for i in range(batch_size):
        # The "wrist" might be index 0 if that's how your data is structured
        wrist_x, wrist_y = XY_batch[i, 0, 0], XY_batch[i, 0, 1]

        # 50% chance to rotate
        if random.random() < 0.5:
            angle = np.radians(random.uniform(-90, 90))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            # Shift so wrist is at origin
            rel_coords = XY_batch[i] - [wrist_x, wrist_y]
            # Rotate
            rotated = rel_coords @ rotation_matrix.T
            # Shift back
            XY_batch[i] = rotated + [wrist_x, wrist_y]

        # 50% chance to flip horizontally
        if random.random() < 0.5:
            # Flip around the wrist's X coordinate
            XY_batch[i, :, 0] = 2 * wrist_x - XY_batch[i, :, 0]

    # Reshape back to (batch_size, 42, 1)
    return XY_batch.reshape(batch_size, 42, 1)
