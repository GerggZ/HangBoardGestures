# gestures/generic_gestures/gesture_recorder.py

import os
import time
import pandas as pd
from datetime import datetime
from src.utils.config import GENERIC_GESTURE_NAMES


class GestureRecorder:
    """
    Module for Recording the landmarks
    """
    def __init__(self, save_folder, verbose: bool = False):
        """
        Initializes things
        """
        self.gesture_idx = None
        self.verbose = verbose
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        self.data = None
        self.gesture_counts = None

        # Define CSV Columns (Gesture IDX, Gesture Name, Handedness, 21 Landmarks X 3 coords
        self.column_names = ["gesture_idx", "gesture_name", "handedness"] + [
            f"{axis}_{i}" for i in range(21) for axis in ['x', 'y', 'z']
        ]

        self.reset_data_df()

    def get_gesture_name(self) -> str:
        """Returns the gesture name based on the current index or 'None' if not selected."""
        if self.gesture_idx is None:
            return "None"
        return GENERIC_GESTURE_NAMES.get(self.gesture_idx, f"Unknown ({self.gesture_idx})")

    def set_gesture_idx(self, num) -> None:
        """Sets gesture index to the number pressed (0-9)."""
        self.gesture_idx = num

        if self.verbose:
            print(f"Switched to Gesture: {self.get_gesture_name()} ({self.gesture_idx})")

    def clear_selection(self) -> None:
        """Clears the currently selected gesture (sets it to None)."""
        self.gesture_idx = None

        if self.verbose:
            print("Gesture selection cleared (None)")

    def record_landmarks(self, results):
        """Records raw hand landmark coordinates (as given by Mediapipe) into the DataFrame."""
        if not results or not results.multi_hand_landmarks:
            # No hands detected
            return None
        if self.gesture_idx is None:
            # No gesture selected
            return None

        for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            # Store gesture index, gesture name, and handedness
            landmark_data = [self.gesture_idx, self.get_gesture_name(), handedness.classification[0].label]

            # Append raw MediaPipe Landmark data
            for landmark in hand_landmarks.landmark:
                landmark_data.extend([landmark.x, landmark.y, landmark.z])

            new_df_row = pd.DataFrame([landmark_data], columns=self.column_names)
            if self.data.empty:
                self.data = new_df_row
            else:
                self.data = pd.concat([self.data, new_df_row], ignore_index=True)

            # Update gesture counts
            self.gesture_counts[self.gesture_idx] += 1
            print(
                f"Captured raw Mediapipe hand data for Gesture: "
                f"{self.get_gesture_name()} ({self.gesture_idx}) "
                f"[{self.gesture_counts[self.gesture_idx]} recorded]"
            )

    def save_to_csv(self) -> None:
        """Saves recorded gestures to a CSV file inside the specified folder."""
        filename = os.path.join(self.save_folder, f"hand_landmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.data.to_csv(filename, index=False)

        self.reset_data_df()

    def reset_data_df(self) -> None:
        """Resets the data DataFrame and counts"""
        self.data = pd.DataFrame(columns=self.column_names)
        self.gesture_counts = {gesture: 0 for gesture in GENERIC_GESTURE_NAMES.keys()}  # Initialize counts to 0
