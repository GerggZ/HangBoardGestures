# gestures/generic_gestures/hand_tracker.py

import cv2
from numpy.typing import NDArray
import mediapipe as mp


class HandTracker:
    """
    Handles hand detection and landmark extraction using Googles MediaPipe
    """
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initializes HandTracker module
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1
        )

    def process_frame(self, frame: NDArray):
        """
        Converts frame from BGR to RGB and processes hand landmarks

        Args:
            - frame (NDArray): Image frame array (in BGR format)

        Returns:
            MediaPipe Hand Landmarks Object Results
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame)
