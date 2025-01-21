# drawing/ui_drawer.py

import cv2
import time
import numpy as np
from numpy.typing import NDArray

from utils.config import HAND_CONNECTIONS, LANDMARK_TO_FINGER, FINGER_COLORS
from camera.base_camera import BaseCamera
from gestures.generic_gestures.hand_tracker import HandTracker
from mediapipe.python.solutions.hands import Hands


class UIDrawer:
    """
    Draws all UI elements on top of camera feed
    """
    def __init__(self) -> None:
        """
        Initializes UIDrawer module
        """
        self.show_save_message = False
        self.save_message_time = 0.
        self.save_message_duration = 0.
        self.prev_time = time.time()

    def get_fps(self) -> float:
        """
        Extracts the fps
        """
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
        self.prev_time = current_time
        return fps

    def draw_overlay(self, frame: NDArray, current_gesture_name: str, current_gesture_idx: int | None, predicted_gesture: str) -> None:
        """
        Draws a semi-transparent box and text showing UI information
        """
        # Define box position & dimensions
        box_x, box_y, box_w, box_h = 10, 10, 270, 150  # Text box size
        text_start_x, text_start_y = box_x + 15, box_y + 30  # Text padding
        line_spacing = 20
        text_transparency = 0.5  # Adjust this (0 = fully transparent, 1 = fully solid)

        # Create an overlay for transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, text_transparency, frame, 1 - text_transparency, 0, frame)

        # Set font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thickness = 0.5, 1
        text_color = (255, 255, 255)

        # UI Text Content
        ui_text = [
            f"Gesture: {current_gesture_name} ({current_gesture_idx})",
            f"Predicted: {predicted_gesture}",
            f"FPS: {self.get_fps(): .2f}",
            "ENTER to Capture",
            "0-9 to Select Gesture",
            "'C' to Clear Selection",
            "'S' to Save, 'Q' to Quit"
        ]

        # Draw each line separately
        for i, text in enumerate(ui_text):
            cv2.putText(frame, text, (text_start_x, text_start_y + i * line_spacing),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Show "Saved!" message if triggered
        if self.show_save_message and (time.time() - self.save_message_time < 0.5):
            cv2.rectangle(overlay, (box_x, box_y + box_h + 10), (box_x + box_w, box_y + box_h + 50), (0, 255, 0), -1)
            cv2.addWeighted(overlay, text_transparency, frame, 1 - text_transparency, 0, frame)
            cv2.putText(frame, "Saved!", (text_start_x, box_y + box_h + 40),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif self.show_save_message:
            self.show_save_message = False

    def trigger_save_message(self) -> None:
        """
        Enables the 'Saved!' message and resets the timer.
        """
        self.show_save_message = True
        self.save_message_time = time.time()

    def flash_screen(self, camera: BaseCamera, tracker: HandTracker, duration: float = 0.1, intensity: float = 0.8, num_steps: int = 5) -> None:
        """
        Applies a white overlay (flash) on top of the live video feed to give optical feedback to data capture

        Args:
            - duration (float): How long the flash should last (start to finish)
            - intensity (float): How strong the white flash is (1.0 = fully white, 0.5 = semi-transparent
            - num_steps (int): How many steps to fade from initial flash back to normal (more steps = smoother fade)
        """
        fade_step_duration = duration / num_steps  # Convert fade duration to seconds

        for step in range(num_steps, -1, -1):  # Gradually decrease overlay intensity
            start_time = time.time()
            while time.time() - start_time < fade_step_duration:
                frame = camera.read_frame()
                fade_alpha = (intensity * step) / num_steps

                h, w, _ = frame.shape
                white_overlay = np.full((h, w, 3), 255, dtype=np.uint8)

                blended = cv2.addWeighted(frame, 1 - fade_alpha, white_overlay, fade_alpha, 0)

                # Continue to detect/draw landmarks (to help it look smoother)
                results = tracker.process_frame(blended)
                self.draw_hand_landmarks(blended, results)

                # Display the blended frame
                cv2.imshow("Hand Tracking UI", blended)
                cv2.waitKey(1)  # Allow OpenCV to refresh the UI

    @staticmethod
    def draw_hand_landmarks(frame: NDArray, results: Hands) -> None:
        """
        Draws detected hand landmarks with colors and connections.
        """
        if not results or not results.multi_hand_landmarks:
            # No hands detected
            return None

        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw colored connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start_lm = hand_landmarks.landmark[start_idx]
                end_lm = hand_landmarks.landmark[end_idx]

                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))

                finger = LANDMARK_TO_FINGER.get(start_idx, 'wrist')
                color = FINGER_COLORS.get(finger, (255, 255, 255))  # Default white

                cv2.line(frame, start_pos, end_pos, color, 2)

            # Draw colored landmarks
            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                finger = LANDMARK_TO_FINGER.get(idx, 'wrist')
                color = FINGER_COLORS.get(finger, (255, 255, 255))  # Default white

                cv2.circle(frame, (cx, cy), 5, color, -1)  # Draw colored landmark



