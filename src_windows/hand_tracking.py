import os
import cv2
import time
import numpy as np
import mediapipe as mp
import pandas as pd
from datetime import datetime


# Gestures for the hand
GESTURE_NAMES = {
    0: "Open Hand",
    1: "Fist",
    2: "Peace Sign",
    3: "Thumbs Up",
    5: "OK Sign",
}

# Connections between hand landmarks (using landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

# Dictionary for finger colors
FINGER_COLORS = {
    'thumb': (0, 0, 255),   # Red
    'index': (0, 255, 0),   # Green
    'middle': (255, 0, 0),  # Blue
    'ring': (0, 255, 255),  # Yellow
    'pinky': (255, 0, 255)  # Magenta
}

# Map landmark index to finger
LANDMARK_TO_FINGER = {
    0: 'wrist',
    1: 'thumb', 2: 'thumb', 3: 'thumb', 4: 'thumb',
    5: 'index', 6: 'index', 7: 'index', 8: 'index',
    9: 'middle', 10: 'middle', 11: 'middle', 12: 'middle',
    13: 'ring', 14: 'ring', 15: 'ring', 16: 'ring',
    17: 'pinky', 18: 'pinky', 19: 'pinky', 20: 'pinky'
}


class HandTracker:
    """Handles hand detection, drawing, and landmark extraction."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame):
        """Converts frame to RGB and processes hand landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(frame_rgb)

    def draw_hand_landmarks(self, frame, results):
        """Draws detected hand landmarks with colors and connections."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape

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


class GestureRecorder:
    """Handles recording and saving hand landmark data."""
    def __init__(self, save_folder="hand_landmarks_data"):
        self.gesture_idx = 0  # Default gesture index
        self.save_folder = save_folder  # Folder to save CSV files
        os.makedirs(self.save_folder, exist_ok=True)

        # Define CSV columns
        self.columns = ["gesture_idx", "gesture_name"] + [f"{axis}_{i}" for i in range(21) for axis in ["x", "y", "z"]]
        self.data = pd.DataFrame(columns=self.columns)

    def get_gesture_name(self):
        """Returns the gesture name based on the current index."""
        return GESTURE_NAMES.get(self.gesture_idx, f"Unknown ({self.gesture_idx})")

    def set_gesture_idx(self, num):
        """Sets gesture index to the number pressed (0-9)."""
        self.gesture_idx = num
        print(f"Switched to Gesture: {self.get_gesture_name()} ({self.gesture_idx})")

    def record_landmarks(self, results):
        """Records raw hand landmark coordinates (as given by Mediapipe) into the DataFrame."""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Store gesture index and name
                landmark_data = [self.gesture_idx, self.get_gesture_name()]

                # Store raw Mediapipe output (normalized by image dimensions, but not by hand bounding box)
                for landmark in hand_landmarks.landmark:
                    cx = landmark.x  # Raw Mediapipe x (0 to 1 relative to image width)
                    cy = landmark.y  # Raw Mediapipe y (0 to 1 relative to image height)
                    cz = landmark.z  # Depth (relative, but not scaled)

                    landmark_data.extend([cx, cy, cz])

                # Create DataFrame row
                new_row = pd.DataFrame([landmark_data], columns=self.columns)

                # If empty, assign directly, else concatenate
                if self.data.empty:
                    self.data = new_row
                else:
                    self.data = pd.concat([self.data, new_row], ignore_index=True)

                print(f"Captured raw Mediapipe hand data for Gesture: {self.get_gesture_name()} ({self.gesture_idx})")

    def save_to_csv(self):
        """Saves recorded gestures to a CSV file inside the specified folder."""
        filename = os.path.join(self.save_folder, f"hand_landmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

        # Reset the DataFrame to clear recorded data
        self.data = pd.DataFrame(columns=self.columns)


class HandTrackingApp:
    """Main application handling the UI and event loop."""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.tracker = HandTracker()
        self.recorder = GestureRecorder()
        self.show_save_message = False  # Track if "Saved!" should be displayed
        self.save_message_time = 0  # Time when the save message was triggered

    def flash_screen(self, fade_duration=0.1):
        """Applies a white overlay on top of the live video feed and fades it out smoothly."""
        flash_intensity = 0.8  # How strong the white flash is (1.0 = fully white, 0.5 = semi-transparent)
        num_steps = 5  # More steps = smoother fade
        fade_duration_ms = int((fade_duration / num_steps) * 1000)  # Convert fade duration to milliseconds

        for step in range(num_steps, -1, -1):  # Gradually decrease overlay intensity
            ret, frame = self.cap.read()  # ✅ Keep capturing new frames from the video feed
            if not ret:
                break

            # Create white overlay and blend with live feed
            h, w, _ = frame.shape
            white_overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
            fade_alpha = (flash_intensity * step) / num_steps
            blended_frame = cv2.addWeighted(frame, 1 - fade_alpha, white_overlay, fade_alpha, 0)

            # Draw hand landmarks and UI
            self.tracker.draw_hand_landmarks(blended_frame, self.tracker.process_frame(blended_frame))
            self.draw_ui(blended_frame)

            cv2.imshow("Hand Tracking UI", blended_frame)
            cv2.waitKey(fade_duration_ms)  # Control fade speed

    def run(self):
        """Main event loop for the hand tracking application."""
        with self.tracker.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process hand tracking
                results = self.tracker.process_frame(frame)
                self.tracker.draw_hand_landmarks(frame, results)

                # Draw UI
                self.draw_ui(frame)

                # Show the output
                cv2.imshow("Hand Tracking UI", frame)

                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF

                if key == 13:  # ENTER key (Capture Data)
                    self.recorder.record_landmarks(results)
                    self.flash_screen()  # Flash white screen and fade back
                elif key == ord('s'):  # 'S' key (Save Data)
                    self.recorder.save_to_csv()
                    self.show_save_message = True  # Enable the "Saved!" message
                    self.save_message_time = time.time()  # Track when the message appeared
                elif key == ord('q'):  # 'Q' key (Quit)
                    break
                elif ord('0') <= key <= ord('9'):  # Number keys 0-9
                    self.recorder.set_gesture_idx(int(chr(key)))  # Convert ASCII to integer

        self.cap.release()
        cv2.destroyAllWindows()

    def draw_ui(self, frame):
        """Draws a semi-transparent UI box with gesture names."""

        # Define box position & dimensions
        box_x, box_y, box_w, box_h = 10, 10, 340, 160
        text_start_x = box_x + 15  # Left padding
        text_start_y = box_y + 30  # Start text inside box
        line_spacing = 30  # Vertical space between lines
        transparency = 0.5  # Adjust this (0 = fully transparent, 1 = fully solid)

        # Create an overlay for transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)

        # Set font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)

        # Get current gesture name
        current_gesture = self.recorder.get_gesture_name()

        # UI Text Content
        ui_text = [
            f"Gesture: {current_gesture} ({self.recorder.gesture_idx})",
            "ENTER to Capture",
            "0-9 to Select Gesture",
            "'S' to Save, 'Q' to Quit"
        ]

        # Draw each line separately
        for i, text in enumerate(ui_text):
            cv2.putText(frame, text, (text_start_x, text_start_y + i * line_spacing),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Show "Saved!" message if triggered
        if self.show_save_message and (time.time() - self.save_message_time < 0.5):
            cv2.rectangle(overlay, (box_x, box_y + box_h + 10), (box_x + box_w, box_y + box_h + 50), (0, 255, 0), -1)
            cv2.addWeighted(overlay, transparency, frame, 1 - transparency, 0, frame)
            cv2.putText(frame, "Saved!", (text_start_x, box_y + box_h + 40),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif self.show_save_message:
            self.show_save_message = False


# Run the application
if __name__ == "__main__":
    app = HandTrackingApp()
    app.run()
