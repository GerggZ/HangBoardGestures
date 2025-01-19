# main.py

import cv2
from camera.windows_camera import WindowsCamera  # or RaspberryPiCamera
from gestures.generic_gestures.hand_tracker import HandTracker
from gestures.generic_gestures.gesture_recorder import GestureRecorder
from gestures.generic_gestures.gesture_predictor import GesturePredictor
from utils.config import GENERIC_GESTURE_NAMES
from drawing.ui_drawer import UIDrawer

from training.train import train_model
from utils.config import LANDMARK_DATA_FOLDER, MODEL_PATH
class HandTrackingApp:
    def __init__(self):
        self.camera = WindowsCamera(camera_source=0, width=640, height=480)
        self.tracker = HandTracker()
        self.recorder = GestureRecorder(save_folder=LANDMARK_DATA_FOLDER)
        self.predictor = GesturePredictor(model_path=MODEL_PATH)

        # The class that draws everything (UI + landmarks)
        self.ui = UIDrawer()

        self.predicted_gesture = "None"

    def run(self):
        self.camera.open_camera()

        while True:
            frame = self.camera.read_frame()

            # 1) Detect hands
            results = self.tracker.process_frame(frame)

            # 2) Draw the actual landmarks on the frame
            self.ui.draw_hand_landmarks(frame, results)

            # 3) Predict gesture
            self.predicted_gesture = self.predictor.predict(results)

            # 4) Draw UI text overlay
            self.ui.draw_overlay(
                frame,
                current_gesture_name=self.recorder.get_gesture_name(),
                current_gesture_idx=self.recorder.gesture_idx,
                predicted_gesture=self.predicted_gesture
            )

            cv2.imshow("Hand Tracking UI", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                self.recorder.clear_selection()
            elif key == ord('s'):
                self.recorder.save_to_csv()
                self.ui.trigger_save_message()
            elif ord('0') <= key <= ord('9'):
                gesture_idx = int(chr(key))
                if gesture_idx in GENERIC_GESTURE_NAMES:
                    self.recorder.set_gesture_idx(gesture_idx)
                else:
                    self.recorder.clear_selection()
            elif key == 13:  # Enter
                self.recorder.record_landmarks(results)
                # A quick white flash
                self.ui.flash_screen(self.camera, self.tracker)

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    train_gesture_tracker = False

    if train_gesture_tracker:
        train_model()
    else:
        app = HandTrackingApp()
        app.run()
