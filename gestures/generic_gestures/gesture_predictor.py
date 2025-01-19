# gestures/generic_gestures/gesture_predictor.py
import os.path

import numpy as np
from keras.models import load_model
from utils.config import GENERIC_GESTURE_NAMES
from utils.scaling import scale_xy_data, convert_landmarks_to_numpy


class GesturePredictor:
    """Handles loading the trained model, extracting hand landmarks, and making predictions."""

    def __init__(self, model_path: str, verbose: bool = False):

        self.verbose = verbose

        if self.verbose: print("Loading gesture recognition model...")

        if os.path.exists(model_path):
            self.model = load_model(model_path)
            if self.verbose: print("Model loaded successfully.")
        else:
            self.model = None
            if self.verbose: print(f"No model found at {model_path}. Predictions disabled")

    def extract_hand_landmarks(self, results):
        """Extracts and normalizes hand landmarks from Mediapipe results."""
        if not results.multi_hand_landmarks:
            return None  # No hands detected

        hand_landmarks_array = convert_landmarks_to_numpy(results.multi_hand_landmarks)
        hand_landmarks_flattened_xy = hand_landmarks_array[:, :, :2].reshape(hand_landmarks_array.shape[0], -1)
        hand_landmarks_flattened_xy_scaled = scale_xy_data(hand_landmarks_flattened_xy)

        return hand_landmarks_flattened_xy_scaled

    def predict(self, results):
        """Uses the trained model to predict the gesture based on extracted hand landmarks."""
        if self.model is None:
            return "Model not trained"
        input_data = self.extract_hand_landmarks(results)
        if input_data is None:
            return "No Hand Detected"

        prediction = self.model.predict(input_data, verbose=0)
        predicted_idx = np.argmax(prediction)
        return GENERIC_GESTURE_NAMES.get(predicted_idx, "Unknown")
