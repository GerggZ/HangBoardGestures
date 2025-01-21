# gestures/generic_gestures/gesture_predictor.py
import os.path

import numpy as np
from numpy.typing import NDArray
import tensorflow as tf
from utils.config import GENERIC_GESTURE_NAMES
from utils.scaling import scale_xy_data, convert_landmarks_to_numpy
from mediapipe.python.solutions.hands import Hands


class GesturePredictor:
    """Handles loading the trained model, extracting hand landmarks, and making predictions."""

    def __init__(self, model_path: str, verbose: bool = False) -> None:

        self.verbose = verbose

        if self.verbose: print("Loading gesture recognition model...")

        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            if self.verbose: print("Model loaded successfully.")
        else:
            self.model = None
            if self.verbose: print(f"No model found at {model_path}. Predictions disabled")

        self.optimized_predict = tf.function(
            self.model,
            input_signature=[tf.TensorSpec(shape=[None, 42], dtype=tf.float32)]
        )

    def extract_hand_landmarks(self, results: Hands) -> NDArray | None:  # typing: ignore
        """Extracts and normalizes hand landmarks from Mediapipe results."""
        if not results.multi_hand_landmarks:
            return None  # No hands detected

        hand_landmarks_array = convert_landmarks_to_numpy(results.multi_hand_landmarks)
        hand_landmarks_flattened_xy = hand_landmarks_array[:, :, :2].reshape(hand_landmarks_array.shape[0], -1)
        hand_landmarks_flattened_xy_scaled = np.asarray(scale_xy_data(hand_landmarks_flattened_xy))

        return hand_landmarks_flattened_xy_scaled

    def predict(self, results: Hands) -> str:
        """Uses the trained model to predict the gesture based on extracted hand landmarks."""
        if self.model is None:
            return "Model not trained"
        input_data = self.extract_hand_landmarks(results)
        if input_data is None:
            return "No Hand Detected"

        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        prediction = self.optimized_predict(input_data)
        predicted_idx = int(np.argmax(prediction))
        predicted_str: str = GENERIC_GESTURE_NAMES.get(predicted_idx, "Unknown")
        return predicted_str
