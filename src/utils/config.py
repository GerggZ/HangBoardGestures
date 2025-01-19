# utils/config.py

import os


CURRENT_NAME = "new_temp"
MODEL_PATH = os.path.join("src", "data", f"{CURRENT_NAME}_cnn_model.keras")
LANDMARK_DATA_FOLDER = os.path.join("src", "data", "hand_landmarks_data", CURRENT_NAME)
TRAINING_LOGS_FOLDER = os.path.join("src", "data", "logs", CURRENT_NAME)

# Gestures for the hand
GENERIC_GESTURE_NAMES = {
    0: "Open Hand",
    1: "Fist",
    2: "Bird",
    3: "Peace",
    4: "OK",
    5: "Rock",
    6: "Llama",
    7: "Ronaldinho"
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
