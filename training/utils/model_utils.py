# training/model_utils.py

from keras.models import Sequential
from keras.layers import Input, Conv1D, Flatten, Dense, Dropout


def build_cnn_model(input_shape, num_classes):
    """
    Creates a 1D CNN model for gesture recognition.
    input_shape is e.g. (42, 1).
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation='relu'),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  # multi-class
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
