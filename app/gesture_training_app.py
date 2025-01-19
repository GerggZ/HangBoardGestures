# app/gesture_training.py

import os
import datetime
from keras.callbacks import TensorBoard

from training.utils.data_utils import (
    load_all_csvs,
    preprocess_data,
    split_data
)
from training.utils.model_utils import build_cnn_model
from training.data_generator import AugmentedGenerator
from training.utils.augment_utils import HandsAugmentor


class TrainModelApp:
    def __init__(self, model_path: str, landmark_data_folder: str, training_logs_folder: str):

        self.model_path = model_path
        self.landmark_data_folder = landmark_data_folder
        self.training_logs_folder = training_logs_folder


    def run(self):
        """Loads data, splits it, trains the CNN model, and evaluates it."""
        # Load and preprocess data
        df = load_all_csvs(data_folder=self.landmark_data_folder)
        X, y, num_classes = preprocess_data(df)

        # Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Get Augmentor ready
        augmentor = HandsAugmentor(
            flip_horizontal_prob=0.5, flip_vertical_prob=0.0, rotation_prob=0.7,
            max_yaw=30,  # (XZ-Plane rotation)
            max_pitch=30,  # (YZ-Plane rotation)
            max_roll=60  # (XY-Plane rotation, in-plane)
        )
        # Build model
        train_shape = 42   # Why don't we use 63? Because we rotate using the Z axis but we don't train with it atm
        model = build_cnn_model(input_shape=(train_shape, 1), num_classes=num_classes)

        train_generator = AugmentedGenerator(X_train, y_train, augmentor=augmentor, batch_size=32)
        val_generator = AugmentedGenerator(X_val, y_val, augmentor=augmentor, batch_size=32)
        test_generator = AugmentedGenerator(X_test, y_test, augmentor=augmentor, batch_size=32)

        # Set up TensorBoard logging
        log_dir = os.path.join(self.training_logs_folder, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

        # Train model with validation set and TensorBoard callback
        model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[tensorboard_callback])

        # Evaluate final test accuracy
        test_loss, test_acc = model.evaluate(test_generator)
        print(f"Final Test Accuracy: {test_acc:.2f}")

        # Save trained model
        model.save(self.model_path)
        print(f"Model saved as {self.model_path}")
        print(f"TensorBoard logs saved to: {log_dir}")
