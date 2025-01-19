# training/trian.py

import os
import datetime
from keras.callbacks import TensorBoard

from training.utils.data_utils import (
    load_all_csvs,
    preprocess_data,
    split_data
)
from training.utils.model_utils import build_cnn_model
from training.data_generator import AugmentedDataGenerator

from utils.config import MODEL_PATH, TRAINING_LOGS_FOLDER


def train_model():
    """Loads data, splits it, trains the CNN model, and evaluates it."""
    # Load and preprocess data
    df = load_all_csvs()
    X, y, num_classes = preprocess_data(df, standardize=False)

    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Build model
    model = build_cnn_model(input_shape=(X_train.shape[1], 1), num_classes=num_classes)

    train_generator = AugmentedDataGenerator(X_train, y_train, batch_size=32, augment=True)
    val_generator = AugmentedDataGenerator(X_val, y_val, batch_size=32, augment=False)
    test_generator = AugmentedDataGenerator(X_test, y_test, batch_size=32, augment=False)

    # Set up TensorBoard logging
    log_dir = os.path.join(TRAINING_LOGS_FOLDER, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

    # Train model with validation set and TensorBoard callback
    model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[tensorboard_callback])

    # Evaluate final test accuracy
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Final Test Accuracy: {test_acc:.2f}")

    # Save trained model
    model.save(MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")
    print(f"TensorBoard logs saved to: {log_dir}")
