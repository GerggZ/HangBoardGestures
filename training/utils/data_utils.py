import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from utils.config import LANDMARK_DATA_FOLDER
from utils.scaling import scale_xy_data


def load_all_csvs(data_folder: str = LANDMARK_DATA_FOLDER) -> pd.DataFrame:
    """
    Loads all CSV gesture data from the specified folder into a single DataFrame
    """
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    dataframes = [pd.read_csv(os.path.join(data_folder, file)) for file in all_files]
    return pd.concat(dataframes, ignore_index=True)


def split_data(X, y, train_size=0.8, test_size=0.1, val_size=0.1, random_state=42, verbose=False):
    """Splits dataset into training, validation, and test sets."""
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=test_size / (test_size + val_size), random_state=random_state
    )

    if verbose:
        print(f"Dataset split:\n"
              f"- Training: {X_train.shape[0]} samples\n"
              f"- Validation: {X_val.shape[0]} samples\n"
              f"- Test: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_data(df, standardize=False):
    """
    Prepares dataset: extracts features (XY), scales X/Y, and optionally standardizes
    """
    if df is None:
        raise ValueError("No dataset provided!")

    # Extract labels (gesture index) and rename to "labels" for clarity
    labels = df["gesture_idx"].astype(int)
    num_classes = len(labels.unique())

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=num_classes)

    # Extract only X and Y values (ignore Z values)
    XY_cols = [col for col in df.columns if 'x_' in col or 'y_' in col]  # Select only X and Y column_names
    XY_data = df[XY_cols].values  # Convert to NumPy array

    # Scale (normalize) X and Y values
    XY_data = scale_xy_data(XY_data)

    # Standardization (optional)
    if standardize:
        mean = np.mean(XY_data, axis=0)  # Compute mean per feature
        std = np.std(XY_data, axis=0)  # Compute standard deviation per feature
        std[std == 0] = 1  # Prevent division by zero

        XY_data = (XY_data - mean) / std  # Standardize

    # Reshape for CNN input: (samples, 42 features, 1 channel)
    XY_data = XY_data.reshape(XY_data.shape[0], XY_data.shape[1], 1)

    return XY_data, labels, num_classes
