import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray
from keras.utils import to_categorical

from utils.scaling import scale_xy_data


def load_all_csvs(data_folder: str) -> pd.DataFrame:
    """
    Loads all CSV gesture data from the specified folder into a single DataFrame
    """
    all_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    dataframes = [pd.read_csv(os.path.join(data_folder, file)) for file in all_files]
    return pd.concat(dataframes, ignore_index=True)


def split_data(
        X: NDArray, y: NDArray,
        train_size: float = 0.8, test_size: float = 0.1, val_size: float = 0.1,
        random_state: int = 42, verbose: bool = False
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
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


def preprocess_data(df: pd.DataFrame) -> tuple[NDArray, NDArray, int]:
    """
    Prepares dataset: extracts features (XY), scales X/Y, and optionally standardizes
    """
    if df is None:
        raise ValueError("No dataset provided!")

    # Extract labels (gesture index) and rename to "labels" for clarity
    labels = df["gesture_idx"].astype(np.int32).to_numpy()
    num_classes = len(np.unique(labels))

    # One-hot encode the labels
    labels = np.asarray(to_categorical(labels, num_classes=num_classes))

    # Extract only X, Y, and Z values (ignore handedness for now)
    XYZ_cols = [col for col in df.columns if col.startswith(('x_', 'y_', 'z_'))]
    XYZ_data = df[XYZ_cols].values  # Convert to NumPy array

    # Reshape for augmentation: (samples, 21 landmarks, 3 coordinates)
    XYZ_data = XYZ_data.reshape(XYZ_data.shape[0], 21, 3, 1)  # Keep the extra channel

    return XYZ_data, labels, num_classes
