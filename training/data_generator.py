# training/data_generator.py

import numpy as np
from keras.utils import Sequence


class AugmentedGenerator(Sequence):
    """
    Generates data batches (X, y) with on-the-fly augmentation.
    """

    def __init__(self, X, y, augmentor, batch_size=32):
        """
        X: np.ndarray of shape (n_samples, 42, 1)
        y: np.ndarray of shape (n_samples, num_classes)
        batch_size: how many samples per batch
        augment: bool, if True applies random augmentation
        """
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = self.X[batch_indices].copy()
        y_batch = self.y[batch_indices]

        X_batch = self.augmentor(X_batch)  # the function from augment_utils.py

        return X_batch, y_batch

    def on_epoch_end(self):
        """
        Shuffles the data at the end of each epoch.
        """
        np.random.shuffle(self.indices)
