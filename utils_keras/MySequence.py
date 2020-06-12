from tensorflow.keras.utils import Sequence
import math
import numpy as np


class MySequence(Sequence):
    def __init__(self, N_datasets, batch_size, X, Y):
        self.N_datasets = N_datasets
        self.batch_size = batch_size
        self.N_iteration = math.ceil(N_datasets / batch_size)

        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X = self.X_train[start_idx:last_idx]
        Y = self.Y_train[start_idx:last_idx]

        return X, Y

    def __len__(self):
        return self.N_iteration


class MySequence_mmap(Sequence):
    def __init__(self, N_datasets, batch_size, X_path, Y_path, Xshape, Yshape):
        self.N_datasets = N_datasets
        self.batch_size = batch_size
        self.N_iteration = math.ceil(N_datasets / batch_size)

        self.X = np.memmap(X_path, dtype=np.float32, mode='r', shape=Xshape)
        self.Y = np.memmap(Y_path, dtype=np.float32, mode='r', shape=Yshape)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X = self.X[start_idx:last_idx]
        Y = self.Y[start_idx:last_idx]

        return X, Y

    def __len__(self):
        return self.N_iteration
