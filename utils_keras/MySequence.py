from tensorflow.keras.utils import Sequence
import math
import numpy as np


class MySequence(Sequence):
    def __init__(self, N_datasets, batch_size, X, Y, shuffle=True):
        self.N_datasets = N_datasets
        self.batch_size = batch_size
        self.N_iteration = math.ceil(N_datasets / batch_size)

        self.X = X
        self.Y = Y

        self.shuffle = shuffle
        if self.shuffle:
            self.order = np.random.permutation(self.N_datasets)
        else:
            self.order = np.arange(self.N_datasets)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        x = self.X[self.order[start_idx:last_idx]]
        y = self.Y[self.order[start_idx:last_idx]]

        return x, y

    def __len__(self):
        return self.N_iteration

    def on_epoch_end(self):
        if self.shuffle:
            self.order = np.random.permutation(self.N_datasets)
