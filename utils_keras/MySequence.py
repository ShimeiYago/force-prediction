from tensorflow.keras.utils import Sequence
import math


class MySequence(Sequence):
    def __init__(self, N_datasets, batch_size, X_train, Y_train):
        self.N_datasets = N_datasets
        self.batch_size = batch_size
        self.N_iteration = math.ceil(N_datasets / batch_size)
        
        self.X_train = X_train
        self.Y_train = Y_train

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X = self.X_train[start_idx:last_idx]
        Y = self.Y_train[start_idx:last_idx]

        return X, Y

    def __len__(self):
        return self.N_iteration
