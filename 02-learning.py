#!/usr/bin/env python

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import LearningRateScheduler
import models.DNN as DNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default="workspace/01-make-dataset/allatom/dataset.npz", help='input trajectory (.npz)')
    parser.add_argument('-o', '--out', default="workspace/02-learning/history.npy", help='output file path(.npy)')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, default=50, help='batch size')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)


    ### prepare data ###

    # expranatory
    x = np.load(args.input)['x']

    # response
    t = np.load(args.input)['y']

    # data split
    x_train, x_val, t_train, t_val = train_test_split(x, t,
                                                      test_size=0.2,
                                                      shuffle=True)

    # learning
    model = DNN.model2()
    lr_callback = LearningRateScheduler(lr_decay(args.epochs))

    hist = model.fit(x_train, t_train,
                     epochs=args.epochs,
                     batch_size=args.batch,
                     verbose=2,
                     validation_data=(x_val, t_val),
                     callbacks=[lr_callback])

    np.save(args.out, hist.history)


class lr_decay:
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def __call__(self, epoch):
        x = 0.01

        if epoch >= (self.n_epochs * 0.5):
            x = 0.001

        if epoch >= (self.n_epochs * 0.75):
            x = 0.0001

        return x


if __name__ == '__main__':
    main()
