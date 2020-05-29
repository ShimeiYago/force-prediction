#!/usr/bin/env python

import numpy as np
import os
import argparse
import tensorflow as tf
import models.DNN as DNN


INPUT_TRAIN = "workspace/02-make-dataset/training.npz"
INPUT_VAL = "workspace/02-make-dataset/validation.npz"

OUTDIR = "workspace/04-learning"
EACH_OUDIR_KEY = "try"
FREQ_SAVE_MODEL = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    args = parser.parse_args()

    # training data
    npz = np.load(INPUT_TRAIN)
    x_train = npz['x']
    t_train = npz['y']
    del npz

    # validation data
    npz = np.load(INPUT_VAL)
    x_val = npz['x']
    t_val = npz['y']
    del npz

    input_dim = x_train.shape[1]

    # model
    model = DNN.model2(input_dim=input_dim, learning_rate=args.lr)

    # learning
    hist = model.fit(x_train, t_train,
                     epochs=args.epochs,
                     batch_size=args.batch,
                     verbose=2,
                     validation_data=(x_val, t_val))


if __name__ == '__main__':
    main()
