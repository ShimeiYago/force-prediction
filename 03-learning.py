#!/usr/bin/env python

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
import models.DNN as DNN

INPUT_TRAIN = "workspace/02-make-dataset/training.npz"
INPUT_VAL = "workspace/02-make-dataset/validation.npz"

OUTDIR_WEIGHT = "workspace/03-learning/weight"
OUTDIR_HISTORY = "workspace/03-learning/history"
OUTDIR_OPTION = "workspace/03-learning/option"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, default=50, help='batch size')
    parser.add_argument('-i', '--learning_index', type=int, default=0, help='learning index')
    parser.add_argument('-r', '--learning_rate', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()

    # make dir
    history_path = os.path.join(OUTDIR_WEIGHT, f'history{args.learning_index:0=3}')
    os.makedirs(OUTDIR_WEIGHT, exist_ok=True)
    os.makedirs(OUTDIR_HISTORY, exist_ok=True)
    os.makedirs(OUTDIR_OPTION, exist_ok=True)

    save_options(args)

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

    # model
    model = DNN.model2(input_dim=x_train.shape[1], learning_rate=args.learning_rate)

    # take over previous weights
    if args.learning_index > 0:
        preindex = args.learning_index - 1
        model.load_weights(os.path.join(OUTDIR_WEIGHT, f'weight{preindex:0=3}.hdf5'))

    # learning
    hist = model.fit(x_train, t_train,
                     epochs=args.epochs,
                     batch_size=args.batch,
                     verbose=2,
                     validation_data=(x_val, t_val))

    # save weights
    weight_path = os.path.join(OUTDIR_WEIGHT, f'weight{args.learning_index:0=3}.hdf5')
    model.save_weights(weight_path)

    # save history
    history_path = os.path.join(OUTDIR_HISTORY, f'history{args.learning_index:0=3}')
    np.save(history_path, hist.history)


def save_options(args):
    filepath = os.path.join(OUTDIR_OPTION, f'option{args.learning_index:0=3}.txt')
    with open(filepath, mode='w') as f:
        f.write(f'Number of epochs:\t{args.epochs}'
                f'\nLearning Rate:\t{args.learning_rate}')


if __name__ == '__main__':
    main()
