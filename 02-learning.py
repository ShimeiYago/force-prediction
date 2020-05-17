#!/usr/bin/env python

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
import models.DNN as DNN

OUTDIR_WEIGHT = "workspace/02-learning/weight"
OUTDIR_HISTORY = "workspace/02-learning/history"
LEARNING_RATE = 0.001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default="workspace/01-make-dataset/allatom/dataset.npz", help='input trajectory (.npz)')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, default=50, help='batch size')
    parser.add_argument('-i', '--learning_index', type=int, default=0, help='learning index')
    args = parser.parse_args()

    # make dir
    history_path = os.path.join(OUTDIR_WEIGHT, f'history{args.learning_index:0=3}')
    os.makedirs(OUTDIR_WEIGHT, exist_ok=True)
    os.makedirs(OUTDIR_HISTORY, exist_ok=True)

    # expranatory
    npz = np.load(args.data)
    x = npz['x']

    # response
    t = np.load(args.data)['y']
    del npz

    # data split
    x_train, x_val, t_train, t_val = train_test_split(x, t,
                                                      test_size=0.2,
                                                      shuffle=True)

    # learning
    model = DNN.model2(input_dim=x.shape[1], learning_rate=LEARNING_RATE)

    # take over previous weights
    if args.learning_index > 0:
        preindex = args.learning_index - 1
        model.load_weights(os.path.join(OUTDIR_WEIGHT, f'weight{preindex:0=3}.hdf5'))
    

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


def get_learning_index():
    objective_dirpath = os.path.join(OUTDIR, 'weights')
    indexlist = [int(f) for f in os.listdir(objective_dirpath) 
                 if os.path.isdir(os.path.join(objective_dirpath, f))]
    
    learning_index = 0
    for i in range(1000):
        if i in indexlist:
            continue

        learning_index = i
        break

    return learning_index


if __name__ == '__main__':
    main()
