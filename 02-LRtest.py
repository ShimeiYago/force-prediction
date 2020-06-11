#!/usr/bin/env python

import os
import argparse
import h5py
import numpy as np
import glob

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger

from utils_keras import DNN
from utils_keras import MySequence
from utils_keras import LRtest

LRLIST = [
    0.0000001, 0.000001, 0.00001, 0.0001,
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.01, 0.02, 0.03, 0.04, 0.05]

INPUTDIR = "workspace/01-make-datasets"
OUTDIR = "workspace/02-lrtest"

TRAIN_NAME = "training"
VAL_NAME = "validation"
EXPLANATORY_NAME = "x"
RESPONSE_NAME = "y"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default=os.path.join(INPUTDIR, 'datasets.hdf5'),
                        help='input datasets')
    parser.add_argument('-b', '--batch', type=int, help='batch size')
    parser.add_argument('--model', type=int, default=1, help='model number')
    args = parser.parse_args()

    # ## path ## #
    keyword = os.path.splitext(os.path.basename(args.input))[0] + f'-model{args.model:02d}'
    outdir = os.path.join(OUTDIR, keyword)
    os.makedirs(outdir, exist_ok=True)
    history_path = os.path.join(outdir, f'{keyword}.csv')

    # ## load dataset ## #
    print("--- loading datasets ---")
    with h5py.File(args.input, mode='r') as f:
        # prepare data
        X_train = f[f'/{TRAIN_NAME}/{EXPLANATORY_NAME}']
        Y_train = f[f'/{TRAIN_NAME}/{RESPONSE_NAME}']

        N_datasets = X_train.shape[0]
        INPUT_DIM = X_train.shape[1]

        # decide batchsize
        if args.batch:
            batchsize = args.batch
        else:
            batchsize = N_datasets // 50

        X_train_mmap = np.memmap(
            os.path.join(outdir, 'x_train.mmap'),
            dtype='float32', mode='w+', shape=X_train.shape)
        Y_train_mmap = np.memmap(
            os.path.join(outdir, 'y_train.mmap'),
            dtype='float32', mode='w+', shape=Y_train.shape)

        # load each batch
        for i in range(0, N_datasets, batchsize):
            X_train_mmap[i:i+batchsize] = X_train[i:i+batchsize]
            Y_train_mmap[i:i+batchsize] = Y_train[i:i+batchsize]
    print("--- datasets have been loaded ---\n")

    # ## callback ## #
    # CSVLogger
    csv_logger = CSVLogger(history_path)
    # learningRateScheduler
    lr_test = LRtest(LRLIST)
    lr_scheduler = LearningRateScheduler(lr_test, verbose=1)

    # ## model ## #
    dnn = DNN(INPUT_DIM)
    model = dnn(args.model)

    # ## datasets generator ## #
    train_generator = MySequence(N_datasets, batchsize, X_train_mmap, Y_train_mmap)

    # ## learning ## #
    try:
        model.fit_generator(
            generator=train_generator,
            epochs=len(LRLIST),
            callbacks=[lr_scheduler, csv_logger],
            verbose=2, shuffle=False)

    except KeyboardInterrupt:
        pass
    finally:
        remove_mmap(outdir)


def remove_mmap(outdir):
    for fp in glob.glob(f"{outdir}/*.mmap"):
        os.remove(fp)


if __name__ == '__main__':
    main()
