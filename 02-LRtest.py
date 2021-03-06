#!/usr/bin/env python

import os
import argparse
import h5py

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
EXPLANATORY_NAME = "x"
RESPONSE_NAME = "y"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default=os.path.join(INPUTDIR, 'datasets.hdf5'),
                        help='input datasets')
    parser.add_argument('-b', '--batch', type=int, help='batch size')
    parser.add_argument('--model', type=int, default=1, help='model number')
    parser.add_argument('--on_memory', action='store_true', help='rapidly but use much memory')
    parser.add_argument('-a', '--atom', type=str, required=True,
                        help='designate atom species name ("CA", "N", "C", etc.)')
    args = parser.parse_args()

    # ## path ## #
    outdir = os.path.join(OUTDIR, args.atom)
    keyword = os.path.splitext(os.path.basename(args.input))[0] + f'-{args.atom}-model{args.model:02d}'
    os.makedirs(outdir, exist_ok=True)
    history_path = os.path.join(outdir, f'{keyword}.csv')

    # ## callback ## #
    # CSVLogger
    csv_logger = CSVLogger(history_path)
    # learningRateScheduler
    lr_test = LRtest(LRLIST)
    lr_scheduler = LearningRateScheduler(lr_test, verbose=1)


    # ## LRtest ## #
    with h5py.File(args.input, mode='r') as f:
        # prepare data
        X_train = f[f'/{args.atom}/{TRAIN_NAME}/{EXPLANATORY_NAME}']
        Y_train = f[f'/{args.atom}/{TRAIN_NAME}/{RESPONSE_NAME}']

        N_datasets = X_train.shape[0]
        INPUT_DIM = X_train.shape[1]

        # model
        dnn = DNN(INPUT_DIM)
        model = dnn(args.model)

        # decide batchsize
        if not args.batch:
            args.batch = N_datasets // 50

        # datasets generator
        train_generator = MySequence(N_datasets, args.batch, X_train, Y_train, args.on_memory)

        # learning
        model.fit_generator(
            generator=train_generator,
            epochs=len(LRLIST),
            callbacks=[lr_scheduler, csv_logger],
            shuffle=True,
            verbose=2)


if __name__ == '__main__':
    main()
