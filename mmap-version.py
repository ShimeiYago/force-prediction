#!/usr/bin/env python

import os
import argparse
import h5py
import numpy as np
import glob

from utils_keras import DNN
from utils_keras.MySequence import MySequence_mmap


INPUTDIR = "workspace/01-make-datasets"
OUTDIR = "workspace/03-learning"
EACH_OUDIR_KEY = "try"
CHECKPOINT_PERIOD = 10

TRAIN_NAME = "training"
VAL_NAME = "validation"
EXPLANATORY_NAME = "x"
RESPONSE_NAME = "y"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default=os.path.join(INPUTDIR, 'datasets.hdf5'),
                        help='input datasets')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--model', type=int, default=1, help='model number')
    args = parser.parse_args()


    # ## load dataset ## #
    print("--- loading datasets ---")
    with h5py.File(args.input, mode='r') as f:
        # prepare data
        X_train = f[f'/{TRAIN_NAME}/{EXPLANATORY_NAME}']
        Y_train = f[f'/{TRAIN_NAME}/{RESPONSE_NAME}']

        N_datasets_train = X_train.shape[0]
        INPUT_DIM = X_train.shape[1]

        Xshape = X_train.shape
        Yshape = Y_train.shape

        # decide batchsize
        if args.batch:
            batchsize = args.batch
        else:
            batchsize = N_datasets_train // 50

        X_train_mmap = np.memmap(
            'x_train.mmap',
            dtype='float32', mode='w+', shape=X_train.shape)
        Y_train_mmap = np.memmap(
            'y_train.mmap',
            dtype='float32', mode='w+', shape=Y_train.shape)

        # load each batch
        for i in range(0, N_datasets_train, batchsize):
            X_train_mmap[i:i+batchsize] = X_train[i:i+batchsize]
            Y_train_mmap[i:i+batchsize] = Y_train[i:i+batchsize]
        
        del X_train_mmap
        del Y_train_mmap
    print("--- datasets have been loaded ---\n")

    # ## model ## #
    dnn = DNN(INPUT_DIM, args.lr)
    model = dnn(args.model)

    # ## datasets generator ## #
    train_generator = MySequence_mmap(N_datasets_train, batchsize, 'x_train.mmap', 'y_train.mmap', Xshape, Yshape)

    # ## learning ## #
    try:
        model.fit_generator(
            generator=train_generator,
            epochs=args.epochs,
            verbose=2, shuffle=True)

    except KeyboardInterrupt:
        pass
    finally:
        remove_mmap('./')


def decide_outdir():
    for i in range(100):
        if f'{EACH_OUDIR_KEY}{i:0=3}' in os.listdir(OUTDIR):
            continue
        return os.path.join(OUTDIR, f'{EACH_OUDIR_KEY}{i:0=3}')


def save_options(args, fp):
    with open(fp, mode='w') as f:
        f.write(
            f'input file:\t{args.input}'
            f'epochs:\t{args.epochs}'
            f'\ninit lr:\t{args.lr}'
            f'\nbatch:\t{args.batch}'
            f'\nmodel number:\t{args.model}')


def remove_mmap(outdir):
    for fp in glob.glob(f"{outdir}/*.mmap"):
        os.remove(fp)


if __name__ == '__main__':
    main()