#!/usr/bin/env python

import os
import argparse
import h5py
import numpy as np
import glob

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

from utils_keras import DNN
from utils_keras import MySequence
from utils_keras import LearningRate_StepDecay


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

    # ## path ## #
    os.makedirs(OUTDIR, exist_ok=True)
    outdir = decide_outdir()
    os.makedirs(outdir, exist_ok=True)
    history_path = os.path.join(outdir, 'history.csv')

    option_path = os.path.join(outdir, 'option.txt')
    save_options(args, option_path)

    weights_dir = os.path.join(outdir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    print(f'results will output into {outdir}\n')

    # ## load dataset ## #
    print("--- loading datasets ---")
    with h5py.File(args.input, mode='r') as f:
        # prepare data
        X_train = f[f'/{TRAIN_NAME}/{EXPLANATORY_NAME}']
        Y_train = f[f'/{TRAIN_NAME}/{RESPONSE_NAME}']
        X_val = f[f'/{VAL_NAME}/{EXPLANATORY_NAME}']
        Y_val = f[f'/{VAL_NAME}/{RESPONSE_NAME}']

        N_datasets_train = X_train.shape[0]
        N_datasets_val = X_val.shape[0]
        INPUT_DIM = X_train.shape[1]

        # decide batchsize
        if args.batch:
            batchsize = args.batch
        else:
            batchsize = N_datasets_train // 50

        X_train_mmap = np.memmap(
            os.path.join(outdir, 'x_train.mmap'),
            dtype='float32', mode='w+', shape=X_train.shape)
        Y_train_mmap = np.memmap(
            os.path.join(outdir, 'y_train.mmap'),
            dtype='float32', mode='w+', shape=Y_train.shape)
        X_val_mmap = np.memmap(
            os.path.join(outdir, 'x_val.mmap'),
            dtype='float32', mode='w+', shape=X_val.shape)
        Y_val_mmap = np.memmap(
            os.path.join(outdir, 'y_val.mmap'),
            dtype='float32', mode='w+', shape=Y_val.shape)

        # load each batch
        for i in range(0, N_datasets_train, batchsize):
            X_train_mmap[i:i+batchsize] = X_train[i:i+batchsize]
            Y_train_mmap[i:i+batchsize] = Y_train[i:i+batchsize]
        for i in range(0, N_datasets_val, batchsize):
            X_val_mmap[i:i+batchsize] = X_val[i:i+batchsize]
            Y_val_mmap[i:i+batchsize] = Y_val[i:i+batchsize]
    print("--- datasets have been loaded ---\n")

    # ## callback ## #
    # ModelCheckpoint
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(weights_dir, "epoch{epoch:03d}.hdf5"),
        save_weights_only=True,
        period=CHECKPOINT_PERIOD)
    # CSVLogger
    csv_logger = CSVLogger(history_path)

    # ## model ## #
    dnn = DNN(INPUT_DIM, args.lr)
    model = dnn(args.model)

    # ## datasets generator ## #
    train_generator = MySequence(N_datasets_train, batchsize, X_train_mmap, Y_train_mmap)
    # val_generator = MySequence(N_datasets_val, batchsize, X_val_mmap, Y_val_mmap, shuffle=False)

    # ## learningRateScheduler ## #
    lr_step_decay = LearningRate_StepDecay(args.epochs, args.lr)
    lr_scheduler = LearningRateScheduler(lr_step_decay)

    # ## learning ## #
    try:
        model.fit_generator(
            generator=train_generator,
            validation_data=(X_val_mmap, Y_val_mmap),
            epochs=args.epochs,
            callbacks=[lr_scheduler, checkpoint, csv_logger],
            verbose=2, shuffle=False)

    except KeyboardInterrupt:
        pass
    finally:
        remove_mmap(outdir)


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
