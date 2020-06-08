#!/usr/bin/env python

import os
import argparse
import h5py

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

    # ## callback ## #
    # ModelCheckpoint
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(weights_dir, "epoch{epoch:03d}.hdf5"),
        save_weights_only=True,
        period=CHECKPOINT_PERIOD)
    # CSVLogger
    csv_logger = CSVLogger(history_path)

    # ## learning ## #
    with h5py.File(args.input, mode='r') as f:
        # prepare data
        X_train = f[f'/{TRAIN_NAME}/{EXPLANATORY_NAME}']
        Y_train = f[f'/{TRAIN_NAME}/{RESPONSE_NAME}']
        X_val = f[f'/{VAL_NAME}/{EXPLANATORY_NAME}']
        Y_val = f[f'/{VAL_NAME}/{RESPONSE_NAME}']

        N_datasets = X_train.shape[0]
        INPUT_DIM = X_train.shape[1]

        # decide batchsize
        if args.batch:
            batchsize = args.batch
        else:
            batchsize = N_datasets // 50

        # model
        dnn = DNN(INPUT_DIM, args.lr)
        model = dnn(args.model)

        # datasets generator
        train_generator = MySequence(N_datasets, batchsize, X_train, Y_train)
        val_generator = MySequence(N_datasets, batchsize, X_val, Y_val)

        # learningRateScheduler
        lr_step_decay = LearningRate_StepDecay(args.epochs, args.lr)
        lr_scheduler = LearningRateScheduler(lr_step_decay)

        # learning
        model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            epochs=args.epochs,
            callbacks=[lr_scheduler, checkpoint, csv_logger],
            verbose=2)


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


if __name__ == '__main__':
    main()
