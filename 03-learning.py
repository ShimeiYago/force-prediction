#!/usr/bin/env python

import os
import argparse
import h5py

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

MAINCHAIN = ['N', 'CA', 'C', 'O']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default=os.path.join(INPUTDIR, 'datasets.hdf5'),
                        help='input datasets')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--model', type=int, default=1, help='model number')

    parser.add_argument('--weight', type=str, help='model weight path if take over')

    parser.add_argument('--on_memory', action='store_true', help='rapidly but use much memory')

    parser.add_argument('-a', '--atom', type=str,
                        help='designate atom species name ("CA", "N", "C", "O")')
    parser.add_argument('--scheduler', action='store_true', help='use scheduler')
    args = parser.parse_args()

    if not args.atom:
        for atom in MAINCHAIN:
            print(f'---------- {atom} ----------')
            learning(args, atom)
    else:
        learning(args, args.atom)


def learning(args, atom):
    # ## path ## #
    outdir = os.path.join(OUTDIR, atom)
    os.makedirs(outdir, exist_ok=True)
    outdir = decide_outdir(outdir)
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
        X_train = f[f'/{atom}/{TRAIN_NAME}/{EXPLANATORY_NAME}']
        Y_train = f[f'/{atom}/{TRAIN_NAME}/{RESPONSE_NAME}']
        X_val = f[f'/{atom}/{VAL_NAME}/{EXPLANATORY_NAME}']
        Y_val = f[f'/{atom}/{VAL_NAME}/{RESPONSE_NAME}']

        N_datasets_train = X_train.shape[0]
        N_datasets_val = X_val.shape[0]
        INPUT_DIM = X_train.shape[1]

        # decide batchsize
        if args.batch:
            batchsize = args.batch
        else:
            batchsize = N_datasets_train // 50

        # model
        dnn = DNN(INPUT_DIM, args.lr)
        model = dnn(args.model)
        if args.weight:
            model.load_weights(args.weight)

        # init epoch
        if args.weight:
            init_epoch = int(os.path.splitext(args.weight)[0][-3:])
        else:
            init_epoch = 0

        # datasets generator
        train_generator = MySequence(N_datasets_train, batchsize, X_train, Y_train, args.on_memory)
        val_generator = MySequence(N_datasets_val, batchsize, X_val, Y_val, args.on_memory)

        # learningRateScheduler
        if args.scheduler:
            lr_step_decay = LearningRate_StepDecay(args.epochs, args.lr)
            lr_scheduler = LearningRateScheduler(lr_step_decay)
        else:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.8,
                patience=10,
                verbose=1,
                min_lr=0.0001
            )

        # learning
        model.fit_generator(
            generator=train_generator,
            validation_data=val_generator,
            epochs=args.epochs,
            initial_epoch=init_epoch,
            callbacks=[lr_scheduler, checkpoint, csv_logger],
            shuffle=True,
            verbose=2)


def decide_outdir(outdir):
    for i in range(100):
        if f'{EACH_OUDIR_KEY}{i:0=3}' in os.listdir(outdir):
            continue
        return os.path.join(outdir, f'{EACH_OUDIR_KEY}{i:0=3}')


def save_options(args, fp):
    with open(fp, mode='w') as f:
        f.write(
            f'input file:\t{args.input}'
            f'\nepochs:\t{args.epochs}'
            f'\ninit lr:\t{args.lr}'
            f'\nbatch:\t{args.batch}'
            f'\nmodel number:\t{args.model}')


if __name__ == '__main__':
    main()
