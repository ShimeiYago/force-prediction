#!/usr/bin/env python

import numpy as np
import os
import argparse
import tensorflow as tf
import models.DNN as DNN


X_TRAIN = "workspace/02-make-dataset/x_train.npz"
X_VAL = "workspace/02-make-dataset/x_val.npz"
Y_TRAIN = "workspace/02-make-dataset/y_train.npz"
Y_VAL = "workspace/02-make-dataset/y_val.npz"

OUTDIR = "workspace/04-learning"
EACH_OUDIR_KEY = "try"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
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

    # training data
    x_train = np.load(X_TRAIN)['x']
    t_train = np.load(Y_TRAIN)['y']
    x_val = np.load(X_VAL)['x']
    t_val = np.load(Y_VAL)['y']

    input_dim = x_train.shape[1]

    # model
    model = DNN.model2(input_dim=input_dim, learning_rate=args.lr)

    # learning
    hist = model.fit(x_train, t_train,
                     epochs=args.epochs,
                     batch_size=args.batch,
                     verbose=2,
                     validation_data=(x_val, t_val))
    
    loss_history = np.array([hist.history['loss'], hist.history['val_loss']]).transpose(1,0)
    np.savetxt(history_path, loss_history, delimiter=",", header="train loss,val_loss")

    model.save_weights(os.path.join(weights_dir, 'weights'))


def decide_outdir():
    for i in range(100):
        if f'{EACH_OUDIR_KEY}{i:0=3}' in os.listdir(OUTDIR):
            continue
        return os.path.join(OUTDIR, f'{EACH_OUDIR_KEY}{i:0=3}')


def save_options(args, fp):
    with open(fp, mode='w') as f:
        f.write(f'Number of epochs:\t{args.epochs}'
                f'\nlr:\t{args.lr}'
                f'\nbatch:\t{args.batch}')


if __name__ == '__main__':
    main()
