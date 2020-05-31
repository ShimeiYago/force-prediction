#!/usr/bin/env python

import numpy as np
import os
import argparse
import models.DNN as DNN


X_TRAIN = "workspace/02-make-dataset/x_train.npz"
X_VAL = "workspace/02-make-dataset/x_val.npz"
Y_TRAIN = "workspace/02-make-dataset/y_train.npz"
Y_VAL = "workspace/02-make-dataset/y_val.npz"

OUTDIR = "workspace/04-learning"
EACH_OUDIR_KEY = "try"

N_ATOMS = 309

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--model', type=int, default=1, help='model number')
    parser.add_argument('--mode', type=int, default=0, help='Methods to devide test-validaiton. (0:run1 and run2. 1:before and after)')
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

    # dataset
    x_train = np.load(X_TRAIN)['x']
    t_train = np.load(Y_TRAIN)['y']

    input_dim = x_train.shape[1]

    if args.mode == 0:
        x_val = np.load(X_VAL)['x']
        t_val = np.load(Y_VAL)['y']
    else:
        x_train = x_train.reshape(N_ATOMS, -1, input_dim).transpose(1,0,2)
        t_train = t_train.reshape(N_ATOMS, -1, 3).transpose(1,0,2)

        train_len = int(x_train.shape[0] * 0.8)

        x_train = x_train[:train_len].reshape(-1, input_dim)
        t_train = t_train[:train_len].reshape(-1, 3)
        x_val = x_train[train_len:].reshape(-1, input_dim)
        t_val = t_train[train_len:].reshape(-1, 3)

    # model
    if args.model == 1:
        model = DNN.model1(input_dim=input_dim, learning_rate=args.lr)
    elif args.model == 2:
        model = DNN.model2(input_dim=input_dim, learning_rate=args.lr)
    elif args.model == 3:
        model = DNN.model3(input_dim=input_dim, learning_rate=args.lr)

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
                f'\nbatch:\t{args.batch}'
                f'\nmodel number:\t{args.model}'
                f'\nmode:\t{args.mode}')


if __name__ == '__main__':
    main()
