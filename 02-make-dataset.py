#!/usr/bin/env python

import os
import numpy as np
import argparse
from utils import zero_padding_array


INPUTDIR = 'workspace/01-preprocess'
OUTDIR = 'workspace/02-make-dataset'
CUTOFF_RADIUS = 1.0
N_ATOMS = 309
DTYPE = 'float64'


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    train_x_descriptor, train_x_atomindex, train_y = main_process('training')

    val_x_descriptor, val_x_atomindex, val_y = main_process('validation')

    # zero padding
    maxlen = max([len(li) for li in train_x_descriptor] + [len(li) for li in val_x_descriptor])
    train_x_descriptor = zero_padding_array(train_x_descriptor, maxlen=maxlen)
    val_x_descriptor = zero_padding_array(val_x_descriptor, maxlen=maxlen)

    # normalize x
    max_reciprocal_radius = max(
        max(train_x_descriptor.reshape(-1, 4)[:, 0]),
        max(val_x_descriptor.reshape(-1, 4)[:, 0]))

    train_x_descriptor = train_x_descriptor / np.array([max_reciprocal_radius, 1, 1, 1])
    val_x_descriptor = val_x_descriptor / np.array([max_reciprocal_radius, 1, 1, 1])

    # normalize y
    y_concat = np.concatenate([train_y, val_y], axis=0).reshape(-1)
    y_mean = np.mean(y_concat, axis=0)
    y_std = np.std(y_concat, axis=0)
    del y_concat

    train_y = (train_y - y_mean) / y_std
    val_y = (val_y - y_mean) / y_std

    # join x
    train_x_descriptor = train_x_descriptor.reshape(train_y.shape[0], -1)
    train_x = np.concatenate([train_x_descriptor, train_x_atomindex], axis=1)
    del train_x_descriptor

    val_x_descriptor = val_x_descriptor.reshape(val_y.shape[0], -1)
    val_x = np.concatenate([val_x_descriptor, val_x_atomindex], axis=1)
    del val_x_descriptor


    # save
    print(f'train_x: {train_x.shape}\ntrain_y: {train_y.shape}')
    np.savez(os.path.join(OUTDIR, 'training.npz'), x=train_x, y=train_y)

    print(f'val_x: {val_x.shape}\nval_y: {val_y.shape}')
    np.savez(os.path.join(OUTDIR, 'validation.npz'), x=val_x, y=val_y)



def main_process(name:str):
    inputdir = os.path.join(INPUTDIR, name)
    print(f'Processing {name} data')

    x_descriptor, x_atomindex, y = [], [], []
    for i in range(N_ATOMS):
        npz = np.load(os.path.join(inputdir, f'trj{i:0=3}.npz'))
        x_descriptor.extend(npz['x'])
        y.extend(npz['y'])
        del npz

    y = np.array(y, dtype=DTYPE)

    # x_atomindex
    n_frames = y.shape[0] // N_ATOMS
    x_atomindex = np.array([[i]*n_frames for i in range(N_ATOMS)]).ravel()
    x_atomindex = np.identity(N_ATOMS)[x_atomindex]  # one-hot

    return x_descriptor, x_atomindex, y


if __name__=='__main__':
    main()