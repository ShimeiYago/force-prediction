#!/usr/bin/env python

import os
import numpy as np
import argparse
from utils import zero_padding_array


INPUTDIR = 'workspace/01-preprocess'
OUTDIR = 'workspace/02-make-dataset'
CUTOFF_RADIUS = 1.0
N_ATOMS = 309
DTYPE = 'float32'


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_lower', type=int, default=None, help='')
    parser.add_argument('--train_upper', type=int, default=None, help='')
    parser.add_argument('--val_lower', type=int, default=None, help='')
    parser.add_argument('--val_upper', type=int, default=None, help='')
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    # ## load x descriptor ## #
    # train
    train_x_descriptor, _, _ = main_process('training', cut=[args.train_lower, args.train_upper], atomindex_bool=False, y_bool=False)

    train_n_data = len(train_x_descriptor)
    maxlen = max([len(li) for li in train_x_descriptor])
    max_reciprocal_radius = max([max(x[:,0]) for x in train_x_descriptor])
    del train_x_descriptor

    # val
    val_x_descriptor, _, _ = main_process('validation', cut=[args.val_lower, args.val_upper], atomindex_bool=False, y_bool=False)

    val_n_data = len(val_x_descriptor)
    maxlen = max([maxlen] + [len(li) for li in val_x_descriptor])
    max_reciprocal_radius = max([max_reciprocal_radius] + [max(x[:,0]) for x in val_x_descriptor])
    del val_x_descriptor


    # ## save x_train ## #
    train_x_descriptor, train_x_atomindex, _ = main_process('training', cut=[args.train_lower, args.train_upper], y_bool=False)
    # zero padding
    train_x_descriptor = zero_padding_array(train_x_descriptor, maxlen=maxlen, dtype=DTYPE)
    # normalize x
    train_x_descriptor = train_x_descriptor / np.array([max_reciprocal_radius, 1, 1, 1])
    train_x_descriptor = train_x_descriptor.astype(DTYPE)
    # join x
    train_x_descriptor = train_x_descriptor.reshape(train_n_data, -1)
    train_x_descriptor = np.concatenate([train_x_descriptor, train_x_atomindex], axis=1)

    print(f'train_x: {train_x_descriptor.shape}')
    np.savez(os.path.join(OUTDIR, 'x_train.npz'), x=train_x_descriptor)
    del train_x_descriptor; del train_x_atomindex


    # ## save x_val ## #
    val_x_descriptor, val_x_atomindex, _ = main_process('validation', cut=[args.val_lower, args.val_upper], y_bool=False)
    # zero padding
    val_x_descriptor = zero_padding_array(val_x_descriptor, maxlen=maxlen, dtype=DTYPE)
    # normalize x
    val_x_descriptor = val_x_descriptor / np.array([max_reciprocal_radius, 1, 1, 1])
    val_x_descriptor = val_x_descriptor.astype(DTYPE)
    # join x
    val_x_descriptor = val_x_descriptor.reshape(val_n_data, -1)
    val_x_descriptor = np.concatenate([val_x_descriptor, val_x_atomindex], axis=1)

    print(f'val_x: {val_x_descriptor.shape}')
    np.savez(os.path.join(OUTDIR, 'x_val.npz'), x=val_x_descriptor)
    del val_x_descriptor; del val_x_atomindex


    # ## save y ## #
    _, _, train_y = main_process('training', cut=[args.train_lower, args.train_upper], descriptor_bool=False, atomindex_bool=False)
    _, _, val_y = main_process('validation', cut=[args.val_lower, args.val_upper], descriptor_bool=False, atomindex_bool=False) 

    # cal mean and std
    y_mean = np.mean(np.concatenate([train_y, val_y], axis=0).reshape(-1), axis=0)
    y_std = np.std(np.concatenate([train_y, val_y], axis=0).reshape(-1), axis=0)

    # normalize y
    train_y = (train_y - y_mean) / y_std
    val_y = (val_y - y_mean) / y_std

    print(f'train_y: {train_y.shape}')
    np.savez(os.path.join(OUTDIR, 'y_train.npz'), y=train_y)

    print(f'val_y: {val_y.shape}')
    np.savez(os.path.join(OUTDIR, 'y_val.npz'), y=val_y)


def main_process(name:str, cut, descriptor_bool=True, atomindex_bool=True, y_bool=True):
    inputdir = os.path.join(INPUTDIR, name)

    x_descriptor, x_atomindex, y = [], [], []
    for i in range(N_ATOMS):
        npz = np.load(os.path.join(inputdir, f'trj{i:0=3}.npz'))
        if descriptor_bool:
            x_descriptor.extend(npz['x'][cut[0]:cut[1]])
        if y_bool:
            y.extend(npz['y'][cut[0]:cut[1]])
        del npz

    if y_bool:
        y = np.array(y, dtype=DTYPE)

    # x_atomindex
    if atomindex_bool:
        n_frames = len(x_descriptor) // N_ATOMS
        x_atomindex = np.array([[i]*n_frames for i in range(N_ATOMS)]).ravel()
        x_atomindex = np.identity(N_ATOMS)[x_atomindex]  # one-hot
        x_atomindex = x_atomindex.astype(DTYPE)

    return x_descriptor, x_atomindex, y


if __name__=='__main__':
    main()