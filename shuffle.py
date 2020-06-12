#!/usr/bin/env python

import os
import argparse
import h5py
import dask.array as da
import numpy as np
import shutil

INPUTDIR = "workspace/01-make-datasets"

TRAIN_NAME = "training"
EXPLANATORY_NAME = "x"
RESPONSE_NAME = "y"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default=os.path.join(INPUTDIR, 'datasets.hdf5'),
                        help='input datasets')
    args = parser.parse_args()

    outpath = os.path.splitext(args.input)[0] + '-shuffled.hdf5'
    shutil.copy(args.input, outpath)

    with h5py.File(outpath, mode='r+') as f:
        X_train = da.from_array(f[f'/{TRAIN_NAME}/{EXPLANATORY_NAME}'])
        Y_train = da.from_array(f[f'/{TRAIN_NAME}/{RESPONSE_NAME}'])

        random_order = np.random.permutation(X_train.shape[0])

        X_train = da.slicing.shuffle_slice(X_train, random_order)
        Y_train = da.slicing.shuffle_slice(Y_train, random_order)

        da.to_hdf5(outpath, f'/{TRAIN_NAME}/{EXPLANATORY_NAME}', X_train)
        da.to_hdf5(outpath, f'/{TRAIN_NAME}/{RESPONSE_NAME}', Y_train)


if __name__ == '__main__':
    main()