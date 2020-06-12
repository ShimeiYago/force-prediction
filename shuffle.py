#!/usr/bin/env python

import os
import argparse
import h5py
import dask.array as da

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

    with h5py.File(args.input, mode='r+') as f:
        # prepare data
        X_train = da.from_array(f[f'/{TRAIN_NAME}/{EXPLANATORY_NAME}'])
        Y_train = da.from_array(f[f'/{TRAIN_NAME}/{RESPONSE_NAME}'])

        ramdom_order = da.random.permutation(X_train.shape[0])

        X_train = X_train[ramdom_order]
        Y_train = Y_train[ramdom_order]

        da.to_hdf5(args.input, f'/{TRAIN_NAME}/{EXPLANATORY_NAME}', X_train)
        da.to_hdf5(args.input, f'/{TRAIN_NAME}/{RESPONSE_NAME}', Y_train)


if __name__ == '__main__':
    main()