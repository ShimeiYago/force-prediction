#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
import dask.array as da
import h5py

from utils01 import ReadXVGs
from utils01 import Nearest2indeces
from utils01 import DiscriptorGenerator


OUTDIR = 'workspace/01-make-datasets'
CUTOFF_RADIUS = 1.0
TRAIN_SIZE = 0.75  # used if validation data is specified

TRAIN_NAME = "training"
VAL_NAME = "validation"
EXPLANATORY_NAME = "x"
RESPONSE_NAME = "y"


def main():
    parser = argparse.ArgumentParser(description='This script create datasets for deep learning.')
    parser.add_argument('-i', '--inputs', action='append', nargs=2, metavar=('coord','force'), required=True, help='two xvg files')
    parser.add_argument('-v', '--inputs_val', action='append', nargs=2, metavar=('coord','force'), 
                        help='if you prepare validation data aside from inputted files, specify the two files')
    parser.add_argument('--init_time', default=0, type=int, help='initial time to use')
    parser.add_argument('--maxlen', type=int, help='max length of trajectory to use')

    parser.add_argument('-o', default=os.path.join(OUTDIR, 'datasets.hdf5'),
                        type=str, help='output file name (.hdf5 is recommended)')
    parser.add_argument('-f', action="store_true",
                        help="force to process whether the output file exist")

    parser.add_argument('--gro', type=str, help='specify .gro path if want to include amino infomation into datasets')

    parser.add_argument('-b', '--batch', type=int,
                        help='batchsize for one process (recommend: the Number of Frames*Atoms, divided by any natural number)')

    parser.add_argument('--atom', type=str,
                        help='designate atom species name if process only one ("CA", "N", "C", etc.)')

    args = parser.parse_args()

    # ## check output file existing ## #
    if not args.f:
        check_output(args.o)

    # ## read data ## #
    print('--- Reading files ---')
    readxvgs = ReadXVGs(args.init_time, args.maxlen, args.gro)
    train_dict = readxvgs(args.inputs)

    # print inputted data shape
    for atomname, trj in train_dict.items():
        print('{}: {}'.format(atomname, trj[0].shape))

    # val data
    if args.inputs_val:
        val_dict = readxvgs(args.inputs_val)
    else:
        train_dict, val_dict = train_val_split(train_dict)

    # reshape only force
    train_dict = {atom: [trj[0], trj[1].reshape(-1, 3)] for atom, trj in train_dict.items()}
    val_dict = {atom: [trj[0], trj[1].reshape(-1, 3)] for atom, trj in val_dict.items()}

    # ## nearrest 2index ## #
    choose_nearest_2indeces = Nearest2indeces(train_dict)
    ab_indeces = {atom: choose_nearest_2indeces(atom) for atom in train_dict.keys()}


    # ## make descriptor ## #
    os.makedirs(OUTDIR, exist_ok=True)

    # if outfile do not exists, create file
    if not os.path.isfile(args.o) or args.f:
        with h5py.File(args.o, mode='w'):
            pass

    # existing datasets
    with h5py.File(args.o, mode='r') as f:
        exising_datasets_list = list(f.keys())

    # instance
    discriptor_generator = DiscriptorGenerator(
        train_dict, val_dict,
        CUTOFF_RADIUS, args.o, args.batch, ab_indeces,
        TRAIN_NAME, VAL_NAME, EXPLANATORY_NAME, RESPONSE_NAME)

    if args.atom:
        atomlist = [args.atom]
    else:
        atomlist = train_dict.keys()
    for atom in atomlist:
        if atom in exising_datasets_list:
            print(f'--- datasets {atom} is alleady existing ---')
            continue

        print(f'--- Creating datasets {atom} ---')
        discriptor_generator(atom)


    # ## shuffle ## #
    print('--- Shuffling Training datasets ---')
    shuffle_traindata(args.o)


def check_output(outpath):
    if os.path.isfile(outpath):
        print(f'Error: "{outpath}" already existing! If you force to do, use the -f option.')
        sys.exit(1)


def train_val_split(datadict):
    train_len = int(datadict[list(datadict.keys())[0]][0].shape[0] * TRAIN_SIZE)

    train_dict, val_dict = {}, {}
    for atomname, trj in datadict.items():
        train_dict[atomname] = [trj[0][:train_len], trj[1][:train_len]]
        val_dict[atomname] = [trj[0][train_len:], trj[1][train_len:]]

    return train_dict, val_dict


def shuffle_traindata(datapath):
    with h5py.File(datapath, mode='r+') as f:
        for atom in f.keys():
            X_train = da.from_array(f[f'/{atom}/{TRAIN_NAME}/{EXPLANATORY_NAME}'])
            Y_train = da.from_array(f[f'/{atom}/{TRAIN_NAME}/{RESPONSE_NAME}'])

            random_order = np.random.permutation(X_train.shape[0])

            X_train = da.slicing.shuffle_slice(X_train, random_order)
            Y_train = da.slicing.shuffle_slice(Y_train, random_order)

            da.to_hdf5(datapath, f'/{atom}/{TRAIN_NAME}/{EXPLANATORY_NAME}', X_train)
            da.to_hdf5(datapath, f'/{atom}/{TRAIN_NAME}/{RESPONSE_NAME}', Y_train)

            print(f'{atom} shuffled')


if __name__ == '__main__':
    main()
