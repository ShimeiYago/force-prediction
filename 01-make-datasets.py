#!/usr/bin/env python

import argparse
import os
import numpy as np

from utils01 import ReadXVGs
from utils01 import make_amino_dict
from utils01 import Nearest2indeces
from utils01 import DiscriptorGenerator


OUTDIR = 'workspace/01-make-datasets'
CUTOFF_RADIUS = 1.0
TRAIN_SIZE = 0.75  # used if validation data is specified


def main():
    parser = argparse.ArgumentParser(description='This script create datasets for deep learning.')
    parser.add_argument('-i', '--inputs', action='append', nargs=2, metavar=('coord','force'), required=True, help='two xvg files')
    parser.add_argument('-v', '--inputs_val', action='append', nargs=2, metavar=('coord','force'), 
                        help='if you prepare validation data aside from inputted files, specify the two files')
    parser.add_argument('--maxlen', type=int, help='max length of trajectory to use')
    parser.add_argument('-o',
                        default=os.path.join(OUTDIR, 'datasets.hdf5'), type=str,
                        help='output file name (.hdf5 is recommended)')
    parser.add_argument('--init_time', default=0, type=int, help='initial time to use')

    parser.add_argument('--gro', type=str, help='specify .gro path if want to include amino infomation into datasets')
    parser.add_argument('--no_atom_index', action="store_true", help='do not include atom index infomation into datasets')
    parser.add_argument('--no_relative_distance', action="store_true", help='do not include relative residue distance into datasets')

    parser.add_argument('-b', '--batch', type=int,
                        help='batchsize for one process (recommend: the Number of Frames*Atoms, divided by any natural number)')
    args = parser.parse_args()

    # ## read data ## #
    readxvgs = ReadXVGs(args.init_time, args.maxlen)
    # train data
    train_coords, train_forces = readxvgs(args.inputs)
    # val data
    if args.inputs_val:
        val_coords, val_forces = readxvgs(args.inputs_val)
    else:
        train_len = int(train_coords.shape[0] * TRAIN_SIZE)
        val_coords, val_forces = train_coords[train_len:], train_forces[train_len:]
        train_coords, train_forces = train_coords[:train_len], train_forces[:train_len]
    # print inputted data shape
    print('--- Read files ---\ntraining data:',
          train_coords.shape, '\nvalidation data:', val_coords.shape)
    # reshape
    train_forces = train_forces.reshape(-1, 3)
    val_forces = val_forces.reshape(-1, 3)

    # ## make residue-symbol dict ## #
    Index2ID = None
    if args.gro:
        Index2ID = make_amino_dict(args.gro)

    # ## define constants ## #
    N_ATOMS = train_coords.shape[1]

    # ## nearrest 2index ## #
    choose_nearest_2indeces = Nearest2indeces(train_coords[0].compute())
    ab_indeces = np.array([choose_nearest_2indeces(i) for i in np.arange(N_ATOMS)])

    # ## make descriptor ## #
    os.makedirs(OUTDIR, exist_ok=True)

    discriptor_generator = DiscriptorGenerator(
        (train_coords, train_forces, val_coords, val_forces),
        N_ATOMS, CUTOFF_RADIUS, args.o, args.batch, ab_indeces,
        Index2ID, args.no_atom_index, args.no_relative_distance)

    discriptor_generator()


if __name__ == '__main__':
    main()
