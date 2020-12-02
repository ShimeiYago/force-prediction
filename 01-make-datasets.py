#!/usr/bin/env python

import argparse
import os
import sys
import h5py

from utils01 import ReadXVGs
from utils01 import GROParser
from utils01 import DiscriptorGenerator


OUTDIR = 'workspace/01-make-datasets'
CUTOFF_RADIUS = 1.0
TRAIN_SIZE = 0.9  # used if validation data is specified

TRAIN_NAME = "training"
VAL_NAME = "validation"
EXPLANATORY_NAME = "x"
RESPONSE_NAME = "y"


def main():
    parser = argparse.ArgumentParser(description='This script create datasets for deep learning.')
    parser.add_argument('-i', '--inputs', action='append', nargs=4, metavar=('coord','force', 'init_time', 'maxlen'),
                        required=True, help='two xvg files, init_time, and maxlen')
    parser.add_argument('-v', '--inputs_val', action='append', nargs=4, metavar=('coord','force', 'init_time', 'maxlen'), 
                        help='if you prepare validation data aside from inputted files, specify the two files')

    parser.add_argument('-o', default=os.path.join(OUTDIR, 'datasets.hdf5'),
                        type=str, help='output file name (.hdf5 is recommended)')
    parser.add_argument('-f', action="store_true",
                        help="force to process whether the output file exist")

    parser.add_argument('--gro', type=str, help='specify .gro path if want to include amino infomation into datasets')

    parser.add_argument('-b', '--batch', type=int,
                        help='batchsize for one process (recommend: the Number of Frames, divided by any natural number)')
    parser.add_argument('--cb', action="store_true", help='mainchain + CB mode')
    parser.add_argument('--only_terminal_rate', type=float, default=0.0, help='to inclease both terminal datasets')
    parser.add_argument('--skip', type=int, default=1, help='read data each skip')
    args = parser.parse_args()

    # ## check output file existing ## #
    if not args.f:
        check_output(args.o)

    # ## parse gro file ## #
    print('--- Reading gro file ---')
    groparser = GROParser(args.gro, CUTOFF_RADIUS, args.cb)
    MAINCHAIN = groparser.mainchains
    N_ATOMS = groparser.n_atoms
    EACH_N_ATOMS = groparser.each_n_atoms
    SLICE_INDECES = groparser.slice_indeces
    ARRANGED_INDECES = groparser.arranged_indeces
    ADJACENT_INDECES = groparser.adjacent_indeces
    AB_INDECES = groparser.ab_indeces
    ATOM_ALIGN = groparser.atom_align
    TARGET_ATOM_INDECES_FOR_XVG = groparser.target_atom_indeces_for_xvg
    RESID_DICT = groparser.resid_dict

    # print target atoms
    print('Traget Atoms:', end="")
    for atom in MAINCHAIN:
        print(f' {atom}({EACH_N_ATOMS[atom]})', end=",")
    print()


    # ## read data ## #
    print('--- Reading trajectory ---')
    readxvgs = ReadXVGs(TARGET_ATOM_INDECES_FOR_XVG, ARRANGED_INDECES, args.skip)
    train_coords, train_forces = readxvgs(args.inputs)

    # val data
    if args.inputs_val:
        val_coords, val_forces = readxvgs(args.inputs_val)
    else:
        train_coords, train_forces, val_coords, val_forces = train_val_split(train_coords, train_forces)

    # print inputted data shape
    print(f'training: {train_coords.shape}\nvalidation: {val_coords.shape}')


    # define batchsize
    if args.batch:
        batchsize = args.batch
    else:
        batchsize = train_coords.shape[0] // 30


    # ## make descriptor ## #
    os.makedirs(OUTDIR, exist_ok=True)

    # if outfile do not exists, create file
    if not os.path.isfile(args.o) or args.f:
        with h5py.File(args.o, mode='w'):
            pass

    # instance
    discriptor_generator = DiscriptorGenerator(
        args.o, batchsize,
        MAINCHAIN, N_ATOMS, EACH_N_ATOMS, SLICE_INDECES,
        ADJACENT_INDECES, AB_INDECES, ATOM_ALIGN, RESID_DICT,
        EXPLANATORY_NAME, RESPONSE_NAME)

    # input dims
    print('--- Inout dimensions ---')
    for atom in MAINCHAIN:
        print("[{}] {} ({})".format(atom, discriptor_generator.INPUTDIMS[atom], discriptor_generator.INPUTDIMS_ONLY_DESCRIPTOR[atom]))

    # process train data
    print('--- Process Training data ---')
    discriptor_generator(train_coords, train_forces, TRAIN_NAME, args.only_terminal_rate)

    # process val data
    print('--- Process Validation data ---')
    discriptor_generator(val_coords, val_forces, VAL_NAME)

    # normalize y
    print('--- Normalize response values ---')
    discriptor_generator.normalize(TRAIN_NAME, VAL_NAME)

    # ## shuffle ## #
    print('--- Shuffling Training datasets ---')
    discriptor_generator.shuffle(TRAIN_NAME)


def check_output(outpath):
    if os.path.isfile(outpath):
        print(f'Error: "{outpath}" already existing! If you force to do, use the -f option.')
        sys.exit(1)


def train_val_split(coords, forces):
    train_len = int(coords.shape[0] * TRAIN_SIZE)

    train_coords, train_forces = coords[:train_len], forces[:train_len]
    val_coords, val_forces = coords[train_len:], forces[train_len:]

    return train_coords, train_forces, val_coords, val_forces


if __name__ == '__main__':
    main()
