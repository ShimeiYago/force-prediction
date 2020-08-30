#!/usr/bin/env python

import argparse
import os
import h5py
import numpy as np

from utils01 import ReadXVGs
from utils01 import GROParser
from utils01 import DiscriptorGenerator
from utils04 import LeapFrog

from utils_keras import DNN


DATASETDIR = "workspace/01-make-datasets"
CUTOFF_RADIUS = 1.0
OUTDIR = "workspace/04-simulate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coord', type=str, help='coord file (.xvg)')
    parser.add_argument('--init_time', type=int, default=0, help='init time to start simulation')
    parser.add_argument('--gro', type=str, help='specify .gro path if want to include amino infomation into datasets')
    parser.add_argument('--dataset', type=str,
                        default=os.path.join(DATASETDIR, 'datasets.hdf5'),
                        help='input datasets')

    parser.add_argument('--model', type=int, default=1, help='model number')
    parser.add_argument('--weights', type=str, nargs=4, required=True, help='model weights (N, CA, C, O)')

    parser.add_argument('--len', type=int, default=5000, help='simulation length')
    parser.add_argument('-o', type=str, default="trj", help='output name')
    parser.add_argument('-k', type=float, default=1, help='spring constant')
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    # ## load gro file ## #
    groparser = GROParser(args.gro, CUTOFF_RADIUS)
    MAINCHAIN = groparser.mainchains
    N_ATOMS = groparser.n_atoms
    EACH_N_ATOMS = groparser.each_n_atoms
    SLICE_INDECES = groparser.slice_indeces
    ARRANGED_INDECES = groparser.arranged_indeces
    ADJACENT_INDECES = groparser.adjacent_indeces
    AB_INDECES = groparser.ab_indeces
    ATOM_ALIGN = groparser.atom_align
    CONNECT_INDECES = groparser.connects_indeces
    INIT_RADIUSES = groparser.init_radiuses

    # ## init strcuct ## #
    init_structs = ReadXVGs(None, None, ARRANGED_INDECES)._read_xvg(args.coord)[args.init_time:args.init_time+2].compute()
    init_structs = init_structs[:, ARRANGED_INDECES, :]

    # ## discriptor generator ## #
    discriptor_generator = DiscriptorGenerator(
        None, None,
        MAINCHAIN, N_ATOMS, EACH_N_ATOMS, SLICE_INDECES,
        ADJACENT_INDECES, AB_INDECES, ATOM_ALIGN,
        None, None)

    # ## read models ## #
    inputdims = discriptor_generator.INPUTDIMS
    models = {}
    for fp in args.weights:
        for atom in MAINCHAIN:
            if f'/{atom}/' in fp:
                dnn = DNN(inputdims[atom], None)
                model = dnn(args.model)
                dnn = DNN(inputdims[atom], None)
                model = dnn(args.model)
                model.load_weights(fp)
                models[atom] = model

    # ## normalization values ## #
    normalization = {}
    with h5py.File(args.dataset, mode='r') as f:
        for atom in MAINCHAIN:
            y_mean, y_std = f[f'/{atom}/normalization'][...]
            normalization[atom] = [y_mean, y_std]


    # ## simulate ## #
    leapfrog = LeapFrog(discriptor_generator, models, normalization, args.k,
                        N_ATOMS, MAINCHAIN, SLICE_INDECES, ATOM_ALIGN,
                        CONNECT_INDECES, INIT_RADIUSES, inputdims,
                        init_structs)

    trj = np.zeros((args.len, N_ATOMS, 3))
    trj[0:2] = init_structs

    for t in range(1, args.len-1):
        trj[t+1] = leapfrog(trj[t-1], trj[t])
        print('\r', t+2, '/', args.len, end="")
    print()

    trj = trj[:, REARRANGED_INDECES, :]


    # ## output ## #
    outnpy = os.path.join(OUTDIR, f"{args.o}.npy")
    np.save(outnpy, trj)

    outamber = os.path.join(OUTDIR, f"{args.o}.amber")
    np.savetxt(outamber, np.multiply(trj, 10).reshape(-1, 3), delimiter=' ', header='header')


if __name__ == '__main__':
    main()
