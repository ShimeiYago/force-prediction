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

SAVE_DISTANCE = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coord', type=str, help='coord file (.xvg)')
    parser.add_argument('--init_time', type=int, default=0, help='init time to start simulation')
    parser.add_argument('--gro', type=str, help='specify .gro path if want to include amino infomation into datasets')
    parser.add_argument('--dataset', type=str,
                        default=os.path.join(DATASETDIR, 'datasets.hdf5'),
                        help='input datasets')

    parser.add_argument('--model', type=int, default=1, help='model number')
    parser.add_argument('--weights', type=str, action='append', required=True, help='model weights (N, CA, CB, C, O)')

    parser.add_argument('--len', type=int, default=500, help='outputed trj length')
    parser.add_argument('--save_distance', type=int, default=1, help='save each steps')
    parser.add_argument('-o', type=str, default="trj", help='output name')
    parser.add_argument('-k', type=float, default=0, help='spring constant')
    parser.add_argument('--scaling', type=int, action='append', nargs=2, metavar=('lower','upper'), help='scaling group range')
    parser.add_argument('--cb', action="store_true", help='mainchain + CB mode')
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    # ## load gro file ## #
    groparser = GROParser(args.gro, CUTOFF_RADIUS, args.cb)
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
    REARRANGED_INDECES = groparser.rearranged_indeces
    RESID_GROUP_INDECES = groparser.resid_group_indeces
    TARGET_ATOM_INDECES_FOR_XVG = groparser.target_atom_indeces_for_xvg

    # ## init strcuct ## #
    init_structs = ReadXVGs(TARGET_ATOM_INDECES_FOR_XVG, ARRANGED_INDECES)._read_xvg(args.coord)[args.init_time:args.init_time+2].compute()
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
                model.load_weights(fp)
                models[atom] = model

    # ## normalization values ## #
    normalization = {}
    with h5py.File(args.dataset, mode='r') as f:
        for atom in MAINCHAIN:
            y_mean, y_std = f[f'/{atom}/normalization'][...]
            normalization[atom] = [y_mean, y_std]

    # scaling group
    if args.scaling:
        scaling_group = [list(range(l, u)) for l, u in args.scaling]
    else:
        scaling_group = []

    # resid group indeces
    group_indeces = []
    for resid_list in scaling_group:
        indeces = sum([RESID_GROUP_INDECES[resid] for resid in resid_list], [])
        group_indeces.append(indeces)

    # ## simulate ## #
    leapfrog = LeapFrog(discriptor_generator, models, normalization, args.k,
                        N_ATOMS, MAINCHAIN, SLICE_INDECES, ATOM_ALIGN,
                        group_indeces,
                        CONNECT_INDECES, INIT_RADIUSES, 
                        discriptor_generator.INPUTDIMS_ONLY_DESCRIPTOR, EACH_N_ATOMS,
                        init_structs)

    trj = []
    t = 0
    pre_struct = init_structs[0]
    current_struct = init_structs[1]

    for i in range(1, args.len*args.save_distance):
        next_struct = leapfrog(pre_struct, current_struct)

        if (i+1)%args.save_distance == 0:
            t += 1
            trj.append(next_struct)
            print('\r', t, '/', args.len, end="")

        pre_struct = current_struct
        current_struct = next_struct
    print()

    trj = np.array(trj)[:, REARRANGED_INDECES, :]
    print(trj.shape)

    # ## output ## #
    outnpy = os.path.join(OUTDIR, f"{args.o}.npy")
    np.save(outnpy, trj)

    outamber = os.path.join(OUTDIR, f"{args.o}.amber")
    np.savetxt(outamber, np.multiply(trj, 10).reshape(-1, 3), delimiter=' ', header='header')


if __name__ == '__main__':
    main()
