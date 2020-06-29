#!/usr/bin/env python

import argparse
import os
import h5py
import numpy as np

from utils01 import ReadXVGs
from utils01 import GROParser
from utils01 import DiscriptorGenerator

from utils_keras import DNN

os.environ['KMP_DUPLICATE_LIB_OK']='True'


DATASETDIR = "workspace/01-make-datasets"
CUTOFF_RADIUS = 1.0
MASS = {'CA': 12.0107, 'C': 12.0107, 'O': 15.999, 'N': 14.0067}
DT = 0.002
OUTPATH = "workspace/04-simulate/trj.npy"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coords', type=str, help='coord file (.xvg)')
    parser.add_argument('--init_time', type=int, default=0, help='init time to start simulation')
    parser.add_argument('--gro', type=str, help='specify .gro path if want to include amino infomation into datasets')
    parser.add_argument('--dataset', type=str,
                        default=os.path.join(DATASETDIR, 'datasets.hdf5'),
                        help='input datasets')

    parser.add_argument('--model', type=int, default=1, help='model number')
    parser.add_argument('--weights', type=str, nargs=4, required=True, help='model weights (N, CA, C, O)')

    parser.add_argument('--len', type=int, default=5000, help='simulation length')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)

    # init strcuct
    init_structs = ReadXVGs(None, None)._read_xvg(args.coords)[args.init_time:args.init_time+2].compute()

    # ## load gro file ## #
    groparser = GROParser(args.gro)
    ATOM_ALIGN = groparser.atom_align
    MAINCHAIN = groparser.mainchains
    N_ATOMS = groparser.n_atoms
    EACH_N_ATOMS = groparser.each_n_atoms
    EACHATOM_INDECES = groparser.eachatom_indeces
    ADJACENT_INDECES, AB_INDECES, MAX_N_ADJACENT = groparser.cal_adjacent(CUTOFF_RADIUS)


    # ## arrange ## #
    arranged_indeces = []
    for atom in MAINCHAIN:
        arranged_indeces.extend(EACHATOM_INDECES[atom])

    init_structs = init_structs[:, arranged_indeces, :]

    i = 0
    ATOM_ALIGN = []
    slice_indeces = {}
    for atom in MAINCHAIN:
        j = i+len(EACHATOM_INDECES[atom])
        slice_indeces[atom] = [i, j]
        ATOM_ALIGN = ATOM_ALIGN + [atom] * (j-i)
        i = j

    EACHATOM_INDECES = {atom: list(range(i, j)) for atom, [i, j] in slice_indeces.items()}

    index_convert_dict = {orig_index: new_index for new_index, orig_index in enumerate(arranged_indeces)}

    for i in range(len(ADJACENT_INDECES)):
        for j in range(len(ADJACENT_INDECES[i])):
            for k in range(len(ADJACENT_INDECES[i][j])):
                ADJACENT_INDECES[i][j][k] = index_convert_dict[ADJACENT_INDECES[i][j][k]]
        
    for i in range(len(AB_INDECES)):
        for j in range(len(AB_INDECES[i])):
            AB_INDECES[i][j] = index_convert_dict[AB_INDECES[i][j]]

    temp_ADJACENT_INDECES = ADJACENT_INDECES
    temp_AB_INDECES = AB_INDECES
    for i, j in enumerate(arranged_indeces):
        ADJACENT_INDECES[i] = temp_ADJACENT_INDECES[j]
        AB_INDECES[i] = temp_AB_INDECES[j]


    # ## discriptor generator ## #
    discriptor_generator = DiscriptorGenerator(
        None, None,
        MAINCHAIN, ATOM_ALIGN, N_ATOMS, EACH_N_ATOMS, EACHATOM_INDECES,
        ADJACENT_INDECES, AB_INDECES, MAX_N_ADJACENT,
        None, None)

    # ## read models ## #
    dnn = DNN(discriptor_generator.INPUTDIM, None)
    Nmodel = dnn(args.model)
    Nmodel .load_weights(args.weights[0])
    CAmodel = dnn(args.model)
    CAmodel.load_weights(args.weights[1])
    Cmodel = dnn(args.model)
    Cmodel.load_weights(args.weights[2])
    Omodel = dnn(args.model)
    Omodel.load_weights(args.weights[3])
    model = {'N': Nmodel, 'CA': CAmodel, 'C': Cmodel, 'O':Omodel}


    # ## normalization values ## #
    with h5py.File("workspace/01-make-datasets/datasets.hdf5", mode='r') as f:
        for atom in MAINCHAIN:
            y_mean, y_std = f[f'/{atom}/normalization'][...]
        

    # ## cal force ## #
    def cal_force(discriptors):
        discriptors = np.tile(discriptors, (N_ATOMS, 1)).reshape(N_ATOMS, -1, 3)
        discriptors = discriptors - discriptors.transpose(1, 0, 2)

        discriptor, rot_matrices = discriptor_generator._descriptor(discriptors)

        forces = []
        for atom in MAINCHAIN:
            i, j = slice_indeces[atom]
            forces.append( model[atom].predict(discriptor[i:j]) )
        forces = np.concatenate(forces)
        
        # rotate
        forces = np.array([np.dot(force, np.linalg.inv(rot_matrix)) for force, rot_matrix in zip(forces, rot_matrices)])
        
        # expand scale
        forces = np.add(np.multiply(forces, y_std), y_mean)
        
        return forces
    

    # ## leap frog ## #
    weights = np.array([MASS[atom] for atom in ATOM_ALIGN]).reshape(-1, 1)

    def leap_frog(struct1, struct2):
        return np.subtract(2*struct2, struct1) + np.divide(cal_force(struct2), weights) * (DT**2)

    
    # ## simulate ## #
    trj = np.zeros((args.len, N_ATOMS, 3))
    trj[0:2] = init_structs

    for t in range(2, args.len):
        trj[t] = leap_frog(trj[t-2], trj[t-1])

    np.save(OUTPATH, trj)

if __name__ == '__main__':
    main()
