#!/usr/bin/env python

import os
import numpy as np

INPUTDIR = 'workspace/01-preprocess'
OUTDIR = 'workspace/02-make-dataset'
CUTOFF_RADIUS = 1.0
N_ATOMS = 309
DTYPE = 'float64'


def main():    
    x_descriptor, x_atomindex, y = [], [], []
    for i in range(N_ATOMS):
        npz = np.load(os.path.join(OUTDIR, f'trj{i:0=3}.npz'))
        x_descriptor.extend(npz['x'])
        y.extend(npz['y'])
        del npz


    x_descriptor = zero_padding_array(x_descriptor)
    y = np.array(y, dtype=DTYPE)

    # x_atomindex
    n_frames = x_descriptor.shape[0] // N_ATOMS
    x_atomindex = np.array([[i]*n_frames for i in range(N_ATOMS)]).ravel()
    x_atomindex = np.identity(N_ATOMS)[x_atomindex]  # one-hot

    # normalize
    x_descriptor = x_descriptor.reshape(-1,4)
    x_descriptor = (x_descriptor - np.mean(x_descriptor,axis=0)) / np.std(x_descriptor,axis=0)
    x_descriptor = x_descriptor.reshape(y.shape[0], -1)

    y = (y - np.mean(y.reshape(-1),axis=0)) / np.std(y.reshape(-1),axis=0)

    # join x
    x = np.concatenate([x_descriptor, x_atomindex], axis=1)

    # save
    print(f'x: {x.shape}\ny: {y.shape}')
    outpath = os.path.join(OUTDIR, 'dataset.npz')
    np.savez(outpath, x=x, y=y)



def zero_padding_array(x:list):
    maxlen = max([len(li) for li in x])

    x = [np.pad(arr, [(0,maxlen-arr.shape[0]), (0,0)], 'constant') for arr in x]
    x = np.array(x, dtype=DTYPE)

    return x



if __name__=='__main__':
    main()