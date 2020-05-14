#!/usr/bin/env python

OUTDIR = 'workspace/01-make-dataset/allatom'
CUTOFF_RADIUS = 1.0
N_ATOMS = 309
EXCEPTION_INDEX = [0, 308]
DTYPE = 'float32'


import os
import sys
import numpy as np


def main():    
    x, y = [], []
    for i in range(N_ATOMS):
        if i in EXCEPTION_INDEX:
            continue

        npz = np.load(os.path.join(OUTDIR, f'trj{i:0=3}.npz'))
        x.extend(npz['x'])
        y.extend(npz['y'])
        del npz


    x = zero_padding_array(x)
    y = np.array(y, dtype=DTYPE)


    ### normalize ###
    x = x.reshape(-1,4)
    x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
    x = x.reshape(y.shape[0], -1)

    y = (y - np.mean(y.reshape(-1),axis=0)) / np.std(y.reshape(-1),axis=0)


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