#!/usr/bin/env python

OUTDIR = 'workspace/01-make-dataset'
CUTOFF_RADIUS = 1.0

import argparse
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from utils import DiscriptorGenerator
from utils import MyProcess


def main():
    parser = argparse.ArgumentParser(description='This script parse xvg-files and output npz-file')
    parser.add_argument('-c', '--coord', default='input/coord.xvg', help='xvg file path describing coordinates of trajectory')
    parser.add_argument('-f', '--force', default='input/force.xvg', help='xvg file path describing forces of trajectory')
    parser.add_argument('-i', required=True, type=int, help='C-alpha index')
    parser.add_argument('-w', default=4, type=int, help='max wokers of multi-process')
    args = parser.parse_args()


    os.makedirs(OUTDIR, exist_ok=True)

    # read xvg-files
    coords = read_xvg(args.coord)
    forces = read_xvg(args.force)


    # check shape
    if coords.shape != forces.shape:
        print("shapes of coord and force files must match")
        sys.exit()
    

    ### discriptor_generater ###
    discriptor_generator = DiscriptorGenerator(coords, args.i, CUTOFF_RADIUS)


    ### parallel process ###
    myprocess = MyProcess(discriptor_generator, coords.shape[0])
    with ProcessPoolExecutor(max_workers=args.w) as executor:
        futures = []
        for step in range(coords.shape[0]):
            futures.append(executor.submit(myprocess, step, coords[step], args.i, forces[step, args.i]))

    results = [f.result() for f in futures]


    ### x (coords) ###
    x = [d for d,_ in results]
    x = zero_padding_array(x)


    ### y (forces) ###
    y = np.array([f for _,f in results])


    ### normalize ###
    x = x.reshape(-1,4)
    x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
    x = x.reshape(y.shape[0], -1)

    y = (y - np.mean(y.reshape(-1),axis=0)) / np.std(y.reshape(-1),axis=0)


    # save
    print(f'\nx: {x.shape}\ny: {y.shape}')
    outpath = os.path.join(OUTDIR, f'trj{args.i:0=3}.npz')
    np.savez(outpath, x=x, y=y)





def read_xvg(filepath:str) -> np.ndarray:
    trj = np.loadtxt(filepath, comments=['#', '@'], delimiter='\t')[:, 1:]

    trj = trj.reshape(trj.shape[0], -1, 3)
    
    return trj



def zero_padding_array(x:list):
    maxlen = max([len(li) for li in x])

    x = np.array([np.pad(arr, [(0,maxlen-arr.shape[0]), (0,0)], 'constant') 
           if arr.shape[0] != 0 
           else np.array([0,0,0,0]*maxlen) 
           for arr in x
    ])

    return x



if __name__=='__main__':
    main()