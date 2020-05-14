#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from utils import DiscriptorGenerator
from utils import MyProcess

OUTDIR = 'workspace/01-make-dataset/allatom'
CUTOFF_RADIUS = 1.0
DTYPE = 'float32'


def main():
    parser = argparse.ArgumentParser(description='This script parse xvg-files and output npz-file')
    parser.add_argument('-c', '--coord', default='input/coord.xvg', help='xvg file path describing coordinates of trajectory')
    parser.add_argument('-f', '--force', default='input/force.xvg', help='xvg file path describing forces of trajectory')
    parser.add_argument('-i', default=0, type=int, help='start index of atom')
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

    for i in range(args.i, coords.shape[1]):
        # discriptor_generater
        discriptor_generator = DiscriptorGenerator(coords, i, CUTOFF_RADIUS)

        # parallel process
        myprocess = MyProcess(discriptor_generator, coords.shape[0], i)
        with ProcessPoolExecutor(max_workers=args.w) as executor:
            futures = []
            for step in range(coords.shape[0]):
                futures.append(executor.submit(myprocess, step, coords[step], i, forces[step, i]))

        results = [f.result() for f in futures]

        # x (coords)
        x = [d for d,_ in results]
        x = zero_padding_array(x)

        # y (forces)
        y = np.array([f for _,f in results], dtype=DTYPE)

        print('')

        # save
        outpath = os.path.join(OUTDIR, f'trj{i:0=3}.npz')
        np.savez(outpath, x=x, y=y)


def read_xvg(filepath: str) -> np.ndarray:
    trj = np.loadtxt(filepath, comments=['#', '@'], delimiter='\t', dtype=DTYPE)[:, 1:]

    trj = trj.reshape(trj.shape[0], -1, 3)

    return trj


def zero_padding_array(x: list):
    maxlen = max([len(li) for li in x])

    x = np.array([np.pad(arr, [(0,maxlen-arr.shape[0]), (0,0)], 'constant') 
        if arr.shape[0] != 0 
        else [[0,0,0,0]]*maxlen
        for arr in x
    ], dtype=DTYPE)

    return x



if __name__=='__main__':
    main()