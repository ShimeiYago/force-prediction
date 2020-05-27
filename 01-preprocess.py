#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from utils import DiscriptorGenerator
from utils import MyProcess
from utils import ReadXVGs
from utils import zero_padding_array

OUTDIR = 'workspace/01-preprocess'
CUTOFF_RADIUS = 1.0
DTYPE = 'float64'


def main():
    parser = argparse.ArgumentParser(description='This script preprocess for deep learning. Parse xvg-files and output npz-file.')
    parser.add_argument('-i', '--inputs', action='append', nargs=2, metavar=('coord','force'), required=True, help='two xvg files')
    parser.add_argument('-v', '--val', action='store_true', default=False, help='process validation data (default is training)')
    parser.add_argument('--init_time', default=0, type=int, help='initial time to use')
    parser.add_argument('-s', default=0, type=int, help='start index of atom')
    parser.add_argument('-w', default=4, type=int, help='max wokers of multi-process')
    args = parser.parse_args()

    name = 'training'
    if args.val:
        name = 'validation'

    # read data
    read_xvgs = ReadXVGs(args.init_time, DTYPE)
    coords, forces = read_xvgs(args.inputs)

    print(f'Processing {name} data.\nCoord:{coords.shape} Force:{forces.shape}\n')

    # process
    outdir = os.path.join(OUTDIR, name)
    main_process(args.s, coords, forces, outdir, args.w)


def main_process(start_index, coords, forces, outdir, max_wokers):
    for i in range(start_index, coords.shape[1]):
        # discriptor_generater
        discriptor_generator = DiscriptorGenerator(coords, i, CUTOFF_RADIUS)

        # parallel process
        myprocess = MyProcess(discriptor_generator, coords.shape[0], i)
        with ProcessPoolExecutor(max_workers=max_wokers) as executor:
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
        outpath = os.path.join(outdir, f'trj{i:0=3}.npz')
        os.makedirs(outdir, exist_ok=True)
        np.savez(outpath, x=x, y=y)


if __name__=='__main__':
    main()