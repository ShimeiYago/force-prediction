#!/usr/bin/env python

OUTDIR = 'workspace/01-make-dataset'
CUTOFF_RADIUS = 1.0

MAX_WOKERS = 4
# MAX_WOKERS = None

import argparse
import os
import sys
import time
import numpy as np
import concurrent.futures as confut


def main():
    parser = argparse.ArgumentParser(description='This script parse xvg-files and output npz-file')
    parser.add_argument('-c', '--coord', default='input/coord.xvg', help='xvg file path describing coordinates of trajectory')
    parser.add_argument('-f', '--force', default='input/force.xvg', help='xvg file path describing forces of trajectory')
    args = parser.parse_args()


    os.makedirs(OUTDIR, exist_ok=True)

    # read xvg-files
    coords = read_xvg(args.coord)
    forces = read_xvg(args.force)


    # check shape
    if coords.shape != forces.shape:
        print("shapes of coord and force files must match")
        sys.exit()
    

    ### parallel process ###
    myprocess = MyProcess(convert_descriptor, coords.shape[0])
    with confut.ProcessPoolExecutor(max_workers=MAX_WOKERS) as executor:
        futures = []
        for step in range(coords.shape[0]):
            for idx in range(coords.shape[1]):
                # if idx==0 or idx==coords.shape[1]-1:
                if idx!=1:
                    continue # pass if atom is edge one

                futures.append(executor.submit(myprocess, step, coords[step], idx))

    x = [f.result() for f in futures]
    x = zero_padding_array(x)

    # y = forces.reshape(-1,3)
    # y = forces[:, 1:-1, :].reshape(-1,3) # delete edge atoms
    y = forces[:, 1, :].reshape(-1,3)

    ### normalize ###
    x = x.reshape(-1,4)
    x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
    x = x.reshape(y.shape[0], -1)

    y = (y - np.mean(y.reshape(-1),axis=0)) / np.std(y.reshape(-1),axis=0)


    # save
    print(f'\nx: {x.shape}\ny: {y.shape}')
    outpath = os.path.join(OUTDIR, 'trj.npz')
    np.savez(outpath, x=x, y=y)





def read_xvg(filepath:str) -> np.ndarray:
    trj = np.loadtxt(filepath, comments=['#', '@'], delimiter='\t')[:, 1:]

    trj = trj.reshape(trj.shape[0], -1, 3)
    
    return trj



def convert_descriptor(struct:np.ndarray, i:int) -> np.ndarray:
    # shift
    Ri = struct - struct[i]

    # rotate
    a,b = choose_nearest_2indexes(struct, i)
    e1 = e_(Ri[a])
    e2 = e_(Ri[b] - np.dot(Ri[b],e1)*e1)
    e3 = np.cross(e1, e2)
    rotation_martix = np.array([e1,e2,e3]).T
    Ri = np.dot(Ri, rotation_martix)

    Ri = np.delete(Ri, obj=i, axis=0)

    # descriptor
    D = np.array([ \
        [1/radius(Ri[j]), Ri[j,0]/radius(Ri[j]), Ri[j,1]/radius(Ri[j]), Ri[j,2]/radius(Ri[j])] \
        for j in range(Ri.shape[0]) \
        if radius(Ri[j])<=CUTOFF_RADIUS \
        ])

    
    return D


def e_(R:np.ndarray) -> np.ndarray:
    return R/radius(R)


def choose_nearest_2indexes(struct:np.ndarray, i:int) -> np.ndarray:
    radiuslist = [radius(struct[i], struct[j]) for j in range(struct.shape[0])]
    return np.argsort(radiuslist)[1:3]


def radius(a:np.ndarray, b=np.array([0,0,0])):
    return np.sqrt(np.sum(np.square(a-b)))


def zero_padding_array(x:list):
    maxlen = max([len(li) for li in x])

    x = [np.pad(arr, [(0,maxlen-arr.shape[0]), (0,0)], 'constant') for arr in x]
    x = np.array(x)

    return x



def choose_nearest_2indexes_each_struct(struct:np.ndarray) -> np.ndarray:
    nearest_indexes = []
    for i in range(struct.shape[0]):
        radiuslist = [radius(struct[i], struct[j]) for j in range(struct.shape[0])]
        nearest_indexes.append(np.argsort(radiuslist)[1:3])

    return np.array(nearest_indexes)



def mean_struct(trj:np.ndarray) -> np.ndarray:
    return np.mean(trj, axis=0)



class MyProcess:
    def __init__(self, func, totalstep:int):
        self.totalstep = totalstep
        self.prev_step = -1
        self.starttime = time.time()
    
    def __call__(self, step:int, *args):
        ret = convert_descriptor(*args)

        if self.prev_step!=step:
            self.prev_step = step
            progress = int((step+1) / self.totalstep * 100)
            elapsed_time = time.time() - self.starttime
            print(f'\rProgress: {progress}% {elapsed_time:.1f}s', end='')
        
        return ret





if __name__=='__main__':
    main()