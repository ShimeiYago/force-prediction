#!/usr/bin/env python

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import os

TOPOPATH = "input/topo.gro"
TRRPATH = "input/fitted.trr"

OUTDIR = 'workspace/check-rmsf'


def main():
    ### load ###
    trj_mdtraj = md.load(TRRPATH, top=TOPOPATH)
    trj = trj_mdtraj.xyz
    topo = trj_mdtraj.topology

    CAindexlist = [atom.index for atom in topo.atoms if atom.name == 'CA']


    ### calucrate RMSF of Ca
    rmsfs = calu_rmsfs(trj[:, CAindexlist, :])


    ### print most largest Ca indexes about RMSF ###
    print('most largest Ca indexes about RMSF')
    print(np.argsort(rmsfs)[::-1][:10])


    ### plot ###
    os.makedirs(OUTDIR, exist_ok=True)

    x = range(1, len(CAindexlist)+1)

    fig = plt.figure()
    plt.plot(x, rmsfs)
    fig.savefig(os.path.join(OUTDIR, 'rmsf.png'))


def calu_rmsfs(trj):
    mean_structure = trj.mean(axis=0)
    
    return np.sqrt(np.mean(np.sum(np.square(trj - mean_structure), axis=2), axis=0))


if __name__=='__main__':
    main()