#!/bin/bash

rsync -r nig:deep-md/force-prediction/workspace/ ./remote-workspace/ --exclude="*.hdf5"

# scp nig:gromacs/1d9v/rmsd.png ./remote-workspace