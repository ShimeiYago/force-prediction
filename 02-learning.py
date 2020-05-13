#!/usr/bin/env python

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

from DNNmodel import model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default="workspace/01-make-dataset/trj.npz", help='input trajectory (.npz)')
    parser.add_argument('-o', '--out', default="workspace/02-learning/history.npy", help='output file path(.npy)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)


    ### prepare data ###

    # expranatory
    x = np.load(args.input)['x']

    # response
    t = np.load(args.input)['y']

    # data split
    x_train, x_val, t_train, t_val = train_test_split(x,t, test_size=0.2, shuffle=True)


    ### modeling ###
    hist = model.fit(x_train, t_train,
                    epochs=1000, batch_size=50,
                    verbose=2,
                    validation_data=(x_val, t_val))


    np.save(args.out, hist.history)


if __name__ == '__main__':
    main()
