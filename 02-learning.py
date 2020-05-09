#!/usr/bin/env python

import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping


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
    model = Sequential()
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(80, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.5))
    # for _ in range(10):
    #     model.add(Dense(100, activation='relu'))
    # # model.add(Dense(100, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(10, activation='relu')) 
    model.add(Dense(3, activation='linear'))      

    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    hist = model.fit(x_train, t_train,
                    epochs=100, batch_size=50,
                    verbose=2,
                    validation_data=(x_val, t_val))


    np.save(args.out, hist.history)


if __name__ == '__main__':
    main()
