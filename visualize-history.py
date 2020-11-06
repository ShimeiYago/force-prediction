#!/usr/bin/env python

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py


INPUTDIR02 = "workspace/02-lrtest"
INPUTDIR03 = "workspace/03-learning"

plt.rcParams["font.size"] = 18

def main():
    # load 02 data
    historydict02 = load_history_csv(glob.glob(INPUTDIR02 + '/*/*.csv'))
    historydict03 = load_history_csv(glob.glob(INPUTDIR03 + '/*/*/history.csv'))

    # plot 02
    for outpath, li in historydict02.items():
        _, history = li
        history = np.array(history)[:, [1, 2]].transpose(1, 0)
        loss = history[0]
        lrs = [str(lr) for lr in history[1]]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.plot(lrs, loss)
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.xticks(rotation=90)
        plt.grid(color='gray')
        plt.savefig(outpath)

    # plot 03
    for outpath, li in historydict03.items():
        atom, history = li

        # root and denormalize
        mean, std = get_mean_std(outpath, atom)
        history = np.sqrt(history)
        history = np.add(np.multiply(history, std), mean)

        history = np.array(history)[:, [1, 3]].transpose(1, 0)
        loss = history[0]
        val_loss = history[1]

        epochs = range(1, len(loss)+1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epochs, loss, label='train')
        ax.plot(epochs, val_loss, label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
        plt.savefig(outpath)
        plt.close()


def load_history_csv(filepath_list):
    historydict = {}
    for filepath in filepath_list:
        filename = os.path.splitext(os.path.basename(filepath))[0] + '.png'
        outpath = os.path.join(os.path.dirname(filepath), filename)

        atom = os.path.basename((os.path.dirname(os.path.dirname(filepath))))

        try:
            historydict[outpath] = [atom, np.loadtxt(filepath, delimiter=',', skiprows=1)]
        except:
            print(f'"{filepath}" are skipped. cannot load the file.')

    return historydict


def get_mean_std(filepath, atom):
    option_path = os.path.join(os.path.dirname(filepath), 'option.txt')
    with open(option_path) as f:
        datasetpath = f.readline().split('\t')[1].strip()

    ## normalization values ## #
    with h5py.File(datasetpath, mode='r') as f:
        y_mean, y_std = f[f'/{atom}/normalization'][...]

    return y_mean, y_std

if __name__ == '__main__':
    main()
