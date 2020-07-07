#!/usr/bin/env python

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

INPUTDIR02 = "workspace/02-lrtest"
INPUTDIR03 = "workspace/03-learning"


def main():
    # load 02 data
    historydict02 = load_history_csv(glob.glob(INPUTDIR02 + '/*/*.csv'))
    historydict03 = load_history_csv(glob.glob(INPUTDIR03 + '/*/*/history.csv'))

    # plot 02
    for outpath, history in historydict02.items():
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
    for outpath, history in historydict03.items():
        history = np.array(history)[:, [1, 3]].transpose(1, 0)
        loss = history[0]
        val_loss = history[1]

        epochs = range(1, len(loss)+1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epochs, loss, label='train')
        ax.plot(epochs, val_loss, label='validation')
        ax.plot(epochs, [1]*len(epochs), color='k', linewidth=0.5)
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

        try:
            historydict[outpath] = np.loadtxt(filepath, delimiter=',', skiprows=1)
        except:
            print(f'"{filepath}" are skipped. cannot load the file.')
        
    return historydict

if __name__ == '__main__':
    main()
