#!/usr/bin/env python

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

INPUTDIR = "workspace/04-learning"
OUTDIR = "workspace/05-visualize-history"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    
    keyname_list = []
    history_list = []
    for filepath in glob.glob(INPUTDIR + '/*/history.csv'):
        keyname_list.append(os.path.basename(os.path.dirname(filepath)))
        history_list.append(np.loadtxt(filepath, delimiter=','))

    # plot
    for keyname, history in zip(keyname_list, history_list):
        loss = history[0]
        val_loss = history[1]

        epochs = range(1, len(loss)+1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epochs, loss, label='train')
        ax.plot(epochs, val_loss, label='validation')
        ax.plot(epochs, [1]*len(epochs), color='k', linewidth = 0.5)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
        plt.savefig(os.path.join(OUTDIR, f'history-{keyname}.png'))


if __name__ == '__main__':
    main()