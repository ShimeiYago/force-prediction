#!/usr/bin/env python

import numpy as np
import os
import glob
import matplotlib.pyplot as plt

INPUTDIR = "workspace/02-learning/history"
OUTDIR = "workspace/03-visualize-history"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    loss = []
    val_loss = []
    for filepath in glob.glob(INPUTDIR + '/*'):
        hist = np.load(filepath, allow_pickle=True).reshape(1)[0]
        loss.extend(hist['loss'])
        val_loss.extend(hist['val_loss'])
    
    epochs = range(1, len(loss)+1)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs, loss, label='train')
    ax.plot(epochs, val_loss, label='validation')
    ax.plot(epochs, [1]*len(epochs), color='k', linewidth = 0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.savefig(os.path.join(OUTDIR, 'history.png'))


if __name__ == '__main__':
    main()