#!/usr/bin/env python

import numpy as np
import os
import argparse
import tensorflow as tf
from utils_tf import MyModel

INPUT_TRAIN = "workspace/02-make-dataset/training.npz"
INPUT_VAL = "workspace/02-make-dataset/validation.npz"

OUTDIR = "workspace/03-LRtest"

LRLIST = [0.00000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=50, help='batch size')
    parser.add_argument('--model_number', type=int, default=0, help='choose Model number')
    args = parser.parse_args()

    # save path
    os.makedirs(OUTDIR, exist_ok=True)
    outdir = os.path.join(OUTDIR, f"Model{args.model_number}")
    history_path = os.path.join(OUTDIR, f'history-Model{args.model_number}.csv')

    # training data
    npz = np.load(INPUT_TRAIN)
    x_train = npz['x']
    t_train = npz['y']
    del npz

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, t_train)).shuffle(10000).batch(args.batch)

    n_data = x_train.shape[0]
    input_dim = x_train.shape[1]

    del x_train; del t_train

    # model
    model = MyModel(input_dim=input_dim, model_number=args.model_number)

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')


    @tf.function
    def train_step(x, t):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(t, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    # ## learning ## #
    loss_history = []
    for epoch, lr in enumerate(LRLIST):
        optimizer.lr = lr
        for x, t in train_ds:
            train_step(x, t)


        epoch_str = str(epoch+1).zfill(len(str(len(LRLIST))))
        template = 'Epoch {}/{}, Loss: {:.5f}, LR: {}'
        print(template.format(len(LRLIST), epoch_str,
                                train_loss.result(), lr))

        loss_history.append([lr, train_loss.result()])
   
        # reset metrics for next epoch
        train_loss.reset_states()
        val_loss.reset_states()
    
    np.savetxt(history_path, np.array(loss_history), delimiter=",", header="LR,train loss")



if __name__ == '__main__':
    main()
