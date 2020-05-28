#!/usr/bin/env python

import numpy as np
import os
import argparse
import tensorflow as tf
from utils_tf import MyModel
from utils_tf import CycleLearningRate

INPUT_TRAIN = "workspace/02-make-dataset/training.npz"
INPUT_VAL = "workspace/02-make-dataset/validation.npz"

OUTDIR = "workspace/04-learning"
EACH_OUDIR_KEY = "try"
FREQ_SAVE_MODEL = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--clr', nargs=2, type=float, metavar=('base_lr', 'max_lr'), help='input base_lr and max_lr if use Cyclical Learning Rate')
    parser.add_argument('--model_number', type=int, default=0, help='choose Model number')
    args = parser.parse_args()


    # save path
    os.makedirs(OUTDIR, exist_ok=True)
    outdir = decide_outdir()
    os.makedirs(outdir, exist_ok=True)
    history_path = os.path.join(outdir, 'history.csv')

    option_path = os.path.join(outdir, 'option.txt')
    save_options(args, option_path)

    weights_dir = os.path.join(outdir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # training data
    npz = np.load(INPUT_TRAIN)
    x_train = npz['x']
    t_train = npz['y']
    del npz

    # validation data
    npz = np.load(INPUT_VAL)
    x_val = npz['x']
    t_val = npz['y']
    del npz

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, t_train)).shuffle(10000).batch(args.batch)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, t_val)).batch(args.batch)

    n_data = x_train.shape[0]
    input_dim = x_train.shape[1]

    del x_train; del t_train; del x_val; del t_val

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


    @tf.function
    def val_step(x, t):
        predictions = model(x)
        v_loss = loss_object(t, predictions)

        val_loss(v_loss)

    # ## learning rate # ##
    if args.clr:
        lr_generator = CycleLearningRate(args.clr[0], args.clr[1], n_data, args.batch)

    else:
        lr_generator = lambda: args.lr
    

    # ## learning ## #
    loss_history = []
    for epoch in range(args.epochs):
        optimizer.lr = lr_generator()
        for x, t in train_ds:
            train_step(x, t)

        for x, t in val_ds:
            val_step(x, t)

        epoch_str = str(epoch+1).zfill(len(str(args.epochs)))
        template = 'Epoch {}/{}, Loss: {:.5f}, Test Loss: {:.5f}'
        print(template.format(args.epochs, epoch_str,
                                train_loss.result(),
                                val_loss.result()))

        loss_history.append([train_loss.result(), val_loss.result()])
   
        # reset metrics for next epoch
        train_loss.reset_states()
        val_loss.reset_states()

        if (epoch+1) % FREQ_SAVE_MODEL == 0:
            weight_path = os.path.join(weights_dir, f'weights{epoch_str}')
            model.save_weights(weight_path)

            np.savetxt(history_path, np.array(loss_history), delimiter=",", header="train loss,val_loss")
    
    weight_path = os.path.join(weights_dir, f'weights{epoch_str}')
    model.save_weights(weight_path)

    np.savetxt(history_path, np.array(loss_history), delimiter=",", header="train loss,val_loss")


def save_options(args, fp):
    with open(fp, mode='w') as f:
        f.write(f'Number of epochs:\t{args.epochs}'
                f'\nModel number:\t{args.model_number}')

        if args.clr:
            f.write(f'\nbase lr:\t{args.clr[0]}'
                    f'\nmax lr:\t{args.clr[1]}')
        else:
            f.write(f'\nlr:\t{args.lr}')


def decide_outdir():
    for i in range(100):
        if f'{EACH_OUDIR_KEY}{i:0=3}' in os.listdir(OUTDIR):
            continue
        return os.path.join(OUTDIR, f'{EACH_OUDIR_KEY}{i:0=3}')


if __name__ == '__main__':
    main()
