#!/usr/bin/env python

file_string = '''
import numpy as np
import tensorflow as tf
from utils import Sample

# Global variable
OUT_SHAPE = 5
INPUT_SHAPE = (Sample.IMG_H, Sample.IMG_W, Sample.IMG_D)


def customized_loss(y_true, y_pred, loss='euclidean'):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = tf.keras.backend.mean(tf.keras.backend.square((y_pred - y_true)), axis=-1) \
                    + tf.keras.backend.sum(tf.keras.backend.square(y_pred), axis=-1)/2 * L2_norm_cost
    # euclidean distance loss
    elif loss == 'euclidean':
        val = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred-y_true), axis=-1))
    return val


def create_model(keep_prob = 0.8):
    model = tf.keras.models.Sequential()

    # NVIDIA's model
    model.add(tf.keras.layers.Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape= INPUT_SHAPE))
    model.add(tf.keras.layers.Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.Dense(OUT_SHAPE, activation='softsign'))

    return model


if __name__ == '__main__':
    # Load Training Data
    x_train = np.load("data/X.npy")
    y_train = np.load("data/y.npy")

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 100
    batch_size = 50

    model = create_model()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.compile(loss=customized_loss, optimizer=tf.keras.optimizers.adam())
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1, callbacks=callbacks_list)
'''