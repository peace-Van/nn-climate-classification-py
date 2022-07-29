#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:40:19 2022

@author: ji
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from custom_activation import CustomActivation

def CustomLoss(y_true, y_pred):
    return -tf.reduce_mean(tf.multiply(y_true, tf.math.log(y_pred)) + tf.multiply(1 - y_true, tf.math.log(1 - y_pred))) * 14


if __name__ == '__main__':

    ipt = keras.Input(shape=(3, 12))
    tanh_branch, logrelu_branch = CustomActivation()(ipt)

    tanh_branch = layers.Conv2D(24, (3, 3), kernel_initializer='he_normal')(tanh_branch)
    tanh_branch = layers.BatchNormalization(beta_regularizer='l2', gamma_regularizer='l2')(tanh_branch)
    tanh_act1 = keras.activations.relu(tanh_branch)
    tanh_act2 = keras.activations.tanh(tanh_branch)
    logrelu_branch = layers.Conv2D(24, (3, 3), kernel_initializer='he_normal')(logrelu_branch)
    logrelu_branch = layers.BatchNormalization(beta_regularizer='l2', gamma_regularizer='l2')(logrelu_branch)
    logrelu_act1 = keras.activations.relu(logrelu_branch)
    logrelu_act2 = keras.activations.tanh(logrelu_branch)

    f1 = layers.Flatten()(layers.MaxPooling2D((1, 12))(tanh_act1))
    f2 = layers.Flatten()(layers.AveragePooling2D((1, 12))(tanh_act1))
    f3 = layers.Flatten()(layers.MaxPooling2D((1, 12))(tanh_act2))
    f4 = layers.Flatten()(layers.AveragePooling2D((1, 12))(tanh_act2))
    f5 = layers.Flatten()(layers.MaxPooling2D((1, 12))(logrelu_act1))
    f6 = layers.Flatten()(layers.AveragePooling2D((1, 12))(logrelu_act1))
    f7 = layers.Flatten()(layers.MaxPooling2D((1, 12))(logrelu_act2))
    f8 = layers.Flatten()(layers.AveragePooling2D((1, 12))(logrelu_act2))

    feats = layers.concatenate((f1, f2, f3, f4, f5, f6, f7, f8))
    feats = layers.Dense(768, kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.02))(feats)
    feats = layers.BatchNormalization(beta_regularizer='l2', gamma_regularizer='l2')(feats)
    feats = layers.PReLU(alpha_initializer=keras.initializers.Constant(0.25))(feats)
    feats = layers.Dropout(0.5)(feats)
    pred = layers.Dense(14, kernel_initializer='he_normal', activation='softmax')(feats)

    full_model = keras.Model(inputs=ipt, outputs=pred)
    full_model.compile(optimizer=keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.ExponentialDecay(0.001, 5200, 0.1)),
                       loss=CustomLoss,
                       metrics=[keras.metrics.CategoricalAccuracy()])
    full_model.summary()
    full_model.save('net')
    keras.utils.plot_model(full_model, show_shapes=True, dpi=300)
