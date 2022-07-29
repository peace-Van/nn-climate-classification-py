#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 10:25:54 2022

@author: ji
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def log_relu(x):
    return tf.math.log1p(tf.nn.relu(x))

class CustomActivation(keras.layers.Layer):
    def __init__(self):
        super(CustomActivation, self).__init__()

        self.mu_t_1 = tf.Variable(initial_value=-18.0)
        self.mu_t_2 = tf.Variable(initial_value=0.0)
        self.mu_t_3 = tf.Variable(initial_value=10.0)
        self.mu_t_4 = tf.Variable(initial_value=22.0)

        self.sigma_t_1 = tf.Variable(initial_value=1.5)
        self.sigma_t_2 = tf.Variable(initial_value=1.5)
        self.sigma_t_3 = tf.Variable(initial_value=1.5)
        self.sigma_t_4 = tf.Variable(initial_value=1.5)

        self.sigma_t_5 = tf.Variable(initial_value=2.0)
        self.sigma_t_6 = tf.Variable(initial_value=2.0)
        self.sigma_t_7 = tf.Variable(initial_value=2.0)
        self.sigma_t_8 = tf.Variable(initial_value=2.0)

        self.mu_p_1 = tf.Variable(initial_value=10.0)
        self.mu_p_2 = tf.Variable(initial_value=40.0)
        self.mu_p_3 = tf.Variable(initial_value=100.0)

        self.sigma_p_1 = tf.Variable(initial_value=10.0)
        self.sigma_p_2 = tf.Variable(initial_value=10.0)
        self.sigma_p_3 = tf.Variable(initial_value=10.0)

        self.sigma_p_4 = tf.Variable(initial_value=20.0)
        self.sigma_p_5 = tf.Variable(initial_value=20.0)
        self.sigma_p_6 = tf.Variable(initial_value=20.0)

    def call(self, inputs):

        res1 = []
        res1.append(tf.math.tanh((inputs[:,0,:] - self.mu_t_1) / self.sigma_t_1))
        res1.append(tf.math.tanh((inputs[:,1,:] - self.mu_p_1) / self.sigma_p_1))
        res1.append(tf.math.tanh((inputs[:,2,:] - self.mu_p_1) / self.sigma_p_1))
        res1.append(tf.math.tanh((inputs[:,0,:] - self.mu_t_2) / self.sigma_t_2))
        res1.append(tf.math.tanh((inputs[:,1,:] - self.mu_p_2) / self.sigma_p_2))
        res1.append(tf.math.tanh((inputs[:,2,:] - self.mu_p_2) / self.sigma_p_2))
        res1.append(tf.math.tanh((inputs[:,0,:] - self.mu_t_3) / self.sigma_t_3))
        res1.append(tf.math.tanh((inputs[:,1,:] - self.mu_p_3) / self.sigma_p_3))
        res1.append(tf.math.tanh((inputs[:,2,:] - self.mu_p_3) / self.sigma_p_3))
        res1.append(tf.math.tanh((inputs[:,0,:] - self.mu_t_4) / self.sigma_t_4))
        res1 = tf.stack(res1, axis=1)
        jan = tf.reshape(res1[:,:,0], (-1,10,1))
        dec = tf.reshape(res1[:,:,11], (-1,10,1))
        res1 = tf.concat((dec, res1, jan), axis=-1)
        res1 = tf.reshape(res1, (-1, 10, 14, 1))

        res2 = []
        res2.append(log_relu((inputs[:,0,:] - self.mu_t_1) / self.sigma_t_5))
        res2.append(log_relu((inputs[:,1,:] - self.mu_p_1) / self.sigma_p_4))
        res2.append(log_relu((inputs[:,2,:] - self.mu_p_1) / self.sigma_p_4))
        res2.append(log_relu((inputs[:,0,:] - self.mu_t_2) / self.sigma_t_6))
        res2.append(log_relu((inputs[:,1,:] - self.mu_p_2) / self.sigma_p_5))
        res2.append(log_relu((inputs[:,2,:] - self.mu_p_2) / self.sigma_p_5))
        res2.append(log_relu((inputs[:,0,:] - self.mu_t_3) / self.sigma_t_7))
        res2.append(log_relu((inputs[:,1,:] - self.mu_p_3) / self.sigma_p_6))
        res2.append(log_relu((inputs[:,2,:] - self.mu_p_3) / self.sigma_p_6))
        res2.append(log_relu((inputs[:,0,:] - self.mu_t_4) / self.sigma_t_8))
        res2 = tf.stack(res2, axis=1)
        jan = tf.reshape(res2[:,:,0], (-1,10,1))
        dec = tf.reshape(res2[:,:,11], (-1,10,1))
        res2 = tf.concat((dec, res2, jan), axis=-1)
        res2 = tf.reshape(res2, (-1, 10, 14, 1))

        return [res1, res2]


if __name__ == '__main__':
    import scipy.io
    d = scipy.io.loadmat('training_set.mat')['t']
    act = CustomActivation()
    data = np.stack(list(d[0:2, 0]))
    a, b = act(data)
    print(a, b)
