#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:32:31 2022

@author: ji
"""

from tensorflow import keras
import scipy.io
import numpy as np
from model import CustomLoss

dataset = scipy.io.loadmat('training_set.mat')['t']
model = keras.models.load_model('net', custom_objects={'CustomLoss': CustomLoss})

X = np.stack(list(dataset[:, 0]))
Y = np.stack([arr.flatten() for arr in dataset[:, 1]])

hist = model.fit(x=X, y=Y, batch_size=128, epochs=30, verbose=2,
                 validation_data=(X, Y), validation_freq=1)

model.save('trainedNet')
