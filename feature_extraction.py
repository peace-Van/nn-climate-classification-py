#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 23:46:19 2022

@author: ji
"""

from tensorflow import keras
import scipy.io
import numpy as np
from model import CustomLoss

dataset = scipy.io.loadmat('test_set.mat')['t']
X = np.stack(list(dataset[:, 0]))
model = keras.models.load_model('trainedNet_70.81', custom_objects={'CustomLoss': CustomLoss})

feature_extraction = keras.Model(inputs=model.input, outputs=model.get_layer('p_re_lu').output)
features = feature_extraction(X, training=False).numpy()

scipy.io.savemat('features.mat', {'features':features}, do_compression=True)
