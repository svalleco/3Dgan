import sys
import h5py

from h5py import File as HDF5File
import numpy as np

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)

from keras.models import Model, Sequential
import math
from tensorflow import py_func, float32, Tensor
import tensorflow as tf

# moved these functions to keras/analysis/utils/GANutils.py
from GANutils import ecal_sum, count, ecal_angle

if K.image_data_format() =='channels_last':
    daxis=(1,2,3)
else:
    daxis=(2,3,4)


# Generator - 7 layers (stronger G = better results), uses ReLu (sparsity), clustered up-sampling operations to help with convergence at the lower distribution tails.
def generator(latent_size=254, return_intermediate=False):
    if K.image_data_format() =='channels_last':
        dim = (9,9,8,8)
        baxis=-1 # axis for BatchNormalization
    else:                       #channels_first
        dim = (8, 9, 9,8) 
        baxis=1 # axis for BatchNormalization
    
    # Clustered up-sampling operations to help with convergence at the lower distribution tails
    loc = Sequential([
        Dense(5184, input_shape=(latent_size,)),
        Reshape(dim),
        UpSampling3D(size=(6, 6, 6)),
        
        # layer 1
        Conv3D(8, (6, 6, 8), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        # layer 2
        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        ####################################### added layers (stronger G = better results)
        
        # layer 3
        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        # layer 4
        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        # layer 5
        ZeroPadding3D((1, 1, 0)),
        Conv3D(6, (3, 3, 5), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        #####################################  
        
        # layer 6
        ZeroPadding3D((1, 1,0)),
        Conv3D(6, (3, 3, 3), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        
        # layer 7
        Conv3D(1, (2, 2, 2),  padding='valid', kernel_initializer='glorot_normal'),
        Activation('relu')
    ])
    latent = Input(shape=(latent_size, ))   
    fake_image = loc(latent)
    loc.summary()
    Model(input=[latent], output=[fake_image]).summary()
    return Model(input=[latent], output=[fake_image])