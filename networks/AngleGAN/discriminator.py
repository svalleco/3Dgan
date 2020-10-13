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


# Discriminator -  - 4 layers, lambda functions, uses LeakyReLu (sparsity & convergence), outputs sigmoid neuron and linear neuron
def discriminator(power=1.0):
    
    # Make sure the dimension ordering is correct 
    if K.image_data_format() =='channels_last':
        dshape=(51, 51, 25,1)
    else:
        dshape=(1, 51, 51, 25)
        daxis=(2,3,4)

    image = Input(shape=dshape) 

    # layer 1 = takes in the image
    x = Conv3D(16, 5, 6, 6, border_mode='same')(image)
    x = LeakyReLU()(x)     # D uses LeakyReLU (needed for convergence), G uses ReLu for sparsity
    x = Dropout(0.2)(x)    # dropout regularization

    # layer 2
    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, 5, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)     # dropout regularization

    # layer 3
    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, 5, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)     # dropout regularization

     # layer 4
    x = Conv3D(8, 5, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)     # dropout regularization

    # one average pooling layer (additional pooling layers resulted in loss in performance)
    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
    dnn.summary()

    # Output
    dnn_out = dnn(image)
    # A sigmoid neuron predicts the typical GAN real/fake probability 
    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    # A linear neuron implements a regression on the primary particle energy following the auxiliary GAN schema
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    
    # Lambda Functions: calculate and constrain (total deposited energy, binned pixel intensity distribution, and incident angle) 
    # according to loss function terms
    ang = Lambda(ecal_angle, arguments={'power':power})(image)
    ecal = Lambda(ecal_sum, arguments={'power':power})(image)
    add_loss = Lambda(count, arguments={'power':power})(image)
    Model(input=[image], output=[fake, aux, ang, ecal, add_loss]).summary()
    return Model(input=[image], output=[fake, aux, ang, ecal, add_loss])
