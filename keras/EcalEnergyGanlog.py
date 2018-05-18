import sys
import h5py

from h5py import File as HDF5File
import numpy as np

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding, Masking)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)
import math
from keras.models import Model, Sequential
import tensorflow as tf
K.set_image_dim_ordering('tf')

# Output shape
def output_of_lambda(input_shape):
    return (input_shape[0], 1)

# Exponent
def expon(a):
   return K.exp(-6.0 * a)

# Ecal sum Calculation
def ecal_sum_log(image):
    image = K.squeeze(image, axis = 4)
    result = K.tf.where(K.equal(image, 0.0),  K.zeros_like(image), expon(image))
    sum = K.sum(result, axis=(1, 2, 3))
    sum = K.expand_dims(sum)
    return sum
   
def discriminator():

    image = Input(shape=(25, 25, 25, 1))

    x = Conv3D(32, 5, 5,5, border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, 5, 5,5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
    image = Input(shape=(25, 25, 25, 1))
    dnn_out = dnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ecal = Lambda(ecal_sum_log)(image)  #Ecal sum output
    Model(input=image, output=[fake, aux, ecal]).summary()
    return Model(input=image, output=[fake, aux, ecal])

def generator(latent_size=200, return_intermediate=False):

    loc = Sequential([
        Dense(64 * 7* 7, input_dim=latent_size),
        Reshape((7, 7,8, 8)),

        Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 2)),

        ZeroPadding3D((2, 2, 0)),
        Conv3D(6, 6, 5, 8, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 3)),

        ZeroPadding3D((1,0,3)),
        Conv3D(6, 3, 3, 8, init='he_uniform'),
        LeakyReLU(),
        Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal'),
        Activation('relu')
    ])
   
    latent = Input(shape=(latent_size, ))    
    fake_image = loc(latent)
    Model(input=[latent], output=fake_image).summary()
    return Model(input=[latent], output=fake_image)
