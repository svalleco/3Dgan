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

K.set_image_dim_ordering('tf')

def ecal_sum(image):
    sum = K.sum(image, axis=(1, 2, 3))
    return sum
   

def discriminator():

    image = Input(shape=(51, 51, 25, 1))

    x = Conv3D(32, 5, 6, 6, border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, 5, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, 5, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    #x = AveragePooling3D((2, 2, 2))(x)

    #x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, 5, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
    dnn.summary()

    dnn_out = dnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    theta = Dense(1, activation='linear', name='theta')(dnn_out)
    #phi = Dense(1, activation='linear', name='phi')(dnn_out)

    ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)))(image)
    Model(input=image, output=[fake, aux, theta, ecal]).summary()
    return Model(input=image, output=[fake, aux, theta, ecal])

def generator(latent_size=200, return_intermediate=False):
    
    loc = Sequential([
        Dense(2592, input_shape=(2,latent_size)),
        Reshape((9, 9, 8, 8)),

        Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(3, 3, 2)),

        ZeroPadding3D((2, 3, 1)),
        Conv3D(6, 5, 8, 8, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 3)),

        ZeroPadding3D((0, 2,0)),
        Conv3D(6, 3, 5, 8, init='he_uniform'),
        LeakyReLU(),
        Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal'),
        Activation('relu')
    ])
    latent = Input(shape=(2, latent_size ))   
    fake_image = loc(latent)
    loc.summary()
    Model(input=[latent], output=fake_image).summary()
    return Model(input=[latent], output=fake_image)

g= generator()
d=discriminator()

