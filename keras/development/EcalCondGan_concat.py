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

def discriminator():

    image = Input(shape=(51, 51, 25, 1))

    x = Conv3D(8, 6, 8, 6, border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, 5, 8, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, 8, 8, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, 8, 8, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv3D(8, 8, 8, 6, border_mode='valid')(x)
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
    #eta = Dense(1, activation='linear', name='eta')(dnn_out)
    theta = Dense(1, activation='linear', name='theta')(dnn_out)

    ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)))(image)
    Model(input=image, output=[fake, aux, theta, ecal]).summary()
    return Model(input=image, output=[fake, aux, theta, ecal])

def generator(latent_size=256, return_intermediate=False):
    
    loc = Sequential([
        Dense(5184, input_shape=(latent_size, )),
        Reshape((9, 9, 8, 8)),

        Conv3D(64, 6, 8, 8, border_mode='same', init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(3, 3, 2)),

        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, 5, 6, 8, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 3)),

        ZeroPadding3D((0, 2, 1)),
        Conv3D(6, 3, 5, 7, init='he_uniform'),
        LeakyReLU(),

        ZeroPadding3D((1, 1, 1)),
        Conv3D(6, 3, 3, 6, init='he_uniform'),
        LeakyReLU(),

        Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal'),
        Activation('relu')
    ])
    latent = Input(shape=(latent_size, ))   
    fake_image = loc(latent)
    loc.summary()
    Model(input=[latent], output=fake_image).summary()
    return Model(input=[latent], output=fake_image)

g= generator()
d=discriminator()

