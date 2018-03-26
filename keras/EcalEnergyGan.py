import sys
import h5py

from h5py import File as HDF5File
import numpy as np

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)

from keras.models import Model, Sequential

K.set_image_dim_ordering('th')

def ecal_sum(image):
    sum = K.sum(image, axis=(2, 3))
    return sum
   

def discriminator():

    image = Input(shape=(1, 25, 25))

    x = Conv2D(32, (5,5), data_format='channels_first', border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding2D((2,2))(x)
    x = Conv2D(8, (5, 5), data_format='channels_first', border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(8, (5,5), data_format='channels_first', border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(8, (5, 5), data_format='channels_first', border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = AveragePooling2D((2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)

    image = Input(shape=(1, 25, 25))

    dnn_out = dnn(image)


    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ecal = Lambda(lambda x: K.sum(x, axis=(2, 3)))(image)
    Model(input=image, output=[fake, aux, ecal]).summary()
    return Model(input=image, output=[fake, aux, ecal])

def generator(latent_size=1024, return_intermediate=False):

    loc = Sequential([
        Dense(64 * 7, input_dim=latent_size),
        Reshape((8, 7,8)),

        Conv2D(64, (6, 8), data_format='channels_first', border_mode='same', init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling2D(size=(2, 2)),

        ZeroPadding2D((2, 0)),
        Conv2D(6, (5, 8), data_format='channels_first', init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling2D(size=(2, 3)),

        ZeroPadding2D((0,3)),
        Conv2D(6, (3, 8), data_format='channels_first', init='he_uniform'),
        LeakyReLU(),
        Conv2D(1, (2, 2), data_format='channels_first', bias=False, init='glorot_normal'),
        Activation('relu')
    ])
   
    latent = Input(shape=(latent_size, ))
     
    fake_image = loc(latent)

    Model(input=[latent], output=fake_image).summary()
    return Model(input=[latent], output=fake_image)
