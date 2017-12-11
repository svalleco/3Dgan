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
from keras.utils import plot_model

K.set_image_dim_ordering('tf')

def ecal_sum(image):
    sum = K.sum(image, axis=(1, 2, 3))
    return sum
   

def discriminator():

    image = Input(shape=(25, 25, 25, 1))

    x = Conv3D(8, 12, 12, 12, border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    #x = ZeroPadding3D((2, 2,2))(x)
    x = Conv3D(16, 8, 8, 8, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    #x = ZeroPadding3D((2, 2, 2))(x)  #added
    x = Conv3D(32, 6, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    #x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(64, 6, 6, 6, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    #x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(128, 4, 4, 4, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv3D(256, 2, 2, 2, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    #x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)
    
    dnn = Model(image, h)
    dnn.summary()
    plot_model(dnn, to_file='dnn.pdf', show_shapes=1)
    image = Input(shape=(25, 25, 25, 1))

    dnn_out = dnn(image)


    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)))(image)
    Model(input=image, output=[fake, aux, ecal])
    return Model(input=image, output=[fake, aux, ecal])

def generator(latent_size=1024, return_intermediate=False):

    loc = Sequential([
        Dense(64 * 64, input_dim=latent_size),
        Reshape((8, 8, 8, 8)),

        Conv3D(128, 3, 3, 3, border_mode='valid', init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 2)),

        #ZeroPadding3D((2, 2, 0)),
        Conv3D(64, 4, 4, 4, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 2)),

        #ZeroPadding3D((2, 2, 3)),   #added
        Conv3D(32, 5, 5, 5, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 2)),

        #ZeroPadding3D((1,0,3)),
        Conv3D(16, 4, 4, 4, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        #UpSampling3D(size=(2, 2, 2)),

        #ZeroPadding3D((1,0,3)),   
        Conv3D(8, 8, 8, 8, init='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        UpSampling3D(size=(2, 2, 2)),

        #ZeroPadding3D((1,0,3)),           
        Conv3D(1, 12, 12, 12, init='he_uniform'),
        LeakyReLU(),
        #BatchNormalization(),
    ])
   
    latent = Input(shape=(latent_size, ))
    loc.summary() 
    plot_model(loc, to_file='loc.pdf', show_shapes=1)
    fake_image = loc(latent)

    Model(input=[latent], output=fake_image)
    return Model(input=[latent], output=fake_image)
