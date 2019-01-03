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

#K.set_image_dim_ordering('channels_first')

def ecal_sum(image):
    sum = K.sum(image, axis=(2, 3, 4))
    return sum
   

def discriminator(keras_dformat='channels_last'):
    print (keras_dformat)
    if keras_dformat =='channels_last':
        dshape=(25, 25, 25,1)
        daxis=(1,2,3)
    else:
        dshape=(1, 25, 25, 25)
        daxis=(2,3,4)
    image = Input(shape=dshape) 
    x = Conv3D(32, (5, 5,5), data_format=keras_dformat, padding='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    x = Conv3D(8, (5, 5, 5), data_format=keras_dformat, padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x, training=True)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, (5, 5,5), data_format=keras_dformat, padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x, training=True)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, (5, 5, 5), data_format=keras_dformat, padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x, training=True)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2), data_format=keras_dformat)(x)
    h = Flatten()(x)

    dnn = Model(image, h)#.summary()
    dnn.summary()
    #image = Input(shape=(25, 25, 25,1))

    dnn_out = dnn(image)


    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ecal = Lambda(lambda x: K.sum(x, axis=daxis))(image)
    Model(input=[image], output=[fake, aux, ecal]).summary()
    return Model(input=[image], output=[fake, aux, ecal])

def generator(latent_size=200, return_intermediate=False, keras_dformat='channels_last'):

     if keras_dformat =='channels_last':
        dim = (7,7,8,8)
     else:
        dim = (8, 7, 7,8)
     
     loc = Sequential([
         Dense(64 * 7* 7, input_dim=latent_size),
         Reshape(dim),
         Conv3D(64, (6, 6, 8), data_format=keras_dformat, padding='same', kernel_initializer='he_uniform'),
         LeakyReLU(),
         BatchNormalization(),
         UpSampling3D(size=(2, 2, 2), data_format=keras_dformat),

         ZeroPadding3D((2, 2, 0)),
         Conv3D(6, (6, 5, 8), data_format=keras_dformat, kernel_initializer='he_uniform'),
         LeakyReLU(),
         BatchNormalization(),
         UpSampling3D(size=(2, 2, 3), data_format=keras_dformat),

         ZeroPadding3D((1,0,3)),
         Conv3D(6, (3, 3, 8), data_format=keras_dformat, kernel_initializer='he_uniform'),
         LeakyReLU(),
         Conv3D(1, (2, 2, 2), data_format=keras_dformat, use_bias=False, kernel_initializer='glorot_normal'),
         Activation('relu')
      
     ])
   
     latent = Input(shape=(latent_size, ))
     loc.summary()
     fake_image = loc(latent)

     Model(input=[latent], output=[fake_image]).summary()
     return Model(input=[latent], output=[fake_image])

def main():
    d = discriminator()
    g = generator()

if __name__ == '__main__':
    main()
