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

K.set_image_dim_ordering('tf')

def ecal_sum(image):
    sum = K.sum(image, axis=(1, 2, 3))
    #sum = K.expand_dims(sum)
    return sum
   
def ecal_angle(image):
    a = K.sum(image, axis=(1, 4))
    maxa = K.argmax(a, axis=1)
    b = np.arange(25)
    p = []
    ang = []
    for i in np.arange(image.shape[0]):    
       p.append( np.polyfit(b, maxa[i], 1))
       ang.append( math.atan(p[i][0]))
    ang = tf.convert_to_tensor(ang, dtype=tf.float32)
    return ang
    
def ecal_angle2(image):
    a = K.sum(image, axis=(1, 4))
    b = K.argmax(a, axis=1)
    c = K.cast(b[:,21] - b[:,3], dtype='float32')/18.0
    d = tf.atan(c)
    #d = K.abs(d)
    d = (3.14158/2.0) - d
    d = K.expand_dims(d)
    return d

def tf_ecal_angle(image):
    out = py_func(ecal_angle2, [image], float32)
    return out

def output_of_lambda(input_shape):
    return (input_shape[0], 1)

def discriminator():
  
    image=Input(shape=(51, 51, 25, 1))

    x = Conv3D(16, 5, 6, 6, border_mode='same')(image)
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
    ang = Dense(100, activation='linear', name= 'ang')(dnn_out)
    ang1 = Dense(1, activation='linear', name='ang1')(ang)
    ang2 = Lambda(ecal_angle2)(image)
    ecal = Lambda(ecal_sum)(image)
    Model(input=image, output=[fake, aux, ang1, ang2, ecal]).summary()
    return Model(input=image, output=[fake, aux, ang1, ang2, ecal])


def generator(latent_size=200, return_intermediate=False):
    
    loc = Sequential([
        Dense(5184, input_shape=(latent_size, )),
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
    latent = Input(shape=(latent_size, ))   
    fake_image = loc(latent)
    loc.summary()
    Model(input=[latent], output=fake_image).summary()
    return Model(input=[latent], output=fake_image)

g= generator()
d=discriminator()
