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

K.set_image_dim_ordering('th')

def ecal_sum(image):
    sum = K.sum(image, axis=(2, 3, 4))
    return sum
   
def ecal_angle(image):

    image = K.squeeze(image, axis=4)
    # size of ecal

    image = K.squeeze(image, axis=1)

    x_shape= K.int_shape(image)[1]
    y_shape= K.int_shape(image)[2]
    z_shape= K.int_shape(image)[3]
    sumtot = K.sum(image, axis=(1, 2, 3))# sum of events

    # ref denotes barycenter as that is our reference point
    x_ref = K.sum(K.sum(image, axis=(2, 3)) * K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32'), axis=1)# sum for x position * x index
    y_ref = K.sum(K.sum(image, axis=(1, 3)) * K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32'), axis=1)
    z_ref = K.sum(K.sum(image, axis=(1, 2)) * K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32'), axis=1)
    x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) * K.cast(x_shape - 1, dtype='float32'), x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
    y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref)* K.cast(y_shape - 1, dtype='float32'), y_ref/sumtot)
    z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref)* K.cast(z_shape - 1, dtype='float32'), z_ref/sumtot)
    #reshape    
    x_ref = K.expand_dims(x_ref, 1)
    y_ref = K.expand_dims(y_ref, 1)
    z_ref = K.expand_dims(z_ref, 1)

    sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z
    x = K.expand_dims(K.arange(x_shape), 0) # x indexes
    x = K.cast(K.expand_dims(x, 2), dtype='float32')
    y = K.expand_dims(K.arange(y_shape), 0)# y indexes
    y = K.cast(K.expand_dims(y, 2), dtype='float32')
    #xsum = K.sum(image, axis=2)
    #barycenter for each z position
    x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
    y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)
    x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum
    z = K.cast(K.arange(z_shape), dtype='float32') * K.ones_like(z_ref) # Make an array of z indexes for all events
    zproj = K.sqrt((x_mid-x_ref)**2.0 + (z - z_ref)**2.0)# projection from z axis
    m = K.tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = K.tf.where(K.tf.less(z, z_ref),  -1 * m, m)# sign inversion
    ang = (math.pi/2.0) - tf.atan(m)# angle correction
    ang = K.mean(ang, axis=1) # meanof angles computed
    ang = K.expand_dims(ang, 1)
    print(K.int_shape(ang))
    return ang

def ecal_angle2(image):
    image = K.squeeze(image, axis=4)
    x_shape= K.int_shape(image)[1]
    y_shape= K.int_shape(image)[2]
    z_shape= K.int_shape(image)[3]
    sumtot = K.sum(image, axis=(1, 2, 3))# sum of events
    x_ref = K.sum(K.sum(image, axis=(2, 3)) * K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32'), axis=1)
    y_ref = K.sum(K.sum(image, axis=(1, 3)) * K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32'), axis=1)
    z_ref = K.sum(K.sum(image, axis=(1, 2)) * K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32'), axis=1)
    x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) * K.cast(x_shape - 1, dtype='float32'), x_ref/sumtot)
    y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref)* K.cast(y_shape - 1, dtype='float32'), y_ref/sumtot)
    z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref)* K.cast(z_shape - 1, dtype='float32'), z_ref/sumtot)
    x_ref = K.expand_dims(x_ref, 1)
    y_ref = K.expand_dims(y_ref, 1)
    z_ref = K.expand_dims(z_ref, 1)
    sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z
    x = K.expand_dims(K.arange(x_shape), 0)
    x = K.cast(K.expand_dims(x, 2), dtype='float32')
    y = K.expand_dims(K.arange(y_shape), 0)
    y = K.cast(K.expand_dims(y, 2), dtype='float32')
    xsum = K.sum(image, axis=1)
    x_mid = K.sum(K.sum(image, axis=1) * x, axis=1)   ## SOFIA: this axis  (supposedly X) was set at 2 instead of 1 for NHWC (=TF) ordering : why????
    y_mid = K.sum(K.sum(image, axis=2) * y, axis=1)
    x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum
    z = K.cast(K.arange(z_shape), dtype='float32') * K.ones_like(z_ref)
    zproj = K.sqrt((x_mid-x_ref)**2.0 + (z - z_ref)**2.0)
    m = K.tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj), (y_mid-y_ref)/(z - z_ref))
    m = K.tf.where(K.tf.less(z, z_ref),  -1 * m, m)
    ang = (math.pi/2.0) - tf.atan(m)
    ang = K.mean(ang, axis=1)
    ang = K.expand_dims(ang, 1)
    print(K.int_shape(ang))
    return ang
                                                                            
# Angle Measure using weighting by z for all events
def ProcAngles2(meas):
   a = np.arange(meas.shape[1])
   avg = np.zeros(meas.shape[0])
   for i in np.arange(1, meas.shape[0]):
      avg[i] = np.sum( meas[i] * a)/ (0.5  * (meas.shape[1] * (meas.shape[1] - 1)))
   return avg

def ecal_ang_3d(image):
    image = K.eval(image)
    ang = ecal_angle3(image)
    ang = tf.convert_to_tensor(ang, dtype=tf.float32)
    return ang

def tf_ecal_angle(image):
    out = py_func(ecal_angle2, [image], float32)
    return out

def output_of_lambda(input_shape):
    return (input_shape[0], 1)

def discriminator():
  
    image=Input(shape=(1, 51, 51, 25))

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
    ang2 = Lambda(ecal_angle)(image)
    ecal = Lambda(ecal_sum)(image)
    Model(input=image, output=[fake, aux, ang1, ang2, ecal]).summary()
    return Model(input=image, output=[fake, aux, ang1, ang2, ecal])


def generator(latent_size=200, return_intermediate=False):
    
    loc = Sequential([
        Dense(5184, input_shape=(latent_size,)),
        Reshape((8, 9, 9, 8)),

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
        Conv3D(6, 3, 5, 8,init='he_uniform'),
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
