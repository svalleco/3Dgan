import sys
import h5py

from h5py import File as HDF5File
import numpy as np

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, Conv3DTranspose, ZeroPadding3D,
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
    x_shape= K.int_shape(image)[1]
    image = K.squeeze(image, axis=1)
    # size of ecal
    x_shape= K.int_shape(image)[1]
    y_shape= K.int_shape(image)[2]
    z_shape= K.int_shape(image)[3]
    sumtot = K.sum(image, axis=(1, 2, 3))# sum of events

    # get 1. where event sum is 0 and 0 elsewhere
    amask = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
    masked_events = K.sum(amask) # counting zero sum events
    # ref denotes barycenter as that is our reference point
    x_ref = K.sum(K.sum(image, axis=(2, 3)) * K.expand_dims(K.arange(0,25, dtype='float32'), 0), axis=1)# sum for x position * x index
    y_ref = K.sum(K.sum(image, axis=(1, 3)) * K.expand_dims(K.arange(0,25, dtype='float32'), 0) , axis=1)
    z_ref = K.sum(K.sum(image, axis=(1, 2)) * K.expand_dims(K.arange(0,25, dtype='float32'), 0), axis=1)
    x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref), x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
    y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref), y_ref/sumtot)
    z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref), z_ref/sumtot)
    #reshape    
    x_ref = K.expand_dims(x_ref, 1)
    y_ref = K.expand_dims(y_ref, 1)
    z_ref = K.expand_dims(z_ref, 1)

    sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz) , K.ones_like(sumz))
    zunmasked_events = K.sum(zmask, axis=1)
    
    x = K.expand_dims(K.arange(0,25, dtype='float32'), 0) # x indexes
    x = K.expand_dims(x, 2)
    y = K.expand_dims(K.arange(0,25, dtype='float32'), 0)# y indexes
    y = K.expand_dims(y, 2)
  
    #barycenter for each z position
    x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
    y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)
    x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    #Angle Calculations
    z = K.arange(0,25, dtype='float32') * K.ones_like(z_ref) # Make an array of z indexes for all events
    zproj = K.sqrt((x_mid-x_ref)**2.0 + (z - z_ref)**2.0)# projection from z axis
    m = K.tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = K.tf.where(K.tf.less(z, z_ref),  -1 * m, m)# sign inversion
    ang = (math.pi/2.0) - tf.atan(m)# angle correction

    ang = ang * zmask # place zero where zsum is zero
    
    ang = K.sum(ang, axis=1)/zunmasked_events # Mean does not include positions where zsum=0
    ang = K.tf.where(K.equal(amask, 0.), ang, 100 * K.ones_like(ang)) # Place a 4 for measured angle where no energy is deposited in events
    
    ang = K.expand_dims(ang, 1)
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
  
    image=Input(shape=(1, 25, 25, 25))

    loc = Sequential([
    Conv3D(4, (16, 16, 16), padding='same',kernel_initializer='he_uniform',input_shape=(1,25,25,25)),
    LeakyReLU(),
    Dropout(0.2),

    Conv3D(8, (8, 8, 8), padding='valid',kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.2),

    Conv3D(16, (4, 4, 4), padding='valid',kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.2),

    Conv3D(32, (4, 4, 4), padding='valid',kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.2),

    Conv3D(64, (2, 2, 2), padding='valid',kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.2),

    Conv3D(64, (2, 2, 2), padding='valid',kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.2),

    AveragePooling3D((2, 2, 2)),
    Flatten()
    ])

    loc.summary()

    dnn_out = loc(image)

    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    ep = Dense(1, activation='linear', name='Ep')(dnn_out)
    ang = Dense(32, activation='linear', name= 'ang')(dnn_out)
    ang1 = Dense(1, activation='linear', name='ang1')(ang)
    print('image shape',image.shape)
    K.squeeze(image, axis=1)
    print('in discr ',K.sum(image, axis=(2, 3)).shape)
    x_shape= K.int_shape(image)[1]
    print('in discr ',K.arange(0,25, dtype='float32').shape)
    print('in discr ',K.expand_dims(K.arange(0,x_shape, dtype='float32'), 0).shape) 
    ang2 = Lambda(ecal_angle)(image)
    print ('ang2', ang2.shape)
    ecal = Lambda(ecal_sum)(image)
    print ('ecal', ecal.shape)
    #Model(input=image, output=[fake, ep, ang1, ang2, ecal]).summary()
    return Model(inputs=image, outputs=[fake, ep, ang1, ang2, ecal])


def generator(latent_size=200, return_intermediate=False):
    
    loc = Sequential([
        Dense(64*2*2*2, input_shape=(latent_size,)),
        Reshape((64, 2, 2, 2)),

        Conv3DTranspose(128, (2, 2, 2), padding='same', kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),

        Conv3DTranspose(64, (2, 2, 2),kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),

        Conv3DTranspose(32, (4, 4, 4),kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),


        Conv3DTranspose(16, (4,4, 4),kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        #ZeroPadding3D((2, 2,2)),
        #UpSampling3D(size=(2, 2, 1)),

        Conv3DTranspose(8, (8,8, 8),kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
 
        Conv3DTranspose(4, (8, 8, 8), kernel_initializer='he_uniform'),
        LeakyReLU(),
        BatchNormalization(),
        Conv3DTranspose(1, (3, 3, 3), kernel_initializer='glorot_normal'),
        Activation('relu')
    ])
    latent = Input(shape=(latent_size, ))   
    fake_image = loc(latent)
    loc.summary()
    Model(inputs=[latent], outputs=fake_image).summary()
    return Model(inputs=[latent], outputs=fake_image) 
#g= generator()
d=discriminator()
