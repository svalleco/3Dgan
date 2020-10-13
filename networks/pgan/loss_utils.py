import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.models import Model
import tensorflow as tf
import math


if K.image_data_format() =='channels_last':
    daxis=(1,2,3)
    dshape=(51, 51, 25,1)
else:
    daxis=(2,3,4)
    dshape=(1, 51, 51, 25)

def prep_dnn(image, dshape=dshape):
    image = Input(shape=dshape) 
    h = Flatten()(image)
    dnn = Model(image, h) # h is flattened x from line 150 in anglegan discriminator = anglearch3dgan.py
    dnn.summary()
    # Output
    dnn_out = dnn(image)
    return dnn_out

# Summming cell energies
def ecal_sum(image, power):
    image = K.pow(image, 1./power)
    sum = K.sum(image, axis=daxis)
    return sum

# Calculating angle from image
def ecal_angle(image, power):
    if K.image_data_format() =='channels_last':
       image = K.squeeze(image, axis=4)
    else: 
       image = K.squeeze(image, axis=1)
    image = K.pow(image, 1./power)
    
    # size of ecal
    x_shape= K.int_shape(image)[1]
    y_shape= K.int_shape(image)[2]
    z_shape= K.int_shape(image)[3]
    sumtot = K.sum(image, axis=(1,2,3))# sum of events
    
    # get 1. where event sum is 0 and 0 elsewhere
    amask = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
    masked_events = K.sum(amask) # counting zero sum events
    
    # ref denotes barycenter as that is our reference point
    x_ref = K.sum(K.sum(image, axis=(2,3)) * (K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32') + 0.5) , axis=1)# sum for x position * x index
    y_ref = K.sum(K.sum(image, axis=(1,3)) * (K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32') + 0.5), axis=1)
    z_ref = K.sum(K.sum(image, axis=(1,2)) * (K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32') + 0.5), axis=1)
    x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) , x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
    y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref) , y_ref/sumtot)
    z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref), z_ref/sumtot)
    
    # reshape    
    x_ref = K.expand_dims(x_ref, 1)
    y_ref = K.expand_dims(y_ref, 1)
    z_ref = K.expand_dims(z_ref, 1)

    sumz = K.sum(image, axis =(1,2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz) , K.ones_like(sumz))
        
    x = K.expand_dims(K.arange(x_shape), 0) # x indexes
    x = K.cast(K.expand_dims(x, 2), dtype='float32') + 0.5
    y = K.expand_dims(K.arange(y_shape), 0)# y indexes
    y = K.cast(K.expand_dims(y, 2), dtype='float32') + 0.5
  
    # barycenter for each z position
    x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
    y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)
    x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    # Angle Calculations
    z = (K.cast(K.arange(z_shape), dtype='float32') + 0.5)  * K.ones_like(z_ref) # Make an array of z indexes for all events
    zproj = K.sqrt(K.maximum((x_mid-x_ref)**2.0 + (z - z_ref)**2.0, K.epsilon()))# projection from z axis with stability check
    m = K.tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = K.tf.where(K.tf.less(z, z_ref),  -1 * m, m)# sign inversion
    ang = (math.pi/2.0) - tf.atan(m)# angle correction
    zmask = K.tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj) , zmask)
    ang = ang * zmask # place zero where zsum is zero
    
    ang = ang * z  # weighted by position
    sumz_tot = z * zmask # removing indexes with 0 energies or angles

    #zunmasked = K.sum(zmask, axis=1) # used for simple mean 
    #ang = K.sum(ang, axis=1)/zunmasked # Mean does not include positions where zsum=0

    ang = K.sum(ang, axis=1)/K.sum(sumz_tot, axis=1) # sum ( measured * weights)/sum(weights)
    ang = K.tf.where(K.equal(amask, 0.), ang, 100. * K.ones_like(ang)) # Place 100 for measured angle where no energy is deposited in events
    
    ang = K.expand_dims(ang, 1)
    return ang