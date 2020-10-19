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
def new_ecal_sum(image, power):
    # process the image by **1/power
    image = tf.pow(image, 1./power)
    # sum the values along the daxis
    sum = tf.math.reduce_sum(image, daxis)   
    return sum

# Calculating angle from image
def ecal_angle(image, power, channels_format):
    # drop an axis
    if channels_format =='channels_last':
        image = tf.squeeze(image, axis=4)
    else: 
        image = tf.squeeze(image, axis=1)
    
    # pre-process the image
    image = tf.pow(image, 1./power)   
     
    # size of ecal
    x_shape= image.get_shape()[1].value
    y_shape= image.get_shape()[2].value
    z_shape= image.get_shape()[3].value
    sumtot = tf.math.reduce_sum(image, (1,2,3))# sum of events

    # get 1. where event sum is 0 and 0 elsewhere
    amask = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(sumtot) , tf.zeros_like(sumtot))
    masked_events = tf.math.reduce_sum(amask) # counting zero sum events
    
    # ref denotes barycenter as that is our reference point
    x_ref = tf.math.reduce_sum(tf.math.reduce_sum(image, (2,3)) * (tf.cast(tf.expand_dims(tf.range(x_shape), 0), dtype='float32') + 0.5), 1)# sum for x position * x index
    y_ref = tf.math.reduce_sum(tf.math.reduce_sum(image, (1,3)) * (tf.cast(tf.expand_dims(tf.range(y_shape), 0), dtype='float32') + 0.5), 1)
    z_ref = tf.math.reduce_sum(tf.math.reduce_sum(image, (1,2)) * (tf.cast(tf.expand_dims(tf.range(z_shape), 0), dtype='float32') + 0.5), 1)
    x_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(x_ref) , x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
    y_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(y_ref) , y_ref/sumtot)
    z_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(z_ref), z_ref/sumtot)
    
    # reshape    
    x_ref = tf.expand_dims(x_ref, 1)
    y_ref = tf.expand_dims(y_ref, 1)
    z_ref = tf.expand_dims(z_ref, 1)

    sumz = tf.math.reduce_sum(image, axis =(1,2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz) , tf.ones_like(sumz))
        
    x = tf.expand_dims(tf.range(x_shape), 0) # x indexes
    x = tf.cast(tf.expand_dims(x, 2), dtype='float32') + 0.5
    y = tf.expand_dims(tf.range(y_shape), 0)# y indexes
    y = tf.cast(tf.expand_dims(y, 2), dtype='float32') + 0.5
  
    # barycenter for each z position
    x_mid = tf.math.reduce_sum(tf.math.reduce_sum(image, axis=2) * x, axis=1)
    y_mid = tf.math.reduce_sum(tf.math.reduce_sum(image, axis=1) * y, axis=1)
    x_mid = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    # Angle Calculations
    z = (tf.cast(tf.range(z_shape), dtype='float32') + 0.5)  * tf.ones_like(z_ref) # Make an array of z indexes for all events
    epsilon = 0.0000007  # replaces k.epsilon(), used as fluff value to prevent /0 errors
    zproj = tf.math.sqrt(tf.math.maximum((x_mid-x_ref)**2.0 + (z - z_ref)**2.0, epsilon))# projection from z axis with stability check
    m = tf.where(tf.math.equal(zproj, 0.0), tf.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = tf.where(tf.math.less(z, z_ref),  -1 * m, m)   # sign inversion
    ang = (math.pi/2.0) - tf.atan(m)   # angle correction
    zmask = tf.where(tf.math.equal(zproj, 0.0), tf.zeros_like(zproj) , zmask)
    ang = ang * zmask # place zero where zsum is zero
    
    ang = ang * z  # weighted by position
    sumz_tot = z * zmask # removing indexes with 0 energies or angles

    #zunmasked = tf.math.reduce_sum(zmask, axis=1) # used for simple mean 
    #ang = tf.math.reduce_sum(ang, axis=1)/zunmasked # Mean does not include positions where zsum=0

    ang = tf.math.reduce_sum(ang, axis=1)/tf.math.reduce_sum(sumz_tot, axis=1) # sum ( measured * weights)/sum(weights)
    ang = tf.where(tf.math.equal(amask, 0.), ang, 100. * tf.ones_like(ang)) # Place 100 for measured angle where no energy is deposited in events
    
    ang = tf.expand_dims(ang, 1)
    return ang