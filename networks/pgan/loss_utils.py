import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.models import Model
import tensorflow as tf
import math
import numpy as np


# dimensions of real_images_input = [z_batch_size, 1=channel, z, x, y]

# !!!NEEDS WORK!!! prep the images for the fake layer in logistic loss function
def prep_dnn(real_image_input):
    image = Input(shape=dshape) 
    h = Flatten()(image)
    dnn = Model(image, h) # h is flattened x from line 150 in anglegan discriminator = anglearch3dgan.py
    dnn.summary()
    # Output
    dnn_out = dnn(image)
    return dnn_out


# prepares the pgan images for the physics calculations (un-inverts, removes channel, reorders axes)
def prep_image(images, power=1.0, channel_included=True, inverted=False):
    # pgan images shape will be either (z=shape/2, x=shape, y=shape) or (channel=1, z=shape/2, x=shape, y=shape)
    if channel_included:    
        images = tf.squeeze(images) # get rid of channel dimension
        
    # switch dimensions ordering from pgan (z,x,y) back to anglegan (x,y,z)
    images = np.moveaxis(images, 1, -1)   # move z back
    # the tensor is now [num_images,x,y,z]
    
    if not inverted:
        # pre-process the images (the og image is inverted in getdataangle())
        images = tf.pow(images, 1./power)
    return images


# Summming cell energies -- called in conditional lambda layer for the discriminator
def ecal_sum(images, size, power=1.0, channel_included=True, inverted=False):   
    images = prep_image(images, power=1.0, channel_included=True, inverted=False)
    daxis = (1,2,3)   # (x,y,z)
    # sum the values along the daxis
    sum = tf.math.reduce_sum(images, daxis)   
    return sum


# Calculating angles from images -- called in conditional lambda layer for the discriminator
def ecal_angle(images, size, power=1.0, channel_included=True, inverted=False):
    images = prep_image(images, power=1.0, channel_included=True, inverted=False)
     
    # size of ecal
    x_shape, y_shape, z_shape = int(size), int(size), int(size/2)
    print('shapes ----- ', x_shape, y_shape, z_shape )
    sumtot = tf.math.reduce_sum(images, (1,2,3))# sum of events
    print('sumtot: ', sumtot)
    
    # get 1. where event sum is 0 and 0 elsewhere
    amask = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(sumtot) , tf.zeros_like(sumtot))
    masked_events = tf.math.reduce_sum(amask) # counting zero sum events
    
    # ref denotes barycenter as that is our reference point
    x_ref = tf.math.reduce_sum(tf.math.reduce_sum(images, (2,3)) * (tf.cast(tf.expand_dims(tf.range(x_shape), 0), dtype='float32') + 0.5), axis=1)# sum for x position * x index
    y_ref = tf.math.reduce_sum(tf.math.reduce_sum(images, (1,3)) * (tf.cast(tf.expand_dims(tf.range(y_shape), 0), dtype='float32') + 0.5), axis=1)
    z_ref = tf.math.reduce_sum(tf.math.reduce_sum(images, (1,2)) * (tf.cast(tf.expand_dims(tf.range(z_shape), 0), dtype='float32') + 0.5), axis=1)
    # return max position if sumtot=0 and divide by sumtot otherwise
    x_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(x_ref), x_ref/sumtot)
    y_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(y_ref), y_ref/sumtot)
    z_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(z_ref), z_ref/sumtot)
    
    # reshape - put in value at the beginning    
    x_ref = tf.expand_dims(x_ref, 1)
    y_ref = tf.expand_dims(y_ref, 1)
    z_ref = tf.expand_dims(z_ref, 1)

    sumz = tf.math.reduce_sum(images, axis =(1,2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz) , tf.ones_like(sumz))
        
    x = tf.expand_dims(tf.range(x_shape), 0) # x indexes
    x = tf.cast(tf.expand_dims(x, 2), dtype='float32') + 0.5
    y = tf.expand_dims(tf.range(y_shape), 0)# y indexes
    y = tf.cast(tf.expand_dims(y, 2), dtype='float32') + 0.5
  
    # barycenter for each z position
    x_mid = tf.math.reduce_sum(tf.math.reduce_sum(images, axis=2) * x, axis=1)
    y_mid = tf.math.reduce_sum(tf.math.reduce_sum(images, axis=1) * y, axis=1)
    x_mid = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    # Angle Calculations
    z = (tf.cast(tf.range(z_shape), dtype='float32') + 0.5)  * tf.ones_like(z_ref) # Make an array of z indexes for all events
    epsilon = 0.0000007  # replaces k.epsilon(), used as fluff value to prevent /0 errors
    zproj = tf.math.sqrt(tf.math.maximum((x_mid-x_ref)**2.0 + (z - z_ref)**2.0, epsilon))# projection from z axis with stability check
    m = tf.where(tf.math.equal(zproj, 0.0), tf.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = tf.where(tf.math.less(z, z_ref),  -1 * m, m)   # sign inversion
    ang = (math.pi/2.0) - tf.atan(m)   # angle correction
    zmask = tf.where(tf.math.equal(zproj, 0.0), tf.zeros_like(zproj), zmask)
    ang = ang * zmask # place zero where zsum is zero
    
    ang = ang * z  # weighted by position
    sumz_tot = z * zmask # removing indexes with 0 energies or angles

    #zunmasked = tf.math.reduce_sum(zmask, axis=1) # used for simple mean 
    #ang = tf.math.reduce_sum(ang, axis=1)/zunmasked # Mean does not include positions where zsum=0

    ang = tf.math.reduce_sum(ang, axis=1)/tf.math.reduce_sum(sumz_tot, axis=1) # sum ( measured * weights)/sum(weights)
    ang = tf.where(tf.math.equal(amask, 0.), ang, 100. * tf.ones_like(ang)) # Place 100 for measured angle where no energy is deposited in events
    
    ang = tf.expand_dims(ang, 1)
    return ang