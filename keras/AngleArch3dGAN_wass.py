import numpy as np
import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, UpSampling2D,Conv3D, ZeroPadding3D, ZeroPadding2D,
                                        AveragePooling3D, AveragePooling2D, SeparableConv2D)
from keras.models import Model, Sequential
from keras.constraints import Constraint
import math
import tensorflow as tf

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
      self.clip_value = clip_value
 
    # clip model weights to hypercube
    def __call__(self, weights):
      return backend.clip(weights, -self.clip_value, self.clip_value)
 
    # get the config
    def get_config(self):
      return {'clip_value': self.clip_value}

const = ClipConstraint(0.01)

# calculate sum of intensities
def ecal_sum(image, daxis):
    sum = K.expand_dims(K.sum(image, axis=daxis))
    return sum

# counts for various bin entries   
def count(image, daxis):
    limits=[0.05, 0.03, 0.02, 0.0125, 0.008, 0.003] # bin boundaries used
    bin1 = K.expand_dims(K.sum(K.tf.where(image > limits[0], K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bin2 = K.expand_dims(K.sum(K.tf.where(K.tf.logical_and(image < limits[0], image > limits[1]), K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bin3 = K.expand_dims(K.sum(K.tf.where(K.tf.logical_and(image < limits[1], image > limits[2]), K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bin4 = K.expand_dims(K.sum(K.tf.where(K.tf.logical_and(image < limits[2], image > limits[3]), K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bin5 = K.expand_dims(K.sum(K.tf.where(K.tf.logical_and(image < limits[3], image > limits[4]), K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bin6 = K.expand_dims(K.sum(K.tf.where(K.tf.logical_and(image < limits[4], image > limits[5]), K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bin7 = K.expand_dims(K.sum(K.tf.where(K.tf.logical_and(image < limits[5], image > 0.0), K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bin8 = K.expand_dims(K.sum(K.tf.where(K.tf.equal(image, 0.0), K.ones_like(image), K.zeros_like(image)), axis=daxis))
    bins = K.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
    return bins
                                        
# angle calculation 
def ecal_angle(image, daxis):
    #image = K.squeeze(image, axis=daxis)# squeeze along channel axis
    
    # get shapes
    x_shape= K.int_shape(image)[1]
    y_shape= K.int_shape(image)[2]
    z_shape= K.int_shape(image)[3]
    sumtot = K.sum(image, axis=(1, 2, 3))# sum of events

    # get 1. where event sum is 0 and 0 elsewhere
    amask = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
    masked_events = K.sum(amask) # counting zero sum events
    
    # ref denotes barycenter as that is our reference point
    x_ref = K.sum(K.sum(image, axis=(2, 3)) * (K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32') + 0.5) , axis=1)# sum for x position * x index
    y_ref = K.sum(K.sum(image, axis=(1, 3)) * (K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32') + 0.5), axis=1)
    z_ref = K.sum(K.sum(image, axis=(1, 2)) * (K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32') + 0.5), axis=1)
    x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) , x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
    y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref) , y_ref/sumtot)
    z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref), z_ref/sumtot)
    #reshape    
    x_ref = K.expand_dims(x_ref, 1)
    y_ref = K.expand_dims(y_ref, 1)
    z_ref = K.expand_dims(z_ref, 1)

    sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz) , K.ones_like(sumz))
        
    x = K.expand_dims(K.arange(x_shape), 0) # x indexes
    x = K.cast(K.expand_dims(x, 2), dtype='float32') + 0.5
    y = K.expand_dims(K.arange(y_shape), 0)# y indexes
    y = K.cast(K.expand_dims(y, 2), dtype='float32') + 0.5
  
    #barycenter for each z position
    x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
    y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)
    x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    #Angle Calculations
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

def discriminator(power=1.0, dformat='channels_last'):
    K.set_image_data_format(dformat)
    if dformat =='channels_last':
        dshape=(51, 51, 25) # sample shape
        daxis=3 # channel axis 
        baxis=-1 # axis for BatchNormalization
        daxis2=(1, 2, 3) # axis for sum
    else:
        dshape=(1, 51, 51, 25) 
        daxis=1 
        baxis=1 
        daxis2=(2, 3, 4)
    image=Input(shape=dshape)

    x = SeparableConv2D(16, (5, 5), padding='same', kernel_constraint=const)(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    #x = ZeroPadding2D((1, 1))(x)
    x = SeparableConv2D(8, (5, 5), padding='valid', kernel_constraint=const)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    #x = ZeroPadding2D((1, 1))(x)
    x = SeparableConv2D(8, (5, 5), padding='valid', kernel_constraint=const)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = SeparableConv2D(8, (5, 5), padding='valid', kernel_constraint=const)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = AveragePooling2D((2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
    dnn.summary()

    dnn_out = dnn(image)
    fake = Dense(1, activation='tanh',name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    inv_image = Lambda(K.pow, arguments={'a':1./power})(image) #get back original image
    ang = Lambda(ecal_angle, arguments={'daxis':daxis})(inv_image) # angle calculation
    ecal = Lambda(ecal_sum, arguments={'daxis':daxis2})(inv_image) # sum of energies
    add_loss = Lambda(count, arguments={'daxis':daxis2})(inv_image) # loss for bin counts
    Model(inputs=[image], outputs=[fake, aux, ang, ecal, add_loss]).summary()
    return Model(inputs=[image], outputs=[fake, aux, ang, ecal, add_loss])


def generator(latent_size=256, return_intermediate=False, dformat='channels_last'):
    if dformat =='channels_last':
        dim = (18, 18, 16) # shape for dense layer
        baxis=-1 # axis for BatchNormalization
    else:
        dim = (8, 9, 9,8)
        baxis=1
    K.set_image_data_format(dformat)
    loc = Sequential([
        Dense(5184, input_shape=(latent_size,)),
        Reshape(dim),
        UpSampling2D(size=(4, 4)),
        
        SeparableConv2D(8, (8, 8), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        #ZeroPadding2D((2, 2)),
        SeparableConv2D(8, (6, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        ####################################### added layers 
        
        #ZeroPadding2D((2, 2)),
        SeparableConv2D(8, (4, 4), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        #ZeroPadding2D((2, 2)),
        SeparableConv2D(8, (3, 3), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        #ZeroPadding2D((1, 1)),
        SeparableConv2D(8, (3, 3), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        #####################################  
        
        #ZeroPadding2D((1, 1)),
        SeparableConv2D(16, (2, 2), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        
        SeparableConv2D(25, (2, 2),  padding='valid', kernel_initializer='glorot_normal', use_bias=False),
        Activation('relu')
    ])
    latent = Input(shape=(latent_size, ))   
    fake_image = loc(latent)
    loc.summary()
    Model(input=[latent], output=[fake_image]).summary()
    return Model(input=[latent], output=[fake_image])

def main():
    dformat= 'channels_last'
    g= generator(dformat=dformat)
    d=discriminator(dformat=dformat)

if __name__ == "__main__":
    main()

                
