import numpy as np
import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)
from keras.models import Model, Sequential
import math
import tensorflow as tf

# calculate sum of intensities
def ecal_sum(image, daxis):
    sum = K.sum(image, axis=daxis)
    return sum

# counts for various bin entries   
def count(image, daxis):
    limits=[0.05, 0.03, 0.02, 0.0125, 0.008, 0.003] # bin boundaries used
    bin1 = K.sum(K.tf.where(image > limits[0], K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin2 = K.sum(K.tf.where(K.tf.logical_and(image < limits[0], image > limits[1]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin3 = K.sum(K.tf.where(K.tf.logical_and(image < limits[1], image > limits[2]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin4 = K.sum(K.tf.where(K.tf.logical_and(image < limits[2], image > limits[3]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin5 = K.sum(K.tf.where(K.tf.logical_and(image < limits[3], image > limits[4]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin6 = K.sum(K.tf.where(K.tf.logical_and(image < limits[4], image > limits[5]), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin7 = K.sum(K.tf.where(K.tf.logical_and(image < limits[5], image > 0.0), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bin8 = K.sum(K.tf.where(K.tf.equal(image, 0.0), K.ones_like(image), K.zeros_like(image)), axis=daxis)
    bins = K.expand_dims(K.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1), axis=-1)
    return bins

def discriminator(power=1.0, dformat='channels_last'):
    K.set_image_data_format(dformat)
    if dformat =='channels_last':
        dshape=(51, 51, 25,1) # sample shape
        daxis=4 # channel axis 
        baxis=-1 # axis for BatchNormalization
        daxis2=(1, 2, 3) # axis for sum
    else:
        dshape=(1, 51, 51, 25) 
        daxis=1 
        baxis=1 
        daxis2=(2, 3, 4)
    image=Input(shape=dshape)

    x = Conv3D(16, (5, 6, 6), padding='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((0, 0, 1))(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=baxis, epsilon=1e-6)(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
    dnn.summary()

    dnn_out = dnn(image)
    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ang = Dense(1, activation='linear', name='ang')(dnn_out)
    inv_image = Lambda(K.pow, arguments={'a':1./power})(image) #get back original image
    #ang = Lambda(ecal_angle, arguments={'daxis':daxis})(inv_image) # angle calculation
    ecal = Lambda(ecal_sum, arguments={'daxis':daxis2})(inv_image) # sum of energies
    add_loss = Lambda(count, arguments={'daxis':daxis2})(inv_image) # loss for bin counts
    Model(inputs=[image], outputs=[fake, aux, ang, ecal, add_loss]).summary()
    return Model(inputs=[image], outputs=[fake, aux, ang, ecal, add_loss])


def generator(latent_size=256, return_intermediate=False, dformat='channels_last'):
    if dformat =='channels_last':
        dim = (9,9,8,8) # shape for dense layer
        baxis=-1 # axis for BatchNormalization
    else:
        dim = (8, 9, 9,8)
        baxis=1
    K.set_image_data_format(dformat)
    loc = Sequential([
        Dense(5184, input_shape=(latent_size,)),
        Reshape(dim),
        UpSampling3D(size=(6, 6, 6)),
        
        Conv3D(8, (6, 6, 8), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        ####################################### added layers 
        
        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        ZeroPadding3D((2, 2, 1)),
        Conv3D(6, (4, 4, 6), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),

        ZeroPadding3D((1, 1, 0)),
        Conv3D(6, (3, 3, 5), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        BatchNormalization(axis=baxis, epsilon=1e-6),
        
        #####################################  
        
        ZeroPadding3D((1, 1,0)),
        Conv3D(6, (3, 3, 3), padding='valid', kernel_initializer='he_uniform'),
        Activation('relu'),
        
        Conv3D(1, (2, 2, 2),  padding='valid', kernel_initializer='glorot_normal'),
        Activation('relu')
    ])
    latent = Input(shape=(latent_size, ))   
    fake_image = loc(latent)
    loc.summary()
    Model(input=[latent], output=[fake_image]).summary()
    return Model(input=[latent], output=[fake_image])

def main():
    dformat= 'channels_first'
    g= generator(dformat=dformat)
    d=discriminator(dformat=dformat)

if __name__ == "__main__":
    main()

                
