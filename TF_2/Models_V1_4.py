# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:54:24 2020

@author: frehm
"""
#Version1_4: activation function after conv3d_transpose layers

import tensorflow as tf

def generator(latent_size=200, keras_dformat='channels_first'):
    
    if keras_dformat =='channels_last':
        dim = (5,5,5,5)
    else:
        dim = (5,5,5,5)

    latent = tf.keras.Input(shape=(latent_size ))  #define Input     
    x = tf.keras.layers.Dense(5*5*5*5, input_dim=latent_size)(latent)   #shape (none, 625) #none is batch size
    x = tf.keras.layers.Reshape(dim) (x)   
    
    #1.Conv Block
    x = tf.keras.layers.Conv3D(64, (5, 5, 5), data_format=keras_dformat, use_bias=False, padding='same')(x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.BatchNormalization() (x)
    
    #2.Conv Block
    x = tf.keras.layers.Conv3DTranspose(32, (5,5,5), strides =(3,3,3), data_format=keras_dformat, padding="same") (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Conv3D(16, (5, 5, 5), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.BatchNormalization() (x)
        
    #3.Conv Block
    x = tf.keras.layers.Conv3DTranspose(32, (5,5,5), strides =(2,2,2), data_format=keras_dformat, padding="same", name="Conv_Trans_1") (x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.Conv3D(32, (4, 4, 4), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.BatchNormalization() (x)
        
    #4.Conv Block
    x = tf.keras.layers.Conv3D(16, (3, 3, 3), data_format=keras_dformat, padding='valid',  kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.LeakyReLU() (x)
    x = tf.keras.layers.BatchNormalization() (x)    
    
    #Output Block
    x = tf.keras.layers.Conv3D(1, (2, 2, 2), data_format=keras_dformat, padding='same', use_bias=False, kernel_initializer='glorot_normal') (x)
#    x = tf.keras.layers.Activation('relu') (x)
    
    return tf.keras.Model(inputs=[latent], outputs=x)   #Model mit Input und Output zurückgeben

#generator().summary()





def discriminator(keras_dformat='channels_first'):
    
    if keras_dformat =='channels_last':
        dshape=(25, 25, 25,1)
        daxis=(1,2,3)
    else:
        dshape=(1, 25, 25, 25)
        daxis=(2,3,4)
        
    image = tf.keras.layers.Input(shape=dshape)     #Input Image

    x = tf.keras.layers.Conv3D(32, (5, 5,5), data_format=keras_dformat, padding='same')(image)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
#    print("1",x.shape)

    x = tf.keras.layers.ZeroPadding3D((2, 2,2))(x)
    x = tf.keras.layers.Conv3D(32, (5, 5, 5), data_format=keras_dformat, padding='valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.ZeroPadding3D((2, 2, 2))(x)
    x = tf.keras.layers.Conv3D(16, (5, 5,5), data_format=keras_dformat, padding='valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.ZeroPadding3D((1, 1, 1))(x)
    x = tf.keras.layers.Conv3D(16, (5, 5, 5), data_format=keras_dformat, padding='valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.AveragePooling3D((2, 2, 2))(x)

    h = tf.keras.layers.Flatten()(x)

    disc = tf.keras.Model(image, h)  
    disc_out = disc(image)

    #Takes Network outputs as input
    fake = tf.keras.layers.Dense(1, activation='sigmoid', name='generation')(disc_out)   #Klassifikator true/fake
    aux = tf.keras.layers.Dense(1, activation='linear', name='auxiliary')(disc_out)       #Soll sich an E_in (Ep) annähern
    #Takes image as input
    ecal = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=daxis))(image)    #Energie, die im Netzwerk steckt, Summe über gesamtes NEtzwerk
    
    #tf.keras.Model(inputs=image, outputs=[fake, aux, ecal]).summary()
    return tf.keras.Model(inputs=image, outputs=[fake, aux, ecal])

#discriminator().summary()






