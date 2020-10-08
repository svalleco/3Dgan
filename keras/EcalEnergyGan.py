import sys
import h5py

from h5py import File as HDF5File
import numpy as np
import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_first')

def ecal_sum(image):
    sum = K.sum(image, axis=(2, 3))
    return sum
   

def discriminator():

    image = tf.keras.Input(shape=(1, 25, 25))

    x = tf.keras.layers.Conv2D(32, (5,5), data_format='channels_first', padding='same')(image)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.ZeroPadding2D((2,2))(x)
    x = tf.keras.layers.Conv2D(8, (5, 5), data_format='channels_first', padding='valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.ZeroPadding2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (5,5), data_format='channels_first', padding='valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(8, (5, 5), data_format='channels_first', padding='valid')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    h = tf.keras.layers.Flatten()(x)

    dnn = tf.keras.Model(image, h)

    image = tf.keras.Input(shape=(1, 25, 25))

    dnn_out = dnn(image)


    fake = tf.keras.layers.Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = tf.keras.layers.Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ecal =tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=(2, 3)))(image)
    tf.keras.Model(inputs=image, outputs=[fake, aux, ecal]).summary()
    return tf.keras.Model(inputs=image, outputs=[fake, aux, ecal])

def generator(latent_size=1024, return_intermediate=False):

    loc = tf.keras.Sequential([
        tf.keras.layers.Dense(64 * 7, input_dim=latent_size),
        tf.keras.layers.Reshape((8, 7,8)),

        tf.keras.layers.Conv2D(64, (6, 8), data_format='channels_first', padding='same',  kernel_initializer='he_uniform'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.UpSampling2D(size=(2, 2)),

        tf.keras.layers.ZeroPadding2D((2, 0)),
        tf.keras.layers.Conv2D(6, (5, 8), data_format='channels_first',  kernel_initializer='he_uniform'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.UpSampling2D(size=(2, 3)),

        tf.keras.layers.ZeroPadding2D((0,3)),
        tf.keras.layers.Conv2D(6, (3, 8), data_format='channels_first',  kernel_initializer='he_uniform'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(1, (2, 2), data_format='channels_first', use_bias=False,  kernel_initializer='glorot_normal'),
        tf.keras.layers.Activation('relu')
    ])
   
    latent = tf.keras.Input(shape=(latent_size, ))
     
    fake_image = loc(latent)

    tf.keras.Model(inputs=[latent], outputs=fake_image).summary()
    return tf.keras.Model(inputs=[latent], outputs=fake_image)
