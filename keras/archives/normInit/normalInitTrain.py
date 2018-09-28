#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import keras
import argparse
import os
os.environ['LD_LIBRARY_PATH'] = os.getcwd()
from six.moves import range
import sys

import h5py 
import numpy as np
def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

if __name__ == '__main__':

    import keras.backend as K

    K.set_image_dim_ordering('tf')

    from keras.layers import Input
    from keras.models import Model
    from keras.optimizers import Adadelta, Adam, RMSprop
    from keras.utils.generic_utils import Progbar
    from sklearn.cross_validation import train_test_split

    import tensorflow as tf
    config = tf.ConfigProto(log_device_placement=True)
  
    from ecalvegan import generator
    from ecalvegan import discriminator 

    g_weights = 'params_generator_epoch_' 
    d_weights = 'params_discriminator_epoch_' 

    nb_epochs = 50 
    batch_size = 128
    latent_size = 200
    verbose = 'false'
    poly = [-0.03140347, 1.952197, 0.0827042 ]
    print (poly)
    generator=generator(latent_size)
    discriminator=discriminator()

    nb_classes = 2

    print('[INFO] Building discriminator')
    discriminator.summary()
    #discriminator.load_weights('veganweights/params_discriminator_epoch_019.hdf5')
    discriminator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[10, 0.3, 0.1]
        #loss=['binary_crossentropy', 'kullback_leibler_divergence']
    )

    # build the generator
    print('[INFO] Building generator')
    generator.summary()
    #generator.load_weights('veganweights/params_generator_epoch_019.hdf5')
    generator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )

    latent = Input(shape=(latent_size, ), name='combined_z')
     
    fake_image = generator( latent)

    discriminator.trainable = False
    fake, aux, ecal = discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ecal],
        name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[10, 0.3, 0.1]
    )


    d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
    e=d.get('target')
    X=np.array(d.get('ECAL'))
    y=(np.array(e[:,1]))
    print(X.shape)
    #print('*************************************************************************************')
    print(y)
    print('*************************************************************************************')
   
    #Y=np.sum(X, axis=(1,2,3))
    #print(Y)
    #print('*************************************************************************************')

    
   # remove unphysical values
    X[X < 1e-6] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    # tensorflow ordering
    X_train =np.array(np.expand_dims(X_train, axis=-1))
    X_test = np.array(np.expand_dims(X_test, axis=-1))
    y_train= np.array(y_train)/100
    y_test=np.array(y_test)/100
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print('*************************************************************************************')


    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    X_train = X_train.astype(np.float32)  
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    ecal_train = np.sum(X_train, axis=(1, 2, 3))
    ecal_test = np.sum(X_test, axis=(1, 2, 3))

    print(X_train.shape)
    print(X_test.shape)
    print(ecal_train.shape)
    print(ecal_test.shape)
    print('*************************************************************************************')
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        for index in range(nb_batches):
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, nb_batches))

            noise = np.random.normal(0, 1, (batch_size, latent_size))

            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            energy_batch = y_train[index * batch_size:(index + 1) * batch_size]
            ecal_batch = ecal_train[index * batch_size:(index + 1) * batch_size]

            print(image_batch.shape)
            print(ecal_batch.shape)
            sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.polyval(poly, sampled_energies)
            generated_images = generator.predict(generator_ip, verbose=0)

         #   loss_weights=[np.ones(batch_size), 0.05 * np.ones(batch_size)]
             
            real_batch_loss = discriminator.train_on_batch(image_batch, [bit_flip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [bit_flip(np.zeros(batch_size)), sampled_energies, ecal_ip])
                #    print(real_batch_loss)
                 #   print(fake_batch_loss)

#            fake_batch_loss = discriminator.train_on_batch(disc_in_fake, disc_op_fake, loss_weights)

            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)

            gen_losses = []

            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = np.polyval(poly, sampled_energies)

                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

        print('\nTesting for epoch {}:'.format(epoch + 1))

        noise = np.random.normal(0, 1, (nb_test, latent_size))

        sampled_energies = np.random.uniform(0.1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = generator.predict(generator_ip, verbose=False)
        ecal_ip = np.polyval(poly, sampled_energies)
        sampled_energies = np.squeeze(sampled_energies, axis=(1,))
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        ecal = np.concatenate((ecal_test, ecal_ip))
        print(ecal.shape)
        print(y_test.shape)
        print(sampled_energies.shape)
        aux_y = np.concatenate((y_test, sampled_energies), axis=0)
        print(aux_y.shape)
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        noise = np.random.normal(0, 1, (2 * nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (2 * nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = np.polyval(poly, sampled_energies)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(generator_ip,
                                                [trick, sampled_energies.reshape((-1, 1)), ecal_ip], verbose=False, batch_size=batch_size)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights('veganweights/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
        discriminator.save_weights('veganweights/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)

        pickle.dump({'train': train_history, 'test': test_history},
open('dcgan-history.pkl', 'wb'))
