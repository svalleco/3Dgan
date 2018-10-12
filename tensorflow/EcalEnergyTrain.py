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


def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--nbepochs', action='store', type=int, default=1, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/eos/project/d/dshep/LCD/V1/*scan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--weightsdir', action='store', type=str, default='weights2D', help='Directory to store weights.')
    return parser

if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.python.keras.layers import Input
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.optimizers import Adadelta, Adam, RMSprop
    from sklearn.model_selection import train_test_split

    parser = get_parser()
    params = parser.parse_args()
    datapath = params.datapath #Data path on EOS CERN
    weightdir = params.weightsdir

    # tf.flags.DEFINE_string("d", 'test', "data file")
    # tf.flags.DEFINE_integer("bs", 128, "inference batch size")
    # tf.flags.DEFINE_integer("num_inter_threads", 1, "number of inter_threads")
    # tf.flags.DEFINE_integer("num_intra_threads", 56, "number of intra_threads")
    # tf.flags.DEFINE_integer("num_epochs", 2, "number of epochs")
    # FLAGS = tf.flags.FLAGS

    # session_config = tf.ConfigProto(log_device_placement=True, inter_op_parallelism_threads=FLAGS.num_inter_threads, intra_op_parallelism_threads=FLAGS.num_intra_threads)
    # session = tf.Session(config=session_config)
    # K.set_session(session)
  
    from EcalEnergyGan import generator, discriminator
    #from EcalEnergyGan_16f import generator, discriminator

    g_weights = 'params_generator_epoch_' 
    d_weights = 'params_discriminator_epoch_' 
    keras_dformat = 'channels_first'
    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    
    generator=generator(latent_size,keras_dformat=keras_dformat)
    discriminator=discriminator(keras_dformat=keras_dformat)

    nb_classes = 2
    print (tf.__version__)
    print('[INFO] Building discriminator')
    discriminator.trainable = False
    discriminator.summary()
    discriminator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[6, 0.2, 0.1]
        #loss=['binary_crossentropy', 'kullback_leibler_divergence']
    )

    # build the generator
    print('[INFO] Building generator')
    generator.summary()
    generator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )

    latent = Input(shape=(latent_size, ), name='combined_z')
     
    fake_image = generator( latent)

    fake, aux, ecal = discriminator(fake_image)
    combined = Model(
        [latent],
        [fake, aux, ecal],
        name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[6, 0.2, 0.1]
    )

    d=h5py.File(datapath,'r')
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)

    # tensorflow ordering
    X_train =np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
  
    print(X_train.shape)
    if keras_dformat !='channels_last':
       X_train =np.moveaxis(X_train, -1, 1)
       X_test = np.moveaxis(X_test, -1,1)

    y_train= y_train/100
    y_test=y_test/100
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
    if keras_dformat =='channels_last':
        ecal_train = np.sum(X_train, axis=(1, 2, 3))
        ecal_test = np.sum(X_test, axis=(1, 2, 3))
    else:
        ecal_train = np.sum(X_train, axis=(2, 3, 4))
        ecal_test = np.sum(X_test, axis=(2, 3, 4))

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

        epoch_gen_loss = []
        epoch_disc_loss = []
        for index in range(nb_batches):
            print('processed {}/{} batches'.format(index + 1, nb_batches))

            noise = np.random.normal(0, 1, (batch_size, latent_size))

            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            energy_batch = y_train[index * batch_size:(index + 1) * batch_size]
            ecal_batch = ecal_train[index * batch_size:(index + 1) * batch_size]

            print(image_batch.shape)
            print(ecal_batch.shape)
            sampled_energies = np.random.uniform(1, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.multiply(2, sampled_energies)
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
                sampled_energies = np.random.uniform(1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = np.multiply(2, sampled_energies)

                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

        print('\nTesting for epoch {}:'.format(epoch + 1))

        noise = np.random.normal(0, 1, (nb_test, latent_size))

        sampled_energies = np.random.uniform(1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = generator.predict(generator_ip, verbose=False)
        ecal_ip = np.multiply(2, sampled_energies)
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
        sampled_energies = np.random.uniform(1, 5, (2 * nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = np.multiply(2, sampled_energies)

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
        generator.save_weights(weightsdir + '/generator_{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
        discriminator.save_weights(weightsdir + '/discriminator{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)

        pickle.dump({'train': train_history, 'test': test_history},
open('3dgan-history.pkl', 'wb'))
