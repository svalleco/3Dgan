#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code trains 3D gan and saves loss history and weights in result and weights directories respectively. GANutils is also required.
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
import glob
import h5py 
import numpy as np
import time
import math
import argparse
import analysis.utils.GANutils as gan # some common functions for gan

import keras.backend as K
K.set_image_dim_ordering('tf')

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils.generic_utils import Progbar
import tensorflow as tf
config = tf.ConfigProto(log_device_placement=True)
if 'nfshome/' in os.environ.get('HOME'): # Here a check for host can be used
  import setGPU #if Caltech

def main():

    #Architectures to import
    from EcalEnergyGan import generator, discriminator

    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    datapath = params.datapath#Data path 
    nEvents = params.nbEvents#Total events for training
    fitmod = params.mod# Fit to use
    weightdir = params.weightsdir # weight dir
    xscale = params.xscale #scaling of data
    pklfile = params.pklfile # loss history
    # Analysis
    analysis=params.analyse # if analysing
    energies =params.energies # Bins
    resultfile = params.resultfile # analysis result
    print(params)
    gan.safe_mkdir(weightdir)

    # Building discriminator and generator
    d=discriminator()
    g=generator(latent_size)
    Gan3DTrain(d, g, datapath, nEvents, weightdir, pklfile, resultfile, mod=fitmod, nb_epochs=nb_epochs, batch_size=batch_size, latent_size =latent_size , gen_weight=2, aux_weight=0.1, ecal_weight=0.1, xscale = xscale, analysis=analysis, energies=energies)

# This functions loads data from a file and also does any pre processing
def GetprocData(datafile, xscale =1, yscale = 100, limit = 1e-6):
    #get data for training
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    Y=f.get('target')
    X=np.array(f.get('ECAL'))
    Y=(np.array(Y[:,1]))
    X[X < limit] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X = xscale * X
    Y = Y/yscale
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, ecal

def Gan3DTrain(discriminator, generator, datapath, nEvents, WeightsDir, pklfile, resultfile, mod=0, nb_epochs=30, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, analysis=False, energies=[]):
    start_init = time.time()
    verbose = False
    particle = 'Ele'
    f = [0.9, 0.1]
    print('[INFO] Building discriminator')
    #discriminator.summary()
    discriminator.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ecal_weight]
    )

    # build the generator
    print('[INFO] Building generator')
    #generator.summary()
    generator.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )
 
    # build combined Model
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
        loss_weights=[gen_weight, aux_weight, ecal_weight]
    )

    # Getting Data
    Trainfiles, Testfiles = gan.DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
    print('The total data was divided in {} Train files and {} Test files'.format(len(Trainfiles), len(Testfiles)))
    nb_test = int(nEvents * f[1])
    
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ecal_test = GetprocData(dtest, xscale=xscale)
       else:
           if X_test.shape[0] < nb_test:
              X_temp, Y_temp, ecal_temp = GetprocData(dtest, xscale=xscale)
              X_test = np.concatenate((X_test, X_temp))
              Y_test = np.concatenate((Y_test, Y_temp))
              ecal_test = np.concatenate((ecal_test, ecal_temp))
    X_test, Y_test, ecal_test = X_test[:nb_test], Y_test[:nb_test], ecal_test[:nb_test]

    nb_train = int(nEvents * f[0]) #
    total_batches = int(nb_train / batch_size)
    print('In this experiment {} events will be used for training as {}batches'.format(nb_train, total_batches))
    print('{} events will be used for Testing'.format(nb_test))

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    analysis_history = defaultdict(list)
    
    init_time = time.time()- start_init
    print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        X_train, Y_train, ecal_train = GetprocData(Trainfiles[0], xscale=xscale)
        nb_file=1
        nb_batches = int(X_train.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=total_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        file_index = 0
                
        for index in np.arange(total_batches): # Training is controlled by number of batches for events=nEvents
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, total_batches))
            loaded_data = X_train.shape[0]
            used_data = file_index * batch_size
            if (loaded_data - used_data) < batch_size + 1 and (nb_file < len(Trainfiles)):
                X_temp, Y_temp, ecal_temp = GetprocData(Trainfiles[nb_file], xscale=xscale)
                print("\nData file loaded..........",Trainfiles[nb_file])
                nb_file+=1
                X_left = X_train[(file_index * batch_size):]
                Y_left = Y_train[(file_index * batch_size):]
                ecal_left = ecal_train[(file_index * batch_size):]
                X_train = np.concatenate((X_left, X_temp))
                Y_train = np.concatenate((Y_left, Y_temp))
                ecal_train = np.concatenate((ecal_left, ecal_temp))
                nb_batches = int(X_train.shape[0] / batch_size)                
                print("{} batches loaded..........".format(nb_batches))
                file_index = 0

            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            file_index +=1
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
      
            #ecal sum from fit
            ecal_ip = gan.GetEcalFit(sampled_energies, particle,mod, xscale)
            generated_images = generator.predict(generator_ip, verbose=0)        
            real_batch_loss = discriminator.train_on_batch(image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [gan.BitFlip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)
            gen_losses = []
            for _ in np.arange(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))
            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])
        print('The training took {} seconds.'.format(time.time()-epoch_start))
        print('\nTesting for epoch {}:'.format(epoch + 1))
        test_start=time.time()
        noise = np.random.normal(0.1, 1, (nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
        ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
        sampled_energies = np.squeeze(sampled_energies, axis=(1,))
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        ecal = np.concatenate((ecal_test, ecal_ip))
        aux_y = np.concatenate((Y_test, sampled_energies), axis=0)
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        noise = np.random.normal(0.1, 1, (2 * nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (2 * nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
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
        generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
        discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)
        print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, time.time()-test_start, WeightsDir))
        pickle.dump({'train': train_history, 'test': test_history},
        open(pklfile, 'wb'))
        if analysis:
            var = gan.sortEnergy([X_test, Y_test], ecal_test, energies, ang=0)
            noise = np.random.normal(0.1, 1, (nb_test, latent_size))
            generator_ip = np.multiply(Y_test.reshape((-1, 1)), noise)
            generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
                                    
            result = gan.OptAnalysisShort(var, generated_images, energies, ang=0)
            print('Analysing............')
            # All of the results correspond to mean relative errors on different quantities
            analysis_history['total'].append(result[0]) 
            analysis_history['energy'].append(result[1])
            analysis_history['moment'].append(result[2])
            print('Result = ', result)
            pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--model', '-m', action='store', type=str, default='EcalEnergyGan', help='Model architecture to use.')
    parser.add_argument('--nbepochs', action='store', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    #parser.add_argument('--datapath', action='store', type=str, default='/eos/project/d/dshep/LCD/V1/*scan/*.h5', help='HDF5 files to train from.') # CERN EOS
    parser.add_argument('--datapath', action='store', type=str, default='/bigdata/shared/LCD/NewV1/*scan/*.h5', help='HDF5 files to train from.') # Caltech
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    #parser.add_argument('--nbperfile', action='store', type=int, default=10000, help='Number of events in a file.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='weights/3dganWeights', help='Directory to store weights.')
    parser.add_argument('--mod', action='store', type=int, default=1, help='How to calculate Ecal sum corressponding to energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=100, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--pklfile', action='store', type=str, default='results/3dgan_history.pkl', help='File to save losses.')
    parser.add_argument('--resultfile', action='store', type=str, default='results/3dgan_analysis.pkl', help='File to save losses.')
    parser.add_argument('--analyse', action='store_true', default=False, help='Whether or not to perform analysis')
    parser.add_argument('--energies', action='store', type=int, default=[100, 200, 300, 400], help='Energy bins for analysis')
    return parser

if __name__ == '__main__':
    main()
