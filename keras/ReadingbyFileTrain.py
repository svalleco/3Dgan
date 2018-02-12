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
import glob
import h5py 
import numpy as np
import time
import math

def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=200000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))
    
    print ("Found {} files. ".format(len(Files)))
    Filesused = int(math.ceil(nEvents/EventsperFile))
    FileCount=0
   
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")

        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]

        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])

    SampleI=len(Samples.keys())*[int(0)]

    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName][:Filesused]
        NFiles=len(Sample)

        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI

    return out

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
  
    #Architectures to import
    from EcalEnergyGan import generator, discriminator 
   
    g_weights = 'params_generator_epoch_' 
    d_weights = 'params_discriminator_epoch_' 
  
    #Values to be set by user
    nb_epochs = 5 #Total Epochs
    batch_size = 128 #batch size
    latent_size = 200#latent vector size
    verbose = 'false'
    datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path
    num_events = 10000#Events in a file
    nEvents = 100000#Total events for training
    nb_classes = 2
    start_init = time.time()
 
    # Building discriminator and generator

    discriminator=discriminator()
    generator=generator(latent_size)
    print('[INFO] Building discriminator')
    discriminator.summary()
    discriminator.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[8, 0.2, 0.1]
    )

    # build the generator
    print('[INFO] Building generator')
    generator.summary()
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
        loss_weights=[8, 0.2, 0.1]
    )

    # Getting Data
    Trainfiles, Testfiles = DivideFiles(datapath, nEvents=nEvents, EventsperFile = num_events, datasetnames=["ECAL"], Particles =["Ele"])
 
    print(len(Trainfiles), len(Testfiles))
    print(Trainfiles[0])
 
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       d=h5py.File(dtest,'r')
       if index == 0:
           X_test = np.float32(d.get('ECAL'))
           Y_test = np.float32(d.get('target')[:,1]) 
 
       else:
           X_test = np.concatenate((X_test, np.float32(d.get('ECAL'))))
           Y_test = np.concatenate((Y_test, np.float32(d.get('target')[:,1])))

    # Preprocessing of test data
    X_test = np.expand_dims(X_test, axis=-1)
    Y_test= (Y_test)/100
 #   X_test = X_test.astype(np.float32)
 #   Y_test = y_test.astype(np.float32)
    ecal_test = np.sum(X_test, axis=(1, 2, 3))

    print('Test Data loaded of shapes:')
    print(X_test.shape)
    print(Y_test.shape)
    print('*************************************************************************************')

    nb_test = X_test.shape[0]
    nb_train = num_events * len(Trainfiles)# Total events in training files
    total_batches = nb_train / batch_size
    print('total batches = {} with {} events'.format(total_batches, nb_train))

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    init_time = time.time()- start_init
    print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        #Read First Train file
        d=h5py.File(Trainfiles[0],'r')
        X_train=np.float32(d.get('ECAL'))
        Y_train=np.float32(d.get('target')[:,1])
        # remove unphysical values
        X_train[X_train < 1e-6] = 0
        X_train = np.expand_dims(X_train, axis=-1)
        Y_train= (Y_train)/100
        ecal_train = np.sum(X_test, axis=(1, 2, 3))
        print("First train file {} is loaded with shapes:".format(Trainfiles[0]))
        print(X_train.shape)
        print(Y_train.shape)
        print('*************************************************************************************')
        nb_file=1
        nb_batches = int(X_train.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=total_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        file_index = 0
     
        for index in range(total_batches):
            
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, total_batches))
            loaded_data = X_train.shape[0]
            used_data = file_index * batch_size
            if (loaded_data - used_data) < batch_size + 1 and (nb_file < len(Trainfiles)):
            #if (index==nb_file * nb_batches) and (nb_file < len(Trainfiles)):
                d=h5py.File(Trainfiles[nb_file],'r')
                print("\nData file loaded..........",Trainfiles[nb_file])
                X_temp = np.expand_dims(np.float32(d.get('ECAL')), axis=-1)
                X_temp[X_temp < 1e-6] = 0
                Y_temp= np.float32(d.get('target')[:,1])/100
                nb_file+=1
                X_left = X_train[(file_index * batch_size):]
                Y_left = Y_train[(file_index * batch_size):]
                print(Y_left.shape)
                X_train = np.concatenate((X_left, X_temp))
                Y_train = np.concatenate((Y_left, Y_temp))
                ecal_train = np.sum(X_train, axis=(1, 2, 3))
                nb_batches = int(X_train.shape[0] / batch_size)                
                print("{} batches loaded..........".format(nb_batches))
                file_index = 0

            noise = np.random.normal(0, 1, (batch_size, latent_size))
            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            file_index +=1
            print(image_batch.shape)
            print(ecal_batch.shape)
            sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.multiply(2, sampled_energies)
            generated_images = generator.predict(generator_ip, verbose=0)        
            real_batch_loss = discriminator.train_on_batch(image_batch, [bit_flip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [bit_flip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
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
        sampled_energies = np.random.uniform(0.1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = generator.predict(generator_ip, verbose=False)
        ecal_ip = np.multiply(2, sampled_energies)
        sampled_energies = np.squeeze(sampled_energies, axis=(1,))
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        ecal = np.concatenate((ecal_test, ecal_ip))
        print(ecal.shape)
        print(Y_test.shape)
        print(sampled_energies.shape)
        aux_y = np.concatenate((Y_test, sampled_energies), axis=0)
        print(aux_y.shape)
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        noise = np.random.normal(0, 1, (2 * nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (2 * nb_test, 1))
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
        generator.save_weights('veganweights2/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
        discriminator.save_weights('veganweights2/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)

        epoch_time = time.time()-epoch_start
        print("The {} epoch took {} seconds".format(epoch, epoch_time))
        pickle.dump({'train': train_history, 'test': test_history},
open('dcgan-history2.pkl', 'wb'))
