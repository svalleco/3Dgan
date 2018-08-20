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
import argparse
import setGPU #if Caltech

def BitFlip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=200000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    
    Files =sorted( glob.glob(FileSearch))
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
    if len(out[1])==0:
        for ParticleName in Particles:
           out[1].append(Samples[ParticleName][NFiles])
    return out

# This functions loads data from a file and also does any pre processing
def GetDataAngle(datafile, xscale =1, yscale = 100, thetascale=10, phiscale=1):
    #get data for training                                                                                                                             
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    Y=f.get('energy')
    theta = f.get('theta')
    phi = f.get('phi')
    X=np.array(f.get('ECAL'))
    Y=(np.array(Y))/yscale
    theta = (np.array(theta))/thetascale
    phi = (np.array(phi))/phiscale
    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    theta = theta.astype(np.float32)
    phi = phi.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, theta, phi, ecal

def GanTrain3DAngle(discriminator, generator, datapath, EventsperFile, nEvents, WeightsDir, nb_epochs=30, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, theta_weight=0.1, phi_weight=0.1, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_generator_epoch_', xscale=1):
    start_init = time.time()
    verbose = True
    print('[INFO] Building discriminator')
    #discriminator.summary()
    discriminator.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, theta_weight, phi_weight, ecal_weight]
    )

    # build the generator
    print('[INFO] Building generator')
    #generator.summary()
    generator.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )
 
    # build combined Model
    latent = Input(shape=(3, latent_size, ), name='combined_z')   
    fake_image = generator( latent)
    discriminator.trainable = False
    fake, aux, theta, phi, ecal = discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, theta, phi, ecal],
        name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, theta_weight, phi_weight, ecal_weight]
    )

    # Getting Data
    Trainfiles, Testfiles = DivideFiles(datapath, nEvents=nEvents, EventsperFile = EventsperFile, datasetnames=["ECAL"], Particles =["Ele"])
 
    print(Trainfiles)
    print(Testfiles)
   
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, theta_test, phi_test, ecal_test = GetDataAngle(dtest)
       else:
           X_temp, Y_temp, theta_temp, phi_temp, ecal_temp = GetDataAngle(dtest)
           X_test = np.concatenate((X_test, X_temp))
           Y_test = np.concatenate((Y_test, Y_temp))
           theta_test = np.concatenate((theta_test, theta_temp))
           phi_test = np.concatenate((phi_test, phi_temp))
           ecal_test = np.concatenate((ecal_test, ecal_temp))

    print('Test Data loaded of shapes:')
    print(X_test.shape)
    print(Y_test.shape)
    print('*************************************************************************************')

    nb_test = X_test.shape[0]
    nb_train = EventsperFile * len(Trainfiles)# Total events in training files
    total_batches = nb_train / batch_size
    print('Total Training batches = {} with {} events'.format(total_batches, nb_train))

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    init_time = time.time()- start_init
    print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        X_train, Y_train, theta_train, phi_train, ecal_train = GetDataAngle(Trainfiles[0], xscale=xscale)
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
                X_temp, Y_temp, theta_temp, phi_temp, ecal_temp = GetDataAngle(Trainfiles[nb_file], xscale=xscale)
                #print("\nData file loaded..........",Trainfiles[nb_file])
                nb_file+=1
                X_left = X_train[(file_index * batch_size):]
                Y_left = Y_train[(file_index * batch_size):]
                theta_left = theta_train[(file_index * batch_size):]
                phi_left = phi_train[(file_index * batch_size):]
                ecal_left = ecal_train[(file_index * batch_size):]
                X_train = np.concatenate((X_left, X_temp))
                Y_train = np.concatenate((Y_left, Y_temp))
                theta_train = np.concatenate((theta_left, theta_temp))
                phi_train = np.concatenate((phi_left, phi_temp))
                ecal_train = np.concatenate((ecal_left, ecal_temp))
                nb_batches = int(X_train.shape[0] / batch_size)                
                #print("{} batches loaded..........".format(nb_batches))
                file_index = 0

            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            theta_batch = theta_train[(file_index * batch_size):(file_index + 1) * batch_size]
            phi_batch = phi_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            file_index +=1
            batch_shape=energy_batch.shape[0]
            noise = np.random.normal(0, 1, (batch_shape, 3, latent_size))
            cond = np.column_stack((energy_batch, theta_batch, phi_batch))
            cond = np.expand_dims(cond, axis=2)
            generator_ip = np.multiply(cond, noise)
            generated_images = generator.predict(generator_ip, verbose=0)        
            real_batch_loss = discriminator.train_on_batch(image_batch, [BitFlip(np.ones(batch_shape)), energy_batch, theta_batch, phi_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [BitFlip(np.zeros(batch_shape)), energy_batch, theta_batch, phi_batch, ecal_batch])
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_shape)
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_shape, 3, latent_size))
                generator_ip = np.multiply(cond, noise)
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, energy_batch.reshape((-1, 1)), theta_batch, phi_batch, ecal_batch]))
            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

        print('\nTesting for epoch {}:'.format(epoch + 1))
        noise = np.random.normal(0, 1, (nb_test,3, latent_size))
        cond = np.column_stack((Y_test, theta_test, phi_test))
        cond = np.expand_dims(cond, axis=2)
        generator_ip = np.multiply(cond, noise)
        generated_images = generator.predict(generator_ip, verbose=False)
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        theta = np.concatenate((theta_test, theta_test))
        phi = np.concatenate((phi_test, phi_test))
        ecal = np.concatenate((ecal_test, ecal_test))
        aux_y = np.concatenate((Y_test, Y_test), axis=0)
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y, theta, phi, ecal], verbose=False, batch_size=batch_size)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        noise = np.random.normal(0, 1, (2 * nb_test, 3, latent_size))
        cond = np.column_stack((aux_y, theta, phi))
        cond = np.expand_dims(cond, axis=2)

        generator_ip = np.multiply(cond, noise)
        trick = np.ones(2 * nb_test)
        generator_test_loss = combined.evaluate(generator_ip,
                                                [trick, aux_y, theta, phi, ecal], verbose=False, batch_size=batch_size)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s} | {4:5s}| {5:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f} | {5:<5.2f}| {6:<5.2f}'
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

        epoch_time = time.time()-epoch_start
        print("The {} epoch took {} seconds".format(epoch, epoch_time))
        pickle.dump({'train': train_history, 'test': test_history},
    open('dcgan-history-angle.pkl', 'wb'))

def get_parser_var():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--model', '-m', action='store', type=str, default='EcalCondGan', help='Model architecture to use.')
    parser.add_argument('--nbepochs', action='store', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=126, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    parser.add_argument('--nbperfile', action='store', type=int, default=10000, help='Number of events in a file.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='angleweights', help='Directory to store weights.')
    parser.add_argument('--xscale', action='store', type=int, default=2, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    return parser

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
    from EcalCondGan import generator, discriminator

     #Values to be set by user
    parser = get_parser_var()
    params = parser.parse_args()
    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    datapath = params.datapath#Data path on EOS CERN
    EventsperFile = params.nbperfile#Events in a file
    #nEvents = params.nbEvents#Total events for training
    nEvents = 100000
    EventsperFile = params.nbperfile
    weightdir = params.weightsdir
    xscale = params.xscale
    print(params)
 
    # Building discriminator and generator

    d=discriminator()
    g=generator(latent_size)
    GanTrain3DAngle(d, g, datapath, EventsperFile, nEvents, weightdir, nb_epochs=nb_epochs, batch_size=batch_size, gen_weight=2, aux_weight=0.1, theta_weight=0.1, phi_weight=0.1, ecal_weight=0.1, xscale = xscale)
    
