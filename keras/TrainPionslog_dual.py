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
from numpy import ma

# Computing log only when not zero
def logftn(x, f1, f2):
    select= np.where(x>0)
    x[select] = f1 * np.log10(x[select]) + f2 
    return x

def BitFlip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

def DivideFiles(FileSearch="/data/LCD/*/*.h5", Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    
    Files =sorted( glob.glob(FileSearch))
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
        Sample=Samples[SampleName]
        NFiles=len(Sample)

        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI

    return out

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
    X = logftn(X, 1.0/6.0, 1.0)
    return X, Y, ecal

def GetEcalFit(sampled_energies, particle='Ele', mod=0, xscale=1):
    if mod==0:
       return np.multiply(2, sampled_energies)
    elif mod==1:
       if particle == 'Ele':
         root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale
       elif particle == 'Pi0':
         root_fit = [0.0085, -0.094, 2.051]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale


def Gan3DTrain(discriminator1, generator1, discriminator2, generator2, datapath, nEvents, WeightsDir, pklfile, mod=0, nb_epochs=30, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1):
    start_init = time.time()
    verbose = False
    noise_init = 1e-6
    f=[0.9, 0.1]
    change=20
    print('[INFO] Building discriminator1')
    #discriminator.summary()
    discriminator1.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        clipvalue = 10000,
        loss_weights=[gen_weight, aux_weight, ecal_weight]
    )

    # build the generator
    print('[INFO] Building generator1')
    #generator.summary()
    generator1.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )
 
    # build combined Model
    latent = Input(shape=(latent_size, ), name='combined_z')   
    fake_image = generator1( latent)
    discriminator1.trainable = False
    fake, aux, ecal = discriminator1(fake_image)
    combined1 = Model(
        input=[latent],
        output=[fake, aux, ecal],
        name='combined_model'
    )
    combined1.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        #clipvalue= 10000,
        loss_weights=[gen_weight, aux_weight, ecal_weight]
    )

    print('[INFO] Building discriminator2')
    #discriminator.summary()
    discriminator2.compile(
         optimizer=RMSprop(),
         loss=['binary_crossentropy', 'mean_absolute_percentage_error'],
         #clipvalue = 10000,
         loss_weights=[gen_weight, aux_weight]
    )

    # build the generator
    print('[INFO] Building generator2')
    #generator.summary()
    generator2.compile(
         optimizer=RMSprop(),
         loss='binary_crossentropy'
    )

    # build combined Model
    latent = Input(shape=(latent_size, ), name='combined_z')
    fake_image = generator2( latent)
    discriminator2.trainable = False
    fake, aux = discriminator2(fake_image)
    combined2 = Model(
          input=[latent],
          output=[fake, aux],
          name='combined_model'
    )
    combined2.compile(
          optimizer=RMSprop(),
          loss=['binary_crossentropy', 'mean_absolute_percentage_error'],
          #clipvalue= 10000,
          loss_weights=[gen_weight, aux_weight]
    )
                                                        
    # Getting Data
    Trainfiles, Testfiles = DivideFiles(datapath, Fractions=f, datasetnames=["ECAL"], Particles =["Pi0"])
 
    print(len(Trainfiles))
    print(len(Testfiles))
    nb_test= int(nEvents * f[1])
    print(nb_test) 
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ecal_test = GetprocData(dtest, xscale=xscale)
       else:
           print(X_test.shape[0])
           if X_test.shape[0] < nb_test:
              X_temp, Y_temp, ecal_temp = GetprocData(dtest, xscale=xscale)
              X_test = np.concatenate((X_test, X_temp))[:nb_test]
              Y_test = np.concatenate((Y_test, Y_temp))[:nb_test]
              ecal_test = np.concatenate((ecal_test, ecal_temp))[:nb_test]

    print('Test Data loaded of shapes:')
    print(X_test.shape)
    print(Y_test.shape)
    #print(Y_test[:10])
    print('*************************************************************************************')
    
    nb_test = X_test.shape[0]
    print('The data now varies from {} to {}'.format(np.amax(X_test), np.amin(X_test)))
    nb_train = nEvents * f[0]# Total events in training files
    total_batches = int(nb_train / batch_size)
    print('Total Training batches = {} with {} events'.format(total_batches, nb_train))

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    init_time = time.time()- start_init
    d = discriminator2
    g = generator2
    c = combined2
    Flag = False
    print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        if epoch == change:
            print('Changing Networks')
            d = discriminator1
            g = generator1
            c = combined1
            Flag = True
            d.load_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch - 1))
            g.load_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch - 1))
        epoch_start = time.time()
        nb_file=0
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        X_train, Y_train, ecal_train = GetprocData(Trainfiles[nb_file], xscale=xscale)
        #print(Y_train[:10])
        nb_file+=1
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
            if (loaded_data - used_data) < (batch_size + 1 ):
                X_temp, Y_temp, ecal_temp = GetprocData(Trainfiles[nb_file], xscale=xscale)
                #print("\n{} Data file loaded..........{}".format(nb_file, Trainfiles[nb_file]))
                nb_file+=1
                X_left = X_train[(file_index * batch_size):]
                Y_left = Y_train[(file_index * batch_size):]
                ecal_left = ecal_train[(file_index * batch_size):]
                X_train = np.concatenate((X_left, X_temp))
                Y_train = np.concatenate((Y_left, Y_temp))
                ecal_train = np.concatenate((ecal_left, ecal_temp))
                nb_batches = int(X_train.shape[0] / batch_size)                
                #print("{} batches loaded..........".format(nb_batches))
                file_index = 0

            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            file_index +=1
            #print('The image data for {} batch is from {} to {}'.format(index, np.amin(image_batch), np.amax(image_batch)))
            #print('Energy batch = {}'.format(energy_batch[:5]))
            #print('Ecal batch = {}'.format(ecal_batch[:5]))
            noise = np.random.normal(0, noise_init, (batch_size, latent_size))
            sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            #print('The sampled energies are {}'.format(sampled_energies[:5]))
            #ecal sum from fit
            ecal_ip = GetEcalFit(sampled_energies, 'Pi0',mod, xscale)
            #print('Ecal from fit = {}'.format(ecal_ip[:5]))
            generated_images = g.predict(generator_ip, verbose=0) 
            #print('ecal batch', ecal_batch.shape)
            #disc_out = discriminator.predict(image_batch)
            #print('disc out', len(disc_out), len(disc_out[0]))
           # ecal_sum = np.sum(np.exp(-6 *image_batch), axis= (1, 2, 3))
           # print('ecal sum', ecal_sum[:5])       
            #print('The generated data for {} batch is from {} to {}'.format(index, np.amin(generated_images), np.amax(generated_images)))
            if Flag:
                 real_batch_loss = d.train_on_batch(image_batch, [BitFlip(np.ones(batch_size)), energy_batch, ecal_batch])
            else:
                 real_batch_loss = d.train_on_batch(image_batch, [BitFlip(np.ones(batch_size)), energy_batch])
            #print('Real batch loss = {}'.format(real_batch_loss))
            #print('ecal ip',ecal_ip[:5])
            #disc_out = d.predict(image_batch)
            #print('disc out for data', disc_out[2][:5])

            #disc_out = d.predict(generated_images)
            #print('disc out for generated', disc_out[2][:5])
            if Flag:
                fake_batch_loss = d.train_on_batch(generated_images, [BitFlip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            else:
                fake_batch_loss = d.train_on_batch(generated_images, [BitFlip(np.zeros(batch_size)), sampled_energies])
            #print('Fake batch loss = {}'.format(fake_batch_loss))
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, noise_init, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = GetEcalFit(sampled_energies, 'Pi0', mod, xscale)
                if Flag:
                    gen_loss= c.train_on_batch(
                       [generator_ip],
                       [trick, sampled_energies.reshape((-1, 1)), ecal_ip])
                else:
                    gen_loss= c.train_on_batch(
                       [generator_ip],
                       [trick, sampled_energies.reshape((-1, 1))])
                    
                gen_losses.append(gen_loss)
                #print('Generator batch loss = {}'.format(gen_loss))
                #com_out = c.predict(generator_ip)
                #print("#################Generator########################")
                #print('ecal ip',ecal_ip[:5])
                #print('com out', com_out[2][:5])

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

        print('\nTesting for epoch {}:'.format(epoch + 1))
        noise = np.random.normal(0, noise_init, (nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = g.predict(generator_ip, verbose=False)
        ecal_ip = GetEcalFit(sampled_energies, 'Pi0', mod, xscale)
        sampled_energies = np.squeeze(sampled_energies, axis=(1,))
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        ecal = np.concatenate((ecal_test, ecal_ip))
        aux_y = np.concatenate((Y_test, sampled_energies), axis=0)
        if Flag:
            discriminator_test_loss = d.evaluate(
                X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)
        else:
            discriminator_test_loss = d.evaluate(
                                X, [y, aux_y], verbose=False, batch_size=batch_size)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
       # print( discriminator_train_loss)
        noise = np.random.normal(0, noise_init, (2 * nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (2 * nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = GetEcalFit(sampled_energies, 'Pi0', mod, xscale)
        trick = np.ones(2 * nb_test)
        if Flag:
           generator_test_loss = c.evaluate(generator_ip,
                [trick, sampled_energies.reshape((-1, 1)), ecal_ip], verbose=False, batch_size=batch_size)
        else:
           generator_test_loss = c.evaluate(generator_ip,
                [trick, sampled_energies.reshape((-1, 1))], verbose=False, batch_size=batch_size)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        #print (generator_train_loss)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        if Flag:
            print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format('component', *d.metrics_names))
            print('-' * 65)

            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f} '
        else:
            print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format('component', *d.metrics_names))
            print('-' * 65)
            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f} '
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        g.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
        d.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)

        epoch_time = time.time()-epoch_start
        print("The {} epoch took {} seconds".format(epoch, epoch_time))
        pickle.dump({'train': train_history, 'test': test_history},
    open(pklfile, 'wb'))

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--model', '-m', action='store', type=str, default='EcalEnergyGan', help='Model architecture to use.')
    parser.add_argument('--nbepochs', action='store', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/eos/project/d/dshep/LCD/V1/*scan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    parser.add_argument('--nbperfile', action='store', type=int, default=10000, help='Number of events in a file.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='veganweights', help='Directory to store weights.')
    parser.add_argument('--mod', action='store', type=int, default=0, help='How to calculate Ecal sum corressponding to energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--pklfile', action='store', type=str, default='dcgan-history.pkl', help='File to save losses.')
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
  
    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    model = params.model
    #nb_epochs = params.nbepochs #Total Epochs
    nb_epochs = 50
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Data path on Caltech
    #datapath = params.datapath#Data path on EOS CERN default
    EventsperFile = params.nbperfile#Events in a file
    nEvents = params.nbEvents#Total events for training
    #fitmod = params.mod
    fitmod = 1
    #weightdir = params.weightsdir
    weightdir = 'pionweights3'
    #xscale = params.xscale
    xscale = 1 
    #pklfile = params.pklfile
    pklfile = 'dcgan-pion-history3.pkl'
    print(params)
    
    from EcalEnergyGanlog import generator, discriminator
    from EnergyGanlog import generator as generator2
    from EnergyGanlog import discriminator as discriminator2
    # Building discriminator and generator
    d1=discriminator()
    g1=generator(latent_size)
    d2=discriminator2()
    g2=generator2(latent_size)
    Gan3DTrain(d1, g1, d2, g2, datapath, nEvents, weightdir, pklfile, mod=fitmod, nb_epochs=nb_epochs, batch_size=batch_size, gen_weight=2, aux_weight=0.1, ecal_weight=0.2, xscale = xscale)
    
