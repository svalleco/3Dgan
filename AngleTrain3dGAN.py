#Training for GAN
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
#from memory_profiler import profile
import keras.backend as K

K.set_image_dim_ordering('tf')

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils.generic_utils import Progbar
from sklearn.cross_validation import train_test_split

import tensorflow as tf
config = tf.ConfigProto(log_device_placement=True)
                                
def BitFlip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

def safe_mkdir(path):
    #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
      makedirs(path)
      print('{} is created to store weights'.format(path))
    except OSError as exception:
      if exception.errno != EEXIST:
          raise exception

def DivideFiles(FileSearch="/data/LCD/*/*.h4", nEvents=800000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    
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
        Sample=Samples[SampleName]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI

    return out

def GetDataAngle(datafile, xscale =1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4):
    #get data for training                                                                                                                             
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    ang = np.array(f.get(angtype))
    X=np.array(f.get('ECAL'))* xscale
    Y=np.array(f.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ang = ang.astype(np.float32)
    X = np.expand_dims(X, axis=-1)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, ang, ecal

def Gan3DTrainAngle(discriminator, generator, datapath, EventsperFile, nEvents, WeightsDir, pklfile, mod=0, nb_epochs=30, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, ang_weight=10, lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4):
    start_init = time.time()
    verbose = False    
    particle='Ele'
    f = [0.9, 0.1]

    print('[INFO] Building discriminator')
    #discriminator.summary()
    discriminator.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ang_weight, ecal_weight]
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
    fake, aux, ang, ecal= discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ang, ecal],
        name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ang_weight, ecal_weight]
    )

    # Getting Data
    Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
 
    print(Trainfiles)
    print(Testfiles)
    numTest = int(nEvents * f[1])
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh)
       else:
           if X_test.shape[0] < numTest:
              X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh)
              X_test = np.concatenate((X_test, X_temp))
              Y_test = np.concatenate((Y_test, Y_temp))
              ang_test = np.concatenate((ang_test, ang_temp))
              ecal_test = np.concatenate((ecal_test, ecal_temp))
    if X_test.shape[0] > numTest:
        X_test, Y_test, ang_test, ecal_test = X_test[:numTest], Y_test[:numTest], ang_test[:numTest], ecal_test[:numTest]
    else:
        numTest = X_test.shape[0]
    print('Test Data loaded of shapes:')
    print(X_test.shape)
    print(Y_test.shape)
    print('*************************************************************************************')
    print('Ang varies from {} to {} with mean {}'.format(np.amin(ang_test), np.amax(ang_test), np.mean(ang_test)))
    nb_test = numTest
   
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    init_time = time.time()- start_init
    print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        X_train, Y_train, ang_train, ecal_train = GetDataAngle(Trainfiles[0], xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh)
        nb_file=1
  
        epoch_gen_loss = []
        epoch_disc_loss = []
        index = 0
        total_batches = 0
        file_index=0
     
        while nb_file < len(Trainfiles):
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, total_batches))
            loaded_data = X_train.shape[0]
            used_data = file_index * batch_size
            if (loaded_data - used_data) < (batch_size + 1 ):
                X_temp, Y_temp, ang_temp, ecal_temp = GetDataAngle(Trainfiles[nb_file], xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh)
                nb_file+=1
                X_left = X_train[(file_index * batch_size):]
                Y_left = Y_train[(file_index * batch_size):]
                ang_left = ang_train[(file_index * batch_size):]
                ecal_left = ecal_train[(file_index * batch_size):]
                X_train = np.concatenate((X_left, X_temp))
                Y_train = np.concatenate((Y_left, Y_temp))
                ang_train = np.concatenate((ang_left, ang_temp))
                ecal_train = np.concatenate((ecal_left, ecal_temp))
                nb_batches = int(X_train.shape[0] / batch_size)                
                print("{} batches loaded..........".format(nb_batches))
                file_index = 0

            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            ang_batch = ang_train[(file_index * batch_size):(file_index + 1) * batch_size]
            file_index +=1
            noise = np.random.normal(0, 1, (batch_size, latent_size-1))
            noise = np.multiply(energy_batch.reshape(-1, 1), noise) # Same energy as G4
            generator_ip = np.concatenate((ang_batch.reshape(-1, 1), noise), axis=1)
            generated_images = generator.predict(generator_ip, verbose=0)
  
            real_batch_loss = discriminator.train_on_batch(image_batch, [BitFlip(np.ones(batch_size)), energy_batch, ang_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [BitFlip(np.zeros(batch_size)), energy_batch, ang_batch, ecal_batch])

            #print ('real_batch_loss', real_batch_loss)
            #print ('fake_batch_loss', fake_batch_loss)
            
            # if ecal sum has 100% loss then end the training
            if fake_batch_loss[4] == 100.0:
                print("Empty image with Ecal loss equal to 100.0")
                generator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
                discriminator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
                print ('real_batch_loss', real_batch_loss)
                print ('fake_batch_loss', fake_batch_loss)
                sys.exit()
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])
            trick = np.ones(batch_size)
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size-1))
                noise = np.multiply(energy_batch.reshape(-1, 1), noise)
                generator_ip = np.concatenate((ang_batch.reshape(-1, 1), noise), axis=1) # sampled angle same as g4 theta
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, energy_batch.reshape(-1, 1), ang_batch, ecal_batch]))
            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]
            epoch_gen_loss.append(generator_loss)
            #print ('generator_loss', generator_loss)
            index +=1
        print ('Total batches were {}'.format(index))
        print('Time taken by epoch{} was {} seconds.'.format(epoch, time.time()-epoch_start))
        print('\nTesting for epoch {}:'.format(epoch))
        test_start = time.time()
        noise = np.random.normal(0, 1, (nb_test, latent_size-1))
        noise = np.multiply(Y_test.reshape(-1, 1), noise)
        generator_ip = np.concatenate((ang_test.reshape(-1, 1), noise), axis=1)
        generated_images = generator.predict(generator_ip, verbose=False)
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        ang = np.concatenate((ang_test, ang_test))
        ecal = np.concatenate((ecal_test, ecal_test))
        aux_y = np.concatenate((Y_test, Y_test), axis=0)
                
        discriminator_test_loss = discriminator.evaluate( X, [y, aux_y, ang, ecal], verbose=False, batch_size=batch_size)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        noise = np.random.normal(0, 1, (2 * nb_test, latent_size - 1))
        noise = np.multiply(aux_y.reshape(-1, 1), noise)
        generator_ip = np.concatenate((ang.reshape(-1, 1), noise), axis=1)
        trick = np.ones(2 * nb_test)
        generator_test_loss = combined.evaluate(generator_ip,
                [trick, aux_y, ang, ecal], verbose=False, batch_size=batch_size)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)
        ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}'
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
        
        epoch_time = time.time()-test_start
        print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, epoch_time, WeightsDir))
        pickle.dump({'train': train_history, 'test': test_history},
    open(pklfile, 'wb'))

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--nbepochs', action='store', type=int, default=60, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=256, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/data/shared/gkhattak/*Measured3ThetaEscan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    parser.add_argument('--nbperfile', action='store', type=int, default=5000, help='Number of events in a file.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='angleweights', help='Directory to store weights.')
    parser.add_argument('--pklfile', action='store', type=str, default='3dgan-history.pkl', help='Pickle file to store losses.')
    parser.add_argument('--mod', action='store', type=int, default=0, help='How to calculate Ecal sum corressponding to energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--ascale', action='store', type=int, default=1, help='Multiplication factor for angle input')
    return parser

def main():
    #Architectures to import
    from AngleArch3dGAN import generator, discriminator

    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    datapath = params.datapath#Data path 
    EventsperFile = params.nbperfile#Events in a file
    nEvents = params.nbEvents
    fitmod = params.mod # used only for Ecal sum estimation corressponding to sampled primary energies
    ascale = params.ascale
    yscale = params.yscale

    #following can be changed if using from command line
        
    #weightdir = params.weightsdir
    weightdir = 'weights/3Dweights_1loss_25weight' # renamed to keep record
    #pklfile = params.pklfile
    pklfile = '3dgan-history-testing.pkl'
    #xscale = params.xscale
    xscale=1
    nb_epochs = 1
    
    print(params)

    # set for each training
    gen_weight = 3  # weight for generation loss
    aux_weight= 0.1  # weight for primary energy regression loss
    ang_weight= 25   # weight for angle loss
    ecal_weight = 0.1 # weight for ecal loss
    thresh = 0 # threshold for data
    angtype = 'mtheta'
 
    # Building discriminator and generator
    safe_mkdir(weightdir)
    d=discriminator()
    g=generator(latent_size)
    Gan3DTrainAngle(d, g, datapath, EventsperFile, nEvents, weightdir, pklfile, mod=fitmod, nb_epochs=nb_epochs, batch_size=batch_size,
                    latent_size=latent_size, gen_weight=gen_weight, aux_weight=aux_weight, ang_weight=ang_weight, ecal_weight=ecal_weight,
                    xscale = xscale, angscale=ascale, yscale=yscale, thresh=thresh, angtype=angtype)
    
if __name__ == '__main__':
    main()
