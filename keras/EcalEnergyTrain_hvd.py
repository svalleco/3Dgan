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
import random
import psutil
import socket
import time
from tensorflow.python.client import timeline
from keras.callbacks import CallbackList
import analysis.utils.GANutils as gan # some common functions for gan

#import setGPU #if Caltech
def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception


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
    return out

# This functions loads data from a file and also does any pre processing
def GetData(datafile, xscale =1, yscale = 100, dimensions = 3, keras_dformat="channels_last"):
    #get data for training                                                                                         
    if hvd.rank()==0:
        print('Loading Data from .....', datafile)                              
    f=h5py.File(datafile,'r')
    
    X=np.array(f.get('ECAL'))
    
    Y=f.get('target')
    Y=(np.array(Y[:,1]))

    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    if dimensions == 2:
        X = np.sum(X, axis=(1))
        X = xscale * X

    Y = Y.astype(np.float32)
    Y = Y/yscale
    if keras_dformat !='channels_last':
       X =np.moveaxis(X, -1, 1)
       ecal = np.sum(X, axis=(2, 3, 4))
    else:
       ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, ecal


def GetEcalFit(sampled_energies, mod=1, xscale=1):
    if mod==0:
      return np.multiply(2, sampled_energies)
    elif mod==1:
      root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
      ratio = np.polyval(root_fit, sampled_energies)
      return np.multiply(ratio, sampled_energies) * xscale


def genbatches(a,n):
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n]


def randomize(a, b, c):
    assert a.shape[0] == b.shape[0]
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    return shuffled_a, shuffled_b, shuffled_c


def GanTrain(discriminator, generator, opt,run_options, run_metadata, global_batch_size, warmup_epochs, datapath, EventsperFile, nEvents, WeightsDir, resultfile, energies,mod=0, nb_epochs=30, batch_size=128, latent_size=128, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_generator_epoch_', xscale=1, verbose=True, keras_dformat='channels_last', analysis=True):
    start_init = time.time()
    verbose = False
    if hvd.rank()==0:
        print('[INFO] Building discriminator')
    
    discriminator.compile(
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ecal_weight],options=run_options,run_metadata=run_metadata
    )

    #build the generator
    if hvd.rank()==0:
        print('[INFO] Building generator')
    
    generator.compile(
        optimizer=opt,
        loss='binary_crossentropy',options=run_options,run_metadata=run_metadata
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
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ecal_weight],options=run_options,run_metadata=run_metadata
    )
    discriminator.trainable = True # workaround for a k2 bug
    
    gcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])
   
    dcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    ccb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    gcb.set_model( generator )
    dcb.set_model( discriminator )
    ccb.set_model( combined )

    gcb.on_train_begin()
    dcb.on_train_begin()
    ccb.on_train_begin()
    

    datapath = '/eos/user/g/gkhattak/FixedAngleData/*.h5'
    # Getting Data
    Trainfiles, Testfiles = DivideFiles(datapath, nEvents=nEvents, EventsperFile = EventsperFile, datasetnames=["ECAL"], Particles =["Ele"])
    print(Trainfiles)
    print(Testfiles)
    if hvd.rank()==0:
        print("Train files: {0} \nTest files: {1}".format(Trainfiles, Testfiles))
    
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ecal_test = GetData(dtest, keras_dformat=keras_dformat)
       else:
           X_temp, Y_temp, ecal_temp = GetData(dtest, keras_dformat=keras_dformat)
           X_test = np.concatenate((X_test, X_temp))
           Y_test = np.concatenate((Y_test, Y_temp))
           ecal_test = np.concatenate((ecal_test, ecal_temp))
    
    for index, dtrain in enumerate(Trainfiles):
        if index == 0:
            X_train, Y_train, ecal_train = GetData(dtrain, keras_dformat=keras_dformat)
        else:
            X_temp, Y_temp, ecal_temp = GetData(dtrain, keras_dformat=keras_dformat)
            X_train = np.concatenate((X_train, X_temp))
            Y_train = np.concatenate((Y_train, Y_temp))
            ecal_train = np.concatenate((ecal_train, ecal_temp))

    print("On hostname {0} - After init using {1} memory".format(socket.gethostname(), psutil.Process(os.getpid()).memory_info()[0]))

    nb_test = X_test.shape[0]
    assert X_train.shape[0] == EventsperFile * len(Trainfiles), "# Total events in training files"
    nb_train = X_train.shape[0]# Total events in training files
    total_batches = int(nb_train / global_batch_size)
    if hvd.rank()==0:
        print('Total Training batches = {} with {} events'.format(total_batches, nb_train))

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    if hvd.rank()==0:
        print('Initialization time was {} seconds'.format(time.time() - start_init))

    for epoch in range(nb_epochs):
        epoch_start = time.time()
        if hvd.rank()==0:
           print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        
        randomize(X_train, Y_train, ecal_train)

        epoch_gen_loss = []
        epoch_disc_loss = []
        
        image_batches = genbatches(X_train, batch_size)
        energy_batches = genbatches(Y_train, batch_size)
        ecal_batches = genbatches(ecal_train, batch_size)
        

        for index in range(total_batches):
            start = time.time()         
            image_batch = next(image_batches)
            energy_batch = next(energy_batches)
            ecal_batch = next(ecal_batches)

            noise = np.random.normal(0, 1, (batch_size, latent_size))
            sampled_energies = np.random.uniform(0.1, 5,( batch_size, 1))
            generator_ip = np.multiply(sampled_energies, noise)
            # ecal sum from fit
            ecal_ip = GetEcalFit(sampled_energies, mod, xscale)

            generated_images = generator.predict(generator_ip, verbose=0)        
            real_batch_loss = discriminator.train_on_batch(image_batch, [BitFlip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [BitFlip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            #print('real batch loss ={}'.format(real_batch_loss))
            #print('fake batch loss ={}'.format(fake_batch_loss))
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = GetEcalFit(sampled_energies, mod, xscale)
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))
            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

            if (index % 100)==0: # and hvd.rank()==0:
                # progress_bar.update(index)
                print('processed {}/{} batches in {}'.format(index + 1, total_batches, time.time() - start))
        
        # save weights every epoch
        if hvd.rank()==0:
           safe_mkdir(WeightsDir)

           print ("saving weights of gen")
           generator.save_weights(WeightsDir + '/generator_{0}{1:03d}.hdf5'.format(g_weights, epoch), overwrite=True)
            
           print ("saving weights of disc")
           discriminator.save_weights(WeightsDir + '/discriminator_{0}{1:03d}.hdf5'.format(d_weights, epoch), overwrite=True)

           epoch_time = time.time()-epoch_start
           print("The {} epoch took {} seconds".format(epoch, epoch_time))

           #print('The training took {} seconds.'.format(time.time()-epoch_start))
           print('\nTesting for epoch {}:'.format(epoch + 1))
        
        test_start=time.time()
        noise = np.random.normal(0.1, 1, (nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
        ecal_ip = GetEcalFit(sampled_energies, mod, xscale)
        sampled_energies = np.squeeze(sampled_energies, axis=(1,))

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        ecal = np.concatenate((ecal_test, ecal_ip))
        aux_y = np.concatenate((Y_test, sampled_energies), axis=0)
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        discriminator_test_loss_true = discriminator.evaluate(
            X_test, [np.array([1] * nb_test), Y_test, ecal_test], verbose=False, batch_size=batch_size)
        print('discriminator_test_loss_true=', discriminator_test_loss_true)

        discriminator_test_loss_fake = discriminator.evaluate(
            generated_images, [np.array([0] * nb_test), sampled_energies, ecal_ip], verbose=False, batch_size=batch_size) 
        print('discriminator_test_loss_fake=', discriminator_test_loss_fake)

        noise = np.random.normal(0.1, 1, (2 * nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (2 * nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        ecal_ip = GetEcalFit(sampled_energies, mod, xscale)
        trick = np.ones(2 * nb_test)
        generator_test_loss = combined.evaluate(generator_ip,
                            [trick, sampled_energies.reshape((-1, 1)), ecal_ip], verbose=False, batch_size=batch_size)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        pickle.dump({'train': train_history, 'test': test_history}, open('/gkhattak/hvd-history.pkl', 'wb'))
        
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
        if analysis:
              analysis_history = defaultdict(list)
              noise_test = np.random.normal(0., 1., (nb_test, latent_size))
              ep_test = np.expand_dims(Y_test, axis=-1)
              generator_ip_test = np.multiply(ep_test, noise_test)
              generated_images_test = g.predict(generator_ip_test, verbose=0)
              if keras_dformat !='channels_last':
                 generated_images_test = np.swapaxes(generated_images_test, -1, 1)
                 X_test =  np.swapaxes(X_test, -1, 1)
              var = gan.sortEnergy([X_test, Y_test], ecal_test, energies, ang=0)
              print('Analysing............')
              result = gan.OptAnalysisShort(var, generated_images_test, energies, ang=0)
              # All of the results correspond to mean relative errors on different quantities
              analysis_history['total'].append(result[0]) 
              analysis_history['energy'].append(result[1])
              analysis_history['moment'].append(result[2])
              print('Result = ', result)
              pickle.dump({'results': analysis_history}, open(resultfile, 'wb')) 
    

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--model', '-m', action='store', type=str, default='EcalEnergyGan', help='Model architecture to use.')
    parser.add_argument('--nbepochs', action='store', type=int, default=25, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/eos/project/d/dshep/LCD/V1/*scan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    parser.add_argument('--nbperfile', action='store', type=int, default=10000, help='Number of events in a file.')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='/gkhattak/hvdweights/', help='Directory to store weights.')
    parser.add_argument('--mod', action='store', type=int, default=0, help='How to calculate Ecal sum corressponding to energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=100, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--learningRate', '-lr', action='store', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--optimizer', action='store', type=str, default='RMSprop', help='Keras Optimizer to use.')
    parser.add_argument('--intraop', action='store', type=int, default=9, help='Sets onfig.intra_op_parallelism_threads and OMP_NUM_THREADS')
    parser.add_argument('--interop', action='store', type=int, default=1, help='Sets config.inter_op_parallelism_threads')
    parser.add_argument('--warmupepochs', action='store', type=int, default=5, help='No wawrmup epochs')
    parser.add_argument('--channel_format', action='store', type=str, default='channels_last', help='NCHW vs NHWC')
    parser.add_argument('--analysis', action='store', type=bool, default=False, help='Calculate optimisation function')
    return parser


if __name__ == '__main__':

    import keras.backend as K


    from keras.layers import Input
    from keras.models import Model
    from keras.optimizers import Adadelta, Adam, RMSprop
    from keras.utils.generic_utils import Progbar
    #from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split

    import tensorflow as tf
    import horovod.keras as hvd

    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    print(params)

    d_format = params.channel_format
 
    if d_format == 'channels_first':
        print('Setting th channel ordering (NCHW)')
        K.set_image_dim_ordering('th')
    else:
        print('Setting tf channel ordering (NHWC)')
        K.set_image_dim_ordering('tf')
    config = tf.ConfigProto()#(log_device_placement=True)
    config.intra_op_parallelism_threads = params.intraop
    config.inter_op_parallelism_threads = params.interop
    os.environ['KMP_BLOCKTIME'] = str(1)
    os.environ['KMP_SETTINGS'] = str(1)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact'
    # os.environ['KMP_AFFINITY'] = 'balanced'
    os.environ['OMP_NUM_THREADS'] = str(params.intraop)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)
    K.set_session(tf.Session(config=config))
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()


    # Initialize Horovod.
    hvd.init()

    #Architectures to import
    from EcalEnergyGan import generator, discriminator

    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
        # Analysis
    analysis=params.analysis # if analysing
    energies =[100, 200, 300, 400] # Bins
    resultfile = '3dgan_analysis.pkl' # analysis result

    global_batch_size = batch_size * hvd.size()
    print(hvd.size())
    print("Global batch size is: {0} / batch size is: {1}".format(global_batch_size, batch_size))
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Data path on Caltech
    datapath = params.datapath#Data path on EOS CERN
    EventsperFile = params.nbperfile#Events in a file
    nEvents = params.nbEvents#Total events for training
    fitmod = 1 #params.mod
    weightdir = params.weightsdir
    xscale = params.xscale
    warmup_epochs = params.warmupepochs
    mod=0
    opt = getattr(keras.optimizers, params.optimizer)
    #opt = RMSprop()
    opt = opt(params.learningRate * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    # Building discriminator and generator
    d=discriminator(keras_dformat=d_format)
    g=generator(latent_size,keras_dformat=d_format)
    
    GanTrain(d, g, opt, run_options, run_metadata, global_batch_size, warmup_epochs, datapath, EventsperFile, nEvents, weightdir, resultfile, energies, mod=fitmod, nb_epochs=nb_epochs, batch_size=batch_size, latent_size=latent_size, gen_weight=8, aux_weight=0.2, ecal_weight=0.1, xscale = xscale, verbose=verbose, keras_dformat=d_format, analysis=analysis )
    to = timeline.Timeline(run_metadata.step_stats)
    trace = to.generate_chrome_trace_format()
    with open('full_train_trace.json', 'w') as out:
          out.write(trace)

