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
from keras.callbacks import CallbackList
kv2 = keras.__version__.startswith('2')
import argparse
import os
os.environ['LD_LIBRARY_PATH'] = os.getcwd()
from six.moves import range
import sys
#import glob
import h5py 
import numpy as np
import time
import math
import argparse

if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a check for host can be used
    tlab = True
else:
    tlab= False
    
try:
    import setGPU #if Caltech
except:
    pass

#from memory_profiler import profile # used for memory profiling
import keras.backend as K
import analysis.utils.GANutils as gan

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils.generic_utils import Progbar
import horovod.keras as hvd
# printing versions of software used
#print('keras version:', keras.__version__)
#print('python version:', sys.version)
import tensorflow as tf
#print('tensorflow version', tf.__version__)
#print('numpy version', np.version.version)
def genbatches(a,n):
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n]


def randomize(a, b, c, d):
    assert a.shape[0] == b.shape[0]
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    shuffled_d = d[permutation]
    return shuffled_a, shuffled_b, shuffled_c, shuffled_d

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
    nEvents = params.nbEvents
    ascale = params.ascale
    yscale = params.yscale
    weightdir = params.weightsdir
    pklfile = params.pklfile # loss history
    resultfile = params.resultfile # optimization metric history
    xscale = params.xscale
    xpower = params.xpower
    analyse=params.analyse # if analysing
    energies =params.energies # Bins
    resultfile = params.resultfile # analysis result
    loss_weights=params.lossweights
    thresh = params.thresh # threshold for data
    angtype = params.angtype
    warmup_epochs = params.warmupepochs

    d_format = params.channel_format

    if d_format == 'channels_first':
        print('Setting th channel ordering (NCHW)')
        K.set_image_dim_ordering('th')
        K.set_image_data_format('channels_first')
    else:
        print('Setting tf channel ordering (NHWC)')
        K.set_image_dim_ordering('tf')
        K.set_image_data_format('channels_last')

 
    config = tf.ConfigProto(log_device_placement=True)
    config.intra_op_parallelism_threads = params.intraop
    config.inter_op_parallelism_threads = params.interop
    os.environ['KMP_BLOCKTIME'] = str(1)
    os.environ['KMP_SETTINGS'] = str(1)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact'
    # os.environ['KMP_AFFINITY'] = 'balanced'
    # os.environ['OMP_NUM_THREADS'] = str(params.intraop)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)
    K.set_session(tf.Session(config=config))
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()


    if tlab:
      datapath = '/gkhattak/*Measured3ThetaEscan/*.h5'
      weightdir = '/gkhattak/weights/3dgan_weights'
      pklfile = '/gkhattak/results/3dgan_history.pkl'
      resultfile = '/gkhattak/results/3dgan_analysis.pkl'

    print(params)
    #initialize Horovod
    hvd.init()
 
    opt = getattr(keras.optimizers, params.optimizer)
    opt = opt(params.learningRate * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    global_batch_size = batch_size * hvd.size()
    print("Global batch size is: {0} / batch size is: {1}".format(global_batch_size, batch_size))
    # Building discriminator and generator
    gan.safe_mkdir(weightdir)
    d=discriminator(xpower)
    g=generator(latent_size)
    Gan3DTrainAngle(d, g, opt, datapath, nEvents, weightdir, pklfile, global_batch_size=global_batch_size, nb_epochs=nb_epochs, batch_size=batch_size,
                    latent_size=latent_size, loss_weights=loss_weights, xscale = xscale, xpower=xpower, angscale=ascale,
                    yscale=yscale, thresh=thresh, angtype=angtype, analyse=analyse, resultfile=resultfile,
                    energies=energies, warmup_epochs=warmup_epochs)

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--nbepochs', action='store', type=int, default=60, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=256, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/data/shared/gkhattak/*Measured3ThetaEscan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Total Number of events used for Training')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='weights/3dgan_weights', help='Directory to store weights.')
    parser.add_argument('--pklfile', action='store', type=str, default='results/3dgan_history.pkl', help='Pickle file to store losses.')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
    parser.add_argument('--xpower', action='store', type=float, default=0.85, help='pre processing of cell energies by raising to a power')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--ascale', action='store', type=int, default=1, help='Multiplication factor for angle input')
    parser.add_argument('--resultfile', action='store', type=str, default='results/3dgan_analysis.pkl', help='File to save losses.')
    parser.add_argument('--analyse', action='store_true', default=False, help='Whether or not to perform analysis')
    parser.add_argument('--energies', action='store', type=int, default=[0, 110, 150, 190], help='Energy bins for analysis')
    parser.add_argument('--lossweights', action='store', type=int, default=[3, 0.1, 25, 0.1, 0.1], help='loss weights =[gen_weight, aux_weight, ang_weight, ecal_weight, add loss weight]')
    parser.add_argument('--thresh', action='store', type=int, default=0, help='Threshold for cell energies')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle to use for Training. It can be theta, mtheta or eta')
    parser.add_argument('--learningRate', '-lr', action='store', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--optimizer', action='store', type=str, default='RMSprop', help='Keras Optimizer to use.')
    parser.add_argument('--intraop', action='store', type=int, default=9, help='Sets onfig.intra_op_parallelism_threads and OMP_NUM_THREADS')
    parser.add_argument('--interop', action='store', type=int, default=1, help='Sets config.inter_op_parallelism_threads')
    parser.add_argument('--warmupepochs', action='store', type=int, default=5, help='No wawrmup epochs')
    parser.add_argument('--channel_format', action='store', type=str, default='channels_first', help='NCHW vs NHWC')
    parser.add_argument('--analysis', action='store', type=bool, default=False, help='Calculate optimisation function')
    return parser

def mapping(x):
    return x
    
if K.image_data_format() !='channels_last':
   daxis = (2,3,4)
else:
   daxis = (1,2,3)
def hist_count(x, p=1):
    bin1 = np.sum(np.where(x>(0.05**p) , 1, 0), axis=daxis)
    bin2 = np.sum(np.where((x<(0.05**p)) & (x>(0.03**p)), 1, 0), axis=daxis)
    bin3 = np.sum(np.where((x<(0.03**p)) & (x>(0.02**p)), 1, 0), axis=daxis)
    bin4 = np.sum(np.where((x<(0.02**p)) & (x>(0.0125**p)), 1, 0), axis=daxis)
    bin5 = np.sum(np.where((x<(0.0125**p)) & (x>(0.008**p)), 1, 0), axis=daxis)
    bin6 = np.sum(np.where((x<(0.008**p)) & (x>(0.003**p)), 1, 0), axis=daxis)
    bin7 = np.sum(np.where((x<(0.003**p)) & (x>0.), 1, 0), axis=daxis)
    bin8 = np.sum(np.where(x==0, 1, 0), axis=daxis)
    bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
    bins[np.where(bins==0)]=1
    return bins

#get data for training
def GetDataAngle(datafile, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4):
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
    if K.image_data_format() !='channels_last':
       X =np.moveaxis(X, -1, 1)
       ecal = np.sum(X, axis=(2, 3, 4))
    else:
       ecal = np.sum(X, axis=(1, 2, 3))
    if xpower !=1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal

def Gan3DTrainAngle(discriminator, generator, opt, datapath, nEvents, WeightsDir, pklfile, global_batch_size, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], warmup_epochs=0):
    start_init = time.time()
    verbose = False    
    particle='Ele'
    f = [0.9, 0.1]
    loss_ftn = hist_count
    if hvd.rank()==0:
        print('[INFO] Building discriminator')
    #discriminator.summary()
    discriminator.compile(
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )

    # build the generator
    if hvd.rank()==0:
        print('[INFO] Building generator')
    #generator.summary()
    generator.compile(
        optimizer=opt,
        loss='binary_crossentropy'
    )
 
    # build combined Model
    latent = Input(shape=(latent_size, ), name='combined_z')   
    fake_image = generator( latent)
    discriminator.trainable = False
    fake, aux, ang, ecal, add_loss= discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ang, ecal, add_loss],
        name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )
    if kv2: 
        discriminator.trainable = True #workaround for keras 2 bug
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

    # Getting Data
    Trainfiles, Testfiles = gan.DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
    if hvd.rank()==0:
        print(Trainfiles)
        print(Testfiles)
    nb_Test = int(nEvents * f[1]) # The number of test files calculated from fraction of nEvents
    nb_Train = int(nEvents * f[0]) # The number of train files calculated from fraction of nEvents
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
       else:
           if X_test.shape[0] < nb_Test:
              X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
              X_test = np.concatenate((X_test, X_temp))
              Y_test = np.concatenate((Y_test, Y_temp))
              ang_test = np.concatenate((ang_test, ang_temp))
              ecal_test = np.concatenate((ecal_test, ecal_temp))
    if X_test.shape[0] > nb_Test:
        X_test, Y_test, ang_test, ecal_test = X_test[:nb_Test], Y_test[:nb_Test], ang_test[:nb_Test], ecal_test[:nb_Test]
    else:
        nb_Test = X_test.shape[0] # the nb_test maybe different if total events are less than nEvents
    for index, dtrain in enumerate(Trainfiles):
        if index == 0:
            X_train, Y_train, ang_train, ecal_train = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
        else:
            X_temp, Y_temp, ang_temp, ecal_temp = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
            X_train = np.concatenate((X_train, X_temp))
            Y_train = np.concatenate((Y_train, Y_temp))
            ang_train = np.concatenate((ang_train, ang_temp))
            ecal_train = np.concatenate((ecal_train, ecal_temp))

    nb_train = X_train.shape[0]# Total events in training files
    total_batches = nb_train / global_batch_size
    if hvd.rank()==0:
        print('Total Training batches = {} with {} events'.format(total_batches, nb_train))


    if hvd.rank()==0:
       print('Test Data loaded of shapes:')
       print(X_test.shape)
       print(Y_test.shape)
       print('*************************************************************************************')
       print('Ang varies from {} to {} with mean {}'.format(np.amin(ang_test), np.amax(ang_test), np.mean(ang_test)))
       print('Cell varies from {} to {} with mean {}'.format(np.amin(X_test[X_test>0]), np.amax(X_test[X_test>0]), np.mean(X_test[X_test>0])))
       if analyse:
          var = gan.sortEnergy(X_test, Y_test, ang_test, ecal_test, energies)
       train_history = defaultdict(list)
       test_history = defaultdict(list)
       analysis_history = defaultdict(list)
       init_time = time.time()- start_init
       print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        if hvd.rank()==0:
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
 
        epoch_gen_loss = []
        epoch_disc_loss = []
        randomize(X_train, Y_train, ecal_train, ang_train)

        epoch_gen_loss = []
        epoch_disc_loss = []
        
        image_batches = genbatches(X_train, batch_size)
        energy_batches = genbatches(Y_train, batch_size)
        ecal_batches = genbatches(ecal_train, batch_size)
        ang_batches = genbatches(ang_train, batch_size)
        for index in range(int(total_batches)):
            start = time.time()         
            image_batch = next(image_batches) 
            energy_batch = next(energy_batches)
            ecal_batch = next(ecal_batches)
            ang_batch = next(ang_batches)
            add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower), axis=-1)
            noise = np.random.normal(0, 1, (batch_size, latent_size-2))
            generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1)
            generated_images = generator.predict(generator_ip, verbose=0)
  
            real_batch_loss = discriminator.train_on_batch(image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [gan.BitFlip(np.zeros(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])

            #if ecal sum has 100% loss then end the training
            if fake_batch_loss[4] == 100.0 and index >10:
                if hvd.rank()==0:
                    print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
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
                generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1) # sampled angle same as g4 theta
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, energy_batch.reshape(-1, 1), ang_batch, ecal_batch, add_loss_batch]))
            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]
            epoch_gen_loss.append(generator_loss)
            #print ('generator_loss', generator_loss)
            index +=1

            # Used at design time for debugging
            #print('real_batch_loss', real_batch_loss)
            #print ('fake_batch_loss', fake_batch_loss)
            #disc_out = discriminator.predict(image_batch)
            #print('disc_out')
            #print(np.transpose(disc_out[4][:5].astype(int)))
            #print('add_loss_batch')
            #print(np.transpose(add_loss_batch[:5]))

        # Testing  
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        if hvd.rank()==0:
            if analyse:
                result = gan.OptAnalysisShort(var, generated_images, energies)
                print('Analysing............')
                analysis_history['total'].append(result[0])
                analysis_history['energy'].append(result[1])
                analysis_history['moment'].append(result[2])
                analysis_history['angle'].append(result[3])
                print('Result = ', result)
                pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

            print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s} | {6:8s}'.format('component', *discriminator.metrics_names))
            print('-' * 65)
            ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}| {6:<10.2f}'
            print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))

        # save weights every epoch
            generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
            discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)
        
            epoch_time = time.time()-test_start
            pickle.dump({'train': train_history}, open(pklfile, 'wb'))

if __name__ == '__main__':
    main()
