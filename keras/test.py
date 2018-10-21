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
    parser.add_argument('--weightsdir', action='store', type=str, default='weights2D', help='Directory to store weights.')
    parser.add_argument('--mod', action='store', type=int, default=0, help='How to calculate Ecal sum corressponding to energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
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
    from sklearn.cross_validation import train_test_split

    import tensorflow as tf

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
   
    # Initialize Horovod.

    #Architectures to import
    from EcalEnergyGan import generator, discriminator

    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
        # Analysis
    analysis=params.analysis # if analysing
    energies =[100, 200, 300, 400] # Bins
    resultfile = '3dgan_analysis.pkl' # analysis result

    global_batch_size = batch_size 
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
    # Building discriminator and generator
    g=generator(latent_size,keras_dformat=d_format)
    g.load_weights('params_generator_epoch_041.hdf5')

    Trainfiles, Testfiles = DivideFiles(datapath, nEvents=nEvents, EventsperFile = EventsperFile, datasetnames=["ECAL"], Particles =["Ele"])

    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ecal_test = GetData(dtest, keras_dformat=d_format)
       else:
           X_temp, Y_temp, ecal_temp = GetData(dtest, keras_dformat=d_format)
           X_test = np.concatenate((X_test, X_temp))
           Y_test = np.concatenate((Y_test, Y_temp))
           ecal_test = np.concatenate((ecal_test, ecal_temp))

    #nb_test = X_test.shape[0]
    nb_test = 100
    X_test, Y_test, ecal_test = X_test[:nb_test], Y_test[:nb_test], ecal_test[:nb_test]

    print('N test', nb_test)

    print (X_test.shape)
    print (Y_test.shape)
    print (ecal_test.shape)
    
    noise = np.random.normal(0., 1., (nb_test, latent_size))
    #sampled_energies = np.random.uniform(0.1, 5,( nb_test, 1))
    ep = np.expand_dims(Y_test, axis=-1)
    generator_ip = np.multiply(ep, noise)
    print (ep)
    #generator_ip = np.multiply(sampled_energies, noise)
            # ecal sum from fit
    #ecal_ip = GetEcalFit(sampled_energies, mod, xscale)
    print ('ecal_test')
    print (ecal_test)
    generated_images = g.predict(generator_ip, verbose=0)
    analysis_history = defaultdict(list)
    if analysis:
       if d_format !='channels_last':
          generated_images = np.swapaxes(generated_images, -1, 1)
          X_test =  np.swapaxes(X_test, -1, 1)
          print ('channel_first', X_test.shape)
       print (X_test.shape)
       var = gan.sortEnergy([X_test, Y_test], ecal_test, energies, ang=0)

       print('Analysing............')
       result = gan.OptAnalysisShort(var, generated_images, energies, ang=0)
        # All of the results correspond to mean relative errors on different quantities
       analysis_history['total'].append(result[0])
       analysis_history['energy'].append(result[1])
       analysis_history['moment'].append(result[2])
       print('Result = ', result)
       pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

