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
import h5py 
import numpy as np
import time
import math
import argparse

if '.cern.ch' in os.environ.get('HOSTNAME'): # Here a check for host can be used to set defaults accordingly
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
import tensorflow as tf

#batch generator
def genbatches(a,n):
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n]

#shuffle 5 arrays simultaneously
def randomize(a, b, c, d, e):
    assert a.shape[0] == b.shape[0]
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    shuffled_d = d[permutation]
    shuffled_e = e[permutation]
    return shuffled_a, shuffled_b, shuffled_c, shuffled_d, shuffled_e

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
    outpath = params.outpath # training output
    nEvents = params.nEvents
    fEvents = params.fEvents
    f = [params.train_f, 1.0-params.train_f]
    ascale = params.ascale
    yscale = params.yscale
    ymin = params.ymin
    ymax = params.ymax
    weightdir = 'weights/3dgan_weights_' + params.name
    pklfile = 'results/3dgan_history_' + params.name + '.pkl'# loss history
    resultfile = 'results/3dgan_analysis' + params.name + '.pkl'# optimization metric history
    prev_gweights = 'weights/' + params.prev_gweights
    prev_dweights = 'weights/' + params.prev_dweights
    xscale = params.xscale
    xpower = params.xpower
    analyse=params.analyse # if analysing
    loss_weights=[params.gen_weight, params.aux_weight, params.ang_weight, params.ecal_weight, params.hist_weight]
    dformat=params.dformat
    thresh = params.thresh # threshold for data
    angtype = params.angtype
    particle = params.particle
    warm = params.warm
    test = params.test
    lr = params.lr
    events_per_file = params.fEvents
    energies = [0, 110, 150, 190]
    warmup_epochs = params.warmupepochs
    if (ymin > 0) or (ymax < 500):
       ylim =[ymin, ymax]
    else:
       ylim =[]
    # assigning paths based on remote
    if tlab:
       if datapath == 'path1':
         datapath = "/gkhattak/data/*Measured3ThetaEscan/*.h5"
       else:
         datapath = "/eos/user/g/gkhattak/VarAngleData/EleEscanprocessed/EleEscan/*.h5"
       outpath = '/gkhattak/'

    if outpath:
       weightdir = outpath + weightdir
       pklfile = outpath + pklfile # loss history
       resultfile = outpath + resultfile# optimization metric history
       prev_gweights = outpath + prev_gweights
       prev_dweights = outpath + prev_dweights


    # printing config
    print('****************** 3DGAN Config ***************************')
    print('epochs= {}, batch size= {}, latent size= {}, verbose ={}, number of events to use= {}'.format(nb_epochs, batch_size, latent_size, verbose, nEvents))
    print('data path={} with {} events/file\nout path= {}, weight dir= {}'.format(datapath, events_per_file, outpath, weightdir))
    print('loss history={}, analysis history={}'.format(pklfile, resultfile))
    print('cell energy scaled by {} and raised to power {} with threshold at {}'.format(xscale, xpower, thresh))
    print('analyze={} at energies={}, loss weights={}, image data format= {}, angle type= {}, particle= {}'.format(analyse, energies, loss_weights, dformat, angtype, particle))   
    K.set_image_data_format(dformat)
    print('Setting image data format to {}'.format(dformat))
    if warm:
       print('Starting from trained weights: discriminator weights={}, generator weights={}'.format(prev_dweights, prev_gweights))
    print('**************** Horovod config ***************************')
    print("using optimizer={} warmup epochs={}".format(params.optimizer, warmup_epochs))
    print('intraop={}, interop= {}'.format(params.intraop, params.interop))

    # hvd config 
    config = tf.compat.v1.ConfigProto(log_device_placement=False)
    config.intra_op_parallelism_threads = params.intraop
    config.inter_op_parallelism_threads = params.interop
    os.environ['KMP_BLOCKTIME'] = str(0)
    os.environ['KMP_SETTINGS'] = str(1)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact'
    os.environ['OMP_NUM_THREADS'] = str(params.intraop)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)
    os.environ['HOROVOD_LOG_LEVEL'] = str(3)
    K.set_session(tf.compat.v1.Session(config=config))
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()
    
    #initialize Horovod
    hvd.init()
    np.random.seed(42 + hvd.rank())
    tf.compat.v1.random.set_random_seed(42 + hvd.rank())
    
    opt = getattr(keras.optimizers, params.optimizer)
    opt = opt(params.lr)# * hvd.size())
    opt = hvd.DistributedOptimizer(opt)
    global_batch_size = batch_size * hvd.size()
    print('Number of nodes: {}'.format(hvd.size()))
    print("Global batch size is: {0} / batch size is: {1}".format(global_batch_size, batch_size))
    print('***********************************************************')

    # Building discriminator and generator
    gan.safe_mkdir(weightdir)
    d=discriminator(xpower, dformat=dformat)
    g=generator(latent_size, dformat=dformat)
  
    #submit training
    Gan3DTrainAngle(discriminator=d, generator=g, opt=opt, datapath=datapath, nEvents=nEvents, fEvents=fEvents, f=f, WeightsDir=weightdir, pklfile=pklfile,
                    global_batch_size=global_batch_size, nb_epochs=nb_epochs, batch_size=batch_size, latent_size=latent_size, 
                    loss_weights=loss_weights, lr=lr, xscale = xscale, xpower=xpower, angscale=ascale,
                    yscale=yscale, ylim=ylim, thresh=thresh, angtype=angtype, analyse=analyse, resultfile=resultfile,
                    energies=energies, dformat=dformat, particle=particle, verbose=verbose, warm=warm, test=test,
                    prev_gweights= prev_gweights, prev_dweights=prev_dweights,  warmup_epochs=warmup_epochs)

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--nbepochs', action='store', type=int, default=60, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=8, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=256, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/data/shared/gkhattak/*Measured3ThetaEscan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--train_f', action='store', type=float, default=0.9, help='fraction of events used in training')
    parser.add_argument('--outpath', action='store', type=str, default='', help='save training history and weights in a different location.')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last')
    parser.add_argument('--nEvents', action='store', type=int, default=400000, help='Total Number of events used for Training')
    parser.add_argument('--fEvents', action='store', type=int, default=5000, help='Total Number of events in a file')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
    parser.add_argument('--xpower', action='store', type=float, default=0.85, help='pre processing of cell energies by raising to a power')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--ymin', action='store', type=int, default=0, help='Minimum primary energy')
    parser.add_argument('--ymax', action='store', type=int, default=500, help='Maximum primary energy')
    parser.add_argument('--ascale', action='store', type=int, default=1, help='Multiplication factor for angle input')
    parser.add_argument('--analyse', action='store_true', default=False, help='Whether or not to perform analysis')
    parser.add_argument('--gen_weight', action='store', type=float, default=3, help='loss weight for generation real/fake loss')
    parser.add_argument('--aux_weight', action='store', type=float, default=0.1, help='loss weight for auxilliary energy regression loss')
    parser.add_argument('--ang_weight', action='store', type=float, default=25, help='loss weight for angle loss')
    parser.add_argument('--ecal_weight', action='store', type=float, default=0.1, help='loss weight for ecal sum loss')
    parser.add_argument('--hist_weight', action='store', type=float, default=0.1, help='loss weight for additional bin count loss')
    parser.add_argument('--thresh', action='store', type=int, default=0, help='Threshold for cell energies')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle to use for Training. It can be theta, mtheta or eta')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle')
    parser.add_argument('--lr', action='store', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--warm', action='store', default=False, help='Start from pretrained weights or random initialization')
    parser.add_argument('--prev_gweights', type=str, default='3dgan_weights_gan_training_epsilon_k2/params_generator_epoch_131.hdf5', help='Initial generator weights for warm start')
    parser.add_argument('--prev_dweights', type=str, default='3dgan_weights_gan_training_epsilon_k2/params_discriminator_epoch_131.hdf5', help='Initial discriminator weights for warm start')
    parser.add_argument('--test', action='store', default=False, help='Include testing for each epoch')
    parser.add_argument('--name', action='store', type=str, default='gan_dist_training', help='Unique identifier can be set for each training')
    parser.add_argument('--optimizer', action='store', type=str, default='RMSprop', help='Keras Optimizer to use.')
    parser.add_argument('--intraop', action='store', type=int, default=9, help='Sets onfig.intra_op_parallelism_threads and OMP_NUM_THREADS')
    parser.add_argument('--interop', action='store', type=int, default=1, help='Sets config.inter_op_parallelism_threads')
    parser.add_argument('--warmupepochs', action='store', type=int, default=0, help='No wawrmup epochs')
    return parser

# A histogram fucntion that counts cells in different bins
def hist_count(x, p=1.0, daxis=(1, 2, 3)):
    limits=np.array([0.05, 0.03, 0.02, 0.0125, 0.008, 0.003]) # bin boundaries used
    limits= np.power(limits, p)
    bin1 = np.sum(np.where(x>(limits[0]) , 1, 0), axis=daxis)
    bin2 = np.sum(np.where((x<(limits[0])) & (x>(limits[1])), 1, 0), axis=daxis)
    bin3 = np.sum(np.where((x<(limits[1])) & (x>(limits[2])), 1, 0), axis=daxis)
    bin4 = np.sum(np.where((x<(limits[2])) & (x>(limits[3])), 1, 0), axis=daxis)
    bin5 = np.sum(np.where((x<(limits[3])) & (x>(limits[4])), 1, 0), axis=daxis)
    bin6 = np.sum(np.where((x<(limits[4])) & (x>(limits[5])), 1, 0), axis=daxis)
    bin7 = np.sum(np.where((x<(limits[5])) & (x>0.), 1, 0), axis=daxis)
    bin8 = np.sum(np.where(x==0, 1, 0), axis=daxis)
    bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
    bins[np.where(bins==0)]=1 # so that an empty bin will be assigned a count of 1 to avoid unstability
    return bins

#get data for training
def GetDataAngle(datafile, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=0,  caxis=-1, dataset = ['ECAL', 'energy', 'ecal_sum', 'bcount'], ylim=[]):
    print ('Loading Data from .....', datafile)
    dataset = dataset + [angtype]
    f=h5py.File(datafile,'r')
    data = {}
    # read from file
    for key in dataset:
     if key in f:
       data[key] = np.array(f.get(key))
    
    # if processed not available
    for key in dataset:
     if key not in f:
       if key==angtype:
         data[key] = gan.measPython(data['ECAL'])
       elif key=='ecal_sum':
         data[key] = np.sum(data['ECAL'], axis=(1, 2, 3))
       elif key=='bcount':
         data[key] = hist_count(np.expand_dims(data['ECAL'], axis=-1))
       else:
         sys.exit('essential data missing ({})'.format(key))
    
    # ecal_sum filtering:
    if len(ylim)>1:
      indexes = np.where((data['ecal_sum']> 10.0) & (data['energy'] > ylim[0]) & (data['energy']< ylim[1]))
    else:
      indexes = np.where(data['ecal_sum']> 10.0)
    
    for key in dataset:
      data[key]= data[key][indexes]

    # pre-processing
    for key in dataset:
       data[key] = data[key].astype(np.float32)
       if key == ('ECAL' or 'ecal_sum'):
         data[key]= data[key] * xscale
         if key == 'ECAL':      
           data[key][data[key]< thresh]=0
           data[key]= np.expand_dims(data[key], axis=caxis) # add channel
           if xpower !=1.:
             data[key] = np.power(data[key], xpower)
         else:
             data[key]= np.expand_dims(data[key], axis=-1)
       else:
         data[key]= np.expand_dims(data[key], axis=-1)
         if key == 'energy':
           data[key]= data[key]/yscale
    return data

def Gan3DTrainAngle(discriminator, generator, opt, datapath, nEvents, fEvents, f, WeightsDir, pklfile, global_batch_size, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='mtheta', yscale=100, ylim=[],thresh=0, analyse=False, resultfile="", energies=[], warmup_epochs=0, dformat='channels_last', particle='Ele', verbose=False, warm=False, test=False, prev_gweights='', prev_dweights=''):
    start_init = time.time() 
    loss_ftn = hist_count # additional loss   
    # apply settings according to data format
    if dformat=='channels_last':
       caxis=-1 # channel axis
       daxis=(1, 2, 3) # axis for sum
    else:
       caxis=1 # channel axis
       daxis=(2, 3, 4) # axis for sum

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
    if hvd.rank()==0:
      if warm:
        generator.load_weights(prev_gweights)
        print('Generator initialized from {}'.format(prev_gweights))
        discriminator.load_weights(prev_dweights)
        print('Discriminator initialized from {}'.format(prev_dweights))

    discriminator.trainable = True #workaround for keras 2 bug
    gcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=65, multiplier=1.), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=65, end_epoch=140, multiplier=1e-1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=140, end_epoch=260, multiplier=3e-2), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=260, end_epoch=380, multiplier=1e-2), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=380, multiplier=1e-3), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=None) \
        ])

    dcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        #hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=65, multiplier=1.), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=65, end_epoch=140, multiplier=1e-1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=140, end_epoch=260, multiplier=3e-2), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=260, end_epoch=380, multiplier=1e-2), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=380, multiplier=1e-3), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=None) \
        ])

    ccb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        #hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=65, multiplier=1.), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=65, end_epoch=140, multiplier=1e-1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=140, end_epoch=260, multiplier=3e-2), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=260, end_epoch=380, multiplier=1e-2), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=380, multiplier=1e-3), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=None) \
        ])

    gcb.set_model( generator )
    dcb.set_model( discriminator )
    ccb.set_model( combined )

    gcb.on_train_begin()
    dcb.on_train_begin()
    ccb.on_train_begin()

    # Getting Data
    Trainfiles, Testfiles = gan.DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
    nb_Test = int(nEvents * f[1]) # The number of test files calculated from fraction of nEvents
    nb_Train = int(nEvents * f[0]) # The number of train files calculated from fraction of nEvents     
    if hvd.rank()==0:
        print('Train files:', Trainfiles)
        print('Test files:', Testfiles)
        print('Number of events to use for training:', nb_Train)
        print('Number of events to use for testing:', nb_Test)
    if test or analyse:
        if hvd.rank()==0:
           print('Loading test data:') 
    
        #Read test data into a single array
        for index, dtest in enumerate(Testfiles):
           if index == 0:
             data_temp = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, caxis=caxis, thresh=thresh, ylim=ylim)
             X_test = data_temp['ECAL']
             Y_test = data_temp['energy']
             ang_test =data_temp[angtype]
             ecal_test = data_temp['ecal_sum']
             bcount_test = data_temp['bcount']
           else:
             if X_test.shape[0] < nb_Test:
                data_temp = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, caxis=caxis, thresh=thresh, ylim=ylim)
                X_test = np.concatenate((X_test, data_temp['ECAL']))
                Y_test = np.concatenate((Y_test, data_temp['energy']))
                ang_test = np.concatenate((ang_test, data_temp[angtype]))
                ecal_test = np.concatenate((ecal_test, data_temp['ecal_sum']))
                bcount_test = np.concatenate((bcount_test, data_temp['bcount']))
        X_test = X_test[:nb_Test]
        Y_test = Y_test[:nb_Test]
        ang_test = ang_test[:nb_Test]
        ecal_test = ecal_test[:nb_Test]
        bcount_test = bcount_test[:nb_Test]
    
    if hvd.rank()==0:
       print('Loading train data:')
    for index, dtrain in enumerate(Trainfiles):
       if index == 0:
            data_temp = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, caxis=caxis, thresh=thresh, ylim=ylim)
            X_train = data_temp['ECAL']
            Y_train = data_temp['energy']
            ang_train = data_temp[angtype]
            ecal_train = data_temp['ecal_sum']
            bcount_train = data_temp['bcount']
       else:
            if X_train.shape[0] < nb_Train:
              data_temp = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, caxis=caxis, thresh=thresh, ylim=ylim)
              X_train = np.concatenate((X_train, data_temp['ECAL']))
              Y_train = np.concatenate((Y_train, data_temp['energy']))
              ang_train = np.concatenate((ang_train, data_temp[angtype]))
              ecal_train = np.concatenate((ecal_train, data_temp['ecal_sum']))
              bcount_train = np.concatenate((bcount_train, data_temp['bcount']))
    X_train = X_train[:nb_Train]
    Y_train = Y_train[:nb_Train]
    ang_train = ang_train[:nb_Train]
    ecal_train = ecal_train[:nb_Train]
    bcount_train = bcount_train[:nb_Train]
    total_batches = nb_Train / global_batch_size
    if hvd.rank()==0:
       print('Total Training batches = {} with {} events'.format(total_batches, nb_Train))
       print('*************************************************************************************')
       print('Ang varies from {} to {} with mean {}'.format(np.amin(ang_train), np.amax(ang_train), np.mean(ang_train)))
       print('Cell varies from {} to {} with mean {}'.format(np.amin(X_train[X_train>0]), np.amax(X_train[X_train>0]), np.mean(X_train[X_train>0])))
       print('*************************************************************************************')
    if hvd.rank()==0 and test:
       print('Test Data loaded of shapes:')
       print(X_test.shape)
       print(Y_test.shape)
       print('*************************************************************************************')
       print('Ang varies from {} to {} with mean {}'.format(np.amin(ang_test), np.amax(ang_test), np.mean(ang_test)))
       print('Cell varies from {} to {} with mean {}'.format(np.amin(X_test[X_test>0]), np.amax(X_test[X_test>0]), np.mean(X_test[X_test>0])))
       print('*************************************************************************************')
       if analyse:
          var=gan.sortEnergy([np.squeeze(X_test), Y_test, ang_test], ecal_test, energies, ang=1)
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    analysis_history = defaultdict(list)
    init_time = time.time()- start_init

    if hvd.rank()==0:
      print('Initialization time is {} seconds'.format(init_time))
    
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        if hvd.rank()==0:
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
 
        epoch_gen_loss = []
        epoch_disc_loss = []
        randomize(X_train, Y_train, ecal_train, ang_train, bcount_train)

        epoch_gen_loss = []
        epoch_disc_loss = []
        
        image_batches = genbatches(X_train, batch_size)
        energy_batches = genbatches(Y_train, batch_size)
        ecal_batches = genbatches(ecal_train, batch_size)
        ang_batches = genbatches(ang_train, batch_size)
        bcount_batches = genbatches(bcount_train, batch_size)
        for index in range(int(total_batches)):
            if hvd.rank()==0:
              if verbose:
                progress_bar.update(index)
              else:
                if index % 100 == 0:
                  print('processed {} batches'.format(index + 1))

            image_batch = next(image_batches) 
            energy_batch = next(energy_batches)
            ecal_batch = next(ecal_batches)
            ang_batch = next(ang_batches)
            bcount_batch = next(bcount_batches)
            noise = np.random.normal(0, 1, (batch_size, latent_size-2))
            generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1)
            generated_images = generator.predict(generator_ip, verbose=0)
            real_batch_loss = discriminator.train_on_batch(image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ang_batch, ecal_batch, bcount_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [gan.BitFlip(np.zeros(batch_size)), energy_batch, ang_batch, ecal_batch, bcount_batch])
            if fake_batch_loss[4] == 100.0 and index >10:
                if hvd.rank()==0:
                    print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
                    generator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
                    discriminator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
                    print ('real_batch_loss', real_batch_loss)
                    print ('fake_batch_loss', fake_batch_loss)
                sys.exit()
            epoch_disc_loss.append([(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)])
            trick = np.ones(batch_size)
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size-2))
                generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1) # sampled angle same as g4 theta
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, energy_batch.reshape(-1, 1), ang_batch, ecal_batch, bcount_batch]))
            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]
            epoch_gen_loss.append(generator_loss)
        
        #comment vali
        if hvd.rank()==0:
          print('Total Training batches were {}'.format(index))
          print('Epoch {} took {} sec'.format(epoch, time.time()-epoch_start))
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        if test:
            if hvd.rank()==0:
              print('\nTesting for epoch {}:'.format(epoch + 1))
            test_start=time.time()
            noise = np.random.normal(0.1, 1, (nb_Test, latent_size-2))
            generator_ip = np.concatenate((Y_test.reshape(-1, 1), ang_test.reshape(-1, 1), noise), axis=1) 
            generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
                        
            X = np.concatenate((X_test, generated_images))
            y = np.array([1] * nb_Test + [0] * nb_Test)
            ecal = np.concatenate((ecal_test, ecal_test))
            aux_y = np.concatenate((Y_test, Y_test), axis=0)
            bcount = np.concatenate((bcount_test, bcount_test), axis=0)
            ang = np.concatenate((ang_test, ang_test))
            discriminator_test_loss = discriminator.evaluate(
                X, [y, aux_y, ang, ecal, bcount], verbose=False, batch_size=batch_size)
            
            generator_test_loss = combined.evaluate(generator_ip,
               [np.array([1] * nb_Test), Y_test, ang_test, ecal_test, bcount_test], verbose=False, batch_size=batch_size)
            
            test_history['generator'].append(generator_test_loss)
            test_history['discriminator'].append(discriminator_test_loss)

            # perform a short evaluation based on mean relative errors
            if analyse:
                var=gan.sortEnergy([np.squeeze(X_test), Y_test, ang_test], ecal_test, energies, ang=1)
                result = gan.OptAnalysisAngle(var, generator, energies, xpower = xpower, concat=2)    
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
        if test:
            print(ROW_FMT.format('generator (test)',
                       *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                       *train_history['discriminator'][-1]))
        if test:
            print(ROW_FMT.format('generator (test)',
                       *test_history['generator'][-1]))

        # save weights and losses for every epoch
        generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                         overwrite=True)
        discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                         overwrite=True)
        
        pickle.dump({'train': train_history}, open(pklfile, 'wb'))
        if test:
           pickle.dump({'test': test_history}, open(pklfile, 'wb'))
           print("The {} epoch test took {} seconds".format(epoch, time.time()-test_start))

if __name__ == '__main__':
    main()
