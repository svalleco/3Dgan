#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import keras
import argparse
#import os
#os.environ['LD_LIBRARY_PATH'] = os.getcwd()
from six.moves import range
import glob
import os
import sys
import math
#from keras.utils.training_utils import multi_gpu_model
import h5py 
import numpy as np
from adlkit.data_provider import H5FileDataProvider
import time

def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

def myNorm(Norms):
    def NormalizationFunction(Ds):
        # converting the data from an ordered-dictionary format to a list
        Ds = [Ds[item] for item in Ds]
        out = []
        # print('DS', Ds)
        # TODO replace with zip function
        for i,D in enumerate(Ds):
            Norm=Norms[i]
            if i == 0:
                D[D < 1e-666666] = 0
                D=np.expand_dims(D, axis=-1)
            if i == 1:
                D = D[:,1]
            if Norm != 0.:
                if isinstance(Norm, float):
                    D /= Norm
                if isinstance(Norm, str) and Norm.lower() == "nonlinear":
                    D = np.tanh(
                        np.sign(Ds[i]) * np.log(np.abs(Ds[i]) + 1.0) / 2.0)
                out.append(D)
        return out
    return NormalizationFunction

def mySetupData(FileSearch,
              ECAL,HCAL,target,
              NClasses,f,Particles,
              BatchSize,
              multiplier,
              ECALShape,
              HCALShape,
              ECALNorm,
              HCALNorm,
              targetNorm,
              delivery_function,
              n_threads,
              NTrain,
              NTest):
    datasets=[]
    shapes=[]
    Norms=[]

    if ECAL:
        datasets.append("ECAL")
        shapes.append((BatchSize*multiplier,)+ECALShape[1:])
        Norms.append(ECALNorm)
    if HCAL:
        datasets.append("HCAL")
        shapes.append((BatchSize*multiplier,)+HCALShape[1:])
        Norms.append(HCALNorm)
    if target:
        datasets.append("target")
#        shapes.append((BatchSize*multiplier,)+(1,5))
        shapes.append((BatchSize*multiplier,)+(1,))
        Norms.append(targetNorm)

    TrainSampleList,TestSampleList=DivideFiles(FileSearch,f,
                                               datasetnames=datasets,
                                               Particles=Particles)
    print('Train Events in files = {}'.format(len(TrainSampleList) * 10000))
    print('Test Events in files = {}'.format(len(TestSampleList) * 10000))

    sample_spec_train = list()
    for item in TrainSampleList:
        sample_spec_train.append((item[0], item[1] , item[2], 1))

    sample_spec_test = list()
    for item in TestSampleList:
        sample_spec_test.append((item[0], item[1] , item[2], 1))

    q_multipler = 20
    read_multiplier = 1
    n_buckets = 4

    Train_genC = H5FileDataProvider(sample_spec_train,
                                    max=math.ceil(float(NTrain)/BatchSize),
                                    batch_size=BatchSize,
                                    process_function=myNorm(Norms),
                                    delivery_function=delivery_function,
                                    n_readers=n_threads,
                                    q_multipler=q_multipler,
                                    n_buckets=n_buckets,
                                    read_multiplier=read_multiplier,
                                    #make_one_hot=True,
                                    sleep_duration=0.1,
                                    wrap_examples=True)

    Test_genC = H5FileDataProvider(sample_spec_test,
                                   max=math.ceil(float(NTest)/BatchSize),
                                   batch_size=BatchSize,
                                   process_function=myNorm(Norms),
                                   delivery_function=delivery_function,
                                   n_readers=n_threads,
                                   q_multipler=q_multipler,
                                   n_buckets=n_buckets,
                                   read_multiplier=read_multiplier,
                                   #make_one_hot=True,
                                   sleep_duration=0.1,
                                   wrap_examples=True)

    print ("Class Index Map:", Train_genC.config.class_index_map)

    return Train_genC,Test_genC,Norms,shapes,TrainSampleList,TestSampleList

def DivideFiles(FileSearch="/data/LCD/*/*.h5",Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files = glob.glob(FileSearch)

    print ("Found",len(Files),"files.")
    
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")

        if ParticleName in Particles:
            try:
                Samples[ParticleName].append((F,datasetnames,ParticleName))
            except:
                Samples[ParticleName]=[(F,datasetnames,ParticleName)]

        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    
    out=[] 

    print ("Electron are in ", FileCount ," files.")
    for j in range(len(Fractions)):
        out.append([])
        
    SampleI=len(Samples.keys())*[int(0)]
    
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)

        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+round(NFiles*Frac))
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

    #import tensorflow as tf
    #config = tf.ConfigProto(log_device_placement=True)
  
    from EcalEnergyGan import generator, discriminator

    import tensorflow as tf   
    import horovod.keras as hvd




    g_weights = 'params_generator_epoch_' 
    d_weights = 'params_discriminator_epoch_' 

    nb_epochs = 25
    batch_size = 128
    latent_size = 200
    verbose = 'false'
    nb_classes = 2
    
    ECALShape= None, 25, 25, 25
    HCALShape= None, 5, 5, 60
    #FileSearch="/eos/project/d/dshep/LCD/V1/*scan/*.h5"
    #FileSearch="/afs/cern.ch/work/g/gkhattak/public/Ele_v1*.h5"
    FileSearch="/bigdata/shared/LCD/NewV1/*scan/*.h5"
    #train_file="/tmp/gulrukh-CaloDNN-LCD-TrainEvent-Cache.h5"
    #test_file="/tmp/gulrukh-CaloDNN-LCD-TestEvent-Cache.h5"
    Particles=["Ele"]
    #MaxEvents=int(8.e5)
    NClasses=len(Particles)
    BatchSize= batch_size
    NSamples=BatchSize*10
    ECAL=True
    HCAL=False
    target=True
    delivery_function = None
    ECALNorm= None
    HCALNorm= None
    targetNorm=100.
    multiplier=1
    n_threads=1
    NTest = NTestSamples=50000
    NTrain = 256000
    Train_genC,Test_genC,Norms,shapes,TrainSampleList,TestSampleList= mySetupData(FileSearch,
                                                          ECAL,
                                                          HCAL,
                                                          target,
                                                          NClasses,
                                                          [0.9, 0.1],
                                                          Particles,
                                                          BatchSize,
                                                          multiplier,
                                                          ECALShape,
                                                          HCALShape,
                                                          ECALNorm,
                                                          HCALNorm,
                                                          targetNorm,
                                                          delivery_function,
                                                          n_threads,
                                                          NTrain,
                                                          NTest)

    Train_genC.start()
    Traingen = Train_genC.first().generate()

    Test_genC.start()
    Testgen = Test_genC.first().generate()


   # Initialize Horovod.
    hvd.init()

   # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    import time
    time.sleep( 10* hvd.local_rank())
    import setGPU
    #config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config)


    generator=generator(latent_size)
    discriminator=discriminator()



    print('[INFO] Building discriminator')
    discriminator.summary()
    #discriminator.load_weights('veganweights/params_discriminator_epoch_019.hdf5')

    print('')
    # Add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(RMSprop())

    discriminator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[8, .2, .1]
        #loss=['binary_crossentropy', 'kullback_leibler_divergence']
    )

    # build the generator
    print('[INFO] Building generator')
    generator.summary()
    #generator.load_weights('veganweights/params_generator_epoch_019.hdf5')
    generator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=opt,
        loss='binary_crossentropy'
    )

    latent = Input(shape=(latent_size, ), name='combined_z')
     
    fake_image = generator( latent)

    #fake.name = 'real_fake'
    #aux.name = 'primary_energy'
    #ecal.name = 'ecal_sum'
    discriminator.trainable = False
    discriminator.get_layer(name='generation').name='generation_1'
    generation,  auxiliary, ecal = discriminator(fake_image)
    combined = Model(
       input=[latent],
       output=[generation, auxiliary, ecal],
       name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[8, .2, .1]
    )


    #d=h5py.File("/bigdata/shared/LCD/Pions_fullSpectrum/ChPiEscan_1_2.h5",'r')
    #d=h5py.File("/bigdata/shared/LCD/small_test.h5",'r')
    #d=h5py.File("/bigdata/shared/LCD/Electrons_fullSpectrum/Ele_v1_1_2.h5",'r')
    #e=d.get('target')
    #X=np.array(d.get('ECAL'))
    #y=(np.array(e[:,1]))
    #print(X.shape)
    #print('*************************************************************************************')
    #print(y)
    #print('*************************************************************************************')
   
    #Y=np.sum(X, axis=(1,2,3))
    #print(Y)
    #print('*************************************************************************************')

    
   # remove unphysical values
   # X[X < 1e-6] = 0

    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size=0.1)

    # tensorflow ordering
    #X_train =np.expand_dims(X_train, axis=-1)
    #X_test = np.expand_dims(X_test, axis=-1)
    #y_train= y_train/100
    #y_test=y_test/100
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)
    #print('*************************************************************************************')


    #nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    #X_train = X_train.astype(np.float32)  
    #X_test = X_test.astype(np.float32)
    #y_train = y_train.astype(np.float32)
    #y_test = y_test.astype(np.float32)
    #ecal_train = np.sum(X_train, axis=(1, 2, 3))
    #ecal_test = np.sum(X_test, axis=(1, 2, 3))

    #print(X_train.shape)
    #print(X_test.shape)
    #print(ecal_train.shape)
    #print(ecal_test.shape)
    #print('*************************************************************************************')
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    # Broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #callbacks = [
    #    
    #]

    gcb =hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    dcb =hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    ccb =hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    gcb.set_model( generator )
    dcb.set_model( discriminator )
    ccb.set_model( combined )


    gcb.on_train_begin()
    dcb.on_train_begin()
    ccb.on_train_begin()
  

    intime = time.time()
  
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(NTrain/batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        for index in range(nb_batches):
            X_train, y_train= Traingen.__next__()
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, nb_batches))

            noise = np.random.normal(0, 1, (batch_size, latent_size))

            #image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            #energy_batch = y_train[index * batch_size:(index + 1) * batch_size]
            #ecal_batch = ecal_train[index * batch_size:(index + 1) * batch_size]
            image_batch = X_train.astype(np.float32)
            energy_batch = y_train.astype(np.float32)
            ecal_batch = np.sum(image_batch, axis=(1, 2, 3))


            print(image_batch.shape)
            print(ecal_batch.shape)
            sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
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
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = np.multiply(2, sampled_energies)

                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])
    ttime = time.time() -intime
    print( 'Training time ', ttime)     

#        print('\nTesting for epoch {}:'.format(epoch + 1))

#        noise = np.random.normal(0, 1, (nb_test, latent_size))

#        sampled_energies = np.random.uniform(0, 5, (nb_test, 1))
#        generator_ip = np.multiply(sampled_energies, noise)
#        generated_images = generator.predict(generator_ip, verbose=False)
#        ecal_ip = np.multiply(2, sampled_energies)
#        sampled_energies = np.squeeze(sampled_energies, axis=(1,))
#        X = np.concatenate((X_test, generated_images))
#        y = np.array([1] * nb_test + [0] * nb_test)
#        ecal = np.concatenate((ecal_test, ecal_ip))
#        print(ecal.shape)
#        print(y_test.shape)
#        print(sampled_energies.shape)
#        aux_y = np.concatenate((y_test, sampled_energies), axis=0)
#        print(aux_y.shape)
#        discriminator_test_loss = discriminator.evaluate(
#            X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)
#
#        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

#        noise = np.random.normal(0, 1, (2 * nb_test, latent_size))
#        sampled_energies = np.random.uniform(0, 5, (2 * nb_test, 1))
#        generator_ip = np.multiply(sampled_energies, noise)
#        ecal_ip = np.multiply(2, sampled_energies)
#
#        trick = np.ones(2 * nb_test)

#        generator_test_loss = combined.evaluate(generator_ip,
#                                                [trick, sampled_energies.reshape((-1, 1)), ecal_ip], verbose=False, batch_size=batch_size)

#        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
#
#        train_history['generator'].append(generator_train_loss)
#        train_history['discriminator'].append(discriminator_train_loss)
#
#        test_history['generator'].append(generator_test_loss)
#        test_history['discriminator'].append(discriminator_test_loss)

#        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
#            'component', *discriminator.metrics_names))
#        print('-' * 65)
#
#        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f}'
#        print(ROW_FMT.format('generator (train)',
#                             *train_history['generator'][-1]))
#        print(ROW_FMT.format('generator (test)',
#                             *test_history['generator'][-1]))
#        print(ROW_FMT.format('discriminator (train)',
#                             *train_history['discriminator'][-1]))
#        print(ROW_FMT.format('discriminator (test)',
#                             *test_history['discriminator'][-1]))

        # save weights every epoch
        ## this needs to done only on one process. overwise each worker is writing it
        if hvd.rank()==0:
            print ("saving weights of gen")
            generator.save_weights('weights/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                                   overwrite=True)
            print ("saving weights of disc")        
            discriminator.save_weights('weights/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                       overwrite=True)
            
            pickle.dump({'train': train_history, 'test': test_history},
                        open('pion-dcgan-history.pkl', 'wb'))
