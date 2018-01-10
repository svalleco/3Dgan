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
from DLTools.ThreadedGenerator import DLh5FileGenerator
#from DLTools.GeneratorCacher import * 
import glob
import h5py
import numpy as np
from CaloDNN.NeuralNets.LoadData import *
from adlkit.data_provider.cached_data_providers import GeneratorCacher
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
    print('Train Events in files = {}'.format(len(TrainSampleList) * 10000)
    print('Test Events in files = {}'.format(len(TestSampleList) * 10000)

    sample_spec_train = list()
    for item in TrainSampleList:
        sample_spec_train.append((item[0], item[1] , item[2], 1))

    sample_spec_test = list()
    for item in TestSampleList:
        sample_spec_test.append((item[0], item[1] , item[2], 1))

    q_multipler = 2
    read_multiplier = 1
    n_buckets = 1

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
                                    sleep_duration=1,
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
                                   sleep_duration=1,
                                   wrap_examples=False)

    print ("Class Index Map:", Train_genC.config.class_index_map)

    return Train_genC,Test_genC,Norms,shapes,TrainSampleList,TestSampleList



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

    from ecalvegan import generator
    from ecalvegan import discriminator

    nb_epochs = 30
    batch_size = 128
    latent_size = 200
    verbose = 'false'
    batches = 0
    generator=generator(latent_size)
    discriminator=discriminator()
    nb_classes = 2

    #Generator variables

    ECALShape= None, 25, 25, 25
    HCALShape= None, 5, 5, 60
    FileSearch="/eos/project/d/dshep/LCD/V1/*scan/*.h5"
    #FileSearch="/afs/cern.ch/work/g/gkhattak/public/Ele_v1*.h5"
    train_file="/tmp/gulrukh-CaloDNN-LCD-TrainEvent-Cache.h5"
    test_file="/tmp/gulrukh-CaloDNN-LCD-TestEvent-Cache.h5"
    Particles=["Ele"]
    MaxEvents=int(8.e5)
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
    n_threads=3
    NTest = NTestSamples=20000
    NTrain = 180000
    #This function will setup Generators
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
    Train_gen = Train_genC.first().generate()

    Test_genC.start()
    Test_gen = Test_genC.first().generate()

    Test_cache = GeneratorCacher(Test_gen, BatchSize, max=NSamples,
                          wrap=False,
                          delivery_function=None,
                          cache_filename=train_file,
                          delete_cache_file=False)


    Train_cache = GeneratorCacher(Train_gen, BatchSize, max=NSamples,
                          wrap=True,
                          delivery_function=None,
                          cache_filename=test_file,
                          delete_cache_file=False)

    Traingen = Train_cache.DiskCacheGenerator()
    Testgen = Test_cache.PreloadGenerator()


    g_weights = 'params_generator_epoch_'
    d_weights = 'params_discriminator_epoch_'

    print('[INFO] Building discriminator')
    discriminator.summary()
    #discriminator.load_weights('veganweights/params_discriminator_epoch_019.hdf5')
    discriminator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[8, 0.2, 0.1]
        #loss=['binary_crossentropy', 'kullback_leibler_divergence']
    )

    # build the generator
    print('[INFO] Building generator')
    generator.summary()
    #generator.load_weights('veganweights/params_generator_epoch_019.hdf5')
    generator.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )

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


    nb_train, nb_test = NTrain, NTest
    nb_batches = NTrain/batch_size
    nb_testbatches= NTest/batch_size
    print('*************************************************************************************')
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        if verbose:
            progress_bar = Progbar(target=nb_batches)
        epoch_gen_train_loss = []
        epoch_disc_train_loss = []

        for batches in range(nb_batches):
            X_train, y_train= Traingen.next()
            print('Train generated')
            #batches+=1
            if verbose:
                progress_bar.update(batches)
            else:
                if batches % 100 == 0:
                    print('processed {} batches'.format( nb_batches))

            noise = np.random.normal(0, 1, (batch_size, latent_size))

            image_batch = X_train.astype(np.float32)
            energy_batch = y_train.astype(np.float32)
            ecal_batch = np.sum(image_batch, axis=(1, 2, 3))

            print(image_batch.shape)
            print(ecal_batch.shape)

            sampled_energies = np.random.uniform(0, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.multiply(2, sampled_energies)
            generated_images = generator.predict(generator_ip, verbose=0)

            real_batch_loss = discriminator.train_on_batch(image_batch, [bit_flip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [bit_flip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            epoch_disc_train_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)

            gen_losses = []

            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = np.multiply(2, sampled_energies)

                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))

            epoch_gen_train_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])
        
        print('\nTesting for epoch {}:'.format(epoch + 1))
        epoch_gen_test_loss = []
        epoch_disc_test_loss = []

        for D in range(nb_testbatches):
            X_test, y_test = Testgen.next()
            image_batch = X_train.astype(np.float32)
            energy_batch = y_train.astype(np.float32)
            ecal_batch = np.sum(image_batch, axis=(1, 2, 3))

            noise = np.random.normal(0, 1, (batch_size, latent_size))
            sampled_energies = np.random.uniform(0, 5, (batch_size, 1))
            generator_ip = np.multiply(sampled_energies, noise)
            generated_images = generator.predict(generator_ip, verbose=False)
            ecal_ip = np.multiply(2, sampled_energies)
            sampled_energies = np.squeeze(sampled_energies, axis=(1,))
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            ecal = np.concatenate((ecal_batch, ecal_ip))
            print(ecal.shape)
            print(y_test.shape)
            print(sampled_energies.shape)
            aux_y = np.concatenate((energy_batch, sampled_energies), axis=0)
            print(aux_y.shape)
            discriminator_test_loss = discriminator.evaluate(
                         X, [y, aux_y, ecal], verbose=False, batch_size=batch_size)
            noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
            sampled_energies = np.random.uniform(0, 5, (2 * batch_size, 1))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.multiply(2, sampled_energies)
            trick = np.ones(2 * batch_size)
            generator_test_loss = combined.evaluate(generator_ip,
                                                [trick, sampled_energies.reshape((-1, 1)), ecal_ip], verbose=False, batch_size=batch_size)
            epoch_disc_test_loss.append(discriminator_test_loss)
            epoch_gen_test_loss.append(generator_test_loss)

        discriminator_test_loss = np.mean(np.array(epoch_disc_test_loss), axis=0)
        generator_test_loss = np.mean(np.array(epoch_gen_test_loss), axis=0)

        discriminator_train_loss = np.mean(np.array(epoch_disc_train_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_train_loss), axis=0)

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

        pickle.dump({'train': train_history, 'test': test_history},
open('dcgan-history2.pkl', 'wb'))
