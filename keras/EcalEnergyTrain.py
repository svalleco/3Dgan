#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code trains 3D gan and saves loss history and weights in result and weights directories respectively. GANutils is also required.
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
import glob
import h5py 
import numpy as np
import time
import math
import analysis.utils.GANutils as gan # some common functions for gan

import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils.generic_utils import Progbar
import tensorflow as tf

config = tf.ConfigProto(log_device_placement=True)
from keras.callbacks import TensorBoard

if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a flag for host can be used
    tlab = True
    print('tlab = True')
else:
    tlab= False
    print('tlab = False')
    
try:
    import setGPU #if Caltech
except:
    pass

def write_log(callback, common_tag, tags, logs, batch_no):
    for tag, value in zip(tags, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = common_tag + tag
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def main():

    #Architectures to import
    from EcalEnergyGan import generator, discriminator

    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    tf_flags=params.tf_flags
    datapath = params.datapath#Data path
    dformat = params.dformat
    nEvents = params.nbEvents#Total events for training
    fitmod = params.mod# Fit to use
    weightdir = 'weights/3dgan_weights' + params.name # weight dir
    xscale = params.xscale #scaling of data
    thresh = params.thresh#threshold for energies
    pklfile = 'results/3dgan_history'+ params.name + '.pkl' # loss history
    lossweights = [params.gen_weight, params.aux_weight, params.ecal_weight] # weights for losses [Gen loss, Aux loss, Ecal sum loss]
    # Analysis
    analysis=params.analyse # if analysing
    energies =params.energies # Bins
    resultfile = 'results/3dgan_analysis' + params.name + '.pkl' # analysis result
    if tlab:
       datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5'
       weightdir = '/gkhattak/weights/EnergyWeights/3dgan_weights' + params.name
       pklfile = '/gkhattak/results/3dgan_history' + params.name + '.pkl'
       resultfile = '/gkhattak/results/3dgan_analysis'  + params.name + '.pkl'
    
    print(params)
    gan.safe_mkdir(weightdir)
    print(weightdir)
    # Building discriminator and generator
    K.set_image_data_format(dformat) #setting data format)
    d=discriminator(keras_dformat=dformat)
    g=generator(latent_size=latent_size, keras_dformat=dformat)
    Gan3DTrain(d, g, datapath, nEvents, weightdir, pklfile, resultfile, mod=fitmod, nb_epochs=nb_epochs, batch_size=batch_size, latent_size =latent_size 
               , loss_weights=lossweights, xscale = xscale, thresh=thresh, analysis=analysis, energies=energies, tf_flags=tf_flags, dformat=dformat)

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--nbepochs', action='store', type=int, default=50, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/bigdata/shared/LCD/NewV1/*scan/*.h5', help='HDF5 files to train from.') # Caltech
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Number of Data points to use')
    parser.add_argument('--verbose', action='store_true', default=False, help='Whether or not to use a progress bar')
    parser.add_argument('--tf_flags', action='store', default=False, help='Setting Tensorflow flags')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='Keras format')
    parser.add_argument('--mod', action='store', type=int, default=1, help='How to calculate Ecal sum corressponding to energy.\n [0].. factor 50 \n[1].. Fit from Root')
    parser.add_argument('--xscale', action='store', type=int, default=100, help='Multiplication factor for ecal deposition')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--gen_weight', action='store', type=float, default=2, help='loss weight for generation real/fake loss')
    parser.add_argument('--aux_weight', action='store', type=float, default=0.1, help='loss weight for auxilliary energy regression loss')
    parser.add_argument('--ecal_weight', action='store', type=float, default=0.1, help='loss weight for ecal sum loss')
    parser.add_argument('--analyse', action='store_true', default=True, help='Whether or not to perform analysis')
    parser.add_argument('--energies', action='store', type=int, default=[100, 200, 300, 400], help='Energy bins for analysis')
    parser.add_argument('--name', action='store', type=str, default='train', help='Identifier for current training')
    parser.add_argument('--thresh', action='store', type=int, default=1e-6, help='energy threshold for be used')
    return parser

# Ths functions loads data from a file and also does any pre processing
def GetprocData(datafile, xscale = 1, yscale = 100, thresh = 1e-6, dformat='channels_last'):
    #get data for training
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    Y=f.get('target')
    X=np.array(f.get('ECAL'))
    Y=(np.array(Y[:,1]))
    X[X < thresh] = 0
    if dformat=='channels_last':
      X = np.expand_dims(X, axis=-1)
      daxis=(1, 2, 3)
    else:
      X = np.expand_dims(X, axis=1)
      daxis=(2, 3, 4)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    X = xscale * X
    Y = Y/yscale
    ecal = np.sum(X, axis=daxis)
    return X, Y, ecal

def Gan3DTrain(discriminator, generator, datapath, nEvents, WeightsDir, pklfile, resultfile, mod=0, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[2, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, thresh=1e-6, analysis=False, energies=[], tf_flags=False, dformat='channels_last'):
    if tf_flags:
       tf.flags.DEFINE_string("d", "/data/svalleco/Ele_v1_1_2.h5", "data file")
       tf.flags.DEFINE_integer("bs", 128, "inference batch size")
       tf.flags.DEFINE_integer("num_inter_threads", 1, "number of inter_threads")
       tf.flags.DEFINE_integer("num_intra_threads", 56, "number of intra_threads")
       tf.flags.DEFINE_integer("num_epochs", 2, "number of epochs")
       FLAGS = tf.flags.FLAGS

       session_config = tf.ConfigProto(log_device_placement=True, inter_op_parallelism_threads=FLAGS.num_inter_threads, intra_op_parallelism_threads=FLAGS.num_intra_threads)
       session = tf.Session(config=session_config)
       K.set_session(session)

    start_init = time.time()
    verbose = False
    particle = 'Ele'
    f = [0.9, 0.1]
    print('[INFO] Building discriminator')
    discriminator.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )

    # build the generator
    print('[INFO] Building generator')

    generator.compile(
        optimizer=RMSprop(),
        loss='binary_crossentropy'
    )
    print('disc updates for trainable', discriminator.updates)
    # build combined Model
    latent = Input(shape=(latent_size, ), name='combined_z')   
    fake_image = generator( latent)
    discriminator.trainable=False
    fake, aux, ecal = discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ecal],
        name='combined_model'
    )
    combined.compile(
        optimizer=RMSprop(),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )
    # the discriminator is made trainable again so that non-trainable updates are applied (keras 2 bug)
    discriminator.trainable=True
    log_path = './logs/' + time.strftime("%Y%m%d-%H%M%S")
    callback = TensorBoard(log_path)
    callback.set_model(combined)
    
    # Getting Data
    Trainfiles, Testfiles = gan.DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
    print('The total data was divided in {} Train files and {} Test files'.format(len(Trainfiles), len(Testfiles)))
    nb_test = int(nEvents * f[1])
    if nb_test % batch_size >0:
      nb_test = (nb_test/batch_size) * batch_size
    #Read test data into a single array
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ecal_test = GetprocData(dtest, xscale=xscale, thresh=thresh, dformat=dformat)
       else:
           if X_test.shape[0] < nb_test:
              X_temp, Y_temp, ecal_temp = GetprocData(dtest, xscale=xscale, thresh=thresh, dformat=dformat)
              X_test = np.concatenate((X_test, X_temp))
              Y_test = np.concatenate((Y_test, Y_temp))
              ecal_test = np.concatenate((ecal_test, ecal_temp))
    X_test, Y_test, ecal_test = X_test[:nb_test], Y_test[:nb_test], ecal_test[:nb_test]

    nb_train = int(nEvents * f[0]) #
    total_batches = int(nb_train / batch_size)
    print('In this experiment {} events will be used for training as {}batches'.format(nb_train, total_batches))
    print('{} events will be used for Testing'.format(nb_test))

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    analysis_history = defaultdict(list)
    
    tb_tags = ['total_loss', 'generation_loss', 'auxiliary_loss', 'lambda_loss']

    init_time = time.time()- start_init
    print('Initialization time is {} seconds'.format(init_time))
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
        X_train, Y_train, ecal_train = GetprocData(Trainfiles[0], xscale=xscale, thresh=thresh, dformat=dformat)
        nb_file=1
        nb_batches = int(X_train.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=total_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        file_index = 0
                
        for index in np.arange(total_batches): # Training is controlled by number of batches for events=nEvents
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, total_batches))
            loaded_data = X_train.shape[0]
            used_data = file_index * batch_size
            if (loaded_data - used_data) < batch_size + 1 and (nb_file < len(Trainfiles)):
                X_train = X_train[(file_index * batch_size):]
                Y_train = Y_train[(file_index * batch_size):]
                ecal_train = ecal_train[(file_index * batch_size):]
                X_temp, Y_temp, ecal_temp = GetprocData(Trainfiles[nb_file], xscale=xscale, thresh=thresh, dformat=dformat)
                print("\nData file loaded..........",Trainfiles[nb_file])
                nb_file+=1
                X_train = np.concatenate((X_train, X_temp))
                Y_train = np.concatenate((Y_train, Y_temp))
                ecal_train = np.concatenate((ecal_train, ecal_temp))
                nb_batches = int(X_train.shape[0] / batch_size)                
                print("{} batches loaded..........".format(nb_batches))
                file_index = 0

            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            file_index +=1
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
      
            #ecal sum from fit
            ecal_ip = gan.GetEcalFit(sampled_energies, particle,mod, xscale)
            generated_images = generator.predict(generator_ip, verbose=0)
    
            real_batch_loss = discriminator.train_on_batch(image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [gan.BitFlip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            
            #print("BN weights after training disc.........")
            #for _ in discriminator.layers[1].layers[7].get_weights(): print('BN 1 = {}'.format(str(_)))
            #for _ in discriminator.layers[1].layers[12].get_weights(): print('BN 2 = {}'.format(str(_)))
            #for _ in discriminator.layers[1].layers[17].get_weights(): print('BN 3 = {}'.format(str(_)))
            
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            trick = np.ones(batch_size)
            gen_losses = []
            for _ in np.arange(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_energies = np.random.uniform(0.1, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))
            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])
            #print("BN weights after training combined.........")
            #for _ in discriminator.layers[1].layers[7].get_weights(): print('BN 1 = {}'.format(str(_)))
            #for _ in discriminator.layers[1].layers[12].get_weights(): print('BN 2 = {}'.format(str(_)))
            #for _ in discriminator.layers[1].layers[17].get_weights(): print('BN 3 = {}'.format(str(_)))


        print('The training took {} seconds.'.format(time.time()-epoch_start))
        print('\nTesting for epoch {}:'.format(epoch + 1))
        test_start=time.time()
        noise = np.random.normal(0.1, 1, (nb_test, latent_size))
        sampled_energies = np.random.uniform(0.1, 5, (nb_test, 1))
        generator_ip = np.multiply(sampled_energies, noise)
        generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
        ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
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
        ecal_ip = gan.GetEcalFit(sampled_energies, particle, mod, xscale)
        trick = np.ones(2 * nb_test)
        generator_test_loss = combined.evaluate(generator_ip,
                            [trick, sampled_energies.reshape((-1, 1)), ecal_ip], verbose=False, batch_size=batch_size)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        global_index = epoch * total_batches
        write_log(callback, 'discriminator_train/', tb_tags, discriminator_train_loss, global_index)
        write_log(callback, 'generator_train/', tb_tags, generator_train_loss, global_index)
        write_log(callback, 'discriminator_test/', tb_tags, discriminator_test_loss, global_index)
        write_log(callback, 'generator_test/', tb_tags, generator_test_loss, global_index)


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
        generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
        discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                overwrite=True)
        #discriminator.load_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch))
        print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, time.time()-test_start, WeightsDir))
        pickle.dump({'train': train_history, 'test': test_history}, open(pklfile, 'wb'))
        if analysis:
            var = gan.sortEnergy([np.squeeze(X_test), Y_test], np.squeeze(ecal_test), energies, ang=0)
            noise = np.random.normal(0.1, 1, (nb_test, latent_size))
            generator_ip = np.multiply(Y_test.reshape((-1, 1)), noise)
            generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
            result = gan.OptAnalysisShort(var, generated_images, energies, ang=0)
            print('Analysing............')
            # All of the results correspond to mean relative errors on different quantities
            analysis_history['total'].append(result[0]) 
            analysis_history['energy'].append(result[1])
            analysis_history['moment'].append(result[2])
            print('Result = ', result)
            pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

if __name__ == '__main__':
    main()

