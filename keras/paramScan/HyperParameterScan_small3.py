#!/usr/bin/env python
# -*- coding: utf-8 -*-   

# This file has a function that will take a list of params as input and run training using that. Finally it will run analysis to get a single matric that we will need to optimize


from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import h5py
import os

import numpy as np
import glob

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.plots import plot_convergence

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils.generic_utils import Progbar

from sklearn.cross_validation import train_test_split
K.set_image_dim_ordering('tf')
import tensorflow as tf
config = tf.ConfigProto(log_device_placement=True)
import time
import numpy.core.umath_tests as umath
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
plt.switch_backend('Agg')
import math

def hpscan():
    space = [
         #Integer(25, 25), #name ='epochs'),  
         #Integer(5, 8), #name ='batch_size'),
         #Integer(8, 10), #name='latent size'),
         Categorical([1, 2, 5, 6, 8]), #name='gen_weight'),
         Categorical([0.1, 0.2, 1, 2, 10]), #name='aux_weight'),
         Categorical([0.1, 0.2, 1, 2, 10]), #name='ecal_weight'),
         #Real(10**-5, 10**0, "log-uniform"), #name ='lr'),
         #Real(8, 9), #name='rho'),
         #Real(0, 0.0001), #name='decay'), 
         #Categorical([True,False]), #name='dflag'),
         #Integer(4, 64), #name='df'),
         #Integer(2, 16), #name='dx'),
         #Integer(2, 16), #name='dy'),
         #Integer(2, 16), #name='dz'),
         #Real(0.01, 0.5), #name='dp'),
         #Categorical([True,False]), #name='gflag'),
         #Integer(4, 64), #name='gf'),
         #Integer(2, 16), #name='gx'),
         #Integer(2, 16), #name='gy'),
         #Integer(2, 16)] #name='gz')
           ]
    res_gp = gp_minimize(evaluate, space, n_calls=15, n_random_starts=5, verbose=True, random_state=0)
    "Best score=%.4f" % res_gp.fun
    print("""Best parameters:
    Loss Weights:
    _ Weight Gen loss ={}
    _ Weight Aux loss ={}
    _ Weight Ecal loss ={}
   """.format(res_gp.x[0], res_gp.x[1], res_gp.x[2]))
    plot_convergence(res_gp)

#Function to return a single value for a network performnace metric. The metric needs to be minimized.
def evaluate(params):
   gen_weights = "gen_weights.hdf5"
   disc_weights = "disc_weights.hdf5"
   datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5'
   num_events = 200000
   Trainfiles, Testfiles = GetFiles(datapath, nEvents=num_events, datasetnames=["ECAL"], Particles =["Ele"])
   latent =200
   print( Trainfiles)   
   # Just done to print the parameter setting to screen
   gen_weight, aux_weight, ecal_weight= params
   params1= [gen_weight, aux_weight, ecal_weight]
   print("Generation loss weight={}   Auxilliary loss weight={}   ECAL loss weight={}".format(*params))
   d = discriminator()
   g = generator()
   loss = vegantrain(d, g, Trainfiles, gen_weight = gen_weight, aux_weight = aux_weight, ecal_weight = aux_weight)
   score = analyse(g, True, False, gen_weights, Testfiles)
   return score

# Fuction used to flip random bits for training
def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

# Architecture that can have additional parametrized layer
def discriminator(dflag=0, df=8, dx=5, dy=5, dz=5, dp=0.2):

    image = Input(shape=(25, 25, 25, 1))
    x = image
    if dflag==1:
        x = Conv3D(df, dx, dy, dz, border_mode='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dp)(x)

    x = Conv3D(32, 5, 5, 5, border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(dp)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dp)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, 5, 5,5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dp)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dp)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
       
    dnn_out = dnn(image)

    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
    ecal = Lambda(lambda x: K.sum(x, axis=(1, 2, 3)))(image)
    Model(input=image, output=[fake, aux, ecal])
    return Model(input=image, output=[fake, aux, ecal])

def generator(latent_size=200, gflag=0, gf=8, gx=5, gy=5, gz=5):
    
    latent = Input(shape=(latent_size, ))

    x = Dense(64 * 7* 7)(latent)
    x = Reshape((7, 7,8, 8))(x)
    x = Conv3D(64, 6, 6, 8, border_mode='same', init='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = UpSampling3D(size=(2, 2, 2))(x)

    x = ZeroPadding3D((2, 2, 0))(x)
    x = Conv3D(6, 6, 5, 8, init='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = UpSampling3D(size=(2, 2, 3))(x)

    x = ZeroPadding3D((1,0,3))(x)
    x = Conv3D(6, 3, 3, 8, init='he_uniform')(x)
    x = LeakyReLU()(x)

    if gflag==1:
        x = Conv3D(gf, gx, gy, gz, init='he_uniform', border_mode='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

    x = Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal')(x)
    x = Activation('relu')(x)

    loc = Model(latent, x)
    fake_image = loc(latent)
    Model(input=[latent], output=fake_image)
    return Model(input=[latent], output=fake_image)

def GetFiles(FileSearch="/data/LCD/*/*.h5", nEvents=200000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))
    print ("Found {} files. ".format(len(Files)))
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

def get_data(datafile):
    #get data for training                                                                                                                                                                      
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    y=f.get('target')
    X=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, y, ecal

def sort(data, energies, num_events):
    X = data[0]
    Y = data[1]
    tolerance = 5
    srt = {}
    for energy in energies:
       indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
       if len(indexes) > num_events:
          indexes = indexes[:num_events]
       srt["events_act" + str(energy)] = X[indexes]
       srt["energy" + str(energy)] = Y[indexes]
    return srt

def save_sorted(srt, energies):
    for energy in energies:
       filename = "sorted_{:03d}.hdf5".format(energy)
       with h5py.File(filename ,'w') as outfile:
          outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
          outfile.create_dataset('Target',data=srt["energy" + str(energy)])
       print ("Sorted data saved to ", filename)

def load_sorted(sorted_path):
    sorted_files = sorted(glob.glob(sorted_path))
    energies = []
    srt = {}
    for f in sorted_files:
       energy = int(filter(str.isdigit, f)[:-1])
       energies.append(energy)
       srtfile = h5py.File(f,'r')
       srt["events_act" + str(energy)] = np.array(srtfile.get('ECAL'))
       srt["energy" + str(energy)] = np.array(srtfile.get('Target'))
       print ("Loaded from file", f)
    return energies, srt

def get_gen(energy):
    filename = "Gen_{:03d}.hdf5".format(energy)
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    print ("Generated file ", filename, " is loaded")
    return generated_images

def generate(g, index, latent, sampled_labels):
    noise = np.random.normal(0, 1, (index, latent))
    sampled_labels=np.expand_dims(sampled_labels, axis=1)
    gen_in = sampled_labels * noise
    generated_images = g.predict(gen_in, verbose=False, batch_size=50)
    return generated_images

def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

def get_moments(images, sumsx, sumsy, sumsz, totalE, m):
    ecal_size = 25
    totalE = np.squeeze(totalE)
    index = images.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = umath.inner1d(sumsx, moments) /totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = umath.inner1d(sumsy, moments) /totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = umath.inner1d(sumsz, moments)/totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

# This function will calculate two errors derived from position of maximum along an axis and the sum of ecal along the axis
def analyse(g, read_data, save_data, gen_weights, Testfiles):
   print ("Started")
   num_events=2000
   num_data = 100000
   sortedpath = 'sortedTest_*.hdf5'
   Test = False
   latent= 200
   m = 2
   var = {}
   g =generator(latent)
   if read_data:
     start = time.time()
     energies, var = load_sorted(sortedpath)
     sort_time = time.time()- start
     print ("Events were loaded in {} seconds".format(sort_time))
   else:
     # Getting Data
     events_per_file = 10000
     energies = [50, 100, 200, 250, 300, 400, 500]
     start = time.time()
     for index, dfile in enumerate(Testfiles):
        data = get_data(dfile)
        sorted_data = sort(data, energies, num_events)
        data = None
        if index==0:
          var.update(sorted_data)
        else:
          for key in var:
            var[key]= np.append(var[key], sorted_data[key], axis=0)
     data_time = time.time() - start
     print ("{} events were loaded in {} seconds".format(num_data, data_time))
     if save_data:
        save_sorted(var, energies)
   total = 0
   for energy in energies:
     var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
     total += var["index" + str(energy)]
     data_time = time.time() - start
   print ("{} events were put in {} bins".format(total, len(energies)))
   g.load_weights(gen_weights)

   start = time.time()
   for energy in energies:
     var["events_gan" + str(energy)] = generate(g, var["index" + str(energy)], latent, var["energy" + str(energy)]/100)
   gen_time = time.time() - start
   print ("{} events were generated in {} seconds".format(total, gen_time))

   for energy in energies:
     var["ecal_act"+ str(energy)] = np.sum(var["events_act" + str(energy)], axis = (1, 2, 3))
     var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
     var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
     var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
     var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["events_act" + str(energy)], var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m)
     var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["events_gan" + str(energy)], var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m)
   return matric4(var, energies, m)

def metric4(var, energies, m):

   ecal_size = 25
   metricp = 0
   metrice = 0
   for energy in energies:
     #Relative error on mean moment value for each moment and each axis
     x_act= np.mean(var["momentX_act"+ str(energy)], axis=0)
     x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
     y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
     y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
     z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
     z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
     var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act
     var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
     var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
     #Taking absolute of errors and adding for each axis then scaling by 3
     var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
     #Summing over moments and dividing for number of moments
     var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
     metricp += var["pos_total"+ str(energy)]
     #Take profile along each axis and find mean along events
     sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis=0), np.mean(var["sumsz_act" + str(energy)], axis=0)
     sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
     var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
     var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
     var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
     #Take absolute of error and mean for all events
     var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)]))/ecal_size
     var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)]))/ecal_size
     var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/ecal_size

     var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
     metrice += var["eprofile_total"+ str(energy)]
   metricp = metricp/len(energies)
   metrice = metrice/len(energies)
   tot = metricp + metrice
   print('Energy\t\tEvents\t\tPosition Error\tEnergy Error')
   for energy in energies:
     print ("%d \t\t%d \t\t%f \t\t%f" %(energy, var["index" +str(energy)], var["pos_total"+ str(energy)], var["eprofile_total"+ str(energy)]))
   print(" Total Position Error = %.4f\t Total Energy Profile Error =   %.4f" %(metricp, metrice))
   print(" Total Error =  %.4f" %(tot))
   return(tot, metricp, metrice)

## Training Function
def vegantrain(d, g, Trainfiles, epochs=10, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0):
#dflag=0, df= 16, dx=8, dy=8, dz= 8, dp=0.2, gflag=0, gf= 16, gx=8, gy=8, gz= 8):
    init_start = time.time()
    g_weights = 'params_generator_epoch_'
    d_weights = 'params_discriminator_epoch_'
    verbose = 'false'
    print('[INFO] Building discriminator')
    d.compile(
        optimizer=RMSprop(lr=lr, rho=rho, decay=decay),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ecal_weight]
    )

    # build the generator                                                       
    print('[INFO] Building generator')
    g.compile(
        optimizer=RMSprop(lr=lr, rho=rho, decay=decay),
        loss='binary_crossentropy')
    latent = Input(shape=(latent_size, ), name='combined_z')
    fake_image = g(latent)
    d.trainable = False
    fake, aux, ecal = d(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ecal],
        name='combined_model')
    combined.compile(
        optimizer=RMSprop(lr=lr, rho=rho, decay=decay),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ecal_weight])
    
    nb_train = 10000 * len(Trainfiles)# Total events in training files                         
    total_batches = nb_train / batch_size
    train_history = defaultdict(list)
    for epoch in range(epochs):
        start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, epochs))
        epoch_start = time.time()
        epoch_gen_loss = []
        epoch_disc_loss = []
        X_train, Y_train, ecal_train = get_data(Trainfiles[0])
       
        nb_file=1
        nb_batches = int(X_train.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=total_batches)
        file_index = 0
        for index in range(total_batches):
            if verbose:
                progress_bar.update(index)
            if index % 100 == 0:
               print('processed {}/{} batches'.format(index + 1, nb_batches))

            loaded_data = X_train.shape[0]
            used_data = file_index * batch_size
            if (loaded_data - used_data) < batch_size + 1 and (nb_file < len(Trainfiles)):
                X_temp, Y_temp, ecal_temp= get_data(Trainfiles[nb_file])
                nb_file+=1
                X_left = X_train[(file_index * batch_size):]
                Y_left = Y_train[(file_index * batch_size):]
                ecal_left = ecal_train[(file_index * batch_size):]
#                print(Y_left.shape)
                X_train = np.concatenate((X_left, X_temp))
                Y_train = np.concatenate((Y_left, Y_temp))
                ecal_train = np.concatenate((ecal_left, ecal_temp))
                nb_batches = int(X_train.shape[0] / batch_size)
#                print("{} batches loaded..........".format(nb_batches))
                file_index = 0
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            image_batch = X_train[(file_index * batch_size):(file_index  + 1) * batch_size]
            energy_batch = Y_train[(file_index * batch_size):(file_index + 1) * batch_size]
            ecal_batch = ecal_train[(file_index *  batch_size):(file_index + 1) * batch_size]
            file_index +=1
#            print(image_batch.shape)
#            print(ecal_batch.shape)

            sampled_energies = np.random.uniform(0.1, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.multiply(2, sampled_energies)
            generated_images = g.predict(generator_ip, verbose=0)
            real_batch_loss = d.train_on_batch(image_batch, [bit_flip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = d.train_on_batch(generated_images, [bit_flip(np.zeros(batch_size)), sampled_energies, ecal_ip])
            epoch_disc_loss.append([(a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)])
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

        epoch_time = time.time() - start
        print('Training for {} epoch took {} seconds'.format(epoch, epoch_time))
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)      
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        #print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
         #   'component', *d.metrics_names))
       # print('-' * 65)

        # print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
        # print(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
           
    #save weights at last epoch                                                                                   
    g.save_weights('gen_weights.hdf5'.format(g_weights, epoch), overwrite=True)
    d.save_weights('disc_weights.hdf5'.format(d_weights, epoch), overwrite=True)
    return epoch_gen_loss[nb_batches -1][1]

if __name__ == "__main__":
    hpscan()
