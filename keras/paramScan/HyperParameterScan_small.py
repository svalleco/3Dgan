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

def get_data(datafile):
    #get data for training                                                                                                                                                                        
    print('Loading Data.....')
    start_load = time.time()
    f=h5py.File(datafile,'r')
    y=f.get('target')
    X=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    X[X < 1e-6] = 0
    X = np.expand_dims(X, axis=-1)
    #y=np.expand_dims(y[:,1], axis=-1)
    #y = y/100
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    X_train, X_test, y_train, y_test, ecal_train, ecal_test = train_test_split(X, y, ecal, train_size=0.9)
    load_time = time.time()- start_load
    print('Data is loaded in {:2} seconds.'.format(load_time))
    return X_train, X_test, y_train, y_test, ecal_train, ecal_test
  
## Training Function
def vegantrain(d, g, X_train, y_train, ecal_train, epochs=10, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0):
#dflag=0, df= 16, dx=8, dy=8, dz= 8, dp=0.2, gflag=0, gf= 16, gx=8, gy=8, gz= 8):
    init_start = time.time()
    g_weights = 'params_generator_epoch_'
    d_weights = 'params_discriminator_epoch_'

    #d= discriminator(dflag=dflag, df= df, dx=dx, dy=dy, dz= dz, dp=dp)
    #g= generator(latent_size=latent_size, gflag=gflag, gf=gf, gx=gx, gy=gy, gz=gz)

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
    y_train= y_train/100
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
    
    nb_train= X_train.shape[0]
    train_history = defaultdict(list)
    init_time = time.time() - init_start
    print('Initialization time = {}'.format(init_time))
    for epoch in range(epochs):

        print('Epoch {} of {}'.format(epoch + 1, epochs))
        epoch_start = time.time()
        nb_batches = int(X_train.shape[0] / batch_size)
        
        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            #if index % 100 == 0:
             #       print('processed {}/{} batches'.format(index + 1, nb_batches))

            noise = np.random.normal(0, 1, (batch_size, latent_size))

            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            energy_batch = y_train[index * batch_size:(index + 1) * batch_size]
            ecal_batch = ecal_train[index * batch_size:(index + 1) * batch_size]
            sampled_energies = np.random.uniform(0, 5,( batch_size,1 ))
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
                sampled_energies = np.random.uniform(0, 5, ( batch_size,1 ))
                generator_ip = np.multiply(sampled_energies, noise)
                ecal_ip = np.multiply(2, sampled_energies)
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, sampled_energies.reshape((-1, 1)), ecal_ip]))

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

        epoch_time = time.time() - epoch_start
        #print('Training for one epoch took {} seconds'.format(epoch_time))

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        #print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
         #   'component', *d.metrics_names))
       # print('-' * 65)

        # print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
        # print(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
           
    pickle.dump({'train': train_history},open('dcgan-history.pkl', 'wb'))

    #save weights at last epoch                                                                                          
    g.save_weights('gen_weights.hdf5'.format(g_weights, epoch), overwrite=True)
    d.save_weights('disc_weights.hdf5'.format(d_weights, epoch), overwrite=True)
    return epoch_gen_loss[nb_batches -1][1]

# This function will calculate two errors derived from position of maximum along an axis and the sum of ecal along the axis
def analyse(d, g, X, Y, gen_weights, disc_weights, latent=200):
   print ("Started")
   num_events=1000
   energies=[50, 100, 150, 200, 300, 400, 500] 
   tolerance = 5
   m = 2
   g.load_weights(gen_weights)
   d.load_weights(disc_weights)
   #X = np.concatenate(X_train, X_test)
   Y = np.reshape(Y, (-1, 1))
   #ecal = np.concatenate(ecal_train, ecal_test)
   # Initialization of parameters  
   var = {}
   for energy in energies:
     var["index" + str(energy)] = 0
     var["events_act" + str(energy)] = np.zeros((num_events, 25, 25, 25))
     var["max_pos_act_" + str(energy)] = np.zeros((num_events, 3))
     var["sumact" + str(energy)] = np.zeros((num_events, 3, 25))
     var["energy_sampled" + str(energy)] = np.zeros((num_events, 1))
     var["max_pos_gan_" + str(energy)] = np.zeros((num_events, 3))
     var["sumgan" + str(energy)] = np.zeros((num_events, 3, 25))
     var["x_act" + str(energy)] = np.zeros((num_events, m))
     var["y_act" + str(energy)] =np.zeros((num_events, m))
     var["z_act" + str(energy)] =np.zeros((num_events, m))
     var["x_gan" + str(energy)] =np.zeros((num_events, m))
     var["y_gan" + str(energy)] =np.zeros((num_events, m))
     var["z_gan" + str(energy)] =np.zeros((num_events, m))
          
   ## Sorting data in bins                                                         
   size_data = int(X.shape[0])
   #print ("Sorting data")
   #print (X.shape)
   #print (Y.shape)
   #print (Y[:10])
   for i in range(size_data):
     for energy in energies:
        if Y[i][0] > energy-tolerance and Y[i][0] < energy+tolerance and var["index" + str(energy)] < num_events:
            var["events_act" + str(energy)][var["index" + str(energy)]]= np.squeeze(X[i])
            var["energy_sampled" + str(energy)][var["index" + str(energy)]] = Y[i]/100
            var["index" + str(energy)]= var["index" + str(energy)] + 1
            
        #print('The energy {} has {} events'.format(energy, var["index" + str(energy)]))
   # Generate images
   for energy in energies:        
        noise = np.random.normal(0, 1, (var["index" + str(energy)], latent))
        #print(energy, var["index" + str(energy)], noise.shape)
        sampled_labels = var["energy_sampled" + str(energy)]
        generator_in = np.multiply(sampled_labels, noise)
        generated_images = g.predict(generator_in, verbose=False, batch_size=100)
        var["events_gan" + str(energy)]= np.squeeze(generated_images)
        var["isreal_gan" + str(energy)], var["energy_gan" + str(energy)], var["ecal_gan"] = np.array(d.predict(generated_images, verbose=False, batch_size=100))
        var["isreal_act" + str(energy)], var["energy_act" + str(energy)], var["ecal_act"] = np.array(d.predict(np.expand_dims(var["events_act" + str(energy)], -1), verbose=False, batch_size=100))
        #print(var["events_gan" + str(energy)].shape)
        #print(var["events_act" + str(energy)].shape)
# calculations                                                                                        
   for energy in energies:
     for j in range(var["index" + str(energy)]):
        var["max_pos_act_" + str(energy)][j] = np.unravel_index(var["events_act" + str(energy)][j].argmax(), (25, 25, 25))
        var["sumact" + str(energy)][j, 0] = np.sum(var["events_act" + str(energy)][j], axis=(1,2))
        var["sumact" + str(energy)][j, 1] = np.sum(var["events_act" + str(energy)][j], axis=(0,2))
        var["sumact" + str(energy)][j, 2] = np.sum(var["events_act" + str(energy)][j], axis=(0,1))
        var["max_pos_gan_" + str(energy)][j] = np.unravel_index(var["events_gan" + str(energy)][j].argmax(), (25, 25, 25))
        var["sumgan" + str(energy)][j, 0] = np.sum(var["events_gan" + str(energy)][j], axis=(1,2))
        var["sumgan" + str(energy)][j, 1] = np.sum(var["events_gan" + str(energy)][j], axis=(0,2))
        var["sumgan" + str(energy)][j, 2] = np.sum(var["events_gan" + str(energy)][j], axis=(0,1))
     var["totalE_act" + str(energy)] = np.sum(var["events_act" + str(energy)][:var["index" + str(energy)]], axis=(1, 2, 3))
     var["totalE_gan" + str(energy)] = np.sum(var["events_gan" + str(energy)][:var["index" + str(energy)]], axis=(1, 2, 3))
     # Moments Computations                                                                                          
     ecal_size = 25
     ECAL_midX = np.zeros(var["index" + str(energy)])
     ECAL_midY = np.zeros(var["index" + str(energy)])
     ECAL_midZ = np.zeros(var["index" + str(energy)])
     for i in range(m):
        relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
        moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
        sumx = var["sumact" + str(energy)][0:(var["index" + str(energy)]), 0]
        ECAL_momentX = umath.inner1d(sumx, moments)/var["totalE_act" + str(energy)]
        if i==0: ECAL_midX = ECAL_momentX.transpose()
        var["x_act"+ str(energy)][:var["index" + str(energy)],i]= ECAL_momentX
     for i in range(m):
        relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
        moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
        ECAL_momentY = umath.inner1d(var["sumact" + str(energy)][:var["index" + str(energy)], 1], moments)/var["totalE_act" + str(energy)]
        if i==0: ECAL_midY = ECAL_momentY.transpose()
        var["y_act"+ str(energy)][:var["index" + str(energy)],i]= ECAL_momentY
     for i in range(m):
        relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
        moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
        ECAL_momentZ = umath.inner1d(var["sumact" + str(energy)][:var["index" + str(energy)], 2], moments)/var["totalE_act" + str(energy)]
        if i==0: ECAL_midZ = ECAL_momentZ.transpose()
        var["z_act"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentZ

     ECAL_midX = np.zeros(var["index" + str(energy)])
     ECAL_midY = np.zeros(var["index" + str(energy)])
     ECAL_midZ = np.zeros(var["index" + str(energy)])
     for i in range(m):
        relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
        moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
        ECAL_momentX = umath.inner1d(var["sumgan" + str(energy)][:var["index" + str(energy)], 0], moments)/var["totalE_gan" + str(energy)]
        if i==0: ECAL_midX = ECAL_momentX.transpose()
        var["x_gan"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentX
     for i in range(m):
        relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
        moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
        ECAL_momentY = umath.inner1d(var["sumgan" + str(energy)][:var["index" + str(energy)], 1], moments)/var["totalE_gan" + str(energy)]
        if i==0: ECAL_midY = ECAL_momentY.transpose()
        var["y_gan"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentY
     for i in range(m):
        relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
        moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
        ECAL_momentZ = umath.inner1d(var["sumgan" + str(energy)][:var["index" + str(energy)], 2], moments)/var["totalE_gan" + str(energy)]
        if i==0: ECAL_midZ = ECAL_momentZ.transpose()
        var["z_gan"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentZ
   metricp = 0
   metrice = 0
   for energy in energies:
       #Relative error on mean moment value for each moment and each axis                                          
       x_act= np.sum(var["x_act"+ str(energy)], axis=0)/ var["index" + str(energy)]
       x_gan= np.sum(var["x_gan"+ str(energy)], axis=0)/ var["index" + str(energy)]
       y_act= np.sum(var["y_act"+ str(energy)], axis=0)/ var["index" + str(energy)]
       y_gan= np.sum(var["y_gan"+ str(energy)], axis=0)/ var["index" + str(energy)]
       z_act= np.sum(var["z_act"+ str(energy)], axis=0)/ var["index" + str(energy)]
       z_gan= np.sum(var["z_gan"+ str(energy)], axis=0)/ var["index" + str(energy)]
       var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act
       var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
       var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
       var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
       #Summing over moments and dividing for number of moments                                                    
       var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m

       metricp += var["pos_total"+ str(energy)]

       #Take profile along each axis and find mean along events                                                    
       sumact = np.mean(var["sumact" + str(energy)][:var["index" + str(energy)]], axis=0)
       sumgan = np.mean(var["sumgan" + str(energy)][:var["index" + str(energy)]], axis=0)
       var["eprofile_error"+ str(energy)] = np.divide((sumact - sumgan), sumact)
       var["eprofile_total"+ str(energy)]= np.sum(np.absolute(var["eprofile_error"+ str(energy)]), axis=1)/ecal_size
       var["eprofile_total"+ str(energy)]= np.sum(var["eprofile_total"+ str(energy)])/3
       metrice += var["eprofile_total"+ str(energy)]
   metricp = metricp/len(energies)
   metrice = metrice/len(energies)
  
   tot = metricp + metrice

   #for energy in energies:
      # print ("%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f \t\t%f \t\t%f" %(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]), str(np.unravel_index(var["events_gan" + str(energy)].argmax(), (var["index" + str(energy)], 25, 25, 25))), np.mean(var["events_gan" + str(energy)]), np.amin(var["events_gan" + str(energy)]), var["pos_total"+ str(energy)], var["eprofile_total"+ str(energy)]))
   #print(" Position Error = %.4f\t Energy Profile Error =   %.4f" %(metricp, metrice))
   #print(" Total Error =  %.4f" %(tot))
   return(tot)

#Function to return a single value for a network performnace metric. The metric needs to be minimized.
def objective(params):
   gen_weights = "gen_weights.hdf5"
   disc_weights = "disc_weights.hdf5"
   datafile = "/nfshome/gkhattak/Ele_v1_1_2.h5"
   latent =200
   X_train, X_test, y_train, y_test, ecal_train, ecal_test= get_data(datafile)
   
   # Just done to print the parameter setting to screen
   gen_weight, aux_weight, ecal_weight= params
   #params1= [gen_weight, aux_weight, ecal_weight]
   #print("Generation loss weight={}   Auxilliary loss weight={}   ECAL loss weight={}".format(*params))
   d = discriminator()
   g = generator()
   loss = vegantrain(d, g, X_train, y_train, ecal_train, gen_weight = gen_weight, aux_weight = aux_weight, ecal_weight = aux_weight)
   score = analyse(d, g, X_train, y_train, gen_weights, disc_weights)
   return score

def hpscan():
    space = [
         #Integer(25, 25), #name ='epochs'),  
         #Integer(5, 8), #name ='batch_size'),
         #Integer(8, 10), #name='latent size'),
         Categorical([1, 2, 5, 6, 8]), #name='gen_weight'),
         Categorical([0.1, 0.2, 1, 2]), #name='aux_weight'),
         Categorical([0.1, 1.0, 10.0]), #name='ecal_weight'),
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
    res_gp = gp_minimize(objective, space, n_calls=15, n_random_starts=5, n_jobs=3, verbose=True, random_state=0)
    "Best score=%.4f" % res_gp.fun
    print("""Best parameters:
    Loss Weights:
    _ Weight Gen loss ={}
    _ Weight Aux loss ={}
    _ Weight Ecal loss ={}
   """.format(res_gp.x[0], res_gp.x[1], res_gp.x[2]))
                                         

def analysisTest(X, y, weightpath, resultfile):
    from arch16 import generator, discriminator
    genpath = weightpath +  "/params_generator*.hdf5"
    discpath = weightpath + "/params_discriminator*.hdf5"
    
    gen_weights=[]
    disc_weights=[]
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
    for f in sorted(glob.glob(discpath)):
      disc_weights.append(f)
    print(len(gen_weights))
    print(len(disc_weights))
    print(X.shape)
    print(y.shape)
    results = []
    d = discriminator()
    g = generator(200)
#    for i in range(len(gen_weights)):
    for i in range(2):                                                                                            
       results.append(analyse(d, g, X, y, gen_weights[i], disc_weights[i]))

#    for i in range(len(gen_weights)):
    for i in range(2):                                                                                            
       print ('The results for......',gen_weights[i])
       print (" The result for {} = {:.2f} , {:.2f}, {:.2f}".format(i, results[i][0], results[i][1], results[i][2]\
))
    total=[]
    pos_e=[]
    energy_e=[]
    for item in results:
        total.append(item[0])
        pos_e.append(item[1])
        energy_e.append(item[2])
    plt.figure()
    plt.plot(total, label = 'Total')
    plt.plot(pos_e, label = 'Max pos error')
    plt.plot(energy_e , label = 'Energy profile error')
    plt.legend()
    plt.xticks(np.arange(0, 30, 1.0))
    plt.savefig(resultfile)

if __name__ == "__main__":
    #hpscan()
    datafile = "/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5"
    weightpath = "/afs/cern.ch/work/g/gkhattak/weights/arch16weights"
    resultfile = 'result_train.pdf'          
    #X_train, X_test, y_train, y_test, ecal_train, ecal_test= get_data(datafile)
    #analysisTest(X_train, y_train, weightpath, resultfile)
    hpscan()
    plot_convergence(res_gp)
