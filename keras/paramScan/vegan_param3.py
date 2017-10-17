#!/usr/bin/env python
# -*- coding: utf-8 -*-   

# This file has a function that will take a list of params as input and run training using that. Finally it will run analysis to get a single metric that we will need to optimize


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

#this does not work so far
#from skopt import gp_minimize

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

# Fuction used to flip random bits for training
def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

# Architecture that can have additional parametrized layer
def discriminator(dflag=0, df=8, dx=5, dy=5, dz=5):

    image = Input(shape=(25, 25, 25, 1))

    x = Conv3D(32, 5, 5, 5, border_mode='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2,2))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, 5, 5,5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    if dflag==1:
        x = Conv3D(df, dx, dy, dz, border_mode='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, 5, 5, 5, border_mode='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    h = Flatten()(x)

    dnn = Model(image, h)
    dnn.summary()
    
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

    if gflag==1:
        x = Conv3D(gf, gx, gy, gz, init='he_uniform', border_mode='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

    x = ZeroPadding3D((1,0,3))(x)
    x = Conv3D(6, 3, 3, 8, init='he_uniform')(x)
    x = LeakyReLU()(x)
    x = Conv3D(1, 2, 2, 2, bias=False, init='glorot_normal')(x)
    x = Activation('relu')(x)

    loc = Model(latent, x)
    loc.summary()
    fake_image = loc(latent)
    Model(input=[latent], output=fake_image)
    return Model(input=[latent], output=fake_image)

## Training Function
def vegantrain(epochs=30, batch_size=128, latent_size=200, gen_weight=6, aux_weight=0.2, ecal_weight=0.1, lr=0.001, rho=0.9, decay=0.0, dflag=0, df= 16, dx=8, dy=8, dz= 8, gflag=0, gf= 16, gx=8, gy=8, gz= 8):
    g_weights = 'params_generator_epoch_'
    d_weights = 'params_discriminator_epoch_'

    d= discriminator(dflag, df= df, dx=dx, dy=dy, dz= dz)
    g= generator(latent_size=latent_size, gflag=gflag, gf=gf, gx=gx, gy=gy, gz=gz)

    print('[INFO] Building discriminator')
    d.summary()
    d.compile(
        optimizer=RMSprop(lr, rho=rho, decay=decay),
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=[gen_weight, aux_weight, ecal_weight]
    )

    # build the generator                                                       
    print('[INFO] Building generator')
    g.summary()
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
    
    #get data for training
    f=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
    e=f.get('target')
    X=np.array(f.get('ECAL'))
    y=(np.array(e[:,1]))
    # remove unphysical values                                       
    X[X < 1e-6] = 0
    X_train = X
    y_train = y                                                                                                                                                                                    
    X_train =np.array(np.expand_dims(X_train, axis=-1))
    y_train= np.array(y_train)/100
       
    nb_train= X_train.shape[0]
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    ecal_train = np.sum(X_train, axis=(1, 2, 3))
    
    train_history = defaultdict(list)
    
    for epoch in range(epochs):

        print('Epoch {} of {}'.format(epoch + 1, epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        
        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, nb_batches))

            noise = np.random.normal(0, 1, (batch_size, latent_size))

            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            energy_batch = y_train[index * batch_size:(index + 1) * batch_size]
            ecal_batch = ecal_train[index * batch_size:(index + 1) * batch_size]

            #print(image_batch.shape)
            #print(ecal_batch.shape)
            sampled_energies = np.random.uniform(0, 5,( batch_size,1 ))
            generator_ip = np.multiply(sampled_energies, noise)
            ecal_ip = np.multiply(2, sampled_energies)
            generated_images = g.predict(generator_ip, verbose=0)
            real_batch_loss = d.train_on_batch(image_batch, [bit_flip(np.ones(batch_size)), energy_batch, ecal_batch])
            fake_batch_loss = d.train_on_batch(generated_images, [bit_flip(np.zeros(batch_size)), sampled_energies, ecal_ip])

            epoch_disc_loss.append([
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

            epoch_gen_loss.append([
                (a + b) / 2 for a, b in zip(*gen_losses)
            ])

        #The testing portion may have to be removed but for the time being it is good to have it there

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
            'component', *d.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f}'
        print(ROW_FMT.format('generator (train)', *train_history['generator'][-1]))
       # print(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1]))
       # print(ROW_FMT.format('discriminator (test)',  *test_history['discriminator'][-1]))
        
        pickle.dump({'train': train_history},open('dcgan-history.pkl', 'wb'))

    #save weights at last epoch                                                                                          
    g.save_weights('gen_weights.hdf5'.format(g_weights, epoch), overwrite=True)
    d.save_weights('disc_weights.hdf5'.format(d_weights, epoch), overwrite=True)

# This function will calculate two errors derived from position of maximum along an axis and the sum of ecal along the axis
def analyse(gen_weights, disc_weights, datafile, latent=200, gflag=0, gf=8, gx=5, gy=5, gz=5, dflag=0, df=8, dx=5, dy=5, dz=5):
   print ("Started")
   num_events=3000
   data=h5py.File(datafile,'r')
   X=np.array(data.get('ECAL'))
   y=np.array(data.get('target'))
   Y=np.expand_dims(y[:,1], axis=-1)
   X[X < 1e-6] = 0
   print("Data is loaded")
   energies=[50, 100, 150, 200, 300, 400, 500] 
   tolerance = 5
   g = generator(latent_size=latent, gflag=0, gf=8, gx=5, gy=5, gz=5)
   g.load_weights(gen_weights)
   d = discriminator(dflag=0, df=8, dx=5, dy=5, dz=5)
   d.load_weights(disc_weights)

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
          
   ## Sorting data in bins                                                         
   size_data = int(X.shape[0])
   print ("Sorting data")
   print(Y[:10])
   for i in range(size_data):
     for energy in energies:
        if Y[i][0] > energy-tolerance and Y[i][0] < energy+tolerance and var["index" + str(energy)] < num_events:
            var["events_act" + str(energy)][var["index" + str(energy)]]= X[i]
            var["energy_sampled" + str(energy)][var["index" + str(energy)]] = Y[i]/100
            var["index" + str(energy)]= var["index" + str(energy)] + 1
   # Generate images
   for energy in energies:        
        noise = np.random.normal(0, 1, (var["index" + str(energy)], latent))
        sampled_labels = var["energy_sampled" + str(energy)]
        generator_in = np.multiply(sampled_labels, noise)
        generated_images = g.predict(generator_in, verbose=False, batch_size=100)
        var["events_gan" + str(energy)]= np.squeeze(generated_images)
        var["isreal_gan" + str(energy)], var["energy_gan" + str(energy)], var["ecal_gan"] = np.array(d.predict(generated_images, verbose=False, batch_size=100))
        var["isreal_act" + str(energy)], var["energy_act" + str(energy)], var["ecal_act"] = np.array(d.predict(np.expand_dims(var["events_act" + str(energy)], -1), verbose=False, batch_size=100))
        print(var["events_gan" + str(energy)].shape)
        print(var["events_act" + str(energy)].shape)
# calculations                                                                                        
   for j in range(num_events):
    for energy in energies:
      var["max_pos_act_" + str(energy)][j] = np.unravel_index(var["events_act" + str(energy)][j].argmax(), (25, 25, 25))
      var["sumact" + str(energy)][j, 0] = np.sum(var["events_act" + str(energy)][j], axis=(1,2))
      var["sumact" + str(energy)][j, 1] = np.sum(var["events_act" + str(energy)][j], axis=(0,2))
      var["sumact" + str(energy)][j, 2] = np.sum(var["events_act" + str(energy)][j], axis=(0,1))
      var["max_pos_gan_" + str(energy)][j] = np.unravel_index(var["events_gan" + str(energy)][j].argmax(), (25, 25, 25))
      var["sumgan" + str(energy)][j, 0] = np.sum(var["events_gan" + str(energy)][j], axis=(1,2))
      var["sumgan" + str(energy)][j, 1] = np.sum(var["events_gan" + str(energy)][j], axis=(0,2))
      var["sumgan" + str(energy)][j, 2] = np.sum(var["events_gan" + str(energy)][j], axis=(0,1))
   #### Generate Data table to screen                                                                    
   print ("Actual Data")
   print ("Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t")
   for energy in energies:
       print ("%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), str(np.unravel_index(var["events_act" + str(energy)].argmax(), (var["index" + str(energy)], 25, 25, 25))), np.mean(var["events_act" + str(energy)]), np.amin(var["events_act" + str(energy)])))

   #### Generate GAN table to screen                                                                 
                                                                                                      
   print ("Generated Data")
   print ("Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\tPosition error\t\t Energy Profile error\t\t")
   metricp = 0
   metrice = 0
   for energy in energies:
       var["pos_error"+ str(energy)] = var["max_pos_act_" + str(energy)] - var["max_pos_gan_" + str(energy)]
       var["pos_total"+ str(energy)] = np.sum(var["pos_error"+ str(energy)]**2)/num_events
       metricp += var["pos_total"+ str(energy)]
       var["eprofile_error"+ str(energy)]= var["sumact" + str(energy)] - var["sumgan" + str(energy)]
       var["eprofile_total"+ str(energy)]= np.sum(var["eprofile_error"+ str(energy)]**2)/num_events
       var["eprofile_total"+ str(energy)]= 400000 * var["eprofile_total"+ str(energy)]/(energy* energy)
       metrice += var["eprofile_total"+ str(energy)]
   tot = metricp + metrice
   for energy in energies:
       print ("%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f \t\t%f \t\t%f" %(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]), str(np.unravel_index(var["events_gan" + str(energy)].argmax(), (var["index" + str(energy)], 25, 25, 25))), np.mean(var["events_gan" + str(energy)]), np.amin(var["events_gan" + str(energy)]), var["pos_total"+ str(energy)], var["eprofile_total"+ str(energy)]))
       print(" Position Error = %.4f\t Energy Profile Error =   %.4f" %(metricp, metrice))
       print(" Total Error =  %.4f" %(tot))
   return(tot)

#Function to return a single value for a network performnace metric. The metric needs to be minimized.
def objective(params):
   gen_weights = "gen_weights.hdf5"
   disc_weights = "disc_weights.hdf5"
   datafile = "/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5"
   
   # Just done to print the parameter setting to screen
   epochs, batch_size, latent, gen_weight, aux_weight, ecal_weight, lr, rho, decay, dflag, df, dx, dy, dz, gflag, gf, gx, gy, gz= params
   params1= [1*epochs, pow(2,batch_size), latent, gen_weight, aux_weight, ecal_weight, pow(10,lr), rho * 0.1, decay, dflag, df, dx, dy, dz, gflag, gf, gx, gy, gz]
   print(len(params1))
   print("epochs= {}   batchsize={}   Latent space={}\nGeneration loss weight={}   Auxilliary loss weight={}   ECAL loss weight={}\nLearning rate={}   rho={}   decay={}\nDiscriminator: extra layer={}  filters={}  x={}  y={}  z{\
}\nGenerator: extra layer={}  filters={}  x={}  y={}  z{}\n".format(*params1))

   vegantrain(1*epochs, pow(2,batch_size), latent, gen_weight, aux_weight, ecal_weight, pow(10,lr), rho * 0.1, decay, dflag, df, dx, dy, dz, gflag, gf, gx, gy, gz)
   score = analyse(gen_weights, disc_weights, datafile, latent, dflag, df, dx, dy, dz, gflag, gf, gx, gy, gz)
   return score

def main():
    space = [(3, 5), #epochs x 10 
         (5, 8), #batch_size power of 2
         [256, 512], #latent size
         (1, 10), #gen_weight
         (0.01, 0.1), #aux_weight
         (0.01, 0.1), #ecal_weight
         (-8, -1), #lr
         (2, 9), #rho
         [0, 0.001], #decay 
         [True,False], # dflag
         (4, 64), #df
         (2, 16), #dx
         (2, 16), #dy
         (2, 16), #dz
         [True,False],#gflag
         (4, 64), #gf
         (2, 16), #gx
         (2, 16), #gy
         (2, 16), #gz
        ]
    datafile = "/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5"
    gen_weights1="veganweights/params_generator_epoch_019.hdf5"
    disc_weights1="veganweights/params_discriminator_epoch_019.hdf5"
    gen_weights2="veganweights/params_generator_epoch_029.hdf5"
    disc_weights2="veganweights/params_discriminator_epoch_029.hdf5"
    gen_weights3="veganweights/params_generator_epoch_039.hdf5"
    disc_weights3="veganweights/params_discriminator_epoch_039.hdf5"

    params1 =[1, 7, 256, 10, 0.1, 0.2, -3, 9 , 0, False, 12, 5, 5, 5, False, 12, 5, 5, 5]
    #tot1 = objective(params1)
    #print (" The negative performance metric for first = %.4f"%(tot1))
    tot1 = analyse(gen_weights1, disc_weights1, datafile)
    tot2 = analyse(gen_weights2, disc_weights2, datafile)
    tot3 = analyse(gen_weights3, disc_weights3, datafile)
    print (" The negative performance metric for first = %.4f"%(tot1))
    print (" The negative performance metric for first = %.4f"%(tot2))
    print (" The negative performance metric for first = %.4f"%(tot3))
if __name__ == "__main__":
    main()
