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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
plt.switch_backend('Agg')
import numpy as np
import glob
import numpy.core.umath_tests as umath

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

def main():
    datafile = "/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5"
    genpath = "/afs/cern.ch/work/g/gkhattak/newweights/params_generator*.hdf5"
    discpath = "/afs/cern.ch/work/g/gkhattak/newweights/params_discriminator*.hdf5"
    resultplot = 'Newresult.pdf'
    resultfile = 'Newresult.txt'
    gen_weights=[]
    disc_weights=[]
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
    for f in sorted(glob.glob(discpath)):
      disc_weights.append(f)

    results = []
    file = open(resultfile,'w')

#    for i in range(len(gen_weights)-2,len(gen_weights)):                                                          
    for i in range(len(gen_weights)):
#    for i in range(2):                                                                                            
       results.append(analyse(gen_weights[i], disc_weights[i], datafile))
       file.write("{}\t{}\t{}\n".format(results[i][0], results[i][1], results[i][2]))
#    for i in range(2):                                                                                            
    for i in range(len(gen_weights)):
#    for i in range(2):                                                                                            
       print ('The results for......',gen_weights[i])
       print (" The result for {} = {:.4f} , {:.4f}, {:.4f}".format(i, results[i][0], results[i][1], results[i][2]))
    file.close()
    total=[]
    pos_e=[]
    energy_e=[]
    mine = 100
    minp = 100
    mint = 100
    num = 0
    for item in results:
        total.append(item[0])
        if item[0]< mint:
           mint = item[0]
           mint_n = num
        pos_e.append(item[1])
        if item[1]< minp:
           minp = item[1]
           minp_n = num
        energy_e.append(item[2])
        if item[2]< mine:
           mine = item[2]
           mine_n = num
        num = num + 1

    plt.figure()
    plt.plot(total, label = 'Total')
    plt.plot(pos_e, label = 'Pos error (Moments)')
    plt.plot(energy_e , label = 'Energy profile error')
    plt.legend(title='Min total error{:.4f}({})\nPosition error {:.4f}({})\nEnergy error {:.4}({})'.format(mint, mint_n, minp, minp_n, mine, mine_n))
    plt.xticks(np.arange(0, len(gen_weights), 2))
    plt.ylim(0, 1.0)
    plt.savefig(resultplot)


# Fuction used to flip random bits for training
def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x
 
# This function will calculate two errors derived from position of maximum along an axis and the sum of ecal along the axis
def analyse(gen_weights, disc_weights, datafile, latent=200, gflag=0, gf=8, gx=5, gy=5, gz=5, dflag=0, df=8, dx=5, dy=5, dz=5):
   print ("Started")
   num_events=3000
   data=h5py.File(datafile,'r')
   X=np.array(data.get('ECAL'))
   Y=np.expand_dims(np.array(data.get('target')[:,1]), axis=-1)
   X[X < 1e-6] = 0
   print("Data is loaded")
   energies=[50, 100, 150, 200, 300, 400, 500] 
   tolerance = 5
   from ecalvegan import generator, discriminator
   g = generator(latent_size=200)
   g.load_weights(gen_weights)
   d = discriminator()
   d.load_weights(disc_weights)
   m = 2
   print('Generator...........', gen_weights)
   print('Discriminator...........', disc_weights)

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
   for i in range(size_data):
     for energy in energies:
        if Y[i][0] > energy-tolerance and Y[i][0] < energy+tolerance and var["index" + str(energy)] < num_events:
            var["events_act" + str(energy)][var["index" + str(energy)]]= X[i]
            var["energy_sampled" + str(energy)][var["index" + str(energy)]] = Y[i]/100
            var["index" + str(energy)]= var["index" + str(energy)] + 1
   # Generate images
   for energy in energies:        
        noise = np.random.normal(0, 1, (var["index" + str(energy)], latent))
        sampled_labels = var["energy_sampled" + str(energy)][:var["index" + str(energy)]]
        generator_in = np.multiply(sampled_labels, noise)
        generated_images = g.predict(generator_in, verbose=False, batch_size=100)
        var["events_gan" + str(energy)]= np.squeeze(generated_images)
        var["isreal_gan" + str(energy)], var["energy_gan" + str(energy)], var["ecal_gan"] = np.array(d.predict(generated_images, verbose=False, batch_size=100))
        var["isreal_act" + str(energy)], var["energy_act" + str(energy)], var["ecal_act"] = np.array(d.predict(np.expand_dims(var["events_act" + str(energy)], -1), verbose=False, batch_size=100))
        print(var["events_gan" + str(energy)].shape)
        print(var["events_act" + str(energy)].shape)
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
       #Taking absolute of errors and adding for each axis then scaling by 3
       var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
       #Summing over moments and dividing for number of moments
       var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
       metricp += var["pos_total"+ str(energy)]

       #Take profile along each axis and find mean along events
       sumact = np.mean(var["sumact" + str(energy)][:var["index" + str(energy)]], axis=0)
       sumgan = np.mean(var["sumgan" + str(energy)][:var["index" + str(energy)]], axis=0)
       maxact = np.amax(sumact, axis=1)
       maxact = np.reshape(maxact, (-1, 1))
       maxact = np.repeat(maxact, 25, axis=1)
       var["eprofile_error"+ str(energy)] = np.divide((sumact - sumgan), sumact)

       #Take absolute of error and mean for all events
       var["eprofile_total"+ str(energy)]= np.sum(np.absolute(var["eprofile_error"+ str(energy)]), axis=1)/ecal_size
       var["eprofile_total"+ str(energy)]= np.sum(var["eprofile_total"+ str(energy)])/3
       metrice += var["eprofile_total"+ str(energy)]
   metricp = metricp/len(energies)
   metrice = metrice/len(energies)    
   tot = metricp + metrice
   
   for energy in energies:
       print ("%d \t\t%d \t\t%f \t\t%f" %(energy, var["index" +str(energy)], var["pos_total"+ str(energy)], var["eprofile_total"+ str(energy)]))
   print(" Total Position Error = %.4f\t Total Energy Profile Error =   %.4f" %(metricp, metrice))
   print(" Total Error =  %.4f" %(tot))
   return(tot, metricp, metrice)

if __name__ == "__main__":
    main()
