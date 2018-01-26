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
   y=np.array(data.get('target'))
   Y=np.expand_dims(y[:,1], axis=-1)
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
#   print ("Sorting data")
#   print(Y[:10])
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
         #print('retive index', relativeIndices.transpose().shape)
         moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
         sumx = var["sumact" + str(energy)][0:(var["index" + str(energy)]), 0]
         #print (moments.shape)                                                                    
         #print (sumx.shape)                                                                       
         #print (totalE.shape)                                                                     
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
       #var["pos_error"+ str(energy)] = var["max_pos_act_" + str(energy)] - var["max_pos_gan_" + str(energy)]
       #pos_error = var["max_pos_act_" + str(energy)] - var["max_pos_gan_" + str(energy)]
       #var["pos_error"+ str(energy)]= np.divide(pos_error, var["max_pos_act_" + str(energy)], out=np.zeros_like(pos_error), where=var["max_pos_act_" + str(energy)]!=0)
       #var["pos_total"+ str(energy)] = np.sum(var["pos_error"+ str(energy)]**2)/num_events
       #print('moment =', var["x_act"+ str(energy)].shape)
       var["posx_error"+ str(energy)]= (np.mean(var["x_act"+ str(energy)][:var["index" + str(energy)]], axis=0)- np.mean(var["x_gan"+ str(energy)][:var["index" + str(energy)]], axis=0))/np.mean(var["x_act"+ str(energy)][:var["index" + str(energy)]], axis=0)
       var["posy_error"+ str(energy)]= (np.mean(var["y_act"+ str(energy)][:var["index" + str(energy)]], axis=0)- np.mean(var["y_gan"+ str(energy)][:var["index" + str(energy)]], axis=0))/np.mean(var["y_act"+ str(energy)][:var["index" + str(energy)]], axis=0)
       var["posz_error"+ str(energy)]= (np.mean(var["z_act"+ str(energy)][:var["index" + str(energy)]], axis=0)- np.mean(var["z_gan"+ str(energy)][:var["index" + str(energy)]], axis=0))/np.mean(var["x_act"+ str(energy)][:var["index" + str(energy)]], axis=0)
       print (var["posx_error"+ str(energy)].shape)
       #var["posx_error"+ str(energy)]= np.divide((var["x_act"+ str(energy)]-var["x_gan"+ str(energy)]), var["x_act"+ str(energy)])
       #var["posy_error"+ str(energy)]= np.divide((var["y_act"+ str(energy)]-var["y_gan"+ str(energy)]), var["y_act"+ str(energy)])
       #var["posz_error"+ str(energy)]= np.divide((var["z_act"+ str(energy)]-var["z_gan"+ str(energy)]), var["x_act"+ str(energy)])
       var["pos_error"+ str(energy)]= ((var["posx_error"+ str(energy)])**2 + (var["posy_error"+ str(energy)])**2 + (var["posz_error"+ str(energy)])**2)/3
       var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])
       metricp += var["pos_total"+ str(energy)]
       Ecal = np.tile(var["totalE_act" + str(energy)], (25,3, 1))
       Ecal = Ecal.transpose()
       #print('Ecal', Ecal[1])
       #print(var["totalE_act" + str(energy)][:3])
       var["eprofile_error"+ str(energy)]= np.divide((var["sumact" + str(energy)] - var["sumgan" + str(energy)]), Ecal)
       var["eprofile_total"+ str(energy)]= np.sum(var["eprofile_error"+ str(energy)]**2)
       var["eprofile_total"+ str(energy)]= var["eprofile_total"+ str(energy)]/var["index" + str(energy)]
       metrice += var["eprofile_total"+ str(energy)]
   metricp = metricp/len(energies)
   metrice = metrice/len(energies)    
   tot = metricp + metrice
   
   for energy in energies:
       print ("%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f \t\t%f \t\t%f" %(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]), str(np.unravel_index(var["events_gan" + str(energy)].argmax(), (var["index" + str(energy)], 25, 25, 25))), np.mean(var["events_gan" + str(energy)]), np.amin(var["events_gan" + str(energy)]), var["pos_total"+ str(energy)], var["eprofile_total"+ str(energy)]))
   print(" Position Error = %.4f\t Energy Profile Error =   %.4f" %(metricp, metrice))
   print(" Total Error =  %.4f" %(tot))
   return(tot, metricp, metrice)

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
    genpath = "/afs/cern.ch/work/g/gkhattak/caltech8p2p1weights/params_generator*.hdf5"
    discpath = "/afs/cern.ch/work/g/gkhattak/caltech8p2p1weights/params_discriminator*.hdf5"
    gen_weights=[]
    disc_weights=[]
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
    for f in sorted(glob.glob(discpath)):
      disc_weights.append(f)

    """disc_weights.append("veganweights/params_discriminator_epoch_019.hdf5")
    gen_weights.append("veganweights/params_generator_epoch_029.hdf5")
    disc_weights.append("veganweights/params_discriminator_epoch_029.hdf5")
    gen_weights.append("veganweights/params_generator_epoch_039.hdf5")
    disc_weights.append("veganweights/params_discriminator_epoch_039.hdf5")"""
    print(len(gen_weights))
    print(len(disc_weights))
    #params1 =[1, 7, 256, 10, 0.1, 0.2, -3, 9 , 0, False, 12, 5, 5, 5, False, 12, 5, 5, 5]
    #tot1 = objective(params1)
    #print (" The negative performance metric for first = %.4f"%(tot1))
    results = []
    for i in range(len(gen_weights)):
#    for i in range(2):
       results.append(analyse(gen_weights[i], disc_weights[i], datafile))
    
    for i in range(len(gen_weights)):
#    for i in range(2):
       print ('The results for......',gen_weights[i])
       print (" The result for {} = {:.2f} , {:.2f}, {:.2f}".format(i, results[i][0], results[i][1], results[i][2]))
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
    plt.savefig('result.pdf')
if __name__ == "__main__":
    main()
