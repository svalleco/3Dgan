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
import time
import math
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
from ecalvegan import generator


def main():
#    datafile = "/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5"
#    genpath = "/afs/cern.ch/work/g/gkhattak/newweights/params_generator*.hdf5"
#    discpath = "/afs/cern.ch/work/g/gkhattak/newweights/params_discriminator*.hdf5"
    datapath = "/bigdata/shared/LCD/NewV1/*scan/*.h5"
    genpath = "/nfshome/gkhattak/keras/veganweights/params_generator*.hdf5"
    sorted_path = 'sorted_*.hdf5'
    filename = 'rootfit2p1p1'
    g= generator()
    gen_weights=[]
    disc_weights=[]
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
    metrics = []
    metric = 3
    """
    # Commented part will be used if compared to other metrics
    metrics.append(metric)
    fig = 1
    resultfile = filename + '_' + str(metric) + 'result.txt'
    resultplot = filename + '_'+ str(metric) + 'result.pdf'
    results1 = GetResults(metric, resultfile, gen_weights, g, datapath)
    PlotResults(results1, metric, resultplot, fig)
    """
    metric+=1
    metrics.append(metric)
    fig+=1
    resultfile = filename + '_' + str(metric) + 'result.txt'
    resultplot = filename + '_' + str(metric) + 'result.pdf'
    results2 = GetResults(metric, resultfile, gen_weights, g, datapath)
    PlotResults(results2, metric, resultplot, fig)
    """
    fig+=1
    resultplot = filename + '_all_' + 'result.pdf'
    PlotResultsAll([results1, results2], metrics, resultplot, fig)
    """

def PlotResultsAll(results, metrics, resultplotall, fig):
    plt.figure(fig)
    plt.ylim(0, 1.0)

    for result, metric in zip(results, metrics):
      total=[]
      pos_e=[]
      energy_e=[]
      mine = 100
      minp = 100
      mint = 100
      num = 0
      for item in result:
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
      
      plt.xticks(np.arange(0, len(total), 5))
      plt.plot(total, label = 'metric' + str(metric) + 'Tot min{:.4f}({})'.format(mint, mint_n))
      plt.plot(pos_e, label = 'metric' + str(metric) + 'Pos min{:.4f}({})'.format(minp, minp_n))
      plt.plot(energy_e , label = 'metric' + str(metric) + 'E min{:.4f}({})'.format(mine, mine_n))
      plt.legend()
    plt.savefig(resultplotall)
    print ('The plot is saved to {}.'.format(resultplotall))

def GetResults(num, resultfile, gen_weights, g, datapath):
    results= []
    file = open(resultfile,'w')
    if num==3:
        metric = metric3
    elif num==4: 
        metric = metric4                                                      
    for i in range(len(gen_weights)):
       if i==0:
         results.append(analyse(g, True, False, gen_weights[i], datapath, metric)) # For the first time when sorted data is not saved we can make use opposite flags
       else:
         results.append(analyse(g, True, False, gen_weights[i], datapath, metric))
       file.write("{}\t{}\t{}\n".format(results[i][0], results[i][1], results[i][2]))
    #print all results together at end                                                                               
    for i in range(len(gen_weights)):                                                                                            
       print ('The results for ......',gen_weights[i])
       print (" The result for {} = {:.4f} , {:.4f}, {:.4f}".format(i, results[i][0], results[i][1], results[i][2]))
    file.close
    print ('The results are saved to {}.'.format(resultfile))
    return results

def PlotResults(results, metric, resultplot, fig):
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

    plt.figure(fig)
    plt.plot(total, label = 'Total')
    plt.plot(pos_e, label = 'Pos error (Moments)')
    plt.plot(energy_e , label = 'Energy profile error')
    plt.legend(title='Min total error{:.4f}({})\nPosition error {:.4f}({})\nEnergy error {:.4}({})'.format(mint, mint_n, minp, minp_n, mine, mine_n))
    plt.xticks(np.arange(0, len(total), 5))
    plt.ylim(0, 1.0)
    plt.savefig(resultplot)
    print ('The plots are saved to {}.'.format(resultplot))

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
def analyse(g, read_data, save_data, gen_weights, datapath, optimizer):
   print ("Started")
   num_events=2000
   num_data = 100000
   sortedpath = 'sorted_*.hdf5'
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
     Trainfiles, Testfiles = GetFiles(datapath, nEvents=num_data, EventsperFile = events_per_file, datasetnames=["ECAL"], Particles =["Ele"]) 
     if Test:
        data_files = Testfiles
     else:
        data_files = Trainfiles + Testfiles
     start = time.time()
     for index, dfile in enumerate(data_files):
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
   return optimizer(var, energies, m)                                        
 
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

def metric3(var, energies, m):
   ecal_size = 25
   metricp = 0
   metrice = 0
   for energy in energies:
     #Relative error on mean moment value for each moment and each axis
     x_act= np.sum(var["momentX_act"+ str(energy)], axis=0)/ var["index"+ str(energy)]
     x_gan= np.sum(var["momentX_gan"+ str(energy)], axis=0)/ var["index"+ str(energy)]
     y_act= np.sum(var["momentY_act"+ str(energy)], axis=0)/ var["index"+ str(energy)]
     y_gan= np.sum(var["momentY_gan"+ str(energy)], axis=0)/ var["index"+ str(energy)]
     z_act= np.sum(var["momentZ_act"+ str(energy)], axis=0)/ var["index"+ str(energy)]
     z_gan= np.sum(var["momentZ_gan"+ str(energy)], axis=0)/ var["index"+ str(energy)]
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


if __name__ == "__main__":
    main()
