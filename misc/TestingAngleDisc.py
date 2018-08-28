import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
import GANutils as gan
from keras.optimizers import RMSprop
#import ROOTutils as r
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from keras import backend as K
import tensorflow as tf

def main():
   datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5"
   datafile = "/eos/project/d/dshep/LCD/DDHEP/EleEscan_RandomAngle_1_MERGED/EleEscan_RandomAngle_1_1.h5"
   numdata = 10
   numevents=10
   ypoint1 = 3
   ypoint2 = 21
   filename = '/error_angle'
   #datafiles = GetDataFiles(datapath, numdata, Particles=['Ele'])
   plotsdir = 'meas_plots{}_{}'.format(ypoint1, ypoint2)
   filename = plotsdir + filename
   gan.safe_mkdir(plotsdir)
   x, y, theta, alpha= GetAngleData(datafile, numdata)
   x, y, theta, alpha = x[:numdata], y[:numdata], theta[:numdata], alpha[:numdata]
   from EcalCondGan6 import discriminator
   d = discriminator()
   d.compile(optimizer=RMSprop(), loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error'], loss_weights=[2, 0.1, 0.1, 0.1]
    )

   g, aux_energy, aux_theta, ecal = d.predict(x)
   
   print aux_theta.shape

def GetAllData(datafiles):
   for index, datafile in enumerate(datafiles):
       if index == 0:
          x, y, theta, eta = GetAngleData(datafile)
       else:
          x_temp, y_temp, theta_temp, eta_temp = GetAngleData(datafile)
          x = np.concatenate((x, x_temp), axis=0)
          y = np.concatenate((y, y_temp), axis=0)
          theta = np.concatenate((theta, theta_temp), axis=0)
          eta = np.concatenate((eta, eta_temp), axis=0)
   return x, y, theta, eta

def GetDataFiles(FileSearch="/data/LCD/*/*.h5", nEvents=800000, EventsperFile = 10000, Particles=[], MaxFiles=-1):
   print ("Searching in :",FileSearch)
   Files =sorted( glob.glob(FileSearch))
   print ("Found {} files. ".format(len(Files)))
   Filesused = int(math.ceil(nEvents/EventsperFile))
   if Filesused < 1:
      Filesused = 1
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
   SampleI=len(Samples.keys())*[int(0)]
   for i,SampleName in enumerate(Samples):
       Sample=Samples[SampleName][:Filesused]
       NFiles=len(Sample)
   return Sample

def GetAngleData(datafile, num_events=10000):
    #get data for training                                                                                                                     
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy'))[:num_events]
    x=np.array(f.get('ECAL'))[:num_events]
    theta = np.array(f.get('theta'))[:num_events]
    x[x < 1e-6] = 0
    alpha = (math.pi/2)*np.ones_like(theta) - theta
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y, theta, alpha
   
if __name__ == "__main__":
    main()
