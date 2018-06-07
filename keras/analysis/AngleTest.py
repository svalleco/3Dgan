#import ROOT
#from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
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
import setGPU

def main():
   datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5"
   datafile = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_2_1.h5"
   numdata = 10000
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
   from  EcalCondGanAngle import discriminator
   d = discriminator()
   d.compile(optimizer=RMSprop(), loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error'], loss_weights=[2, 0.1, 0.1, 0.1] )
   g, aux_energy, aux_theta, ecal = d.predict(x)  
   aux_theta = np.squeeze(aux_theta) 
   error = aux_theta - alpha
   print aux_theta
   print alpha
   print error
   PlotHistError(error, alpha, 'approx', 'Error_hist.pdf', 1)
   PlotAngleMeasure(aux_theta, alpha, 'Error_scatter.pdf', 2)

def PlotHistError(error,  angle, label, outfile, fig, degree=False):
   bins = np.arange(-2,2, 0.01)
   plt.figure(fig)
   print error.shape
   if degree:
      angle = np.degrees(angle)
      m = np.degrees(m)
      error = np.degrees(error)
      unit = 'degrees'
   else:
      unit = 'radians'
   plt.hist(error, bins=bins, label='{} error {:.4f}({:.4f})'.format(label, np.mean(error), np.std(error)))
   plt.legend()
   plt.xlabel('Angle ({})'.format(unit))
   plt.ylabel('Error ({})'.format(unit))
   print(' The error histogram is saved in {}.'.format(outfile))
   plt.savefig(outfile)

def PlotAngleMeasure(measured, angle, outfile, fig=1, degree=False, yp1 = 3, yp2 = 21):
   error = angle - measured
   if degree:
      angle = np.degrees(angle)
      measured = np.degrees(measured)
      error = np.degrees(error)
      unit = 'degrees'
   else:
      unit = 'radians'

   plt.figure(fig)
   plt.scatter(angle, error, label='Error Mean={:.4f}, std={:.4f}'.format(np.mean(np.absolute(error)), np.std(error)))
   plt.legend()
   plt.xlabel('Angle ({})'.format(unit))
   plt.ylabel('Error ({})'.format(unit))
   print(' The error scatter plot is saved in {}.'.format(outfile))
   plt.savefig(outfile)


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
    alpha = (math.pi/2) - theta 
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y, theta, alpha
   
if __name__ == "__main__":
    main()
