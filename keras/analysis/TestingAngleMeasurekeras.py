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
import ROOTutils as r
import matplotlib
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
plt.switch_backend('Agg')

def main():
   #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path                                          
   datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5"
   numdata = 10000
   numevents=10
   ypoint1 = 3 
   ypoint2 = 21
   filename = '/error_angle'
   datafiles = GetDataFiles(datapath, 10000, Particles=['Ele'])
   plotsdir = 'meas_plots_keras{}_{}'.format(ypoint1, ypoint2)
   filename = plotsdir + filename
   gan.safe_mkdir(plotsdir)
   X, Y, theta, alpha= GetAngleData(datafiles[0], num_data=numdata)
   m = Meas(X, yp1 = ypoint1, yp2 = ypoint2)
   fig=1
   angle = theta
   PlotAngleMeasure(m, angle, filename + '_1.pdf', fig=fig, yp1 = ypoint1, yp2 = ypoint2)
   fig+=1
   PlotHistError([m], angle, ['Fit', 'Approx'], filename + '_hist.pdf', fig=fig)
   fig+=1
   indexes = np.where(np.absolute(angle - m)> 0.5)
   events = X[indexes]
   angle = angle[indexes]
   meas = m[indexes]
   x = X.shape[1]
   y = X.shape[2]
   z = X.shape[3]
   for i in np.arange(min(numevents, events.shape[0])):
      PlotEvent(events[i], x, y, z, angle[i], meas[i], meas[i], plotsdir + '/Event1_{}.pdf'.format(i), i, fig= fig, yp1 = ypoint1, yp2 = ypoint2)
      fig+=1
   print('Plots are saved in {}.'.format(plotsdir))
  
def PlotHistError(meas,  angle, labels, outfile, fig, degree=False):
   bins = np.arange(-2,2, 0.01)
   plt.figure(fig)
   for m, label in zip(meas, labels):
     error = angle-m
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
   plt.savefig(outfile)

def PlotEvent(event, x, y, z, angle, m1, m2, outfile, n, fig, yp1 = 3, yp2 = 21):
   print 'event {}'.format(n)
   array = np.sum(np.squeeze(event), axis=(0))
   y = np.arange(25)
   maxy = np.argmax(array, axis=0)
   plt.figure(fig)
   plt.scatter(y, maxy, label = 'maxy G4angle = {}'.format(angle))
   p = np.polyfit(y, maxy, 1)
   ytan = (maxy[yp2]-maxy[yp1]) / np.float32(yp2 - yp1)
   pfit= np.polyval(p, y)
   plt.scatter(y, pfit, label = 'fit angle = {}'.format(m1) )
   a = [yp1, yp2]
   b = [maxy[yp1], maxy[yp2]]
   plt.plot(a, b, label = 'Approx angle = {:.4f}\n yp1={} yp2={}'.format(m2, yp1, yp2))
   plt.legend()
   plt.ylim =(0, y)
   plt.savefig(outfile)

def PlotAngleMeasure(measured, angle, outfile, fig=1, degree=False, yp1 = 3, yp2 = 21):
   error = np.absolute(angle - measured)
   if degree:
      angle = np.degrees(angle)
      measured = np.degrees(measured)
      error = np.degrees(error)
      unit = 'degrees'
   else:
      unit = 'radians'
   print(measured[:5])
   print(angle[:5])
   print(error[:5])
   plt.figure(fig)
   plt.scatter(angle, error, label='Error Mean={:.4f}, std={:.4f}'.format(np.mean(error), np.std(error)))
   plt.legend()
   plt.xlabel('Angle ({})'.format(unit))
   plt.ylabel('Error ({})'.format(unit))
   plt.savefig(outfile)

def GetAngleData(datafile, num_data):
    #get data for training                                                                                  
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy'))[:num_data]
    x=np.array(f.get('ECAL'))[:num_data, 13:38, 13:38, :]
    theta = np.array(f.get('theta'))[:num_data] 
    x[x < 1e-4] = 0
    alpha = (math.pi/2)*np.ones_like(theta) - theta
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y, theta, alpha

def Meas(events, yp1 = 3, yp2 = 21):
    images = tf.convert_to_tensor(events, dtype=tf.float32)
    m = math.pi/2 + ecal_angle(images, yp1, yp2)
    m = np.squeeze(m)
    return m

def ecal_angle(image, p1, p2):
    a = K.sum(image, axis=(1, 4))
    b = K.argmax(a, axis=1)
    c = K.cast(b[:,p1] - b[:,p2], dtype='float32')/(p2 - p1)
    d = tf.atan(c)
    d = K.expand_dims(d)
    return K.eval(d)

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

if __name__ == "__main__":
    main()
