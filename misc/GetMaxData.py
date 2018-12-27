import os
import h5py
import numpy as np
import math
import time
import glob
import sys
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')
from analysis.utils import GANutils as gan
import setGPU
import keras.backend as K
import tensorflow as tf

def main():
   datapath = '/data/shared/gkhattak/EleMeasured3ThetaEscan/*.h5'
   Particles = ['Ele']
   datafiles = gan.GetDataFiles(datapath, Particles=Particles)
   maxval = 0
   for f in datafiles:
      out = GetAngleData(f, ['ECAL'])
      x= out[0] #inv_mapping(out[0])
      max_ecal = np.amax(np.sum(x, axis=(1, 2, 3)))
      if max_ecal > maxval:
         maxval = max_ecal
         print('Get max deposition to {}........'.format(maxval))
   print('The maximum deposition in entire data set was {}.'.format(maxval))

def GetAngleData(datafile, datatype=['ECAL', 'energy']):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   out = []
   for d in datatype:
      out.append(np.array(f.get(d)))
      return out

def safe_log10(x):
   out = 1. * x
   out[np.where(out>0)] = np.log10(out[np.where(out>0)])
   return out

def mapping(x):
      p0 = 6.82245e-02
      p1 = -1.70385
      p2 = 6.27896e-01
      p3 = 1.39350
      p4 = 2.26181
      p5 = 1.23621e-01
      p6 = 4.30815e+01
      p7 = -8.20837e-02
      p8 = -1.08072e-02
      res = 1. * x
      res[res<1e-7]=0
      log10x = safe_log10(res)
      res = (p0 /(1+np.power(np.abs((log10x-p1)/p2),2*p3))) * ((p4+res*(p7+res*p8))* np.sin(p5*(log10x-p6)))
      return res

def inv_mapping(x):
   out = np.where(x<1e-7, 0, 1./mapping(x))
   return out
                        
if __name__ == "__main__":
    main()
