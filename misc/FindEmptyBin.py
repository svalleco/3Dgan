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
   count = 0
   for f in datafiles:
      out = GetAngleData(f, ['ECAL'])
      bins=hist_count(out[0])
      zero = np.where(bins==0, 1, 0)
      sumzero = np.sum(zero, axis=(0, 1))
      if sumzero > 0:
        index = np.where(zero==1)
        print(index)
        print('The position of empty bins is {} event and {} bin'.format(index[0], index[1]))
      count = count+ sumzero
      print('Number of empty bins is {}'.format(count))
    

def GetAngleData(datafile, datatype=['ECAL', 'energy']):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   out = []
   for d in datatype:
      out.append(np.array(f.get(d)))
      return out
                        
def hist_count(x):
      x=np.expand_dims(x, axis=4)
      #bin1 = np.sum(np.where(x> 0.2, 1, 0), axis=(1, 2, 3))
      #bin2 = np.sum(np.where((x<0.2) & (x>0.08) , 1, 0), axis=(1, 2, 3))
      bin3 = np.sum(np.where((x>0.05) , 1, 0), axis=(1, 2, 3))
      bin4 = np.sum(np.where((x<0.05) & (x>0.03), 1, 0), axis=(1, 2, 3))
      bin5 = np.sum(np.where((x<0.03) & (x>0.02), 1, 0), axis=(1, 2, 3))
      bin6 = np.sum(np.where((x<0.02) & (x>0.0125), 1, 0), axis=(1, 2, 3))
      bin7 = np.sum(np.where((x<0.0125) & (x>0.008), 1, 0), axis=(1, 2, 3))
      bin8 = np.sum(np.where((x<0.008) & (x>0.003), 1, 0), axis=(1, 2, 3))
      bin9 = np.sum(np.where((x<0.003) & (x>0.), 1, 0), axis=(1, 2, 3))
      bin10 = np.sum(np.where(x==0, 1, 0), axis=(1, 2, 3))
      return np.concatenate([bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10], axis=1)


if __name__ == "__main__":
    main()
