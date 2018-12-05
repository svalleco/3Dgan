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
      x, y, z = gan.GetAngleData(f)
      if np.amax(x) > maxval:
         maxval = np.amax(x)
   print('The maximum deposition in entire data set was {}.'.format(maxval))
   
if __name__ == "__main__":
    main()
