#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import glob
import h5py
import numpy as np
import time
import math
import argparse
import setGPU #if Caltech

sys.path.insert(0,'/nfshome/gkhattak/3Dgan/')

import analysis.utils.GANutils as tr

def main():
   datapath = '/data/shared/gkhattak/*Measured3ThetaEscan/*.h5'
   particle = 'Ele'
   # Getting Data
   Trainfiles, Testfiles = tr.DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
   X, Y, ang = tr.GetAngleData(Testfiles[0], thresh=0)
   X_sqrt = np.sqrt(X)
   print('Data varies from {} to {}'.format(np.amin(X[X>0]), np.amax(X)))
   print('SQRT of Data varies from {} to {}'.format(np.amin(X_sqrt[X_sqrt>0]), np.amax(X_sqrt)))

def preproc(x):
   return x

if __name__ == '__main__':
   main()
