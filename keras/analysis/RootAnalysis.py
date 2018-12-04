from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import sys
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
from utils.GANutils import perform_calculations_multi, safe_mkdir #Common functions from GANutils.py
import utils.ROOTutils as my
import utils.RootPlotsGAN as pl
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/keras')

def main():
   #Architectures 
   from EcalEnergyGan import generator, discriminator
   disc_weights="../weights/3Dweights/params_discriminator_epoch_041.hdf5"
   gen_weights= "../weights/3Dweights/params_generator_epoch_041.hdf5"

   plots_dir = "results/test_plots/"
   latent = 200
   num_data = 100000
   num_events = 2000
   m = 3
   cell=1 # if producing the cell energy histogram
   energies=[0, 50, 100, 200, 250, 300, 400, 500]
   particle='Ele'
   datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path caltech
   #datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5' # Training data CERN EOS
   sortdir = 'SortedData'
   gendir = 'Gen'  
   discdir = 'Disc' 
   Test = True
   stest = False 
   save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
   read_data = False # True if loading previously sorted data  
   save_gen =  False # True if saving generated data. 
   read_gen = False # True if generated data is already saved and can be loaded
   save_disc = False # True if discriminiator data is to be saved
   read_disc =  False # True if discriminated data is to be loaded from previously saved file
   ifpdf = True # True if pdf are required. If false .C files will be generated
 
   flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   # Lists for different versions comparison. The weights, scales and labels will need to be provided for each version
   dweights = [disc_weights]
   gweights = [gen_weights]
   scales = [100]
   labels = ['']
   d = discriminator()
   g = generator(latent)
   var= perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendir, discdir, num_data, num_events, m, scales, flags, latent, particle)
   pl.get_plots_multi(var, labels, plots_dir, energies, m, len(gweights), ifpdf, stest, cell)

if __name__ == "__main__":
    main()
