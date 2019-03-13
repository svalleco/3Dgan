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

sys.path.insert(0,'/nfshome/gkhattak/3Dgan/analysis')
import utils.GANutils as gan
import utils.RootPlotsGAN as hist


def main():
   datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' # Fixed Training data path
   #datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # path to data
   numdata = 20000 # events read from data
   energies=[50, 100, 150, 200, 300, 400] # energy bins
   num = 10 # single events to plots
   thresh = 0
   opt = 'colz' # drawing option
   plotsdir = 'results/fixed_angle_2D/'  # results location 
   gan.safe_mkdir(plotsdir)               # make directory
   datafiles = gan.GetDataFiles(datapath, Particles=['Ele']) # get list of files
   x, y= gan.GetData(datafiles[0], thresh) # get all data
   # The sorted dict contains events as 'events', the quantity to sort as 'y' and if there is a third it is named 'z'
   yvar = gan.sort([x, y], energies, num_events=1000)  # sort according to energy
   for energy in energies:
      print(yvar["events" + str(energy)].shape)
      edir = plotsdir + '{}GeV'.format(energy)
      gan.safe_mkdir(edir)
      for n in np.arange(num):
         hist.PlotEventFixed(yvar["events" + str(energy)][n], yvar["y" + str(energy)][n], os.path.join(edir, 'Event{}.pdf'.format(n)), n, opt=opt)
         hist.PlotEventCutFixed(yvar["events" + str(energy)][n], yvar["y" + str(energy)][n], os.path.join(edir, 'Event{}_cut.pdf'.format(n)), n, opt=opt)
                   
if __name__ == "__main__":
    main()


   
