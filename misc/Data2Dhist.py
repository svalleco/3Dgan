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
   #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path
   datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # path to data
   numdata = 20000 # events read from data
   energies=[110, 150, 190] # energy bins
   thetas = [62, 90, 118]   # angle bins
   xcuts = [10, 25, 40]     # cuts along x
   ycuts = [10, 25, 40]     # cuts along y
   zcuts = [5, 12, 20]      # cuts along z
   num = 5 # single events to plots
   thresh = 1e-4
   angtype = 'theta'
   opt = 'colz' # drawing option
   plotsdir = 'results/angle_data_plots'  # results location 
   gan.safe_mkdir(plotsdir)               # make directory
      
   datafiles = gan.GetDataFiles(datapath, numdata, Particles=['Ele']) # get list of files
   x, y, theta= gan.GetAllDataAngle(datafiles, numdata, thresh, angtype) # get all data

   # The sorted dict contains events as 'events', the quantity to sort as 'y' and if there is a third it is named 'z'
   yvar = gan.sort([x, y, theta], energies, num_events=1000)  # sort according to energy
   thetavar = gan.sort([x, np.degrees(theta), y], thetas, num_events=1000, tolerance=2) # sort according to angle

   for energy in energies:
      for xcut, ycut, zcut in zip(xcuts, ycuts, zcuts):
         hist.PlotPosCut(yvar["events" + str(energy)], xcut, ycut, zcut, energy, os.path.join(plotsdir, '2Dhits_cut{}_{}.pdf'.format(xcut, energy)))  # make cuts for different positions for different energies
   for t in thetas:
      theta_dir = plotsdir+ '/theta_{}'.format(t) # dir to store n events
      gan.safe_mkdir(theta_dir)
      hist.PlotAngleCut(thetavar["events" + str(t)], t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)), opt=opt)# plot for angle bin
      for n in np.arange(num):
         # individual events will be saved
         hist.PlotEvent(thetavar["events" + str(t)][n], thetavar["z" + str(t)][n]/100, np.radians(thetavar["y" + str(t)][n]), os.path.join(theta_dir, 'Event{}.pdf'.format(n)), n, opt=opt)
                   
if __name__ == "__main__":
    main()


   
