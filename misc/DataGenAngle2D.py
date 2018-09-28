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
import utils.GANutils as gan
import utils.ROOTutils as r
import utils.RootPlotsGAN as pl
import setGPU #if Caltech

sys.path.insert(0,'/nfshome/gkhattak/3Dgan')

def main():
   datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # path to data
   # path to generator weights
   genweight = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_50weight_sqrt/params_generator_epoch_040.hdf5"
   # generator model
   from AngleArch3dGAN_sqrt import generator

   latent = 256 # latent space for generator
   g=generator(latent) # build generator
   g.load_weights(genweight) # load weights
   numdata=120000
   num_events1 = 10000 # Number of events for 0 bin
   num_events2 = 1000  # Number of events for other bins
   tolerance2=0.05
   num=10 # random events generated
   thetamin = np.radians(60)  # min theta
   thetamax = np.radians(120) # maximum theta
   energies=[110, 150, 190] # energy bins
   thetas = [62, 90, 118] # angle bins
   ang = 1 # use all calculation for variable angle
   xscale = 1 # scaling of images
   ascale=1
   post = square # post processing: It can be either scale (without sqrt) or square(sqrt)
   thresh = 0 # if using threshold
   plotsdir = 'results/new_genplots_compare_ep40' # name of folder to save results
   gan.safe_mkdir(plotsdir) # make plot directory
   opt="colz" # option for 2D hist
   angtype='theta'
   datafiles = gan.GetDataFiles(datapath, numdata, Particles=['Ele']) # get list of files
   var = gan.get_sorted_angle(datafiles, energies, True, num_events1, num_events2, angtype=angtype, thresh=thresh) # returning a dict with sorted data.
   for energy in energies:
      edir = os.path.join(plotsdir, 'energy{}'.format(energy))
      gan.safe_mkdir(edir)
      rad = np.radians(thetas)
      for index, a in enumerate(rad):
         adir = os.path.join(edir, 'angle{}'.format(a))
         gan.safe_mkdir(adir)
         indexes = np.where(((var["angle" + str(energy)]) > a - tolerance2) & ((var["angle" + str(energy)]) < a + tolerance2)) # all events with angle within a bin
         # angle bins are added to dict
         var["events_act" + str(energy) + "ang_" + str(index)] = var["events_act" + str(energy)][indexes]
         var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)][indexes]
         var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)][indexes]
         var["index" + str(energy)+ "ang_" + str(index)] = var["events_act" + str(energy) + "ang_" + str(index)].shape[0]
         var["events_gan" + str(energy) + "ang_" + str(index)]= gan.generate(g, var["index" + str(energy)+ "ang_" + str(index)],
                                                                           [var["energy" + str(energy)+ "ang_" + str(index)]/100,
                                                                            (var["angle"+ str(energy)+ "ang_" + str(index)]) * ascale], latent)
         var["events_gan" + str(energy) + "ang_" + str(index)]= post(var["events_gan" + str(energy) + "ang_" + str(index)], xscale)
         ddir = os.path.join(adir, 'G4Data')
         gan.safe_mkdir(ddir)
         gdir =os.path.join(adir, 'GAN')
         gan.safe_mkdir(gdir)
         for n in np.arange(num):
            pl.PlotEvent(var["events_act" + str(energy) + "ang_" + str(index)][n],
                         var["energy" + str(energy) + "ang_" + str(index)][n],
                         var["angle" + str(energy) + "ang_" + str(index)][n],
                         os.path.join(ddir, 'Event{}.pdf'.format(n)), n, opt=opt, label='G4')
            pl.PlotEvent(var["events_gan" + str(energy) + "ang_" + str(index)][n],
                         var["energy" + str(energy) + "ang_" + str(index)][n],
                         var["angle" + str(energy) + "ang_" + str(index)][n],
                         os.path.join(gdir, 'Event{}.pdf'.format(n)), n, opt=opt, label='GAN')
            pl.PlotEventCut(var["events_act" + str(energy) + "ang_" + str(index)][n],
                        var["energy" + str(energy) + "ang_" + str(index)][n],
                        var["angle" + str(energy) + "ang_" + str(index)][n],
                        os.path.join(ddir, 'Event{}_cut.pdf'.format(n)), n, opt=opt, label='G4')
            pl.PlotEventCut(var["events_gan" + str(energy) + "ang_" + str(index)][n],
                        var["energy" + str(energy) + "ang_" + str(index)][n],
                        var["angle" + str(energy) + "ang_" + str(index)][n],
                        os.path.join(gdir, 'Event{}_cut.pdf'.format(n)), n, opt=opt, label='GAN')
                        
            
   print('Plots are saved in {}'.format(plotsdir))
    
def square(n, xscale):
   return np.square(n)/xscale

def scale(n, xscale):
   return n / xscale

def applythresh(n, thresh):
   n[n<thresh]=0
   return n

if __name__ == "__main__":
    main()


   
