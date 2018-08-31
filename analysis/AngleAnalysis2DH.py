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
   genweight = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_50weight_withoutsqrt/params_generator_epoch_059.hdf5"
   from AngleArch3dGAN import generator
   latent = 256
   g=generator(latent)
   g.load_weights(genweight)

   num_events = 100 # Number of events for each bin
   num=10 # random events generated
   f = 57.325
   thetamin = 60/f
   thetamax = 120/f
   energies=[100, 150, 200]
   thetas = [62, 90, 118]
   ang = 1
   xscale = 1
   post = scale
   thresh = 2e-3
   plotsdir = 'results/without_sqrt_genplots_ep59'
   gan.safe_mkdir(plotsdir)
   opt="colz"
   events = {}
   for energy in energies:
     events[str(energy)] = {} 
     for t in thetas:
         sampled_energies=energy/100 * np.ones((num_events))  # scale energy
         sampled_thetas = (np.float(t)/f )* np.ones((num_events)) # get radian
         events[str(energy)][str(t)] = gan.generate(g, num_events, [sampled_energies, sampled_thetas], latent) # generate
         events[str(energy)][str(t)] = post(events[str(energy)][str(t)], xscale)
         events[str(energy)][str(t)] = applythresh(events[str(energy)][str(t)], thresh)
         pl.PlotAngleCut(events[str(energy)][str(t)], t, os.path.join(plotsdir, 'Theta{}_GeV{}.pdf'.format(t, energy)), opt=opt)
     pl.PlotEnergyHistGen(events[str(energy)], os.path.join(plotsdir, 'Hist_GeV{}.pdf'.format(energy)), energy, thetas)
   for t in thetas:
      sampled_energies=np.random.uniform(1, 2, size=(num_events))
      sampled_thetas = (np.float(t) /f )* np.ones((num_events))
      events = gan.generate(g, num_events, [sampled_energies, sampled_thetas], latent)
      events = post(events, xscale)
      events = applythresh(events, thresh)
      theta_dir = plotsdir+ '/theta_{}'.format(t)
      gan.safe_mkdir(theta_dir)
      for n in np.arange(num):
          pl.PlotEvent(events[n], sampled_energies[n], sampled_thetas[n], os.path.join(theta_dir, 'Event{}.pdf'.format(n)), n, opt=opt)
            
      pl.PlotAngleCut(events, t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)), opt=opt)

   sampled_energies=np.random.uniform(1, 2, size=(num))
   sampled_thetas = np.random.uniform(thetamin, thetamax, size=(num))
   events = gan.generate(g, num, [sampled_energies, sampled_thetas], latent)
   events = post(events, xscale)
   events = applythresh(events, thresh)
   for n in np.arange(num):
     pl.PlotEvent(events[n], sampled_energies[n], sampled_thetas[n], os.path.join(plotsdir, 'Event{}.pdf'.format(n)), n, opt=opt)
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


   
