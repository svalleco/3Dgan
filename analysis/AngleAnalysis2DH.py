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
try:
    import setGPU #if Caltech                                                                                
except:
    pass
if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a check for host can be used        
    tlab = True
else:
    tlab= False

sys.path.insert(0,'../')

def main():
   # path to generator weights
   if tlab:
     genweight= "/gkhattak/weights/3Dweights_newbins2/params_generator_epoch_059.hdf5"
   else:
     genweight = "../weights/3dgan_weights_oldtrain/params_generator_epoch_059.hdf5"
   plotsdir = 'results/genplots_oldtrain_ep59' # name of folder to save results
   
   # generator model
   from AngleArch3dGAN import generator

   # some variables that need to be adjusted according to training configuration
   latent = 256 # latent space for generator
   xpower=0.85 # power used for post processing
   concat=1 # mode for concatenating
   num_events = 1000 # Number of events for each bin
   num=10 # random events generated
   thetamin = np.radians(60)  # min theta
   thetamax = np.radians(120) # maximum theta
   energies=[100, 150, 200, 300, 400] # energy bins
   thetas = [62, 90, 118] # angle bins
   ang = 1 # use all calculation for variable angle
   xscale = 1.0 # scaling of images
   post = inv_power # post processing: It can be either scale (without sqrt) or square(sqrt)
   thresh = 0.0 # if using threshold
   opt="colz" # option for 2D hist

   events = {} # initialize list

   g=generator(latent) # build generator
   g.load_weights(genweight) # load weights
   gan.safe_mkdir(plotsdir) # make plot directory  

   ############################################   Different energy and angle bins     #########################################
   for energy in energies:
     events[str(energy)] = {} # create a dict in list
     for t in thetas: # for each angle bin
         sampled_energies=energy/100 * np.ones((num_events))  # scale energy
         sampled_thetas = (np.radians(t))* np.ones((num_events)) # get radian
         events[str(energy)][str(t)] = gan.generate(g, num_events, [sampled_energies, sampled_thetas], latent, concat) # generate
         events[str(energy)][str(t)] = post(events[str(energy)][str(t)], xscale=xscale, xpower=xpower)# post processing
         if thresh > 0:
             events[str(energy)][str(t)] = applythresh(events[str(energy)][str(t)], thresh) # if to apply threshold
         pl.PlotAngleCut(events[str(energy)][str(t)], t, os.path.join(plotsdir, 'Theta{}_GeV{}.pdf'.format(t, energy)), opt=opt) # make plot for angle bin corressponding to each energy bin
     pl.PlotEnergyHistGen(events[str(energy)], os.path.join(plotsdir, 'Hist_GeV{}.pdf'.format(energy)), energy, thetas) # make superimposed weighted histograms along x, y and z for each angle bin

   #################### generate random energy events with a certain theta  ##################################
   for t in thetas:
      sampled_energies=np.random.uniform(1, 2, size=(num_events)) #sampled energies
      sampled_thetas = (np.radians(t))* np.ones((num_events)) # theta
      events = gan.generate(g, num_events, [sampled_energies, sampled_thetas], latent, concat)# generate
      events = post(events, xscale) # apply post processing
      if thresh > 0:
         events = applythresh(events, thresh)# apply threshold
      theta_dir = plotsdir+ '/theta_{}'.format(t) # make directory to have random events with fixed theta
      gan.safe_mkdir(theta_dir)
      for n in np.arange(num):
          pl.PlotEvent(events[n], sampled_energies[n], sampled_thetas[n], os.path.join(theta_dir, 'Event{}.pdf'.format(n)), n, opt=opt)# make plots with fixed theta and random energies
            
      pl.PlotAngleCut(events, t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)), opt=opt) # make plots for all events in a bin

   ####################### generate random energy events ######################################################
   sampled_energies=np.random.uniform(1, 2, size=(num)) # sampled energies
   sampled_thetas = np.random.uniform(thetamin, thetamax, size=(num))# sampled theta
   events = gan.generate(g, num, [sampled_energies, sampled_thetas], latent, concat)
   events = post(events, xscale)
   if thresh>0:
      events = applythresh(events, thresh)
   for n in np.arange(num):
     pl.PlotEvent(events[n], 100*sampled_energies[n], sampled_thetas[n], os.path.join(plotsdir, 'Event{}.pdf'.format(n)), n, opt=opt)
   print('Plots are saved in {}'.format(plotsdir))

def square(n, xscale):
   return np.square(n)/xscale

def scale(n, xscale):
   return n / xscale

def power(n, xpower=1.0, xscale=1.0):
   return np.power(n * xscale, xpower)

def inv_power(n, xpower=1.0, xscale=1.0):
   return (np.power(n, 1./xpower))/xscale

def applythresh(n, thresh):
   n[n<thresh]=0
   return n

if __name__ == "__main__":
    main()


   
