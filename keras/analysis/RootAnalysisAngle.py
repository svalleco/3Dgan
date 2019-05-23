#!/usr/bin/env python
# -*- coding: utf-8 -*-
## This script loads weights into architectures for generator and discriminator. Different Physics quantities are then calculated and plotted for a 100-200 GeV events from LCD variable angle dataset##
from utils.GANutils import perform_calculations_angle  # to calculate different Physics quantities
from utils.RootPlotsGAN import get_plots_angle         # to make plots with ROOT
import os
import h5py
import numpy as np
print(np.__version__)
import math
import sys
import argparse

if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a check for host can be used        
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech                                                                                
except:
    pass

sys.path.insert(0,'../')

def main():
   parser = get_parser()
   params = parser.parse_args()

   datapath =params.datapath
   latent = params.latentsize
   particle= params.particle
   angtype= params.angtype
   plotdir= params.plotdir
   sortdir= params.sortdir
   gendir= params.gendir
   discdir= params.discdir
   nbEvents= params.nbEvents
   binevents= params.binevents
   moments= params.moments
   addloss= params.addloss
   angloss= params.angloss
   concat= params.concat
   cell= params.cell
   corr= params.corr
   test= params.test
   stest= params.stest
   save_data= params.save_data
   read_data= params.read_data
   save_gen= params.save_gen
   read_gen= params.read_gen
   save_disc= params.save_disc
   read_disc= params.read_disc
   ifpdf= params.ifpdf
   grid = params.grid
   leg= params.leg
   statbox= params.statbox
   mono= params.mono
   gweights= [params.gweights]
   dweights= [params.dweights]
   xscales= [params.xscales]
   ascales= [params.ascales]
   dscale = params.dscale
   yscale= params.yscale
   xpowers = [params.xpower]
   thresh = params.thresh
   dformat = params.dformat

   labels=['']
    
   #Architecture 
   from AngleArch3dGAN import generator, discriminator

   if datapath=='path1':
       datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV
       events_per_file = 5000
       energies = [0, 110, 150, 190]
   elif datapath=='path2':
       datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
       events_per_file = 10000
       energies = [0, 50, 100, 200, 250, 300, 400, 500]
   elif datapath=='path3':
       datapath = "/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech
       events_per_file = 10000
       energies = [0, 50, 100, 200, 250, 300, 400, 500]

   if tlab: 
     #Weights
     dweights=["/gkhattak/weights/3dgan_weights/params_discriminator_epoch_059.hdf5"]
     gweights= ["/gkhattak/weights/3dgan_weights/params_generator_epoch_059.hdf5"]
     datapath = '/gkhattak/*Measured3ThetaEscan/*.h5'
     events_per_file = 5000
   angles = [62, 90, 118]
   flags =[test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   
   d = discriminator(xpowers[0])
   g = generator(latent)
   var= perform_calculations_angle(g, d, gweights, dweights, energies, angles, 
                datapath, sortdir, gendir, discdir, nbEvents, binevents, moments, xscales, xpowers,
                ascales, dscale, flags, latent, particle, events_per_file=events_per_file, thresh=thresh*50., angtype=angtype, offset=0.0,
                angloss=angloss, addloss=addloss, concat=concat 
                , pre =taking_power, post =inv_power  # Adding other preprocessing, Default is simple scaling                 
   )
   
   get_plots_angle(var, labels, plotdir, energies, angles, angtype, moments, len(gweights), ifpdf=ifpdf, grid=grid, stest=stest, angloss=angloss, addloss=addloss, cell=cell, corr=corr)

def sqrt(n, scale=1):
   return np.sqrt(n * scale)

def square(n, scale=1):
   return np.square(n)/scale
        
def taking_power(n, scale=1.0, power=1.0):
   return(np.power(n * scale, power))

def inv_power(n, scale=1.0, power=1.0):
   return(np.power(n, 1.0/power))/scale

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, default=256, help='size of random N(0, 1) latent space to sample')    #parser.add_argument('--model', action='store', default=AngleArch3dgan, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='path1', help='HDF5 files to train from.')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--angtype', action='store', type=str, default='theta', help='Angle used.')
    parser.add_argument('--plotdir', action='store', type=str, default='results/3dgan_Analysis_gan_training_ep65/', help='Directory to store the analysis plots.')
    parser.add_argument('--sortdir', action='store', type=str, default='SortedData', help='Directory to store sorted data.')
    parser.add_argument('--gendir', action='store', type=str, default='Gen', help='Directory to store the generated images.')
    parser.add_argument('--discdir', action='store', type=str, default='Disc', help='Directory to store the discriminator outputs.')
    parser.add_argument('--nbEvents', action='store', type=int, default=100000, help='Max limit for events used for Testing')
    parser.add_argument('--eventsperfile', action='store', type=int, default=5000, help='Number of events in a file')
    parser.add_argument('--binevents', action='store', type=int, default=2000, help='Number of events in each bin')
    parser.add_argument('--moments', action='store', type=int, default=3, help='Number of moments to compare')
    parser.add_argument('--addloss', action='store', type=int, default=1, help='If using bin count loss')
    parser.add_argument('--angloss', action='store', type=int, default=1, help='Number of loss terms related to angle')
    parser.add_argument('--concat', action='store', type=int, default=2, help='Modes related to combining conditions with latent 0)not cancatenated.. 1)concatenate angle...3) concatenate energy and angle')
    parser.add_argument('--cell', action='store', type=int, default=0, help='Whether to plot cell energies..0)Not plotted...1)Only for bin with uniform spectrum.....2)For all energy bins')
    parser.add_argument('--corr', action='store', type=int, default=0, help='Plot correlation plots..0)Not plotted...1)detailed features ...2) reduced features.. 3) reduced features for each energy bin')
    parser.add_argument('--test', action='store', default=True, help='Use Test data')
    parser.add_argument('--stest', action='store', default=False, help='Statistics test for shower profiles')
    parser.add_argument('--save_data', action='store', default=False, help='Save sorted data')
    parser.add_argument('--read_data', action='store', default=False, help='Get saved and sorted data')
    parser.add_argument('--save_gen', action='store', default=False, help='Save generated images')
    parser.add_argument('--read_gen', action='store', default=False, help='Get saved generated images')
    parser.add_argument('--save_disc', action='store', default=False, help='Save discriminator output')
    parser.add_argument('--read_disc', action='store', default=False, help='Get discriminator output')
    parser.add_argument('--ifpdf', action='store', default=True, help='Whether generate pdf plots or .C plots')
    parser.add_argument('--grid', action='store', default=False, help='set grid')
    parser.add_argument('--leg', action='store', default=True, help='add legends')
    parser.add_argument('--statbox', action='store', default=True, help='add statboxes')
    parser.add_argument('--mono', action='store', default=False, help='changing line style as well as color for comparison')
    parser.add_argument('--gweights', action='store', type=str, default='../weights/3dgan_weights_gan_training/params_generator_epoch_065.hdf5', help='comma delimited list for paths to Generator weights.')
    parser.add_argument('--dweights', action='store', type=str, default='../weights/3dgan_weights_gan_training/params_discriminator_epoch_065.hdf5', help='comma delimited list for paths to Discriminator weights')
    parser.add_argument('--xscales', action='store', type=int, default=1, help='Multiplication factors for cell energies')
    parser.add_argument('--ascales', action='store', type=int, default=1, help='Multiplication factors for angles')
    parser.add_argument('--dscale', action='store', type=int, default=50, help='Data = dscale * GeV')
    parser.add_argument('--yscale', action='store', default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--xpower', action='store', default=0.85, help='Power of cell energies')
    parser.add_argument('--thresh', action='store', type=int, default=0, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
    return parser
   
# If using reduced Ecal 25x25x25 then use the following function as argument to perform_calculations_angle, Data=GetAngleDataEta_reduced
def GetAngleDataEta_reduced(datafile, thresh=1e-6):
    #get data for training                                                                                        
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))[:, 13:38, 13:38, :]
    Y=np.array(f.get('energy'))
    eta = np.array(f.get('eta')) + 0.6
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, eta, ecal

if __name__ == "__main__":
    main()
