import os
import sys
import h5py
import numpy as np
import math
import time
import glob
from utils.GANutils import perform_calculations_multi, safe_mkdir #Common functions from GANutils.py
import keras
import argparse
import utils.RootPlotsGAN as pl
import keras.backend as K

if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a check for host can be used
    tlab = True
else:
    tlab= False

sys.path.insert(1, os.path.join(sys.path[0], '..'))

def main():

   #Architectures 
   from EcalEnergyGan import generator, discriminator

   import keras.backend as K

   parser = get_parser()
   params = parser.parse_args()

   datapath =params.datapath#Data path
   latent = params.latentsize
   particle= params.particle
   plotsdir= params.plotsdir
   sortdir= params.sortdir
   gendir= params.gendir
   discdir= params.discdir
   nbEvents= params.nbEvents
   binevents= params.binevents
   moments= params.moments
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
   gweights= params.gweights
   dweights= params.dweights
   xscales= params.xscales
   yscale= params.yscale
   energies= params.energies
   thresh = params.thresh
   dformat = params.dformat
   labels=['']
   K.set_image_data_format(dformat)   
   if tlab:
      datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5' # Training data CERN EOS
      gweights = ['/gkhattak/weights/EnergyWeights/3dganWeights_channel_first/params_generator_epoch_000.hdf5']
      dweights = ['/gkhattak/weights/EnergyWeights/3dganWeights_channel_first/params_discriminator_epoch_000.hdf5']
      
   flags =[test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   d = discriminator(keras_dformat=dformat)
   for layer in d.layers[1:]:
       if hasattr(layer, 'layers'):
          for l in layer.layers:
            l.trainable=False
       else:
          layer.trainable=False

   g = generator(latent, keras_dformat=dformat)
   
   var= perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendir, discdir, num_data=nbEvents
         , num_events=binevents, m=moments, scales=xscales, thresh=thresh, flags=flags, latent=latent, particle=particle, dformat=dformat)
   pl.get_plots_multi(var, labels, plotsdir, energies, moments, len(gweights), ifpdf, stest, cell)

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, default=200, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/bigdata/shared/LCD/NewV1/*scan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--plotsdir', action='store', type=str, default='results/Analysis_plots_ch_first/', help='Directory to store the analysis plots.')
    parser.add_argument('--sortdir', action='store', type=str, default='SortedData', help='Directory to store sorted data.')
    parser.add_argument('--gendir', action='store', type=str, default='Gen', help='Directory to store the generated images.')
    parser.add_argument('--discdir', action='store', type=str, default='Disc', help='Directory to store the discriminator outputs.')
    parser.add_argument('--nbEvents', action='store', type=int, default=100000, help='Total Number of events used for Testing')
    parser.add_argument('--binevents', action='store', type=int, default=2000, help='Number of events in each bin')
    parser.add_argument('--moments', action='store', type=int, default=3, help='Number of moments')
    parser.add_argument('--cell', action='store', type=int, default=0, help='Whether to plot cell energies..0)Not plotted...1)Only for bin with uniform spectrum.....2)For all energy bins')
    parser.add_argument('--corr', action='store', default=False, help='Plot correlation plots')
    parser.add_argument('--test', action='store', default=True, help='Use Test data')
    parser.add_argument('--stest', action='store', default=False, help='Statistics test for shower profiles')
    parser.add_argument('--save_data', action='store', default=False, help='Save sorted data')
    parser.add_argument('--read_data', action='store', default=False, help='Get saved and sorted data')
    parser.add_argument('--save_gen', action='store', default=False, help='Save generated images')
    parser.add_argument('--read_gen', action='store', default=False, help='Get saved generated images')
    parser.add_argument('--save_disc', action='store', default=False, help='Save discriminator output')
    parser.add_argument('--read_disc', action='store', default=False, help='Get discriminator output')
    parser.add_argument('--ifpdf', action='store', default=True, help='Whether generate pdf plots or .C plots') 
    parser.add_argument('--gweights', action='store', type=str, default=['../weights/3dganWeights/params_generator_epoch_049.hdf5'], help='list for paths to Generator weights.')
    parser.add_argument('--dweights', action='store', type=str, default=['../weights/3dganWeights/params_discriminator_epoch_049.hdf5'], help='list for paths to Discriminator weights')
    parser.add_argument('--xscales', action='store', type=int, default=[100], help='list for Multiplication factors for all models to be checked')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--energies', action='store', type=int, default=[0, 50, 100, 200, 250, 300, 400, 500], help='Energy bins for analysis')
    parser.add_argument('--thresh', action='store', type=int, default=0, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_first', help='keras image format')
    return parser


if __name__ == "__main__":
    main()
