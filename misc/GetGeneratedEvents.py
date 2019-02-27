from __future__ import print_function
from os import path
import ROOT
import h5py
import numpy as np
import keras.backend as K
import tensorflow as tf
#import tensorflow.python.ops.image_ops_impl as image 
import time
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../analysis')
import utils.GANutils as gan
import utils.ROOTutils as roo
from skimage import measure
import math
from AngleArch3dGAN import generator, discriminator
try:
  import setGPU
except:
  pass

def main():
  latent = 256  #latent space
  power=0.85    #power for cell energies used in training
  thresh =0.0   #threshold used
  get_shuffled= True # whether to make plots for shuffled
  labels =["G4", "GAN"] # labels
  outdir = 'results/GANevents' # dir for results
  gan.safe_mkdir(outdir) 
  datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path     
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  energies =[0, 110, 150, 190]# energy bins
  angles=[62, 90, 118]
  g = generator(latent)       # build generator
  gen_weight1= "../weights/3dgan_weights_bins_pow_p85/params_generator_epoch_059.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[24:], energies, thresh=thresh) # load data in a dict

  # for each energy bin
  for energy in energies:
     for a in angles:
       filename = path.join(outdir, "{}GeV_{}degree.h5".format(energy, a)) # file name
       indexes = np.where(((sorted_data["angle" + str(energy)]) > math.radians(a) - 0.1) & ((sorted_data["angle" + str(energy)]) < math.radians(a) + 0.1))
       events_g4 = sorted_data["events_act" + str(energy)][indexes]
       penergy = sorted_data["energy" + str(energy)][indexes]
       theta = sorted_data["angle" + str(energy)][indexes]
       index= events_g4.shape[0]  # number of events in bin
       # generate images
       generated_images = gan.generate(g, index, [penergy/100., theta]
                                     , latent, concat=1)
       # post processing
       generated_images = np.power(generated_images, 1./power)

       with h5py.File(filename ,'w') as outfile:
         outfile.create_dataset('ECALG4',data=events_g4)
         outfile.create_dataset('ECALGAN',data=generated_images)
         outfile.create_dataset('energy',data=penergy)
         outfile.create_dataset('theta',data=theta)
       print ("Generated data saved to ", filename)
                  
if __name__ == "__main__":
  main()
