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
sys.path.insert(0,'../keras')
sys.path.insert(0,'../keras/analysis')
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
  particle = 'Ele'
  outdir = 'results/GANevents' # dir for results
  gan.safe_mkdir(outdir) 
  datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech
  data_files = gan.GetDataFiles(datapath, [particle]) # get list of files
  energies =[0]# energy bins
  angles=[0]
  g = generator(latent)       # build generator
  gen_weight1= "../keras/weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[-2:], energies, thresh=thresh, angtype='theta') # load data in a dict

  # for each energy bin
  for energy in energies:
     for a in angles:
       edir = path.join(outdir, "{}GeV_{}degree".format(energy, a)) # 
       gan.safe_mkdir(edir)
       ganfile = path.join(edir, "Ele_GAN.h5")
       g4file = path.join(edir, "Ele_GEANT4.h5")
       if a==0:
          indexes = np.where(sorted_data["angle" + str(energy)] > 0)
       else:
          indexes = np.where(((sorted_data["angle" + str(energy)]) > math.radians(a) - 0.1) & ((sorted_data["angle" + str(energy)]) < math.radians(a) + 0.1))
       events_g4 = sorted_data["events_act" + str(energy)][indexes]
       penergy = sorted_data["energy" + str(energy)][indexes]
       theta = sorted_data["angle" + str(energy)][indexes]
       pdgid = np.zeros_like(theta)
       index= events_g4.shape[0]  # number of events in bin
       # generate images
       generated_images = gan.generate(g, index, [penergy/100., theta]
                                     , latent, concat=1)
       # post processing
       generated_images = np.power(generated_images, 1./power)

       with h5py.File(ganfile ,'w') as outfile:
         #outfile.create_dataset('ECALG4',data=events_g4)
         outfile.create_dataset('ECAL',data=generated_images)
         outfile.create_dataset('energy',data=penergy)
         outfile.create_dataset('theta',data=theta)
         outfile.create_dataset('pdgID',data=pdgid)
       print ("Generated data saved to ", ganfile)
       with h5py.File(g4file ,'w') as out2file:
         out2file.create_dataset('ECAL',data=events_g4)
         out2file.create_dataset('energy',data=penergy)
         out2file.create_dataset('theta',data=theta)
         out2file.create_dataset('pdgID',data=pdgid)
       print ("G4 data saved to ", g4file)

                  
if __name__ == "__main__":
  main()
