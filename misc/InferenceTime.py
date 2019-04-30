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
import math
from AngleArch3dGAN_newarch_layers import generator, discriminator
try:
  import setGPU
except:
  pass

def main():
  latent = 256  #latent space
  power=0.85    #power for cell energies used in training
  thresh =0   #threshold used
  concat=2
  num_events=10000
  batch_size=256
  init1=time.time()
  g = generator(latent)       # build generator
  gen_weight1= "../weights/3dgan_weights_newarch_layers/params_generator_epoch_020.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sampled_energies=np.random.uniform(1., 2., size=num_events)
  sampled_angles=np.random.uniform(np.radians(60), np.radians(120), size=num_events)
  init2=time.time()
  generated_image = gan.generate(g, num_events, [sampled_energies, sampled_angles], latent=latent, concat=concat)
  generated_image = np.power(generated_image, 1./0.85)
  end = time.time()
  print('The generator took {} sec to generate {} events taking total of {} sec'.format(end-init2, num_events, end-init1))
  per_shower1= (end-init1)/num_events
  per_shower2= (end-init2)/num_events
  print('1 shower took {} msec and total of {} sec'.format(per_shower2 * 1000, per_shower1 * 1000))
  
if __name__ == "__main__":
  main()
