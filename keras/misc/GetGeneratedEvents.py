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
  particle = 'Pi0'
  outdir = 'results/Gan_G4_Events_Pi0' # dir for results
  gan.safe_mkdir(outdir) 
  datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech
  data_files = gan.GetDataFiles(datapath, [particle]) # get list of files
  energies =[0]# energy bins
  angles=[0]
  g = generator(latent)       # build generator
  gen_weight1= "../keras/weights/3dgan_weights_gan_training_Pi0_2_500GeV_decay_p001/params_generator_epoch_023.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  g4_data = GetAngleData(data_files[0])
  ganfile = path.join(outdir, particle + "_GAN.h5")
  g4file = path.join(outdir, particle + "_GEANT4.h5")
  events_g4 = g4_data['ECAL']
  penergy = g4_data['energy']
  theta = g4_data['theta']
  eta = g4_data['eta']
  phi = g4_data['phi']
  pdgid_g4 = np.ones_like(penergy)
  pdgid_gan = np.zeros_like(penergy)
  index= events_g4.shape[0]  # number of events in bin
  # generate images
  generated_images = gan.generate(g, index, [penergy/100., theta], latent, concat=2)
  # post processing
  generated_images = np.power(generated_images, 1./power)
  generated_images = np.squeeze(generated_images)
  with h5py.File(ganfile ,'w') as outfile:
    outfile.create_dataset('ECAL',data=generated_images)
    outfile.create_dataset('energy',data=penergy)
    outfile.create_dataset('theta',data=theta)
    outfile.create_dataset('pdgID',data=pdgid_gan)
    outfile.create_dataset('phi',data=phi)
    outfile.create_dataset('eta',data=eta)
  print ("Generated data saved to ", ganfile)
  with h5py.File(g4file ,'w') as out2file:
    out2file.create_dataset('ECAL',data=events_g4)
    out2file.create_dataset('energy',data=penergy)
    out2file.create_dataset('theta',data=theta)
    out2file.create_dataset('pdgID',data=pdgid_g4)
    out2file.create_dataset('phi',data=phi)
    out2file.create_dataset('eta',data=eta)
  print ("G4 data saved to ", g4file)

  # get variable angle data                                                                                                        
def GetAngleData(datafile, features=['energy', 'theta', 'phi','eta', 'pdgID']):
    #get data for training                                                                                                       
    print ('Loading Data from .....', datafile)
    data={}
    f=h5py.File(datafile,'r')
    data['ECAL'] =np.array(f.get('ECAL'))
    ecal = np.sum(data['ECAL'], axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    data['ECAL'] = data['ECAL'][indexes]
    data['ECAL'] = data['ECAL'].astype(np.float32)
    for feat in features:
        data[feat] = np.array(f.get(feat))[indexes]
        data[feat] = data[feat].astype(np.float32)
    return data
                
if __name__ == "__main__":
  main()
