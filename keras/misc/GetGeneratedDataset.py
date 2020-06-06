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
  particle = 'Ele'
  num_files=10
  feat = ['energy', 'mtheta', 'theta', 'recoTheta', 'phi', 'recoPhi', 'eta', 'recoEta','pdgID', 'HCAL', 'ECAL_E', 'HCAL_E'] 
  outdir = '/storage/group/gpu/bigdata/gkhattak/{}_generated/'.format(particle) # dir for results
  gan.safe_mkdir(outdir) 
  datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech
  data_files = gan.GetDataFiles(datapath, [particle]) # get list of files
  energies =[0]# energy bins
  angles=[0]
  g = generator(latent)       # build generator
  gen_weight1= "../keras/weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  for i in np.arange(num_files):
    g4_data = GetAngleData(data_files[i], features=feat)
    # generate images
    generated_images = gan.generate(g, g4_data['energy'].shape[0], [g4_data['energy']/100., g4_data['mtheta']], latent, concat=2)
    # post processing
    generated_images = np.power(generated_images, 1./power)
    generated_images = np.squeeze(generated_images)
    print('GAN images vary from {} to {}'.format(np.amin(generated_images[generated_images>0]), np.amax(generated_images)))
    ganfile = path.join(outdir ,'{}_generated_{}.hdf5'.format(particle, i))
    with h5py.File(ganfile ,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      for key in g4_data:
        print(key)
        if key=='mtheta':
          angle = gan.measPython(generated_images)
          outfile.create_dataset(key,data=angle)
        else:
          outfile.create_dataset(key,data=g4_data[key])
    print ("Generated data saved to ", ganfile)

  # get variable angle data                                                                                                        
def GetAngleData(datafile, features=['energy', 'theta', 'phi','eta', 'pdgID']):
    #get data for training                                                                                                       
    print ('Loading Data from .....', datafile)
    data={}
    f=h5py.File(datafile,'r')
    ecal =np.array(f.get('ECAL'))
    ecal_sum = np.sum(ecal, axis=(1, 2, 3))
    indexes = np.where(ecal_sum > 10.0)
    ecal = ecal[indexes]
    print('data images vary from {} to {}'.format(np.amin(ecal[ecal>0]), np.amax(ecal)))
    for feat in features:
        if feat in f:
          data[feat] = np.array(f.get(feat))[indexes]
          data[feat] = data[feat].astype(np.float32)
        elif feat=='mtheta':
          data[feat] = gan.measPython(ecal)
    return data
                
if __name__ == "__main__":
  main()
