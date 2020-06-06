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
  particle = 'ChPi'
  num_files=40
  events_per_file = 10000
  feat = ['HCAL', 'HCAL_E', 'ECAL', 'energy', 'theta', 'phi', 'pdgID', 'eta', "recoEta", "recoPhi", 'recoTheta'] 
  outdir = '~/{}_filter/'.format(particle) # dir for results
  gan.safe_mkdir(outdir) 
  datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech
  data_files = gan.GetDataFiles(datapath, [particle]) # get list of files
  out = 0
  for i in np.arange(num_files):
    if (i==0):
      g4_data = GetAngleData(data_files[i], features=feat)
    else:
      temp_data = GetAngleData(data_files[i], features=feat)
    if g4_data['ECAL'].shape[0] > events_per_file:
      filtfile = outdir + '{}scan_RandomAngle_filt_{}.h5'.format(particle, n)
      n+=1
      with h5py.File(filtfile ,'w') as outfile:
        for key in g4_data:
          print(key)
          outfile.create_dataset(key,data=g4_data[key][:events_per_file])
          g4_data[key] = g4_data[key][:events_per_file]
      print ("Generated data saved to ", filtfile)
      

  # get variable angle data                                                                                                        
def GetAngleData(datafile, features=['HCAL_E', 'ECAL_E','ECAL', 'energy', 'theta', 'phi','eta', 'pdgID']):
    #get data for training                                                                                                       
    print ('Loading Data from .....', datafile)
    data={}
    f=h5py.File(datafile,'r')
    ecal =np.array(f.get('ECAL_E'))
    #ecal_sum = np.sum(ecal, axis=(1, 2, 3))
    indexes = np.where(ecal > 0)
    data['ECAL_E'] = ecal[indexes]
    print(data['ECAL_E'].shape[0])
    for feat in features:
      print(feat)
      data[feat] = np.array(f.get(feat))[indexes]
      data[feat] = data[feat].astype(np.float32)
    #ecal_sum = np.sum(data['ECAL'], axis=(1, 2, 3))
    #hcal_sum = np.sum(data['HCAL'], axis=(1, 2, 3))
    indexes2 = np.where((data['HCAL_E']/(data['ECAL_E'])) < 0.1)
    for key in data:
      data[key] = data[key][indexes2]
      data[key] = data[key].astype(np.float32)
      if key=='energy':
         print('energy vary from {} to {}'.format(np.amin(data[key]), np.amax(data[key])))
    print('{} events left after filtering'.format(data['ECAL_E'].shape[0]))
    return data
                
if __name__ == "__main__":
  main()
