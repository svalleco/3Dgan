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

from ROOT import TTree, TFile, AddressOf, gROOT, std, vector
gROOT.ProcessLine("#include <vector>");

def main():
  latent = 256  #latent space
  power=0.85    #power for cell energies used in training
  thresh =0.0   #threshold used
  get_shuffled= True # whether to make plots for shuffled
  labels =["G4", "GAN"] # labels
  outdir = 'results/GANRoot' # dir for results
  gan.safe_mkdir(outdir) 
  datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path     
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  energies =[150]# energy bins
  angles=[62, 90, 118]
  g = generator(latent)       # build generator
  gen_weight1= "../weights/3dgan_weights_bins_pow_p85/params_generator_epoch_059.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[24:], energies, thresh=thresh, tolerance1=2.5) # load data in a dict

  # for each energy bin
  for energy in energies:
    filename1 = path.join(outdir, "GAN{}GeV.root".format(energy)) # file name
    filename2 = path.join(outdir, "Data{}GeV.root".format(energy)) # file name
    gevents = sorted_data["events_act" + str(energy)]
    penergy = sorted_data["energy" + str(energy)]
    theta = sorted_data["angle" + str(energy)]
    index= gevents.shape[0]  # number of events in bin
    # generate images
    generated_images = gan.generate(g, index, [penergy/100., theta]
                                     , latent, concat=1)
    # post processing
    generated_images = np.power(generated_images, 1./power)

    ecal1 = generated_images/50.
    ecal2 = gevents/50.
       
    ofile1 = ROOT.TFile(filename1,'RECREATE')
    ecalTree = TTree('ecalTree', "ecal Tree")
    en = 0
    ei = ecal1.shape[1]
    ej = ecal1.shape[2]
    ek = ecal1.shape[3]

    vec_x = ROOT.std.vector(int)()
    vec_y = ROOT.std.vector(int)()
    vec_z = ROOT.std.vector(int)()
    vec_E = ROOT.std.vector(float)()

    ecalTree.Branch('x',vec_x)
    ecalTree.Branch('y',vec_y)
    ecalTree.Branch('z',vec_z)
    ecalTree.Branch('E',vec_E)

    for e in range(index):
      ec = ecal1[e]
      vec_x.clear()
      vec_y.clear()
      vec_z.clear()
      vec_E.clear()

      for i in range(ei):
        for j in range(ej):
          for k in range(ek):

            energy = ec[i][j][k]
            if energy > 0:
              vec_E.push_back(energy)
              vec_x.push_back(i)
              vec_y.push_back(j)
              vec_z.push_back(k)
      ecalTree.Fill()
      en += 1

    ofile1.Write()
    ofile1.Close()

    print ("{} GAN events saved to {}".format(index, filename1))

    ofile2 = ROOT.TFile(filename2,'RECREATE')
    ecalTree2 = TTree('ecalTree2', "ecal Tree2")
    vec_x2 = ROOT.std.vector(int)()
    vec_y2 = ROOT.std.vector(int)()
    vec_z2 = ROOT.std.vector(int)()
    vec_E2 = ROOT.std.vector(float)()
    en=0
    ecalTree2.Branch('x',vec_x2)
    ecalTree2.Branch('y',vec_y2)
    ecalTree2.Branch('z',vec_z2)
    ecalTree2.Branch('E',vec_E2)
                                                        
    for e in range(index):
      ec = ecal2[e]
      vec_x2.clear()
      vec_y2.clear()
      vec_z2.clear()
      vec_E2.clear()

      for i in range(ei):
        for j in range(ej):
          for k in range(ek):
            energy = ec[i][j][k]
            if energy > 0:
              vec_E2.push_back(energy)
              vec_x2.push_back(i)
              vec_y2.push_back(j)
              vec_z2.push_back(k)
      ecalTree2.Fill()
      en += 1

    ofile2.Write()
    ofile2.Close()
                                                                                                                                                                                                                                 
    print ("{} Data events saved to {}".format(index, filename2))
                  
if __name__ == "__main__":
  main()
