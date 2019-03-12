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
  thresh =0   #threshold used
  get_shuffled= True # whether to make plots for shuffled
  labels =["G4", "GAN"] # labels
  outdir = 'results/CellsHistGan/' # dir for results
  gan.safe_mkdir(outdir) 
  datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path     
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  energies =[150]# energy bins
  angles=[62, 90, 118]
  g = generator(latent)       # build generator
  gen_weight1= "../weights/3dgan_weights_bins_pow_p85/params_generator_epoch_059.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[24:], energies, thresh=thresh, tolerance1=2.5) # load data in a dict
  print('data center non zero', np.count_nonzero(sorted_data["events_act" + str(150)][:, 25, 25, 12]))
  print('data center min', np.amin(sorted_data["events_act" + str(150)][:, 25, 25, 12]))
  # for each energy bin
  for energy in energies:
    filename = path.join(outdir, "CellProf{}GeV".format(energy)) # file name
    gevents = sorted_data["events_act" + str(energy)]
    penergy = sorted_data["energy" + str(energy)]
    theta = sorted_data["angle" + str(energy)]
    index= gevents.shape[0]  # number of events in bin
    # generate images
    generated_images = gan.generate(g, index, [penergy/100., theta]
                                     , latent, concat=1)
    # post processing
    generated_images = np.power(generated_images, 1./power)

    ecal_g4 = generated_images/50.
    ecal_gan = gevents/50.
    w=2
    print('data center non zero after div', np.count_nonzero(ecal_g4[:,25, 25, 12]))
    print('data center min', np.amin(ecal_g4[:, 25, 25, 12]))
    print('gan non zero', np.count_nonzero(ecal_gan[:, 25, 25, 12]))
        
    ecal_g4 = ecal_g4[:, 25-w:25+w, 25-w:25+w, 12-w:12+w]
    ecal_gan = ecal_gan[:, 25-w:25+w, 25-w:25+w, 12-w:12+w]
    n = ecal_g4.shape[0]
    x_shape = ecal_g4.shape[1]
    y_shape = ecal_g4.shape[2]
    z_shape = ecal_g4.shape[3]
    print('g4 max' , np.amax(ecal_g4))
    print('gan max' , np.amax(ecal_gan))
    print(np.count_nonzero(ecal_g4[:,w, w, w]))
    print(np.count_nonzero(ecal_gan[:, w, w, w]))
    maxe = max(np.amax(ecal_g4), np.amax(ecal_gan))
    cells = x_shape * y_shape * z_shape
    hists_g4=[]
    hists_gan=[]
    c=[]
    print(n, x_shape, y_shape, z_shape, cells)
    for x in np.arange(x_shape):
      for y in np.arange(y_shape):
        for z in np.arange(z_shape):
          maxe = max(np.amax(ecal_g4[:, x, y, z]), np.amax(ecal_gan[:, x, y, z]))
          mine = min(np.amin(ecal_g4[:, x, y, z]), np.amin(ecal_gan[:, x, y, z]))
          hists_g4.append(ROOT.TH1F('histg4{}x_{}y_{}z'.format(x, y, z), 'histg4{}x_{}y_{}z'.format(x, y, z), 100, mine, 1.1 * maxe))
          hists_gan.append(ROOT.TH1F('histgan{}x_{}y_{}z'.format(x, y, z), 'histgan{}x_{}y_{}z'.format(x, y, z), 100, mine, 1.1 * maxe))
    i=0
    for num in np.arange(n):
      for x in np.arange(x_shape):
         for y in np.arange(y_shape):
            for z in np.arange(z_shape):
              hists_g4[i].Fill(ecal_g4[num, x, y, z])
              hists_gan[i].Fill(ecal_gan[num, x, y, z])
              i+=1
      i=0
    for p in np.arange(cells):
      outfile = filename + str(p)
      c.append(ROOT.TCanvas("c"+str(p) ,"c"+str(p) ,200 ,10 ,700 ,500))
      c[p].SetGrid()
      hists_g4[p].SetLineColor(2)
      hists_gan[p].SetLineColor(4)
      hists_g4[p].GetYaxis().SetTitle("Entries")
      hists_g4[p].GetXaxis().SetTitle("Energy [GeV]")
      hists_g4[p].Draw()
      hists_gan[p].Draw('sames')
      c[p].Update()
      c[p].Print(outfile + ".pdf")

    
    print ("histograms saved to {} dir".format(outdir))

if __name__ == "__main__":
  main()
