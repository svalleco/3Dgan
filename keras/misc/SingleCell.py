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
import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as roo
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
  concat =2
  get_shuffled= True # whether to make plots for shuffled
  labels =["MC", "GAN"] # labels
  outdir = 'results/CellsHistGan_0_500GeV_25_25_5/' # dir for results
  gan.safe_mkdir(outdir) 
  #datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path
  datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5"      
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  energies =[0]# energy bins
  angles=[62, 90, 118]
  xc = 25
  yc = 25
  zc= 5
  g = generator(latent)       # build generator
  gen_weight1= "../weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[-3:], energies, thresh=thresh, tolerance1=2.5) # load data in a dict
  print('data center non zero', np.count_nonzero(sorted_data["events_act" + str(0)][:, xc, yc, zc]))
  print('data center min', np.amin(sorted_data["events_act" + str(0)][:, xc, yc, zc]))
  # for each energy bin
  for energy in energies:
    filename = path.join(outdir, "CellProf{}GeV".format(energy)) # file name
    gevents = sorted_data["events_act" + str(energy)]
    penergy = sorted_data["energy" + str(energy)]
    theta = sorted_data["angle" + str(energy)]
    index= gevents.shape[0]  # number of events in bin
    # generate images
    generated_images = gan.generate(g, index, [penergy/100., theta]
                                     , latent, concat=concat)
    # post processing
    generated_images = np.power(generated_images, 1./power)

    ecal_g4 = generated_images/50.
    ecal_gan = gevents/50.
    w=1
    print('data center non zero after div', np.count_nonzero(ecal_g4[:,25, 25, 12]))
    print('data center min', np.amin(ecal_g4[:, 25, 25, 12]))
    print('gan non zero', np.count_nonzero(ecal_gan[:, 25, 25, 12]))
        
    ecal_g4 = ecal_g4[:, xc-w:xc+w+1, yc-w:yc+w+1, zc-w:zc+w+1]
    ecal_gan = ecal_gan[:, xc-w:xc+w+1, yc-w:yc+w+1, zc-w:zc+w+1]
    n = ecal_g4.shape[0]
    x_shape = ecal_g4.shape[1]
    y_shape = ecal_g4.shape[2]
    z_shape = ecal_g4.shape[3]
    print('g4 max' , np.amax(ecal_g4))
    print('gan max' , np.amax(ecal_gan))
    print(np.count_nonzero(ecal_g4[:,w, w,  w]))
    print(np.count_nonzero(ecal_gan[:,w, w, w]))
    maxe = max(np.amax(ecal_g4), np.amax(ecal_gan))
    cells = x_shape * y_shape * z_shape
    hists_g4=[]
    hists_gan=[]
    c=[]
    i=0
    print(n, x_shape, y_shape, z_shape, cells)
    for x in np.arange(x_shape):
      for y in np.arange(y_shape):
        for z in np.arange(z_shape):
          maxe = max(np.amax(ecal_g4[:, x, y, z]), np.amax(ecal_gan[:, x, y, z]))
          mine = min(np.amin(ecal_g4[:, x, y, z]), np.amin(ecal_gan[:, x, y, z]))
          hists_g4.append(ROOT.TH1F('histg4{}x_{}y_{}z'.format(x+xc-w, y+yc-w, z+zc-w), 'histg4{}x_{}y_{}z'.format(x+xc-w, y+yc-w, z+zc-w), 100, 1e-6,  maxe))
          hists_gan.append(ROOT.TH1F('histgan{}x_{}y_{}z'.format(x+xc-w, y+yc-w, z+zc-w), 'histgan{}x_{}y_{}z'.format(x+xc-w, y+yc-w, z+zc-w), 100, 1e-6, maxe))
          hists_g4[i].Sumw2()
          hists_gan[i].Sumw2()
          roo.fill_hist(hists_g4[i], ecal_g4[:, x, y, z])
          roo.fill_hist(hists_gan[i], ecal_gan[:, x, y, z])
          outfile = filename + '{}x_{}y_{}z'.format(x+xc-w, y+yc-w, z+zc-w)
          c.append(ROOT.TCanvas("c"+str(i) ,"c"+str(i) ,200 ,10 ,700 ,500))
          hists_g4[i].SetLineColor(2)
          hists_gan[i].SetLineColor(4)
          hists_g4[i].SetStats(0)
          hists_gan[i].SetStats(0)
          hists_g4[i].GetYaxis().SetTitle("Entries")
          hists_g4[i].GetXaxis().SetTitle("Energy [GeV]")
          ymax = max(hists_g4[i].GetMaximum(), hists_gan[i].GetMaximum())
          hists_g4[i].GetXaxis().SetRangeUser(0, 0.1)#1.1 * ymax)
          hists_g4[i].Draw()
          hists_g4[i].Draw('hist')
          hists_gan[i].Draw('sames hist')
          c[i].Update()
          c[i].Print(outfile + ".pdf")
          i+=1
    
    print ("histograms saved to {}".format(outdir))

if __name__ == "__main__":
  main()
