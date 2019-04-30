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
#from AngleArch3dGAN import generator, discriminator
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
  labels =["G4"]#, "GAN"] # labels
  outdir = 'results/CellsHistG4/' # dir for results
  gan.safe_mkdir(outdir)
  particle='Ele'
  datapath = '/bigdata/shared/LCD/NewV1/*Escan/*Escan*.h5'
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  energies =[0, 50, 100, 150, 200, 300]# energy bins
  sorted_data = gan.get_sorted(data_files[:2], energies, thresh=thresh) # load data in a dict
  ROOT.gStyle.SetOptFit(1011)
  # for each energy bin
  for energy in energies:
    print(energy, "========================================")
    edir = outdir + '{}GeV/'.format(energy)
    gan.safe_mkdir(edir)
    filename = path.join(edir, "CellHist".format(energy)) # file name
    gevents = sorted_data["events_act" + str(energy)]
    penergy = sorted_data["energy" + str(energy)]
    index= gevents.shape[0]  # number of events in bin
    print(index)
    w=2
    midx = int(gevents.shape[1]/2)
    midy = int(gevents.shape[2]/2)
    midz = int(gevents.shape[3]/2)
    ecal_g4 = gevents[:, midx-w:midx+w, midy-w:midy+w, midz-w:midz+w]
    n = ecal_g4.shape[0]
    x_shape = ecal_g4.shape[1]
    y_shape = ecal_g4.shape[2]
    z_shape = ecal_g4.shape[3]
    print('berycenter non zero ={} %'.format( 100.0 * (np.count_nonzero(gevents[:,midx, midy, midz])/float(n))))
    print('berycenter min {} and max {}'.format(np.amin(gevents[:, midx, midy, midz]), np.amax(gevents[:, midx, midy, midz])))
    cells = x_shape * y_shape * z_shape
    hists_g4=[]
    c=[]
    
    for x in np.arange(x_shape):
      for y in np.arange(y_shape):
        for z in np.arange(z_shape):
          maxe = max(np.amax(ecal_g4[:, x, y, z]), np.amax(ecal_g4[:, x, y, z]))
          mine = min(np.amin(ecal_g4[:, x, y, z]), np.amin(ecal_g4[:, x, y, z]))
          hists_g4.append(ROOT.TH1F('histg4_{}x_{}y_{}z'.format(midx-w+x, midy-w+y, midz-w+z), 'histg4_{}x_{}y_{}z'.format(midx-w+x, midy-w+y, midz-w+z), 100, mine, 1.1 * maxe))
    i=0
    for num in np.arange(n):
      for x in np.arange(x_shape):
         for y in np.arange(y_shape):
            for z in np.arange(z_shape):
              hists_g4[i].Fill(ecal_g4[num, x, y, z])
              i+=1
      i=0
    #for p in np.arange(cells):
    p=0
    for x in np.arange(x_shape):
      for y in np.arange(y_shape):
        for z in np.arange(z_shape):
           outfile = filename + 'pos_x{}_y{}_z{}'.format(midx-w+x, midy-w+y, midz-w+z)
           c.append(ROOT.TCanvas("c"+str(p) ,"c"+str(p) ,200 ,10 ,700 ,500))
           c[p].SetGrid()
           hists_g4[p].SetLineColor(2)
           hists_g4[p].GetYaxis().SetTitle("Entries")
           hists_g4[p].GetXaxis().SetTitle("Energy [GeV]")
           hists_g4[p].Draw()
           #hists_gan[p].Draw('sames')
           c[p].Update()
           g1 = ROOT.TF1("m1","gaus",)
           hists_g4[p].Fit(g1)
           c[p].Update()
           c[p].Print(outfile + ".pdf")
           p+=1
           

    print ("histograms saved to {} dir".format(outdir))
     
if __name__ == "__main__":
  main()
