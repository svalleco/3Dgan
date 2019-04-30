from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import math
import time
import glob
import sys
import numpy.core.umath_tests as umath
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/analysis')
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')
sys.path.insert(0,'/nfshome/gkhattak/keras/architectures_tested/')
import utils.GANutils as gan
import utils.ROOTutils as r
from utils.RootPlotsGAN import flip
import setGPU

def main():
   #datapath = "/data/shared/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"
   datapath = "/data/shared/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_8_2.h5"
   #datapath3 = '/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5'
   genweight = "/nfshome/gkhattak/3Dgan/weights/3dgan_weights_newarch_layers_all/params_generator_epoch_007.hdf5"
   out_file='results/sparsity_newarch_layers_ep20_all'
   from AngleArch3dGAN_newarch_layers import generator

   # data post processing to GeV
   numdata = 1000
   x, y, ang=GetAngleData(datapath, numdata, angtype='theta')
   x = x/50.0 # convert to GeV
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))

   # GAN related params
   power=0.85
   latent = 256 # latent space for generator
   g=generator(latent) # build generator
   g.load_weights(genweight) # load weights        
   x_gen = np.squeeze(gan.generate(g, numdata, [y/100, ang], latent, concat=2))
   x_gen = (1./50.) * np.power(x_gen, 1./0.85)

   # Thresholds to use
   thresh=np.arange(-13, 1, 1)
   entries_g4 = np.zeros((thresh.shape[0], x.shape[0]))
   entries_gan =np.zeros((thresh.shape[0], x.shape[0]))
   size = np.float64(x[0].size)

   # Plots related parameters
   labels=['G4', 'GAN']
   mono = False
   leg= True
   energy = 0

   # calculating entries for different threshold applied
   for i in np.arange(thresh.shape[0]):
      t_val = np.power(10.0, thresh[i])
      x_t = np.where(x>t_val, 1, 0)
      entries_g4[i] = np.divide(np.sum(x_t, axis=(1, 2, 3)), size)
      x_gen_t = np.where(x_gen > t_val, 1, 0)
      entries_gan[i] = np.divide(np.sum(x_gen_t, axis=(1, 2, 3)), size)
   plot_sparsity(entries_g4, entries_gan, thresh, out_file, energy, labels, logy=0, min_max=0, ifpdf=True, mono=mono, legend=leg)
   
def GetAngleData(datafile, numevents, ftn=0, scale=1, angtype='theta'):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents]) * scale
   if ftn!=0:
      x = ftn(x)
   ang = np.array(f.get(angtype)[:numevents])
   return x, y, ang

def GetData(datafile, numevents, scale=1, thresh=1e-6):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents])
   x[x<thresh] = 0
   x = x * scale
   return x, y
                              
def GetData2(datafile, numevents, scale=1, thresh=1e-6):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('target')[:numevents, 1])
   x=np.array(f.get('ECAL')[:numevents])
   x[x<thresh] = 0
   x = x * scale
   return x, y
                     
def plot_sparsity(entries1, entries2, thresh, out_file, energy, labels, logy=0, min_max=0, ifpdf=True, mono=False, legend=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()

   title = "Sparsity for electrons with 100-200 GeV primary energy"
   leg = ROOT.TLegend(.8, .8, .9, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   sparsity1 = ROOT.TGraph()
   sparsity2 = ROOT.TGraph()
   sparsity1b = ROOT.TGraph()
   sparsity2b = ROOT.TGraph()
   
   mean1=np.mean(entries1, axis=1)
   mean2=np.mean(entries2, axis=1)
   std1=np.std(entries1, axis=1)
   std2=np.std(entries2, axis=1)
   min1= np.min(entries1, axis=1)
   min2= np.min(entries2, axis=1)
   max1= np.max(entries1, axis=1)
   max2= np.max(entries2, axis=1)
   
   r.fill_graph(sparsity1, thresh, mean1)
   r.fill_graph(sparsity2, thresh, mean2)
   if min_max:
      area1 = np.concatenate((min1, flip(max1, 0)), axis=0)
      area2 = np.concatenate((min2, flip(max2, 0)), axis=0)
      ymax=max(np.max(max2), np.max(max1))
      ylim = 1.1 * ymax
   else:
      area1 = np.concatenate((mean1+std1, flip(mean1-std1, 0)), axis=0)
      area2 = np.concatenate((mean2+std2, flip(mean2-std2, 0)), axis=0)
      ylim = 0.04
   thresh2=np.concatenate((thresh, flip(thresh, 0)), axis=0)
      
   r.fill_graph(sparsity1b, thresh2, area1)
   r.fill_graph(sparsity2b, thresh2, area2)
   sparsity1.GetXaxis().SetTitle("log10(threshold[GeV])")
   sparsity1.GetYaxis().SetTitle("Fraction of cells above threshold")
   sparsity1.SetTitle(title)
   sparsity1.SetLineColor(color)
   sparsity1.Draw('APL')
   sparsity1.GetYaxis().SetRangeUser(0, ylim)
   sparsity2.SetLineColor(color+2)
   if mono: sparsity2.SetLineStyle(2)   
   sparsity2.Draw('PL')

   sparsity1b.SetFillColorAlpha(color, 0.35)
   sparsity2b.SetFillColorAlpha(color+2, 0.35)
   
   sparsity1b.Draw('f')
   sparsity2b.Draw('F')
   
   leg.AddEntry(sparsity1,labels[0] ,"l")
   leg.AddEntry(sparsity2,labels[1] ,"l")
   c1.Update()
   if legend:
     leg.Draw()
     c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C') 

if __name__ == "__main__":
   main()
