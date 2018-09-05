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
import utils.GANutils as gan
import utils.ROOTutils as r
import setGPU

def main():
   datapath = "/data/shared/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"
   genweight = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_25weight_sqrt/params_generator_epoch_049.hdf5"
   # generator model
   from AngleArch3dGAN_sqrt import generator

   numdata = 1000
   outdir = 'epoch_50'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   
   latent = 256 # latent space for generator
   g=generator(latent) # build generator
   g.load_weights(genweight) # load weights        
   x_gen = gan.generate(g, numdata, [y/100, ang], latent)

   labels = ['sqrt G4', 'sqrt GAN', 'G4', 'square GAN']
   plot_ecal_flatten_hist([np.sqrt(x), x_gen, x, np.square(x_gen)], outfile, y, labels)
   plot_ecal_flatten_hist([np.sqrt(x), x_gen, x, np.square(x_gen)],  outfile + '_log', y, labels, logy=1)
   plot_ecal_flatten_hist([np.sqrt(x), x_gen, x, np.square(x_gen)], outfile + '_log_norm', y, labels, logy=1, norm=1)
   print('Histogram is saved in ', outfile)
       

def GetAngleData(datafile, numevents, angtype='theta'):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents])
   ang = np.array(f.get(angtype)[:numevents])
   return x, y, ang

def plot_ecal_flatten_hist(events, out_file, energy, labels, logy=0, norm=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Ecal Flat Histogram for 2-500 GeV"
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (y scale in log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -12, 2))
      hd = hds[i]
      hd.SetStats(0)
      r.BinLogX(hd)
      data = event.flatten()
      r.fill_hist(hd, data)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:                  
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV/50")
        hd.GetYaxis().SetTitle("Count")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        color+=2
      else:
        hd.Draw('sames')
        color+=1
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()
      
   legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')


if __name__ == "__main__":
   main()
