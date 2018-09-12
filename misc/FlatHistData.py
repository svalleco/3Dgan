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
   genweight1 = "/nfshome/gkhattak/3Dgan/weights/params_generator_epoch_032.hdf5"
   #genweight2 = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_25weight_sqrt/params_generator_epoch_010.hdf5"
   #genweight3 = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_25weight_sqrt/params_generator_epoch_020.hdf5"
   genweight4 = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_25weight/params_generator_epoch_040.hdf5"
      
   # generator model
   from AngleArch3dGAN_sqrt import generator as g1
   from AngleArch3dGAN import generator as g2

   numdata = 1000
   scale=10
   outdir = 'sqrt_scale10_5'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   
   latent = 256 # latent space for generator
   g=g1(latent) # build generator
   g.load_weights(genweight1) # load weights        
   x_gen1 = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen2 = postproc(x_gen1 , np.square, 10)

   g = g2(latent)
   g.load_weights(genweight4) # load weights
   x_gen3 = gan.generate(g, numdata, [y/100, ang], latent)
   """
   g.load_weights(genweight3) # load weights
   x_gen3 = gan.generate(g, numdata, [y/100, ang], latent)
      
   g.load_weights(genweight4) # load weights
   x_gen4 = gan.generate(g, numdata, [y/100, ang], latent)
   """   
   labels = ['G4 ', 'GAN sqrt', 'GAN sqrt(square)', 'GAN without sqrt'] #, 'GAN epoch 10', 'GAN epoch 20', 'GAN epoch 40']
   plot_ecal_flatten_hist([x, x_gen1, x_gen2, x_gen3], outfile, y, labels, norm=1)
   plot_ecal_flatten_hist([x, x_gen1, x_gen2, x_gen3], outfile + '_log', y, labels, logy=1, norm=1)
   print('Histogram is saved in ', outfile)
       
def postproc(event, f, scale):
   return f(event)/scale

def sqrt(x):
   epsilon= np.finfo(float).eps
   indexes = np.where(x>0)
   x[indexes] = np.sqrt(x[indexes])
   return x
                

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
        hd.Draw('sames hist')
        color+=2
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
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
