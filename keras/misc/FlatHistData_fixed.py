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
sys.path.insert(0,'../')

import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import setGPU

def main():
   datapath = "/storage/group/gpu/bigdata/LCD/NewV1/EleEscan/EleEscan_1_1.h5"
   numdata = 2000
   scale1=1.0
   scale2=100.0
   from EcalEnergyGan import generator
   dscale =1.
   leg =0
   genweights1="weights_versions/generator_epoch_029.hdf5"
   genweights2="weights_versions/generator_epoch_049_rootfit.hdf5"
   genweights3="weights_versions/generator_epoch_041_scaled.hdf5"   
   outdir = 'results/compare_intensitiesGAN/'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y=gan.GetData(datapath, num_events=numdata)
   print(datapath, numdata)
   print(x.shape)
   print(np.amax(x))
   print('The data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   
   latent = 200 # latent space for generator
   g=generator(latent) # build generator
   g.load_weights(genweights1) # load weights        
   x_gen = gan.generate(g, numdata, [y/100], latent)
      
   g.load_weights(genweights2) # load weights
   x_gen1 = gan.generate(g, numdata, [y/100], latent)
         
   g.load_weights(genweights3) # load weights
   x_gen2 = gan.generate(g, numdata, [y/100], latent)
   x_gen2 = x_gen2/scale2 
   
   #g2=generator(latent)
   #g2.load_weights(genweight3) # load weights
   #x_gen3 = gan.generate(g2, numdata, [y/100, ang], latent)
   #x_gen3 = np.power(x_gen3, 1/0.85)/dscale
   
   """
   latent = 200 # latent space for generator
   g2.load_weights(genweight2) # load weights
   x_gen3 = gan.generate(g2, numdata, [y/100], latent)
   x_gen4 = x_gen3/100
   
   g.load_weights(genweight3) # load weights
   x_gen3 = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen4 = postproc(x_gen3 , np.square, scale)
   """
   labels = ['MC', 'GAN ','GAN scaled by 100']
   plot_ecal_flatten_hist([x, x_gen1, x_gen2], outfile, y, labels, norm=1)
   plot_ecal_flatten_hist([x, x_gen1, x_gen2], outfile + '_log', y, labels, logy=1, norm=1)
   
   print('Histogram is saved in ', outfile)
       
def postproc(event, f, scale):
   return f(event)/scale
def GetData(datafile, thresh=0, num_events=10000):
   #get data for training
    print( 'Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    y=f.get('target')[:num_events]
    x=np.array(f.get('ECAL')[:num_events])
    y=(np.array(y[:,1]))
    print(np.amax(x), thresh)
    if thresh!=0:
       print('loop')
       x[x < thresh] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

def plot_ecal_flatten_hist(events, out_file, energy, labels, logy=0, norm=0, leg=1, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   #c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Cell energy deposits for {} - {} GeV".format(int(np.amin(energy)), int(np.amax(energy)))
   legend = ROOT.TLegend(.11, .6, .3, .89)
   legend.SetBorderSize(0)
   color =1
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      color+= 1
      if color==5:
         color+= 1
      hds.append(ROOT.TH1F(label, "", 100, -12, 0))
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
        hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV")
        hd.GetYaxis().SetTitle("Count")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        hd.Draw('sames hist')
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()
      
   if leg:legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')


if __name__ == "__main__":
   main()
