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
import setGPU

def main():
   datapath = "/data/shared/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"
   #datapath2 = "/data/shared/LCDLargeWindow/fixedangle/EleEscan/EleEscan_1_1.h5"
   #datapath3 = '/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5'
   genweight = "/nfshome/gkhattak/3Dgan/weights/3Dweights_p25/params_generator_epoch_059.hdf5"
   genweight1 = "/nfshome/gkhattak/3Dgan/weights/3Dweights_sqrt_add_loss/params_generator_epoch_059.hdf5"
   genweight2 = "/nfshome/gkhattak/3Dgan/weights/3Dweights_p75/params_generator_epoch_059.hdf5"
   #genweight3 = "/nfshome/gkhattak/3Dgan/weights/3dgan_weights_bins_pow_p85/params_generator_epoch_059.hdf5"
   genweight3 = "/nfshome/gkhattak/3Dgan/keras/weights/3dgan_weights_gan_training_epsilon_k2/params_generator_epoch_131.hdf5"
   # generator model
   from AngleArch3dGAN import generator
   from arch_bin_pow import generator as generator2
   
   numdata = 1000
   scale=1
   dscale =50.
   outdir = 'results/compare_pow'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   #x2, y2=GetData(datapath2, numdata, thresh=0)
   #print('The fixed data varies from {} to {}'.format(np.amin(x2[x2>0]), np.amax(x2)))
   #x3, y3=GetData2(datapath3, numdata, thresh=0)
   #print('The fixed data varies from {} to {}'.format(np.amin(x3[x3>0]), np.amax(x3)))
   x = x/dscale
   latent = 256 # latent space for generator
   g=generator2(latent) # build generator
   g.load_weights(genweight) # load weights        
   x_gen = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen = np.power(x_gen, 1/0.25)/dscale
   
   g.load_weights(genweight1) # load weights
   x_gen1 = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen1 = np.power(x_gen1, 1/0.5)/dscale
   
   g.load_weights(genweight2) # load weights
   x_gen2 = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen2 = np.power(x_gen2, 1/0.75)/dscale

   g2=generator(latent)
   g2.load_weights(genweight3) # load weights
   x_gen3 = gan.generate(g2, numdata, [y/100, ang], latent)
   x_gen3 = np.power(x_gen3, 1/0.85)/dscale
   
   """
   latent = 200 # latent space for generator
   g2.load_weights(genweight2) # load weights
   x_gen3 = gan.generate(g2, numdata, [y/100], latent)
   x_gen4 = x_gen3/100
   
   g.load_weights(genweight3) # load weights
   x_gen3 = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen4 = postproc(x_gen3 , np.square, scale)
   """
   labels = ['G4', 'power 0.25', 'power 0.5', 'power 0.75', 'power 0.85']
   #plot_ecal_flatten_hist([x, x2, x_gen1, x_gen2, x_gen3, x_gen4], outfile, y, labels, norm=1)
   #plot_ecal_flatten_hist([x, x2, x_gen1, x_gen2, x_gen3, x_gen4], outfile + '_log', y, labels, logy=1, norm=1)
   plot_ecal_flatten_hist([x, x_gen, x_gen1, x_gen2, x_gen3], outfile + '_log', y, labels, logy=1, norm=2)
   print('Histogram is saved in ', outfile)
       
def postproc(event, f, scale):
   return f(event)/scale

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
                     

def plot_ecal_flatten_hist(events, out_file, energy, labels, logy=0, norm=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   #c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Cell energy deposits for 100-200 GeV "
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      if label=='G4':
          color =2
      elif label=='power 0.85':
          color = 4
      elif label=='power 0.75':
          color = 3
      elif label=='power 0.5':
          color = 7
      else:
          color = 6
      hds.append(ROOT.TH1F(label, "", 100, -15, 0))
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
        #hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV/50")
        #hd.GetYaxis().SetTitle("Count")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        hd.Draw('sames hist')
        color+=2
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()
      
   #legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')


if __name__ == "__main__":
   main()
