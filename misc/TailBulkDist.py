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
   genweight = "/nfshome/gkhattak/3Dgan/weights/3dgan_weights_newarch3_lr/params_generator_epoch_050.hdf5"

   from AngleArch3dGAN_newarch3 import generator

   numdata = 1000
   scale=1
   outdir = 'results/ecal_tails_bulk_newarch3_lr/'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   x = x/50.0 # convert to GeV
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))

   power=0.85
   latent = 256 # latent space for generator
   concat=2
   
   g=generator(latent) # build generator
   g.load_weights(genweight) # load weights        
   x_gen1 = gan.generate(g, numdata, [y/100, ang], latent, concat=concat)
   x_gen1 = (1./50.) * np.power(x_gen1, 1./0.85)
   
   labels = ['G4', 'GAN']
   plot_ecal_flatten_hist([x, x_gen1], outfile, y, labels)
   plot_ecal_flatten_hist([x, x_gen1], outfile + '_log', y, labels, logy=1)
   plot_ecal_flatten_hist([x[:, 10:40, 10:40, :], x_gen1[:, 10:40, 10:40, :]], outfile + 'bulk', y, labels, logy=0)
   plot_ecal_flatten_hist([x[:, 10:40, 10:40, :], x_gen1[:, 10:40, 10:40, :]], outfile + 'bulk_log', y, labels, logy=1)
   plot_ecal_flatten_hist([x[:, :10, :10, :], x_gen1[:, :10, :10, :]], outfile + 'tail1', y, labels, set_range=1, logy=0)
   plot_ecal_flatten_hist([x[:, :10, :10, :], x_gen1[:, :10, :10, :]], outfile + 'tail1_log', y, labels, set_range=1,logy=1)
   plot_ecal_flatten_hist([x[:, 40:, 40:, :], x_gen1[:, 40:, 40:, :]], outfile + 'tail2', y, labels, set_range=1,logy=0)
   plot_ecal_flatten_hist([x[:, 40:, 40:, :], x_gen1[:, 40:, 40:, :]], outfile + 'tail2_log', y, labels, set_range=1,logy=1)
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
                     

def plot_ecal_flatten_hist(events, out_file, energy, labels, logy=0, norm=0, set_range=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Cell energy deposits for 100-200 GeV "
   legend = ROOT.TLegend(.1, .1, .3, .2)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -12, 2))
      hd = hds[i]
      #hd.SetStats(0)
      r.BinLogX(hd)
      data = event.flatten()
      r.fill_hist(hd, data)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:                  
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Ecal Single cell depositions [GeV]")
        hd.GetYaxis().SetTitle("Entries")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        hd.Draw('sames hist')
        color+=2
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
        if set_range:
         maxbin = hd.GetMaximumBin()
         val = hd.GetBinContent(maxbin)
         hds[0].SetMaximum(1.1 * val)
        color+=1
        c1.Update()
        r.stat_pos(hd)
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
