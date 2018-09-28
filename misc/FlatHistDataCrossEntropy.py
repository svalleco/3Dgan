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
   genweight = "/nfshome/gkhattak/3Dgan/weights/params_generator_epoch_013.hdf5"
   genweight2 = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_25weight_sqrt/params_generator_epoch_042.hdf5"
   #genweight3 = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_25weight_sqrt/params_generator_epoch_025.hdf5"
   # generator model
   from AngleArch3dGAN_sqrt import generator
   #from EcalEnergyGan import generator as generator2
   #from AngleArch3dGAN import generator3

   numdata = 1000
   scale=1
   outdir = 'results/withaux42_woutaux_ep13'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   #x2, y2=GetData(datapath2, numdata, thresh=0)
   #print('The fixed data varies from {} to {}'.format(np.amin(x2[x2>0]), np.amax(x2)))
   #x3, y3=GetData2(datapath3, numdata, thresh=0)
   #print('The fixed data varies from {} to {}'.format(np.amin(x3[x3>0]), np.amax(x3)))
   
   latent = 256 # latent space for generator
   g=generator(latent) # build generator
   g.load_weights(genweight) # load weights        
   x_gen1 = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen2 = postproc(x_gen1, np.square, scale)
   
   """
   latent = 200 # latent space for generator
   g2.load_weights(genweight2) # load weights
   x_gen3 = gan.generate(g2, numdata, [y/100], latent)
   x_gen4 = x_gen3/100
   """
   g.load_weights(genweight2) # load weights
   x_gen3 = gan.generate(g, numdata, [y/100, ang], latent)
   x_gen4 = postproc(x_gen3 , np.square, scale)

   labels = ['G4', 'GAN without Aux(raw)', 'GAN without Aux(sq)', 'GAN (raw)', 'GAN (sq)']
   #plot_ecal_flatten_hist([x, x2, x_gen1, x_gen2, x_gen3, x_gen4], outfile, y, labels, norm=1)
   #plot_ecal_flatten_hist([x, x2, x_gen1, x_gen2, x_gen3, x_gen4], outfile + '_log', y, labels, logy=1, norm=1)
   plot_ecal_flatten_hist([x, x_gen1, x_gen2, x_gen3, x_gen4], outfile + '_log', y, labels, logy=1, norm=2)
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
   c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Cell energy deposits for 100-200 GeV "
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -12, 4))
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
      entropy = get_cross_entropy(events[0], event)
      legend.AddEntry(hd,label+ " CE={}".format(entropy) ,"l")
      c1.Modified()
      c1.Update()
      
   legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')

def get_cross_entropy(event1, event2):
   return np.sum(-1 * (event1.flatten()) * log_ftn(event2.flatten()))

def log_ftn(x):
   indexes=np.where(x>0)
   x[indexes] = np.log(x[indexes])
   return x

if __name__ == "__main__":
   main()
