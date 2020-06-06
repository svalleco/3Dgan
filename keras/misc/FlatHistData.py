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
sys.path.insert(0,'../keras/analysis')
sys.path.insert(0,'../')

import utils.GANutils as gan
import utils.ROOTutils as r
import setGPU

def main():
   datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_1_1.h5"
   datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/ChPiEscan/ChPiEscan_RandomAngle_1_1.h5"
   datapath3 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/Pi0Escan/Pi0Escan_RandomAngle_1_1.h5"
   datapath4 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/GammaEscan/GammaEscan_RandomAngle_1_1.h5"
   
   numdata = 2000
   scale=1
   dscale =50.
   leg =0   
   outdir = 'results/compare_particles_leg0'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   print(datapath, numdata)
   print(x.shape)
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   #x2, y2=GetData(datapath2, numdata, thresh=0)
   #print('The fixed data varies from {} to {}'.format(np.amin(x2[x2>0]), np.amax(x2)))
   #x3, y3=GetData2(datapath3, numdata, thresh=0)
   #print('The fixed data varies from {} to {}'.format(np.amin(x3[x3>0]), np.amax(x3)))
   x = x/dscale

   x2, y2, ang2=GetAngleData(datapath2, numdata)
   x2 = x2/dscale

   x3, y3, ang3=GetAngleData(datapath3, numdata)
   x3 = x3/dscale

   x4, y4, ang4=GetAngleData(datapath4, numdata)
   x4 = x4/dscale


   #latent = 256 # latent space for generator
   #g=generator2(latent) # build generator
   #g.load_weights(genweight) # load weights        
   #x_gen = gan.generate(g, numdata, [y/100, ang], latent)
   #x_gen = np.power(x_gen, 1/0.25)/dscale
   
   #g.load_weights(genweight1) # load weights
   #x_gen1 = gan.generate(g, numdata, [y/100, ang], latent)
   #x_gen1 = np.power(x_gen1, 1/0.5)/dscale
   
   #g.load_weights(genweight2) # load weights
   #x_gen2 = gan.generate(g, numdata, [y/100, ang], latent)
   #x_gen2 = np.power(x_gen2, 1/0.75)/dscale

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
   labels = ['Ele', 'ChPi', 'Pi0', 'Gamma']
   #plot_ecal_flatten_hist([x, x2, x_gen1, x_gen2, x_gen3, x_gen4], outfile, y, labels, norm=1)
   #plot_ecal_flatten_hist([x, x2, x_gen1, x_gen2, x_gen3, x_gen4], outfile + '_log', y, labels, logy=1, norm=1)
   plot_ecal_flatten_hist([x, x2, x3, x4], outfile + '', y, labels, logy=0, norm=2, leg=leg)
   plot_ecal_flatten_hist([x, x2, x3, x4], outfile + '_logy', y, labels, logy=1, norm=2, leg=leg)
   
   plot_ecal_flatten_hist([x], outfile + labels[0], y, [labels[0]], logy=0, norm=2, leg=leg)
   plot_ecal_flatten_hist([x], outfile + labels[0]+'_logy', y, [labels[0]], logy=1, norm=2, leg=leg)

   plot_ecal_flatten_hist([x2], outfile + labels[1], y, [labels[1]], logy=0, norm=2, leg=leg)
   plot_ecal_flatten_hist([x2], outfile + labels[1]+'_logy', y, [labels[1]], logy=1, norm=2, leg=leg)

   plot_ecal_flatten_hist([x3], outfile + labels[2], y, [labels[2]], logy=0, norm=2, leg=leg)
   plot_ecal_flatten_hist([x3], outfile + labels[2]+'_logy', y, [labels[2]], logy=1, norm=2, leg=leg)

   plot_ecal_flatten_hist([x4], outfile + labels[3], y, [labels[3]], logy=0, norm=2, leg=leg)
   plot_ecal_flatten_hist([x4], outfile + labels[3]+'_logy', y, [labels[3]], logy=1, norm=2, leg=leg)

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
                     

def plot_ecal_flatten_hist(events, out_file, energy, labels, logy=0, norm=0, leg=1, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   #c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Cell energy deposits for {} - {} GeV".format(int(np.amin(energy)), int(np.amax(energy)))
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =1
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      color+= 1
      if color==5:
         color+= 1
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
