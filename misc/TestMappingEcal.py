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
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')
import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import setGPU

def main():
   datapath = "/data/shared/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"

   numdata = 1000
   scale=1
   outdir = 'results/test_mapping'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   out=GetAngleData(datapath, numdata, ['ECAL', 'energy'])
   x = out[0]
   y = out[1]
   maxval = 26951.5
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   x2 = inv_mapping(x)
   print('The mapped data varies from {} to {}'.format(np.amin(x2[x2>0]), np.amax(x2)))
   x3= x * x2
   print('The weighted by mapped data varies from {} to {}'.format(np.amin(x3[x3>0]), np.amax(x3)))
   x4 = x3/maxval
   print('The weighted by mapping function data varies from {} to {}'.format(np.amin(x4[x4>0]), np.amax(x4)))
   labels = ['G4', 'G4 mapped']
   #plot_ecal_flatten_hist([x], outfile + '_G4', y, [labels[0]], logy=1, norm=2)
   #plot_ecal_mapped_hist([x2], outfile + '_mapped', y, [labels[1]], logy=1, norm=2)
   plot_ecal_mapped_hist([x4], outfile + '_mapped_weighted_normalized', y, ['G4 weighted by mapped and normalized'], logy=1, norm=2)
   #plot_ecal_flatten_hist([x, x2], outfile + '_G4_mapped', y, labels, logy=1, norm=2)
   print('Histogram is saved in ', outfile)
       
def postproc(event, f, scale):
   return f(event)/scale

def safe_log10(x):
   out = 1. * x
   out[np.where(out>0)] = np.log10(out[np.where(out>0)])
   return out

def mapping(x):
      p0 = 6.82245e-02
      p1 = -1.70385
      p2 = 6.27896e-01
      p3 = 1.39350
      p4 = 2.26181
      p5 = 1.23621e-01
      p6 = 4.30815e+01
      p7 = -8.20837e-02
      p8 = -1.08072e-02
      res = 1. * x
      res[res<1e-7]=0
      log10x = safe_log10(res)
      res = (p0 /(1+np.power(np.abs((log10x-p1)/p2),2*p3))) * ((p4+res*(p7+res*p8))* np.sin(p5*(log10x-p6)))
      return res

def inv_mapping(x):
   out = np.where(x<1e-7, 0, 1./mapping(x))
   return out
   
def GetAngleData(datafile, numevents, datatype):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   out = []
   for d in datatype:
      out.append(np.array(f.get(d)[:numevents]))
   return out

def plot_ecal_flatten_hist(events, out_file, energy, labels, logy=0, norm=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   ROOT.gPad.SetLogx()
   title = "Cell energy deposits for 100-200 GeV "
   legend = ROOT.TLegend(.1, .8, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -12, 7))
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

def plot_ecal_mapped_hist(events, out_file, energy, labels, logy=0, norm=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   ROOT.gPad.SetLogx()

   title = "Cell energy deposits for mapped(inverse of fit) 100-200 GeV "
   legend = ROOT.TLegend(.1, .7, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -4, 7))
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
        hd.GetXaxis().SetTitle("Ecal Single cell mapped")
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
