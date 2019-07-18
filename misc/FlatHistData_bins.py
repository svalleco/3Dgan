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
   outdir = 'results/data_bins_spectrum'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath) #, ftn=np.sqrt)
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   labels = ['G4 Var. Angle']
   bins=[0.05, 0,03, 0.02, 0.0125, 0.008, 0.003]
   bin_edges = [15, 0.05, 0,03, 0.02, 0.0125, 0.008, 0.003, 0., 0.]
   scale = 50.
   bins = [_/scale for _ in bins]
   bin_edges = [_/scale for _ in bin_edges]
   p=[int(np.amin(y)), int(np.amax(y))]   
   plot_ecal_hist_bins([x/scale], bins, outfile, y, labels, logy=0, norm=2, p=p, ifpdf=False)
   print('Histogram is saved in ', outfile)
         
def GetAngleData(datafile, numevents=1000, ftn=0, scale=1, angtype='theta'):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents]) * scale
   if ftn!=0:
      x = ftn(x)
   ang = np.array(f.get(angtype)[:numevents])
   return x, y, ang

def plot_ecal_flatten_hist_bins(events, bins, out_file, energy, labels, logy=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Cell energy {:d}-{:d} GeV Primary".format(p[0], p[1])
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -8, 1))
      hd = hds[i]
      hd.SetStats(0)
      r.BinLogX(hd)
      data = event.flatten()
      r.fill_hist(hd, data)
      lines=[]
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
      for ln, b in enumerate(bins):
        lines.append(ROOT.TLine(b, 0, b, 0.06))
        lines[ln].Draw('sames')
      color+=2
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()
      
   legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')

def plot_ecal_hist_bins(events, bins_edges, out_file, energy, labels, logy=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                              
   c1.SetGrid()
   ROOT.gPad.SetLogx()

   title = "Cell energy {:d}-{:d} GeV Primary".format(p[0], p[1])
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", len(bin_edges)_1, bins_edges))
      hd = hds[i]
      hd.SetStats(0)
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
      color+=2
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
