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
   datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_1_1.h5"
      
   numdata = 1000
   scale=1
   dscale =50.
   outdir = 'results/g4_distribution_bins'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   
   bins=[0.05, 0.03, 0.02, 0.0125, 0.008, 0.003]
   bins.extend([0, -0.003])
   bins = sorted(bins)
   bins.extend([np.amax(x)])
   edges = np.zeros(len(bins))
   for i, bin in enumerate(bins):
     edges[i] = bins[i]/dscale
   labels = ['Monte Carlo']
   plot_ecal_flatten_hist_bins(x/dscale, edges, outfile, y, labels, logy=0, norm=2, ifpdf=True)
   plot_ecal_flatten_hist_bins(x/dscale, edges, outfile +'_log', y, labels, logy=1, norm=2, ifpdf=True) 
   print('Histogram is saved in ', outfile)
       
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

def plot_ecal_flatten_hist_bins(events, bins, out_file, energy, labels, logy=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   #c1.SetGrid()
   ROOT.gPad.SetLogx()      
   title = "Cell energy distribution"
   legend = ROOT.TLegend(.1, .8, .3, .9)
   color =2
   print bins
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
        hd.GetXaxis().SetTitle("Ecal Single cell depositions [GeV]")
        hd.GetYaxis().SetTitle("Normalized Count")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        hd.Draw('sames hist')
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
      for ln, b in enumerate(bins):
        lines.append(ROOT.TLine(b, 0, b, 0.05))
        lines[ln].Draw('sames')
      color+=2
      legend.AddEntry(hd,label ,"l")
      legend.AddEntry(lines[0],'bins' ,"l")
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
