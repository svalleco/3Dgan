from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
import GANutils as gan
import ROOTutils as r

def main():
   datapath = "/data/shared/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_2_2.h5"
   numdata = 100
   outfile = 'EcalFlat'
   x, y=GetAngleData(datapath, numdata)
   plot_ecal_flatten_hist(x, outfile, y)
   plot_ecal_flatten_hist(x, outfile + '_log', y, logy=1)
   plot_ecal_flatten_hist(x, outfile + '_log_norm', y, logy=1, norm=1)
   print('Histogram is saved in ', outfile)
       

def GetAngleData(datafile, numevents):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents])
   return x, y

def plot_ecal_flatten_hist(event, out_file, energy, logy=0, norm=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   ROOT.gPad.SetLogx()
   title = "Ecal Flat Histogram for 2-500 GeV"
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (y scale in log)"
   hd = ROOT.TH1F("Geant4", "", 100, -8, 2)
   #hd.Sumw2()
   #hd.Scale(1,"width")
   r.BinLogX(hd)
   data = event.flatten()
   
   r.fill_hist(hd, data)
   if norm:
      #data = data/data.shape[0]
      if hd.Integral()!=0:
         hd.Scale(1/hd.Integral())
                        
   hd.SetTitle(title)
   hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV/50")
   hd.GetYaxis().SetTitle("Count")
   hd.GetYaxis().CenterTitle()
   hd.Draw()
   legend = ROOT.TLegend(.3, .85, .4, .9)
   legend.AddEntry(hd,"Geant4","l")
   c1.Modified()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')


if __name__ == "__main__":
   main()
