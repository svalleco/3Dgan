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
   datapath = "/data/shared/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_2_1.h5"
   datapath2 = "/data/shared/LCDLargeWindow/fixedangle/EleEscan/EleEscan_1_1.h5"
   datapath3 = '/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5'

   numdata = 5000
   scale=1
   outdir = 'results/data_compare_ecal'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata) #, ftn=np.sqrt)
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   x2, y2=GetData(datapath2, numdata, thresh=0)
   print('The fixed angle new data varies from {} to {}'.format(np.amin(x2[x2>0]), np.amax(x2)))
   x3, y3=GetData2(datapath3, numdata, thresh=0)
   print('The fixed angle old  varies from {} to {}'.format(np.amin(x3[x3>0]), np.amax(x3)))
   min_p = np.amin([np.amin(y), np.amin(y2), np.amin(y3)])
   max_p = np.amax([np.amax(y), np.amax(y2), np.amax(y3)])
   p = [int(min_p), int(max_p)]
   print(p) 

   labels = ['G4 Var. Angle', 'G4 Fixed Angle(new)', 'G4 Fixed Angle(old)']
   
   ecal1 = np.sum(x, axis=(1, 2, 3))
   ecal2 = np.sum(x2, axis=(1, 2, 3))
   ecal3 = np.sum(x3, axis=(1, 2, 3))
   sf1= np.sum(x, axis=(1, 2, 3))/(y *50.)
   sf2= np.sum(x2, axis=(1, 2, 3))/(y2 * 50.)
   sf3= np.sum(x3, axis=(1, 2, 3))/y3
   plot_ecal_hist([ecal1/50., ecal2/50., ecal3], outfile, y, labels, logy=0, norm=2, p=p)
   plot_ecal_hist([ecal1/50., ecal2/50., ecal3], outfile + '_log', y, labels, logy=1, norm=2, p=p)
   print('Histogram is saved in ', outfile)
   print('Histogram is saved in ', outfile+ '_log')
       
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
                     

def plot_ecal_hist(events, out_file, energy, labels, logy=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   #ROOT.gPad.SetLogx()
   
   title = "Sampling fraction for {:d}-{:d} GeV Primary".format(p[0], p[1])
   legend = ROOT.TLegend(.1, .1, .3, .3)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 50, 0., p[1]*1.1/50.))
      hd = hds[i]
      hd.SetStats(0)
      #r.BinLogX(hd)
      data = event
      r.fill_hist(hd, data)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:                  
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Ecal sum GeV")
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

def plot_ecal_prof(events, energies, out_file, energy, labels, logy=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   #ROOT.gPad.SetLogx()

   title = "Sampling fraction for {:d}-{:d} GeV Primary".format(p[0], p[1])
   legend = ROOT.TLegend(.1, .1, .3, .3)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, e, label) in enumerate(zip(events, energies, labels)):
      hds.append(ROOT.TProfile(label, "", 100, p[0], p[1]*1.1))
      hd = hds[i]
      hd.SetStats(0)
      #r.BinLogX(hd)
      data = event
      r.fill_profile(hd, data, e)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Ep GeV")
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
