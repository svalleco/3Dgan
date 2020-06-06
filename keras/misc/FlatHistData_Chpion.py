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
   #datapath = "/data/shared/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"
   #datapath2 = "/data/shared/LCDLargeWindow/fixedangle/EleEscan/EleEscan_1_1.h5"
   datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/ChPiEscan/ChPiEscan_RandomAngle_1_10.h5"
   numdata = 1000
   scale=1
   dscale =50.
   outdir = 'results/ecal_pion'
   labels =['G4 ChPi']
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   x = x/dscale
   ecal = np.sum(x, axis=(1, 2, 3))
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   print('The ecal varies from {} to {}'.format(np.amin(ecal[ecal>0]), np.amax(ecal)))
   plot_ecal_flatten_hist([x], outfile + '_log', y, labels, logy=1, norm=2)
   plot_ecal_flatten_hist([x], outfile,  y, labels, logy=0, norm=2)
   plot_ecal_hist([ecal], outfile + '_sum', y, labels, logy=0, norm=2)
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

def plot_ecal_hist(events, out_file, energy, labels, logy=0, norm=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   #c1.SetGrid() 
   title = "Ecal sum for 100-200 GeV "
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (ecal, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, 0, 1.1 * (np.amax(ecal))))
      hd = hds[i]
      hd.SetStats(0)
      r.fill_hist(hd, ecal)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:
        hd.SetTitle(title)
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
