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
#sys.path.insert(0,'/nfshome/gkhattak/keras/architectures_tested/')
import utils.GANutils as gan
import utils.ROOTutils as r
import setGPU

def main():
   datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_*.h5"
   datapath2 = '../../Ele_generated_0.hdf5'
   datafiles =  sorted( glob.glob(datapath))
   numdata = 5000
   scale=1
   dscale = 50.
   angle = [62, 118]
   outdir = 'results/data_compare_ecal_angles'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   for i in range(10):
     if i ==0:
       x, y, ang=GetAngleData(datafiles[i], numdata) #, ftn=np.sqrt)
       indexes = np.where((ang>(np.radians(angle[0])-0.1)) & ((ang<(np.radians(angle[0])+0.1))))
       x1 = x[indexes]
       y1 = y[indexes]
       ang1 = ang[indexes]
   
       x1 = x1/dscale
       indexes = np.where((ang>(np.radians(angle[1])-0.1)) & ((ang<(np.radians(angle[1])+0.1))))
       x2 = x[indexes]
       y2 = y[indexes]
       ang2 = ang[indexes]
   
       x2 = x2/dscale 

     else:
       x, y, ang=GetAngleData(datafiles[i], numdata)
       indexes = np.where((ang>(np.radians(angle[0])-0.1)) & ((ang<(np.radians(angle[0])+0.1))))
       x1 = np.concatenate((x1, x[indexes]), axis=0)
       y1 = np.concatenate((y1, y[indexes]), axis=0)
       ang1 = np.concatenate((ang1, ang[indexes]), axis=0)

       indexes = np.where((ang>(np.radians(angle[1])-0.1)) & ((ang<(np.radians(angle[1])+0.1))))
       x2 = np.concatenate((x2, x[indexes]), axis=0)
       y2 = np.concatenate((y2, y[indexes]), axis=0)
       ang2 = np.concatenate((ang2, ang[indexes]), axis=0)
   x1 = x1/dscale
   x2 = x2/dscale
   print(x1.shape[0])
   print(x2.shape[0])
   min_p = np.amin([np.amin(y), np.amin(y2)])
   max_p = np.amax([np.amax(y), np.amax(y2)])
   p = [int(min_p), int(max_p)]
   print(p) 

   labels = ['G4 ' + str(angle[0]), 'G4 ' + str(angle[1])]
   
   ecal1 = np.sum(x1, axis=(1, 2, 3))
   ecal2 = np.sum(x2, axis=(1, 2, 3))
   
   print('The G4 data sum varies from {} to {}'.format(np.amin(ecal1), np.amax(ecal2)))
   print('The generated data sum varies from {} to {}'.format(np.amin(ecal2), np.amax(ecal2)))
   sf1= ecal1/y1
   sf2= ecal2/y2
   
   
   plot_ecal_prof([sf1, sf2], [y1, y2], outfile+'_prof', y, labels, logy=0, norm=0, p=p)
   plot_ecal_prof([ecal1, ecal2], [y1, y2], outfile + 'prof_log', y, labels, logy=1, norm=0, p=p)
   plot_ecal_hist([ecal1, ecal2], outfile+'_', labels, logy=0, norm=0, p=p)
   plot_ecal_hist([ecal1, ecal2], outfile + '_log',  labels, logy=1, norm=0, p=p)

   print('Histogram are saved in ', outdir)
          
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
                     

def plot_ecal_hist(events, out_file, labels, logy=0, norm=0, ifpdf=True, p=[2, 500]):
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
      hds.append(ROOT.TH1F(label, "", 100, 0., 12))
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
      #hd.SetStats(0)
      #r.BinLogX(hd)
      data = event
      r.fill_profile(hd, data, e)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("E_p [GeV]")
        hd.GetYaxis().SetTitle("S_F")
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
