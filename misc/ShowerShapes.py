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
   numdata = 1000
   scale=1
   outdir = 'results/showerShapes/'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   x = x/50.0 # convert to GeV
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   plot_energy_hist(x, outfile + '_1000event_log', n=1000, rep=1, log=1)
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
                     
def plot_energy_hist(events, out_file, n=1, rep=1, log=0, p=[2, 500], ifpdf=True, stest=True):
   canvas = ROOT.TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                                                                           
   canvas.SetTitle('Weighted Histogram for energy deposition along x, y, z axis')
   canvas.SetGrid()
   color = 2
   canvas.Divide(2,2)
   events=np.squeeze(events)
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
   events=events[:n]
   array1x= np.sum(events, axis=(2, 3))
   array1y= np.sum(events, axis=(1, 3))
   array1z= np.sum(events, axis=(1, 2))
   
   h1x = ROOT.TH1F('G4x' , '', x, 0, x)
   h1y = ROOT.TH1F('G4y' , '', y, 0, y)
   h1z = ROOT.TH1F('G4z' , '', z, 0, z)

   h1x.SetLineColor(color)
   h1y.SetLineColor(color)
   h1z.SetLineColor(color)

   #h1x.Sumw2()
   #h1y.Sumw2()
   #h1z.Sumw2()
   h1x.SetStats(0)
   h1y.SetStats(0)
   h1z.SetStats(0)
   color+=2
   canvas.cd(1)
   if log:
      ROOT.gPad.SetLogy()
   
   r.fill_hist_wt(h1x, array1x)
   h1x=r.normalize(h1x)
   h1x.Draw('hist')
   #h1x.Draw('sames hist')
   h1x.GetXaxis().SetTitle("position along x axis")
   h1x.GetYaxis().SetTitle("energy deposition")
   canvas.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   r.fill_hist_wt(h1y, array1y)
   h1y=r.normalize(h1y)
   h1y.Draw('hist')
   #h1y.Draw('sames hist')
   h1y.GetXaxis().SetTitle("position along y axis")
   h1y.GetYaxis().SetTitle("energy deposition")
   canvas.cd(3)
   if log:
      ROOT.gPad.SetLogy()
   r.fill_hist_wt(h1z, array1z)
   h1z=r.normalize(h1z)
   h1z.Draw('hist')
   #h1z.Draw('sames hist')
   h1z.GetXaxis().SetTitle("position along z axis")
   h1z.GetYaxis().SetTitle("energy deposition")
   canvas.cd(4)
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file + '.pdf')
   else:
      canvas.Print(out_file + '.C')

if __name__ == "__main__":
   main()
