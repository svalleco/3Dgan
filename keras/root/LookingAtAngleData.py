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
   #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path
   datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5"
   numdata = 100000
   numevents = 1000
   energies=[10, 250, 400]
   phis = [2, 10, 18]
   thetas = [65, 90, 118]
   xcuts = [10, 25, 40] 
   ycuts = [10, 25, 40]
   zcuts = [5, 12, 20]
   datafiles = GetDataFiles(datapath, numdata, Particles=['Ele'])
   plotsdir = 'angle_plots'
   gan.safe_mkdir(plotsdir)
   x, y, theta, phi, angle= GetAllData(datafiles)
   yvar = gan.sort([x, y], energies, num_events=numevents)
   thetavar = gan.sort([x, theta], thetas, num_events=numevents, tolerance=2)
   phivar = gan.sort([x, phi], phis, num_events=numevents, tolerance=2)
   for energy in energies:
      for xcut, ycut, zcut in zip(xcuts, ycuts, zcuts):
         PlotCut(yvar["events_act" + str(energy)], xcut, ycut, zcut, energy, os.path.join(plotsdir, '2Dhits_cut{}_{}.pdf'.format(xcut, energy)))
         print('The number of events in {} GeV bin is {}.'.format(energy, yvar["events_act" + str(energy)].shape[0]))
   for t in thetas:
      PlotAngleCut(thetavar["events_act" + str(t)], t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)))
      print('The number of events in {} theta bin is {}.'.format(t, thetavar["events_act" + str(t)].shape[0]))
   for p in phis:
      PlotAngleCut(phivar["events_act" + str(p)], p, os.path.join(plotsdir, 'Phi_cut{}.pdf'.format(p)))
      print('The number of events in {} phi bin is {}.'.format(p, phivar["events_act" + str(p)].shape[0])) 

def PlotCut(events, xcut, ycut, zcut, energy, out_file):
   canvas = TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   hx = ROOT.TH2F('x_{}GeV_x={}cut'.format(str(energy), str(xcut)), '', 51, 0, 51, 25, 0, 25)
   hy = ROOT.TH2F('y_{}GeV_y={}cut'.format(str(energy), str(ycut)), '', 51, 0, 51, 25, 0, 25)
   hz = ROOT.TH2F('z_{}GeV_z={}cut'.format(str(energy), str(zcut)), '', 51, 0, 51, 51, 0, 51)
   gStyle.SetPalette(1)
   gPad.SetLogz()
   FillHist2D_wt(hx, events[:, xcut, :, :])
   FillHist2D_wt(hy, events[:, :, ycut, :])
   FillHist2D_wt(hz, events[:, :, :, zcut])
   canvas.cd(1)
   hx.Draw("colz")
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw("colz")
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw("colz")
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   canvas.Update()
   r.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotAngleCut(events, ang, out_file):
   canvas = TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500) 
   canvas.Divide(2,2)
   hx = ROOT.TH2F('x=25_{}cut'.format(str(ang)), '', 51, 0, 51, 25, 0, 25)
   hy = ROOT.TH2F('y=25_{}cut'.format(str(ang)), '', 51, 0, 51, 25, 0, 25)
   hz = ROOT.TH2F('z=12_{}cut'.format(str(ang)), '', 51, 0, 51, 51, 0, 51)
   FillHist2D_wt(hx, events[:, 25, :, :])
   FillHist2D_wt(hy, events[:, :, 25, :])
   FillHist2D_wt(hz, events[:, :, :, 12])
   canvas.cd(1)
   hx.Draw("colz")
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw("colz")
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw("colz")
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   canvas.Update()
   r.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def FillHist2D_wt(hist, array):
   array= np.squeeze(array)
   dim1 = array.shape[0]
   dim2 = array.shape[1]
   dim3 = array.shape[2]
   bin1 = np.arange(dim1)
   bin2 = np.arange(dim2)
   bin3 = np.arange(dim3)
   count = 0
   for j in bin2:
     for k in bin3:
        for i in bin1:
            hist.Fill(j, k, array[i, j, k])
            count+=1
   print('For {} events with {} xbins and {} ybins the countis {}'.format(dim1, dim2, dim3, count))
   #hist.Sumw2()

def GetAngleData(datafile):
    #get data for training
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy'))
    x=np.array(f.get('ECAL'))
    phi = np.array(f.get('phi')) * 57.325
    theta = np.array(f.get('theta')) * 57.325
    angle = np.array(f.get('openingAngle')) * 57.325
    x[x < 1e-6] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    
    return x, y, theta, phi, angle
   
def GetAllData(datafiles):
   for index, datafile in enumerate(datafiles):
       if index == 0:
          x, y, theta, phi, angle = GetAngleData(datafile)
       else:
          x_temp, y_temp, theta_temp, phi_temp, angle_temp = GetAngleData(datafile)
          x = np.concatenate((x, x_temp), axis=0)
          y = np.concatenate((y, y_temp), axis=0)
          theta = np.concatenate((theta, theta_temp), axis=0)
          phi = np.concatenate((phi, phi_temp), axis=0)
          angle = np.concatenate((angle, angle_temp), axis=0)
   return x, y, theta, phi, angle  

def GetDataFiles(FileSearch="/data/LCD/*/*.h5", nEvents=800000, EventsperFile = 10000, Particles=[], MaxFiles=-1):
   print ("Searching in :",FileSearch)
   Files =sorted( glob.glob(FileSearch))
   print ("Found {} files. ".format(len(Files)))
   Filesused = int(math.ceil(nEvents/EventsperFile))
   FileCount=0
   Samples={}
   for F in Files:
       FileCount+=1
       basename=os.path.basename(F)
       ParticleName=basename.split("_")[0].replace("Escan","")
       if ParticleName in Particles:
           try:
               Samples[ParticleName].append(F)
           except:
               Samples[ParticleName]=[(F)]
       if MaxFiles>0:
           if FileCount>MaxFiles:
               break
   SampleI=len(Samples.keys())*[int(0)]
   for i,SampleName in enumerate(Samples):
       Sample=Samples[SampleName][:Filesused]
       NFiles=len(Sample)
   return Sample

if __name__ == "__main__":
    main()


   
