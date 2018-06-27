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
   numdata = 10000
   numevents = 2000
   num=10
   f = 57.325
   energies=[10, 250, 400]
   phis = [2, 10, 18]
   thetas = [62, 90, 118]
   xcuts = [10, 25, 40] 
   ycuts = [10, 25, 40]
   zcuts = [5, 12, 20]
   events = np.random.randint(0, numdata, size=num)
   datafiles = GetDataFiles(datapath, numdata, Particles=['Ele'])
   plotsdir = 'angle_plots_col_new'
   gan.safe_mkdir(plotsdir)
   x, y, theta= GetAllData(datafiles)
   yvar = SortTheta([x, y, theta], energies, num_events=numevents)
   thetavar = gan.sort([x, theta], thetas, num_events=numevents, tolerance=2.0/f)
   for energy in energies:
      for xcut, ycut, zcut in zip(xcuts, ycuts, zcuts):
         PlotCut(yvar["events_act" + str(energy)], xcut, ycut, zcut, energy, os.path.join(plotsdir, '2Dhits_cut{}_{}.pdf'.format(xcut, energy)))
         print('The number of events in {} GeV bin is {}.'.format(energy, yvar["events_act" + str(energy)].shape[0]))   
      thetayvar=gan.sort([yvar["events_act" + str(energy)], yvar["theta_act"+ str(energy)]], thetas, num_events=numevents, tolerance=2)
      for t in thetas:
        PlotAngleCut(thetayvar["events_act" + str(t)], t, os.path.join(plotsdir, 'Theta{}_GeV{}.pdf'.format(t, energy)))
      plot_energy_hist_root(thetayvar, os.path.join(plotsdir, 'Thetashape_GeV{}.pdf'.format(energy)), energy, thetas, log=0, ifC=True)
   for t in thetas:
      PlotAngleCut(thetavar["events_act" + str(t)], t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)))
      print('The number of events in {} theta bin is {}.'.format(t, thetavar["events_act" + str(t)].shape[0]))
   for i, n in enumerate(events):
      PlotEvent(x[n], y[n], theta[n], os.path.join(plotsdir, 'Event{}.pdf'.format(i)), n)
   
def plot_energy_hist_root(events, out_file, energy, thetas, log=0, ifC=False):
   canvas = TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make  
   canvas.SetGrid()
   canvas.Divide(2,2)
   hx=[]
   hy=[]
   hz=[]
   color = 2
   leg = ROOT.TLegend(0.1,0.6,0.7,0.9)
   for i, theta in enumerate(thetas):
      event=events["events_act" + str(theta)]
      num=event.shape[0]
      sumx, sumy, sumz=gan.get_sums(event)
      print sumx.shape
      print sumy.shape
      print sumz.shape
      x=sumx.shape[1]
      y=sumy.shape[1]
      z=sumz.shape[1]
      hx.append( ROOT.TH1F('G4x' + str(theta), '', x, 0, x))
      hy.append( ROOT.TH1F('G4y' + str(theta), '', y, 0, y))
      hz.append( ROOT.TH1F('G4z' + str(theta), '', z, 0, z))
      hx[i].SetLineColor(color)
      hy[i].SetLineColor(color)
      hz[i].SetLineColor(color)
      canvas.cd(1)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hx[i], sumx)
      if i==0:
         hx[i].DrawNormalized('hist')
      else:
         hx[i].DrawNormalized('hist sames')
      canvas.cd(2)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hy[i], sumy)
      if i==0:
         hy[i].DrawNormalized('hist')
      else:
         hy[i].DrawNormalized('hist sames')
      canvas.cd(3)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hz[i], sumz)
      if i==0:
         hz[i].DrawNormalized('hist')
      else:
          hz[i].DrawNormalized('hist sames')
      canvas.cd(4)
      leg.AddEntry(hx[i], '{}theta {}events'.format(theta, num),"l")
      canvas.Update()
      color+= 1
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file + '.pdf')
   if ifC:
      canvas.Print(out_file + '.C')

def SortTheta(data, energies, flag=False, num_events=2000, tolerance=5):
    X = data[0]
    Y = data[1]
    theta = data[2]
    srt = {}
    for energy in energies:
       if energy == 0 and flag:
          srt["events_act" + str(energy)] = X[:10000] # More events in random bin
          srt["energy" + str(energy)] = Y[:10000]
          print srt["events_act" + str(energy)].shape
       else:
          indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
          print(len(indexes))
          srt["events_act" + str(energy)] = X[indexes][:num_events]
          srt["energy" + str(energy)] = Y[indexes][:num_events]
          srt["theta_act"+ str(energy)] = theta[indexes][:num_events]
    return srt

def PlotEvent(event, energy, theta, out_file, n):
   canvas = TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500) #make 
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}theta'.format(energy, theta), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}theta'.format(energy, theta), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}theta'.format(energy, theta), '', x, 0, x, y, 0, y)
   gStyle.SetPalette(1)
   gPad.SetLogz()
   event = np.expand_dims(event, axis=0)
   FillHist2D_wt(hx, np.sum(event, axis=1))
   FillHist2D_wt(hy, np.sum(event, axis=2))
   FillHist2D_wt(hz, np.sum(event, axis=3))
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

def PlotCut(events, xcut, ycut, zcut, energy, out_file):
   canvas = TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]

   hx = ROOT.TH2F('x_{}GeV_x={}cut'.format(str(energy), str(xcut)), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{}GeV_y={}cut'.format(str(energy), str(ycut)), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{}GeV_z={}cut'.format(str(energy), str(zcut)), '', x, 0, x, y, 0, y)
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
   gStyle.SetOptFit (1111) 
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]

   canvas.Divide(2,2)
   hx = ROOT.TH2F('x=25_{}cut'.format(str(ang)), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y=25_{}cut'.format(str(ang)), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z=12_{}cut'.format(str(ang)), '', x, 0, x, y, 0, y)
   FillHist2D_wt(hx, np.sum(events, axis=1))
   FillHist2D_wt(hy, np.sum(events, axis=2))
   FillHist2D_wt(hz, np.sum(events, axis=3))
   canvas.cd(1)
   hx.Draw("colz")
   #hx.Fit("pol2")
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
   array= np.squeeze(array, axis=3)
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
    theta = np.array(f.get('theta')) * 57.325
    x[x < 1e-4] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y, theta
   
def GetAllData(datafiles):
   for index, datafile in enumerate(datafiles):
       if index == 0:
          x, y, theta = GetAngleData(datafile)
       else:
          x_temp, y_temp, theta_temp = GetAngleData(datafile)
          x = np.concatenate((x, x_temp), axis=0)
          y = np.concatenate((y, y_temp), axis=0)
          theta = np.concatenate((theta, theta_temp), axis=0)
   return x, y, theta

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


   
