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
import GANutilsAngle as gan
import ROOTutils as r
import setGPU #if Caltech

def main():
   genweight = "angleweights/params_generator_epoch_049.hdf5"
   from EcalCondGan4 import generator
   g=generator()
   g.load_weights(genweight)
   num_events = 1000
   num=10
   f = 57.325
   thetamin = 60/f
   thetamax = 120/f

   energies=[100, 250, 400]
   thetas = [60, 90, 120]
   xcuts = [10, 25, 40] 
   ycuts = [10, 25, 40]
   zcuts = [5, 12, 20]
   plotsdir = 'angle_plots_gan'
   gan.safe_mkdir(plotsdir)
   opt="colz"  
   for energy in energies:
      for xcut, ycut, zcut in zip(xcuts, ycuts, zcuts):
         sampled_energies=energy/100 * np.ones((num_events))
         sampled_thetas = np.random.uniform(thetamin, thetamax, size=(num_events))
         events = gan.generate(g, num_events, sampled_energies, sampled_thetas)  
         PlotCut(events, xcut, ycut, zcut, energy, os.path.join(plotsdir, '2Dhits_cut{}_{}.pdf'.format(xcut, energy)), opt=opt)
      tevents =[]
      for t in thetas:
         sampled_energies=energy/100 * np.ones((num_events))
         sampled_thetas = (t /f )* np.ones((num_events))
         events = gan.generate(g, num_events, sampled_energies, sampled_thetas)
         tevents.append(events)
         PlotAngleCut(events, t, os.path.join(plotsdir, 'Theta{}_GeV{}.pdf'.format(t, energy)), opt=opt)
      plot_energy_hist_gen(tevents, os.path.join(plotsdir, 'GeV{}_hist.pdf'.format(energy)), energy, thetas, log=0, ifC=False)
   for t in thetas:
         sampled_energies=np.random.uniform(0.2, 5, size=(num_events))
         sampled_thetas = (t /f )* np.ones((num_events))
         events = gan.generate(g, num_events, sampled_energies, sampled_thetas)
         PlotAngleCut(events, t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)), opt=opt)

   sampled_energies=np.random.uniform(0.2, 5, size=(num))
   sampled_thetas = np.random.uniform(thetamin, thetamax, size=(num))
   events = gan.generate(g, num, sampled_energies, sampled_thetas)

   for n in np.arange(num):
     PlotEvent(events[n], 25, 25, 12, sampled_energies[n], sampled_thetas[n], os.path.join(plotsdir, 'Event{}.pdf'.format(n)), n, f, opt=opt)
   
def plot_energy_hist_gen(events, out_file, energy, thetas, log=0, ifC=False):
   canvas = TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make  
   canvas.SetGrid()
   canvas.Divide(2,2)
   hx=[]
   hy=[]
   hz=[]
   color = 2
   leg = ROOT.TLegend(0.1,0.6,0.7,0.9)
   print (len(events))
   for i, theta in enumerate(thetas):
      event=events[i]
      num=event.shape[0]
      sumx, sumy, sumz=gan.get_sums(event)
      x=sumx.shape[1]
      y=sumy.shape[1]
      z=sumz.shape[1]
      hx.append( ROOT.TH1F('GANx' + str(theta), '', x, 0, x))
      hy.append( ROOT.TH1F('GANy' + str(theta), '', y, 0, y))
      hz.append( ROOT.TH1F('GANz' + str(theta), '', z, 0, z))
      hx[i].SetLineColor(color)
      hy[i].SetLineColor(color)
      hz[i].SetLineColor(color)
      canvas.cd(1)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hx[i], sumx)
      if i==0:
         hx[i].Draw('hist')
      else:
         hx[i].Draw('hist sames')
      canvas.cd(2)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hy[i], sumy)
      if i==0:
         hy[i].Draw('hist')
      else:
         hy[i].Draw('hist sames')
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

def PlotEvent(event, x, y, z, energy, theta, out_file, n, f, opt=""):
   canvas = TCanvas("canvas" ,"GAN 2D Hist" ,200 ,10 ,700 ,500) #make 
   canvas.Divide(2,2)
   sum = np.sum(event)     
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}sum_{:.2f}theta'.format(100 * energy, sum, f * theta), '', 51, 0, 51, 25, 0, 25)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}theta'.format(100 * energy, f * theta), '', 51, 0, 51, 25, 0, 25)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}theta'.format(100 * energy, f * theta), '', 51, 0, 51, 51, 0, 51)
   gStyle.SetPalette(1)
   gPad.SetLogz()
   event = np.expand_dims(event, axis=0)
   print(event.shape)
   print(event[0, 25, 25, 15])
   FillHist2D_wt(hx, event[:, x, :, :])
   FillHist2D_wt(hy, event[:, :, y, :])
   FillHist2D_wt(hz, event[:, :, :, z])
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   canvas.Update()
   r.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotAngleCut(events, ang, out_file, opt=""):
   canvas = TCanvas("canvas" ,"GAN 2D Hist" ,200 ,10 ,700 ,500) 
   canvas.Divide(2,2)
   hx = ROOT.TH2F('x=25_{}cut'.format(str(ang)), '', 51, 0, 51, 25, 0, 25)
   hy = ROOT.TH2F('y=25_{}cut'.format(str(ang)), '', 51, 0, 51, 25, 0, 25)
   hz = ROOT.TH2F('z=12_{}cut'.format(str(ang)), '', 51, 0, 51, 51, 0, 51)
   FillHist2D_wt(hx, events[:, 25, :, :])
   FillHist2D_wt(hy, events[:, :, 25, :])
   FillHist2D_wt(hz, events[:, :, :, 12])
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
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


   
