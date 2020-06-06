from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import sys
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
import setGPU #if Caltech
import heapq
sys.path.insert(0,'../keras')
import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import analysis.utils.RootPlotsGAN as pl

def main():
   #datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # path to data
   datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5"
   numdata=120000
   num_events1 = 10000 # Number of events for 0 bin
   num_events2 = 10000  # Number of events for other bins
   tolerance2=0.05
   num=10 # random events generated
   thetamin = np.radians(60)  # min theta
   thetamax = np.radians(120) # maximum theta
   energies=[100, 200, 300, 400, 500] # energy bins
   thetas = [62, 118] # angle bins
   ang = 1 # use all calculation for variable angle
   particle = 'Ele'
   ecalscale=50. # scaling in original data set
   post = inv_power # post processing: It can be either scale (without sqrt) or square(sqrt)
   thresh = 0. # if using threshold
   plotsdir = 'results/2D_images_g4_outliers_sl15_2/'# name of folder to save results
   gan.safe_mkdir(plotsdir) # make plot directory
   opt="colz" # option for 2D hist
   angtype='mtheta'
   numfiles=-9
   sl=15
   datafiles = gan.GetDataFiles(datapath, Particles=[particle]) # get list of files
   var = gan.get_sorted_angle(datafiles[numfiles:], energies, True, num_events1, num_events2, angtype=angtype, thresh=0.0)#, Data=GetDataAngle2) # returning a dict with sorted data.
   for energy in energies:
      edir = os.path.join(plotsdir, 'energy{}'.format(energy))
      gan.safe_mkdir(edir)
      rad = np.radians(thetas)
      for index, a in enumerate(rad):
         adir = os.path.join(edir, 'angle{}'.format(thetas[index]))
         gan.safe_mkdir(adir)
         if a==0:
            var["events_act" + str(energy) + "ang_" + str(index)] = np.squeeze(var["events_act" + str(energy)]/ecalscale)
            var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)]
            var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)]
            var["index" + str(energy)+ "ang_" + str(index)] = var["events_act" + str(energy) + "ang_" + str(index)].shape[0]
         else:
            indexes = np.where(((var["angle" + str(energy)]) > a - tolerance2) & ((var["angle" + str(energy)]) < a + tolerance2)) # all events with angle within a bin
            var["events_act" + str(energy) + "ang_" + str(index)] = var["events_act" + str(energy)][indexes]/ecalscale
            var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)][indexes]
            var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)][indexes]
            var["index" + str(energy)+ "ang_" + str(index)] = var["events_act" + str(energy) + "ang_" + str(index)].shape[0]
            
         var["events_act" + str(energy) + "ang_" + str(index)] = applythresh(var["events_act" + str(energy) + "ang_" + str(index)], thresh)
         print('energy = {} angle={} events={}'.format(energy, thetas[index], var["events_act" + str(energy) + "ang_" + str(index)].shape[0]))
         if thetas[index] < 90:
            indexes = getlargestindex(var["events_act" + str(energy) + "ang_" + str(index)], num, sl=20, start=0)
         else:
            indexes = getlargestindex(var["events_act" + str(energy) + "ang_" + str(index)], num, sl=20, start=-1)
         events_filt = var["events_act" + str(energy) + "ang_" + str(index)][indexes]
         energy_filt = var["energy" + str(energy) + "ang_" + str(index)][indexes]
         angle_filt = var["angle" + str(energy) + "ang_" + str(index)][indexes]
         for n in np.arange(num):
            PlotEvent(events_filt[n], 
                      energy_filt[n],
                      angle_filt[n],
                      os.path.join(adir, 'Event{}'.format(n)), n, opt=opt, logz=1)
            #plot_energy_hist(np.expand_dims(events_filt[n], axis=0), os.path.join(adir, 'shapes{}'.format(n)), log=1) 
   print('Plots are saved in {}'.format(plotsdir))
    
def power(n, xscale=1, xpower=1):
   return np.power(n/xscale, xpower)

def inv_power(n, xscale=1, xpower=1):
   return np.power(n, 1./xpower) / xscale

def applythresh(n, thresh):
   n[n<thresh]=0
   return n

def getlargestindex(events, n, sl=10, start=0):
   if start ==0:
     ysum = np.sum(events[:, :, :sl, :], axis=(1, 2, 3))
   else:
     ysum = np.sum(events[:, :, -1 * sl:, :], axis=(1, 2, 3))
   mean = np.mean(ysum)
   std = np.std(ysum)
   print('mean ={} std={} mean+std = {}'.format(mean, std, mean+std))
   indexes = heapq.nlargest(n, range(len(ysum)), ysum.take)
   print(ysum[indexes])
   return indexes

#get data for training
def GetDataAngle2(datafile, xscale =1, xpower=1, yscale = 1, angscale=1, angtype='theta', offset=0.0, thresh=1e-4, daxis=-1):
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))* xscale
    Y=np.array(f.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where((ecal > 10.0) & (Y > 150) & (Y < 350))
    print('From {} events {} passed'.format(Y.shape[0], indexes[0].shape[0]))
    X=X[indexes]
    Y=Y[indexes]
    ecal = ecal[indexes]
    if angtype in f:
      ang = np.array(f.get(angtype))[indexes]
    else:
      ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal

def PlotEvent(event, energy, theta, out_file, n, opt="", unit='degrees', label="", logz=0, log=1, ifpdf=True):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,900 ,400) #make
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   color=2
   lsize = 0.04 # axis label size
   tsize = 0.08 # axis title size
   tmargin = 0.02
   bmargin = 0.15
   lmargin = 0.15
   rmargin = 0.17
   
   title = ROOT.TPaveLabel(0.1,0.95,0.9,0.99,"Ep = {:.2f} GeV #theta ={:.2f} #circ".format(energy, theta))
   title.SetFillStyle(0)
   title.SetLineColor(0)
   title.SetBorderSize(1)
   title.Draw()
   graphPad = ROOT.TPad("Graphs","Graphs",0.01,0.01,0.95,0.95)
   graphPad.Draw()
   graphPad.cd()
   graphPad.Divide(3, 2)
   hx1 = ROOT.TH2F('x1_{:.2f}GeV_{:.2f}'.format(energy, theta), '', y, 0, y, z, 0, z)
   hy1 = ROOT.TH2F('y1_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, z, 0, z)
   hz1 = ROOT.TH2F('z1_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, y, 0, y)
   hx1.SetStats(0)
   hy1.SetStats(0)
   hz1.SetStats(0)
  
   ROOT.gStyle.SetPalette(1)
   event = np.expand_dims(event, axis=0)
   r.FillHist2D_wt(hx1, np.sum(event, axis=1))
   r.FillHist2D_wt(hy1, np.sum(event, axis=2))
   r.FillHist2D_wt(hz1, np.sum(event, axis=3))
 
   Min = 1e-4
   Max = 1e-1
   graphPad.cd(1)
   if logz: ROOT.gPad.SetLogz(1)
   hx1.Draw('col')
   hx1.GetXaxis().SetTitle("Y")
   hx1.GetYaxis().SetTitle("Z")
   hx1.GetYaxis().CenterTitle()
   hx1.GetXaxis().SetLabelSize(lsize)
   hx1.GetYaxis().SetLabelSize(lsize)
   hx1.GetXaxis().SetTitleSize(tsize)
   hx1.GetYaxis().SetTitleSize(tsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(0)
   hx1.SetMinimum(Min)
   hx1.SetMaximum(Max)
   canvas.Update()
   canvas.Update()
   graphPad.cd(2)
   if logz: ROOT.gPad.SetLogz(1)
   hy1.Draw('col')
   hy1.GetXaxis().SetTitle("X")
   hy1.GetYaxis().SetTitle("Z")
   hy1.GetYaxis().CenterTitle()
   hy1.GetXaxis().SetLabelSize(lsize)
   hy1.GetYaxis().SetLabelSize(lsize)
   hy1.GetXaxis().SetTitleSize(tsize)
   hy1.GetYaxis().SetTitleSize(tsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(0)
   hy1.SetMinimum(Min)
   hy1.SetMaximum(Max)

   canvas.Update()
   canvas.Update()
   graphPad.cd(3)
   if logz: ROOT.gPad.SetLogz(1)
   hz1.Draw(opt)
   hz1.GetXaxis().SetTitle("X")
   hz1.GetYaxis().SetTitle("Y")
   hz1.GetYaxis().CenterTitle()
   hz1.GetXaxis().SetLabelSize(lsize)
   hz1.GetYaxis().SetLabelSize(lsize)
   hz1.GetXaxis().SetTitleSize(tsize)
   hz1.GetYaxis().SetTitleSize(tsize)
   hz1.GetZaxis().SetLabelSize(lsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(rmargin)
   hz1.SetMinimum(Min)
   hz1.SetMaximum(Max)

   canvas.Update()
   
   
   array1x= np.sum(event, axis=(2, 3))
   array1y= np.sum(event, axis=(1, 3))
   array1z= np.sum(event, axis=(1, 2))

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
   graphPad.cd(5)
   if log:
      ROOT.gPad.SetLogy()

   r.fill_hist_wt(h1x, array1x)
   h1x=r.normalize(h1x)
   h1x.Draw('hist')
   h1x.GetXaxis().SetTitle("X")
   h1x.GetYaxis().SetTitle("energy deposition")
   h1x.GetXaxis().SetLabelSize(lsize)
   h1x.GetYaxis().SetLabelSize(lsize)
   h1x.GetXaxis().SetTitleSize(tsize)
   h1x.GetYaxis().SetTitleSize(tsize)
   h1x.GetZaxis().SetLabelSize(lsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(0)

   graphPad.cd(4)
   if log:
      ROOT.gPad.SetLogy()
   r.fill_hist_wt(h1y, array1y)
   h1y=r.normalize(h1y)
   h1y.Draw('hist')
   
   h1y.GetXaxis().SetTitle("Y")
   h1y.GetYaxis().SetTitle("energy deposition")
   h1y.GetXaxis().SetLabelSize(lsize)
   h1y.GetYaxis().SetLabelSize(lsize)
   h1y.GetXaxis().SetTitleSize(tsize)
   h1y.GetYaxis().SetTitleSize(tsize)
   h1y.GetZaxis().SetLabelSize(lsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(0)

   graphPad.cd(6)
   if log:
      ROOT.gPad.SetLogy()
   r.fill_hist_wt(h1z, array1z)
   h1z=r.normalize(h1z)
   h1z.Draw('hist')
   
   h1z.GetXaxis().SetTitle("Z")
   h1z.GetYaxis().SetTitle("energy deposition")
   h1z.GetXaxis().SetLabelSize(lsize)
   h1z.GetYaxis().SetLabelSize(lsize)
   h1z.GetXaxis().SetTitleSize(tsize)
   h1z.GetYaxis().SetTitleSize(tsize)
   h1z.GetZaxis().SetLabelSize(lsize)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(rmargin)

   canvas.Update()
   if ifpdf:
      canvas.Print(out_file + '.pdf')
   else:
      canvas.Print(out_file + '.C')


if __name__ == "__main__":
    main()


   
