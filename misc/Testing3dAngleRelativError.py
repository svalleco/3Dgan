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
import matplotlib
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
plt.switch_backend('Agg')
import setGPU

def main():
   #datapath = '/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5' #Training data path       
   #datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5"
   datapath = '/bigdata/shared/gkhattak/*scan/*.h5'
   numdata = 10000
   numevents=10
   ypoint1 = 3 
   ypoint2 = 21
   datafiles = GetDataFiles(datapath, 10000, Particles=['Ele'])
   plotsdir = 'weighted_3d'
   weights = ['none', 'energy', 'position', 'energy and position']
   filename = plotsdir + '/'
   gan.safe_mkdir(plotsdir)
   X, Y, theta, alpha= GetAngleData(datafiles[0], num_data=numdata)
   angles =[]
   fig=1
   for i, w in enumerate(weights):
     angle = Meas3d_weighted(X, i)  
     angles.append(angle)
     PlotAngleMeasure(theta, angle, filename + 'relative_error_weight_{}'.format(w.replace(" ", "_")), w)
     fig+=1
   PlotAngleMeasureAll(theta, angles, filename + 'relative_error_all', weights)
   fig+=1
   PlotAngleMeasureAll(theta, angles, filename + 'error_all', weights, relative=0)
   fig+=1
      
   print('{} Plots are saved in {}'.format(fig, plotsdir))

def PlotAngleMeasure(actual, calculated, outfile, label, ifpdf=1):
   error = (actual - calculated)/actual
   unit = 'radians'
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
   c1.SetGrid()
   legend = ROOT.TLegend(.1, .7, .4, .9)
   #legend.SetTextSize(0.024)
   Aprof= ROOT.TProfile("Aprof","Aprof", 100, 1., 2.2)
   Aprof.SetStats(0)
   r.fill_profile(Aprof, error, actual)
   Aprof.SetTitle("Relative Error for theta using {} weight".format(label))
   Aprof.GetXaxis().SetTitle("Global Theta")
   Aprof.GetYaxis().SetTitle("(Global theta - Measured)/Global theta")
   #Aprof.GetYaxis().SetRangeUser(-0.15, 0.15)
   Aprof.GetYaxis().CenterTitle()
   Aprof.Draw()
   c1.Update()
   legend.AddEntry(Aprof,"weighted by {}".format(label),"l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(outfile + '.pdf')
   else:
      c1.Print(outfile + '.C')

def PlotAngleMeasureAll(actual, calculated, outfile, label, ifpdf=1, relative=1):
   unit = 'radians'
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
   c1.SetGrid()
   legend = ROOT.TLegend(.1, .7, .4, .9)
   #legend.SetTextFont(132)
   #legend.SetTextSize(0.024)
   Aprofs = []
   color = 2
   for i, angle in enumerate(calculated):
     Aprofs.append(ROOT.TProfile("Aprof" + str(i),"Aprof"+ str(i), 100, 1.0, 2.2))
     Aprofs[i].SetStats(0)
     error = (actual-calculated[i])
     if relative:
        error = error/actual
     r.fill_profile(Aprofs[i], error, actual)
     if i==0:
        if relative:
           Aprofs[i].SetTitle("Relative Error for theta using different weights")
           Aprofs[i].GetYaxis().SetTitle("(Global theta - Measured)/Global theta")
        else:
           Aprofs[i].SetTitle("Error for theta using different weights")
           Aprofs[i].GetYaxis().SetTitle("(Global theta - Measured) radians")
        Aprofs[i].GetXaxis().SetTitle("Global Theta ({}".format(unit))
        Aprofs[i].GetYaxis().CenterTitle()
        Aprofs[i].Draw()
        Aprofs[i].SetLineColor(color)
        #Aprofs[i].GetYaxis().SetRangeUser(-0.15, 0.15)
        c1.Update()
        legend.AddEntry(Aprofs[i],"Weighted by {}".format(label[i]),"l")
     else:
        Aprofs[i].Draw('sames')
        Aprofs[i].SetLineColor(color)
        c1.Update()
        legend.AddEntry(Aprofs[i],"Weighted by {}".format(label[i]),"l")
     color+=2
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(outfile + '.pdf')
   else:
      c1.Print(outfile + '.C')
                                                                  
   
def Meas3d_weighted(image, mod=0):
   print(image.shape)
   image = np.squeeze(image)
   x_shape= image.shape[1]
   y_shape= image.shape[2]
   z_shape= image.shape[3]
   sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
   indexes = np.where(sumtot > 0)
   amask = np.ones_like(sumtot)
   amask[indexes] = 0
   masked_events = np.sum(amask) # counting zero sum events
   x_ref = np.sum(np.sum(image, axis=(2, 3)) * np.expand_dims(np.arange(x_shape) + 0.5, axis=0), axis=1)
   y_ref = np.sum(np.sum(image, axis=(1, 3)) * np.expand_dims(np.arange(y_shape) + 0.5, axis=0), axis=1)
   z_ref = np.sum(np.sum(image, axis=(1, 2)) * np.expand_dims(np.arange(z_shape) + 0.5, axis=0), axis=1)

   x_ref[indexes] = x_ref[indexes]/sumtot[indexes]
   y_ref[indexes] = y_ref[indexes]/sumtot[indexes]
   z_ref[indexes] = z_ref[indexes]/sumtot[indexes]
   sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z

   x = np.expand_dims(np.arange(x_shape) + 0.5, axis=0)
   x = np.expand_dims(x, axis=2)
   y = np.expand_dims(np.arange(y_shape) + 0.5, axis=0)
   y = np.expand_dims(y, axis=2)
   x_mid = np.sum(np.sum(image, axis=2) * x, axis=1)
   y_mid = np.sum(np.sum(image, axis=1) * y, axis=1)
   indexes = np.where(sumz > 0)

   zmask = np.zeros_like(sumz)
   zmask[indexes] = 1
   zunmasked_events = np.sum(zmask, axis=1)

   x_mid[indexes] = x_mid[indexes]/sumz[indexes]
   y_mid[indexes] = y_mid[indexes]/sumz[indexes]
   z = np.arange(z_shape) + 0.5# z indexes
   x_ref = np.expand_dims(x_ref, 1)
   y_ref = np.expand_dims(y_ref, 1)
   z_ref = np.expand_dims(z_ref, 1)

   zproj = np.sqrt((x_mid-x_ref)**2.0  + (z - z_ref)**2.0)
   m = (y_mid-y_ref)/zproj
   z = z * np.ones_like(z_ref)
   indexes = np.where(z<z_ref)
   m[indexes] = -1 * m[indexes]
   ang = (math.pi/2.0) - np.arctan(m)
   ang = ang * zmask
   if mod==0:
      ang = np.sum(ang, axis=1)/zunmasked_events
   elif mod==1:
      wang = ang * sumz
      sumz_tot = sumz 
      ang = np.sum(wang, axis=1)/np.sum(sumz_tot, axis=1)
      print mod, ang[0]
   elif mod==2:
      wang = ang * z
      ang = np.sum(wang, axis=1)/np.sum(z, axis=1)
   elif mod==3:
      print(z[0])
      print(sumz[0])
      w = z * sumz
      print(w[0])
      wang = ang * w
      ang = np.sum(wang, axis=1)/np.sum(w, axis=1)
      print mod, ang[0]
   indexes = np.where(amask>0)
   ang[indexes] = 100.
   return ang

def GetAngleData_reduced(datafile, num_data=10000):
    #get data for training                                                                                  
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy'))[:num_data]
    x=np.array(f.get('ECAL'))[:num_data, 13:38, 13:38, :]
    theta = np.array(f.get('theta'))[:num_data] 
    x[x < 1e-4] = 0
    alpha = (math.pi/2)*np.ones_like(theta) - theta
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y, theta, alpha

def GetAngleData(datafile, num_data=10000):
    #get data for training          
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy'))[:num_data]
    x=np.array(f.get('ECAL'))[:num_data]
    theta = np.array(f.get('theta'))[:num_data]
    x[x < 1e-4] = 0
    alpha = (math.pi/2)*np.ones_like(theta) - theta
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y, theta, alpha

def GetAllData(datafiles):
   for index, datafile in enumerate(datafiles):
       if index == 0:
          x, y, theta, eta = GetAngleData(datafile)
       else:
          x_temp, y_temp, theta_temp, eta_temp = GetAngleData(datafile)
          x = np.concatenate((x, x_temp), axis=0)
          y = np.concatenate((y, y_temp), axis=0)
          theta = np.concatenate((theta, theta_temp), axis=0)
          eta = np.concatenate((eta, eta_temp), axis=0)
   return x, y, theta, eta

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
