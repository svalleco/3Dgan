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
plt.switch_backend('Agg')

def main():
   #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path                                          
   datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5"
   numdata = 10000
   datafiles = GetDataFiles(datapath, numdata, Particles=['Ele'])
   plotsdir = 'meas4_plots'
   centerfile = 'center_hist.pdf'
   angfile = 'ang.pdf'
   gan.safe_mkdir(plotsdir)
   print len(datafiles)
   numevents = 100
   X, Y, theta= GetAngleDataReduced(datafiles[0], numdata=numevents) # get data
   zsum = np.sum(X, axis =(1, 2)) # sums per events per z, shape = (numevents, z)
   #sintheta = np.sin(theta) 
   cx, cy, ct = Meas3(X) # Get co ordinates of berycenters (x, y) and corressponding angle measured for each bin along z
   n = 10 # single event index
   fig= 0
      
   # Single event plots
   PlotHistcenter(cx[n], os.path.join(plotsdir,'x' + centerfile), fig=fig) # plot x coordinates  of berycenter
   fig +=1
   PlotHistcenter(cy[n], os.path.join(plotsdir,'y' + centerfile), fig=fig) # plot y coordinates  of berycenter
   fig +=1
   PlotHisttan(ct[n], zsum[n], os.path.join(plotsdir, 'hist_' + angfile), theta[n], fig=fig, angleproc= ProcAngle4) # histogram of angle computed
   fig+=1  
   PlotTan(ct[n], zsum[n], os.path.join(plotsdir, 'scat_' + angfile), theta[n], fig=fig, angleproc= ProcAngle4) # scatter plot of angle computed
   fig+=1
   
   # All events plots
   PlotHistError(ct, theta, zsum, 'New', os.path.join(plotsdir, 'errorhist_' + angfile), fig=fig, angleproc= ProcAngles4) # Error histogram 
   fig+=1
   PlotAngleMeasure(ct, theta, zsum, os.path.join(plotsdir, 'errorscat_' + angfile), fig=fig, angleproc= ProcAngles4) # Error scatter plot
   fig+=1
   print '{} plots were saved in {}'.format(fig, plotsdir)

# Histogram of berycenters for an event
def PlotHistcenter(x, outfile, fig=0): 
   bins = np.arange(0, 25)
   plt.figure(fig)
   plt.hist(x, bins=bins , label='x of berycenter')
   plt.legend()
   plt.xlabel('xcenter ')
   plt.savefig(outfile)

# Histogram of calculated angle for an event
def PlotHisttan(t, zsum, outfile, theta, fig, angleproc):
   plt.figure(fig)
   m = angleproc(t, zsum)
   label = 'tan actual ({}) Calculated ({})'.format(theta, m)
   plt.hist(t, bins='auto', label=label) 
   plt.legend()
   plt.xlabel('angle (radians)'.format(theta, m))
   plt.savefig(outfile)

#Scatter plot of measured angle going along z for an event
def PlotTan(t, zsum, outfile, theta, fig, angleproc):
   plt.figure(fig)
   bins = np.arange(t.shape[0])
   m = angleproc(t, zsum)
   label = 'tan actual ({}) Calculated ({})'.format(theta, m)
   plt.scatter(bins, t, label=label) 
   plt.legend()
   plt.xlabel('angle (radians)')
   plt.savefig(outfile)

# Angle using average for z for all events
def ProcAngles1(meas, zsum):
   a = np.arange(meas.shape[1])
   avg = np.zeros(meas.shape[0])
   for i in np.arange(1, meas.shape[0]):
      avg[i] = np.sum( meas[i] )/ 24
   return avg

# Angle using average for z for single event
def ProcAngle1(meas, zsum):
   a = np.arange(meas.shape[0])
   avg = np.sum( meas )/ 24
   return avg

# Angle Measure using weighting by z for all events
def ProcAngles2(meas, zsum):
   a = np.arange(meas.shape[1])
   avg = np.zeros(meas.shape[0])
   for i in np.arange(1, meas.shape[0]):
      avg[i] = np.sum( meas[i] * a)/ (0.5  * (meas.shape[1] * (meas.shape[1] - 1)))
   return avg
      
# Angle Measure using weighting by z for all events
def ProcAngle2(meas, zsum):
   a = np.arange(meas.shape[0])
   avg = np.sum( meas * a)/ (0.5  * (meas.shape[0] * (meas.shape[0] - 1)))
   return avg

# Angle Measure using weighting by z and energy deposited per layer for all events
def ProcAngles3(meas, zsum):
   a = np.arange(meas.shape[1])
   avg = np.zeros(meas.shape[0])
   for i in np.arange(1, meas.shape[0]):
      ztot = np.sum(zsum[i])
      num = meas[i] * a
      num = num * zsum[i]
      avg[i] = np.sum( num)/(ztot * 0.5  * (meas.shape[1] * (meas.shape[1] - 1)))
   return avg

# Angle Measure using weighting by z and energy deposited per layer for single event 
def ProcAngle3(meas, zsum):
   a = np.arange(meas.shape[0])
   ztot = np.sum(zsum)
   avg = np.sum( meas * a * zsum)/ (ztot * 0.5  * (meas.shape[0] * (meas.shape[0] - 1)))
   return avg

# Angle Measure using weighting by z and energy deposited per layer for all events                                                                                                                                
def ProcAngles4(meas, zsum):
   a = meas.shape[1]
   avg = np.zeros(meas.shape[0])
   for i in np.arange(1, meas.shape[0]):
      avg[i] = meas[i, a-1]
   return avg

# Angle Measure using weighting by z and energy deposited per layer for single event 
def ProcAngle4(meas, zsum):
   a = meas.shape[0]
   avg = meas[a-1]
   return avg

# Error Histogram
def PlotHistError(meas,  angle, zsum, label, outfile, fig, angleproc, degree=False):
   bins = np.arange(-2,2, 0.01)
   plt.figure(fig)
   m = angleproc(meas, zsum)
   error = angle-m
   if degree:
      angle = np.degrees(angle)
      m = np.degrees(m)
      error = np.degrees(error)
      unit = 'degrees'
   else:
      unit = 'radians'
   plt.hist(error, bins=bins, label='{} error {:.4f}({:.4f})'.format(label, np.mean(error), np.std(error)))
   plt.legend()
   plt.xlabel('Error ({})'.format(unit))
   plt.savefig(outfile)
   
def PlotAngleMeasure(m, angle, zsum, outfile, fig, angleproc, degree=False):
   measured = angleproc(m, zsum)
   error = abs(angle - measured)
   if degree:
      angle = np.degrees(angle)
      measured = np.degrees(measured)
      error = np.degrees(error)
      unit = 'degrees'
   else:
      unit = 'radians'
   plt.figure(fig)
   plt.scatter(angle, error, label='Error Mean={:.4f}, std={:.4f}'.format(np.mean(error), np.std(error)))
   plt.legend()
   plt.xlabel('Angle ({})'.format(unit))
   plt.ylabel('Absolute Error ({})'.format(unit))
   plt.savefig(outfile)

def GetAngleDataReduced(datafile, numdata=1000):
    #get data for training                                                                                         
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy')[:numdata])
    x=np.array(f.get('ECAL'))[:numdata, 13:38, 13:38, :]
    theta = np.array(f.get('theta')[:numdata]) 
    x[x < 1e-4] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y, theta

def Meas3(events, mod=0):
    a = []
    events = np.squeeze(events, axis=4)
    x = np.zeros((events.shape[0], events.shape[3])) # shape = (num events, z)
    y = np.zeros((events.shape[0], events.shape[3]))
    ang = np.zeros((events.shape[0], events.shape[3]))
    for i in np.arange(events.shape[0]): # Looping over events
       event = events[i]
       for j in np.arange(events.shape[3]): # Looping over z
          sum = np.sum(event[:, :, j])
          x[i, j] = 0
          y[i, j] = 0
          for k in np.arange(events.shape[1]):  # Looping over x
             for l in np.arange(events.shape[2]): # Looping over y
               x[i, j] = x[i, j] + event[k, l, j] * k
               y[i, j] = y[i, j] + event[k, l, j] * l
          if sum > 0:                         # check for zero sum
            x[i, j] = x[i, j]/sum
            y[i, j] = y[i, j]/sum
          #print i, j, x[i, j], y[i, j]
          if j >0:
            #ang[i, j] = np.arctan(math.sqrt((x[i, j] - x[i, 0])**2 + (y[i, j] - y[i, 0])**2 + j**2))
            #ang2[i, j] = np.arctan(math.sqrt(((y[i, j] - y[i, 0])**2 + j**2) / ((x[i, j] - x[i, 0])**2+ j**2)))
            #ang[i, j] = np.arctan( math.sqrt((x[i, j] - x[i, 0])**2 + (y[i, j] - y[i, 0])**2 )/j)
            #ang[i, j] = math.pi/2 - ang[i, j]
            ang[i, j] = np.arctan((y[i, j] - y[i, 0])/j)   
            ang[i, j] = math.pi/2 - ang[i, j]
            #hp = math.sqrt((x[i, j] - x[i, 0])**2 + (y[i, j] - y[i, 0])**2 )
            #if hp > 0 and hp < j:
            #    ang[i, j] =  np.arcsin(j/hp) 
    return x, y, ang

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
