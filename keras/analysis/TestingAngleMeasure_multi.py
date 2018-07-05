# This is used for test and verification of angle measurement

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
   #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path Caltech                                          
   datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5" #Training data path eos
   datafiles = GetDataFiles(datapath, Particles=['Ele'])
   plotsdir = '3dAngleTest' # directory for plots
   gan.safe_mkdir(plotsdir)
   print len(datafiles)
   numevents = 1000 # Number of events to be used works with numevents < 10000 if more are required then code has to be modified
   fig=1
   # p1 and p2 meaningful only for some implementations
   p1=0 
   p2=24
   
   X, Y, theta= GetAngleData(datafiles[0], numdata=numevents) # get data
   x_shape = X.shape[1]
   y_shape = X.shape[2]
   z_shape = X.shape[3]
   zsum = np.sum(X, axis =(1, 2)) # sums per events per z, shape = (numevents, z)
   # creating weights
   w1 = np.expand_dims(np.ones(z_shape), axis=0)
   w2 = np.expand_dims(np.arange(z_shape), axis=0)
   weight = [w1, w2, w2 * zsum, zsum]
  
   Meas = [Meas1, Meas2, Meas5_4] # Measurement function to be used
   Proc = [ProcAngles1, ProcAngles2, ProcAngles3, ProcAngles5] # processing of angle/z to get a final value
   labels = ['Unweighted Mean', 'Weighted by z', 'Weighted by z and energy', 'Weighted by energy'] # labels corresponding to weighting scheme
   names = ['0', 'Z', 'ZE', 'E'] # indexes corresponding to weighting scheme
   for i, M in enumerate(Meas): # loop through all functions in list
     if i<=1: # First two functions return single values
       ct = M(X) # Measured Angle
       Plot2DHist2(theta, ct, '2DTheta Computed by Meas{}'.format(i), os.path.join(plotsdir, 'Meas{}_gtheta_histogram.pdf'.format(i)))
       fig +=1
     else:
       cx, cy, ct = M(X, p1, p2) # Get co ordinates of berycenters (x, y) and corressponding angle measured for each bin along z
       theta_calc1 = Proc[0](ct, zsum) # Process angle per z to single mean value
       for j, P in enumerate(Proc): # Loop through different processing functions
         theta_calc = P(ct, zsum) # calculate final angle
         Plot2DHist(theta_calc1, theta_calc, labels[j], os.path.join(plotsdir, 'Meas{}_weighted{}.pdf'.format(i, names[j]))) # plot weight vs. unweighted
         fig +=1
         Plot2DHist2(theta, theta_calc, 'Calculated 3D angle {}'.format(labels[j]), os.path.join(plotsdir, 'Meas{}_weighted{}_GlobalTheta.pdf'.format(i, j))) # plot weighted against global theta
         fig +=1
         Plot2DHisttheta(theta, ct, weight[j], 'Global Theta - Calculated ({}) '.format(labels[j]), os.path.join(plotsdir, 'Meas{}_Diff_z_weighted{}.pdf'.format(i, j))) # plot weight diff against z
         fig +=1
   print '{} plots were saved in {}'.format(fig, plotsdir)

def Plot2DHist(meas1, meas2, label, out_file):
   canvas = TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500)
   hist = ROOT.TH2F('hist', '{} vs. unweighted'.format(label), 50, 1, 2.5, 50, 1, 2.5)
   n = meas1.shape[0]
   legend = ROOT.TLegend(.1, .8, .3, .9)
   for i in np.arange(n):
     hist.Fill(meas1[i], meas2[i])
   hist.GetXaxis().SetTitle("Average")
   hist.GetYaxis().SetTitle(label)
   hist.Draw('colz')
   canvas.Update()
   legend.AddEntry(hist, label,"l")
   canvas.Update()
   legend.Draw()
   canvas.Update()
   canvas.Print(out_file)

def Plot2DHist2(meas1, meas2, label, out_file):
   canvas = TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500)
   hist = ROOT.TH2F('hist', '{} vs. Global Theta'.format(label), 50, 1, 2.5, 50, 1, 2.5)
   n = meas1.shape[0]
   legend = ROOT.TLegend(.1, .8, .3, .9)
   for i in np.arange(n):
     hist.Fill(meas1[i], meas2[i])
   hist.GetXaxis().SetTitle("Global Theta")
   hist.GetYaxis().SetTitle(label)
   hist.Draw('colz')
   canvas.Update()
   legend.AddEntry(hist, label,"l")
   canvas.Update()
   legend.Draw()
   canvas.Update()
   canvas.Print(out_file)

# Unweighted Angle 
def ProcAngles1(meas, zsum):
   avg = np.mean(meas[:, 1:], axis=1)
   return avg

# Angle weighted by z 
def ProcAngles2(meas, zsum):
   index = np.arange(meas.shape[1])
   totw = 0.5  * (meas.shape[1] * (meas.shape[1] - 1)) 
   avg = np.sum(index * meas, axis=1)/totw
   return avg

# Angle Measure weighted by z and energy deposited per layer for all events                                                                 
def ProcAngles3(meas, zsum):
   index = np.arange(meas.shape[1])
   totz = np.sum(zsum * index, axis=1)
   avg = np.sum(meas * zsum * index, axis=1)/ totz
   return avg

# Angle Measure using angle calculated by last layer for all events                                                                                                                                
def ProcAngles4(meas, zsum):
   a = meas.shape[1]
   avg = meas[:, a-1]
   return avg

# Angle Measure using weighting by energy deposited per layer for all events                                                                                                                                                                                                       
def ProcAngles5(meas, zsum):
   ztot = np.sum(zsum, axis=1)
   avg = np.sum(meas * zsum, axis=1)/ztot
   return avg

def Plot2DHisttheta(meas1, meas2, w, label, out_file):
   canvas = TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500)
   diff = np.expand_dims(meas1, 1) - meas2
   diff = w * diff
   max2 = np.amax(diff)
   min2 = np.amin(diff)
   hist = ROOT.TH2F('hist', label, 25, 0, 25, 50, min2, 1.1 * max2)
   n = meas2.shape[1]
   z = np.arange(n)
   legend = ROOT.TLegend(.1, .8, .4, .9)
   for i in np.arange(meas1.shape[0]):
     for j in z:
        hist.Fill(z[j], diff[i, j])
   hist.GetXaxis().SetTitle("z position")
   hist.GetYaxis().SetTitle(label)
   hist.Draw('colz')
   canvas.Update()
   legend.AddEntry(hist, label,"l")
   canvas.Update()
   legend.Draw()
   canvas.Update()
   canvas.Print(out_file)

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

def GetAngleData(datafile, numdata=1000):
    #get data for training                                                                                                  
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy')[:numdata])
    x=np.array(f.get('ECAL')[:numdata])
    theta = np.array(f.get('theta')[:numdata])
    x[x < 1e-4] = 0
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    theta = theta.astype(np.float32)
    return x, y, theta

def Meas1(events, yp1 = 3, yp2 = 21): # angle using Python fit 
    a = []
    start = time.time()
    for i in np.arange(events.shape[0]):
       event = np.sum(np.squeeze(events[i]), axis=(0))
       y = np.arange(25)
       maxy = np.argmax(event, axis=0)
       p = np.polyfit(y[yp1:yp2], maxy[yp1:yp2], 1)
       angle = math.pi/2 - math.atan(p[0])
       a.append(angle)
    print ('The time taken by Meas1 = {} seconds'.format(time.time()-start))    
    return np.array(a)

def Meas2(events, yp1 = 3, yp2 = 21): # angle using two point approximation
    a = []
    start = time.time()
    for i in np.arange(events.shape[0]):
       event = np.sum(np.squeeze(events[i]), axis=(0))
       y = np.arange(25)
       maxy = np.argmax(event, axis=0)
       tan = (maxy[yp2]-maxy[yp1]) / np.float(yp2 - yp1)
       angle = math.pi/2 - math.atan(tan)
       a.append(angle)
    print ('The time taken by Meas2 = {} seconds'.format(time.time()-start))
    return np.array(a)

def Meas3(events, p1=0, p2=24): # 2D angle
    a = []
    start = time.time()
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

          if j >0:
            ang[i, j] = np.arctan((y[i, j] - y[i, 0])/j)
            ang[i, j] = math.pi/2 - ang[i, j]
    print ('The time taken by Meas3 = {} seconds'.format(time.time()-start))
    return x, y, ang

def Meas4(events, p1=0, p2=24): # 3d angle using sine inverse
    a = []
    start = time.time()
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
          if j >0:
            ang[i, j] = np.arcsin((y[i, j] - y[i, 0])/math.sqrt((x[i, j] - x[i, 0])**2 + (y[i, j] - y[i, 0])**2 + j**2))
            ang[i, j] = (math.pi/2) - ang[i, j] 
    print ('The time taken by Meas4 = {} seconds'.format(time.time()-start))
    return x, y, ang

def Meas5(events, p1=0, p2=24): # 3d angle using tan inverse
    a = []
    start = time.time()
    #events = np.squeeze(events, axis=4)
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
          if j > p1:
            zproj = np.sqrt((y[i, j] - y[i, p1])**2 + (j-p1)**2)
            ang[i, j] =  (np.arctan((y[i, j] - y[i, p1])/zproj))
    print ('The time taken by Meas5 = {} seconds'.format(time.time()-start))
    return x, y, ang

def Meas5_3(image, p1 =0, p2=24): # Using numpy and index matrixes
     start = time.time()
     x_shape= image.shape[1]
     y_shape= image.shape[2]
     z_shape= image.shape[3]
     ang = np.zeros((image.shape[0], z_shape))
     y_index = np.arange(y_shape)   # vector of indexes
     y_index = np.tile(y_index, (x_shape))         # repeat x shape times
     y_index = np.reshape(y_index, (y_shape, x_shape))  # reshape to make 2d
     x_index = np.transpose(y_index) # 2d x indexes
     y_index = np.repeat(y_index[:, :, np.newaxis], z_shape, axis=2)# project along z
     y_index = np.expand_dims(y_index, 0) # Add axis=0 to be able to multiply with events
     x_index = np.repeat(x_index[:, :, np.newaxis], z_shape, axis=2)
     x_index = np.expand_dims(x_index, 0) # Add axis=0
     z_index = np.arange(z_shape)
     z_index = np.repeat(z_index, 2601) # repeat x * y times
     z_index = np.reshape(z_index, (z_shape, x_shape, y_shape)) # reshape
     z_index = np.swapaxes(z_index, 0, 2) # adjust indexes
     z_index = np.expand_dims(z_index, 0)
     sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
     sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z
     x_mid = np.sum(x_index * image, axis = (1, 2)) # sum events * x_index along z
     y_mid = np.sum(y_index * image, axis = (1, 2)) # sum events * y_index along z
     indexes = np.where(sumz > 0)
     x_mid[indexes] = x_mid[indexes]/sumz[indexes]
     y_mid[indexes] = y_mid[indexes]/sumz[indexes]
     z = np.arange(z_shape) # z indexes
     x_ref = np.sum(x_index * image, axis = (1, 2, 3))/sumtot
     y_ref = np.sum(y_index * image, axis = (1, 2, 3))/sumtot
     z_ref = np.sum(z_index * image, axis = (1, 2, 3))/sumtot
     x_ref = np.expand_dims(x_ref, 1)
     y_ref = np.expand_dims(y_ref, 1)
     z_ref = np.expand_dims(z_ref, 1)

     zproj = np.sqrt((x_mid-x_ref)**2.0 + + (z - z_ref)**2.0)
     m = (y_mid-y_ref)/zproj
     z_ref=np.squeeze(z_ref)
     z_ref=z_ref.astype(int)
     print(z_ref.shape)
     for i in np.arange(image.shape[0]):
       m[i, :z_ref[i]] = -1 * m[i, :z_ref[i]]
     ang = (math.pi/2.0) - np.arctan(m)
     print ('The time taken by Meas5_2 = {} seconds'.format(time.time()-start))
     return x_mid, y_mid, ang

def Meas5_4(image, p1=0, p2=24): # Working version:p1 and p2 are not used. 3D angle with barycenter as reference point
     start = time.time()
     x_shape= image.shape[1]
     y_shape= image.shape[2]
     z_shape= image.shape[3]

     sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
     x_ref = np.sum(np.sum(image, axis=(2, 3)) * np.expand_dims(np.arange(x_shape), axis=0), axis=1)/sumtot
     y_ref = np.sum(np.sum(image, axis=(1, 3)) * np.expand_dims(np.arange(y_shape), axis=0), axis=1)/sumtot
     z_ref = np.sum(np.sum(image, axis=(1, 2)) * np.expand_dims(np.arange(z_shape), axis=0), axis=1)/sumtot
     
     sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z
     x = np.expand_dims(np.arange(x_shape), axis=0)
     x = np.expand_dims(x, axis=2)
     y = np.expand_dims(np.arange(y_shape), axis=0)
     y = np.expand_dims(y, axis=2)

     x_mid = np.sum(np.sum(image, axis=2) * x, axis=1)
     y_mid = np.sum(np.sum(image, axis=1) * y, axis=1)
   
     indexes = np.where(sumz > 0)
     x_mid[indexes] = x_mid[indexes]/sumz[indexes]
     y_mid[indexes] = y_mid[indexes]/sumz[indexes]
     z = np.arange(z_shape) # z indexes                                                                                                                                                                                                                                                   
     x_ref = np.expand_dims(x_ref, 1)
     y_ref = np.expand_dims(y_ref, 1)
     z_ref = np.expand_dims(z_ref, 1)

     zproj = np.sqrt((x_mid-x_ref)**2.0 + + (z - z_ref)**2.0)
     m = (y_mid-y_ref)/zproj
     z_ref=np.squeeze(z_ref)
     z_ref=z_ref.astype(int)
     print(z_ref.shape)
     for i in np.arange(image.shape[0]):
       m[i, :z_ref[i]] = -1 * m[i, :z_ref[i]]
     ang = (math.pi/2.0) - np.arctan(m)
     print ('The time taken by Meas5_2 = {} seconds'.format(time.time()-start))
     return x_mid, y_mid, ang

def GetAllData(datafiles, numdata): # to be used if more than 10000 events
   for index, datafile in enumerate(datafiles):
       if index == 0:
          if numdata < 10000:
            x, y, theta = GetAngleData(datafile, numdata)
          else:
            x, y, theta = GetAngleData(datafile)
       else:
          if x.shape[0] < numdata:
            x_temp, y_temp, theta_temp, eta_temp = GetAngleData(datafile)
            x = np.concatenate((x, x_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)
            theta = np.concatenate((theta, theta_temp), axis=0)
   return x[:numdata], y[:numdata], theta[num_data]


# Get list of all data files
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
       Sample=Samples[SampleName]
       NFiles=len(Sample)
   return Sample

if __name__ == "__main__":
    main()
