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
   numdata = 1000
   numevents=10
   ypoint1 = 3 
   ypoint2 = 21
   datafiles = GetDataFiles(datapath, 10000, Particles=['Ele'])
   plotsdir = 'keras_vskeras'
   filename = plotsdir + '/'
   gan.safe_mkdir(plotsdir)
   #Get data
   X, Y, theta, alpha= GetAngleData(datafiles[0], num_data=numdata)
   x, y, l, m, m_meas = Meas2(X)  # keras
   event_id = 3
   fig=1
   PlotHistcenter(x[event_id], filename + 'x_center_hist_keras.pdf', 'x', fig)
   fig+=1
   PlotHistcenter(y[event_id], filename + 'y_center_hist_keras.pdf', 'y', fig)
   fig+=1
   Plotcenter(x[event_id], filename + 'x_center_keras.pdf', 'x', fig)
   fig+=1
   Plotcenter(y[event_id], filename + 'y_center_keras.pdf', 'y', fig)
   fig+=1
   Plotcenter(l[event_id], filename + 'l_keras.pdf', 'global {:.4f}'.format(theta[event_id]), fig)
   fig+=1
   PlotAngle(m[event_id], filename + 'angle_keras.pdf', 'global {:.4f}'.format(theta[event_id]), fig)
   fig+=1
   PlotHistMeas(m[event_id], 'global {:.4f} keras'.format(theta[event_id]), filename + 'keras_angle_hist_event.pdf', fig=fig)
   fig+=1
   #m_meas = ProcAngles2(m)
   PlotHistMeas(m_meas, 'keras', filename + 'keras_angle_hist_all.pdf', fig=fig)
   fig+=1
   PlotHistError([m_meas], theta, ['3d Measured '], filename + 'Error_keras_hist.pdf', fig=fig)
   fig+=1
   PlotAngleMeasure(m_meas, theta, ['3d Measured', 'Theta'], filename + 'actual_keras.pdf', fig=fig)
   fig+=1
      
   x2, y2, l2, m2, m2_meas = Meas2_prev(X) # keras 2
   PlotHistcenter(x2[event_id], filename + 'x_center_hist_keras2.pdf', 'global {:.4f} x'.format(theta[event_id]), fig)
   fig+=1
   PlotHistcenter(y2[event_id], filename + 'y_center_hist_keras2.pdf', 'global {:.4f}y'.format(theta[event_id]), fig)
   fig+=1
   Plotcenter(x2[event_id], filename + 'x_center_keras2.pdf', 'global {:.4f} x'.format(theta[event_id]), fig)
   fig+=1
   Plotcenter(y2[event_id], filename + 'y_center_keras2.pdf', 'global {:.4f} y'.format(theta[event_id]), fig)
   fig+=1
   Plotcenter(l2[event_id], filename + 'l_keras2.pdf', 'global {:.4f} '.format(theta[event_id]), fig)
   fig+=1
   PlotAngle(m2[event_id], filename + 'angle_keras2.pdf', 'global {:.4f}'.format(theta[event_id]), fig)
   fig+=1
      
   PlotHistMeas(m2[event_id], 'global {} keras2'.format(alpha[event_id]), filename + 'python_angle_hist_event.pdf', fig=fig)
   fig+=1

   #angle = ProcAngles2(m2)
   PlotHistMeas(m2_meas, 'keras2', filename + 'keras2_angle_hist.pdf', fig=fig)
   fig+=1
   PlotAngleMeasure(m_meas, m2_meas, ['Keras1', 'Keras2'],filename + 'keras1_2.pdf', fig=fig, yp1 = ypoint1, yp2 = ypoint2)
   fig+=1
   PlotHistError([m_meas], m2_meas, ['Keras1', 'Keras2'], filename + 'error_hist.pdf', fig=fig)
   fig+=1
   
   # Plots event with error > 0.5
   indexes = np.where(np.absolute(m2_meas - m_meas)> 0.005)
   events = X[indexes]
   angle = m2_meas[indexes]
   meas = m_meas[indexes]
   x = X.shape[1]
   y = X.shape[2]
   z = X.shape[3]
   for i in np.arange(min(numevents, events.shape[0])):
      PlotEvent(events[i], x, y, z, m2_meas[i], meas[i], meas[i], plotsdir + '/Event1_{}.pdf'.format(i), i, fig= fig, yp1 = ypoint1, yp2 = ypoint2)
      fig+=1
   print('Plots are saved in {}.'.format(plotsdir))

# Histogram of berycenters for an event                                                                                                
def PlotHistcenter(a, outfile, label, fig=0):
   print(a)
   bins = np.arange(0, np.amax(a), 0.25)
   plt.figure(fig)
   plt.hist(a, bins=bins , label= label + ' of barycenter for an event')
   plt.legend()
   plt.xlabel(label + 'center ')
   plt.savefig(outfile)

# Plot of berycenters for an event
def Plotcenter(a, outfile, label, fig=0):
   plt.figure(fig)
   plt.plot(a, label= label + ' of barycenter for an event')
   plt.legend()
   plt.xlabel(label + 'center ')
   plt.savefig(outfile)

# Plot calculated angle for event
def PlotAngle(a, outfile, label, fig=0):
   #bins = np.arange(0, a.shape[0])
   plt.figure(fig)
   plt.plot(a, label= label + 'Angle ')
   plt.legend()
   plt.xlabel(label + 'Angle ')
   plt.savefig(outfile)
                  
def PlotHistError(meas,  angle, labels, outfile, fig, degree=False):
   bins = np.arange(-2,2, 0.01)
   plt.figure(fig)
   for m, label in zip(meas, labels):
     error = angle-m
     print error.shape
     if degree:
        angle = np.degrees(angle)
        m = np.degrees(m)
        error = np.degrees(error)
        unit = 'degrees'
     else:
        unit = 'radians'
     plt.hist(error, bins=bins, label='{} error {:.4f}({:.4f})'.format(label, np.mean(np.absolute(error)), np.std(error)))
   plt.legend()
   plt.ylabel('Count')
   plt.xlabel('Error ({})'.format(unit))
   plt.savefig(outfile)

def PlotHistMeas(meas,  label, outfile, fig, degree=False):
   bins = np.arange(-2,2, 0.01)
   plt.figure(fig)
   unit = label
   plt.hist(meas, bins='auto')
   plt.legend()
   plt.xlabel('Angle ({})'.format(unit))
   plt.ylabel('Count ({})'.format(unit))
   plt.savefig(outfile)

def PlotEvent(event, x, y, z, angle, m1, m2, outfile, n, fig, yp1 = 3, yp2 = 21):
   print 'event {}'.format(n)
   array = np.sum(np.squeeze(event), axis=(0))
   y = np.arange(25)
   maxy = np.argmax(array, axis=0)
   plt.figure(fig)
   plt.scatter(y, maxy, label = 'maxy G4angle = {}'.format(angle))
   p = np.polyfit(y, maxy, 1)
   ytan = (maxy[yp2]-maxy[yp1]) / np.float32(yp2 - yp1)
   pfit= np.polyval(p, y)
   plt.scatter(y, pfit, label = 'fit angle = {}'.format(m1) )
   a = [yp1, yp2]
   b = [maxy[yp1], maxy[yp2]]
   plt.plot(a, b, label = 'Approx angle = {:.4f}\n yp1={} yp2={}'.format(m2, yp1, yp2))
   plt.legend()
   plt.ylim =(0, y)
   plt.savefig(outfile)

def PlotAngleMeasure(measured, angle, labels, outfile, fig=1, degree=False, yp1 = 3, yp2 = 21):
   error = np.absolute(angle - measured)
   if degree:
      angle = np.degrees(angle)
      measured = np.degrees(measured)
      error = np.degrees(error)
      unit = 'degrees'
   else:
      unit = 'radians'
   plt.figure(fig)
   plt.scatter(angle, measured, label='Error Mean={:.4f}, std={:.4f}'.format(np.mean(error), np.std(error)))
   plt.legend()
   plt.xlabel(labels[1] +' Angle ({})'.format(unit))
   plt.ylabel(labels[0] + ' Angle ({})'.format(unit))
   plt.savefig(outfile)

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

def Meas1(events, mod=0):
    a = []
    events = np.squeeze(events, axis=4)
    x = np.zeros((events.shape[0], events.shape[3])) # shape = (num events, z)                                                                                   
    y = np.zeros((events.shape[0], events.shape[3]))
    ang = np.zeros((events.shape[0], events.shape[3]))
    length = np.zeros((events.shape[0], events.shape[3]))
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
            length[i, j] = math.sqrt((x[i, j] - x[i, 0])**2 + (y[i, j] - y[i, 0])**2 + j**2)
            zproj = math.sqrt((x[i, j] - x[i, 0])**2+ j**2)
            ang[i, j] = np.arctan(zproj/length[i, j])
    return x, y, length, ang

def Meas2_2(image):
     image = np.squeeze(image, axis=4)
     x_shape= image.shape[1]
     y_shape= image.shape[2]
     z_shape= image.shape[3]
     ang = np.zeros((image.shape[0], z_shape))
     y_index = np.arange(y_shape)   # vector of indexes
     y_index = np.tile(y_index, (x_shape))         # repeat x shape times
     y_index = np.reshape(y_index, (y_shape, x_shape))  # reshape to make 2d
     x_index = np.transpose(y_index) # 2d x indexes
     #y_index = np.tile(y_index, (1, 1, z_shape)) # repeat along z
     y_index = np.repeat(y_index[:, :, np.newaxis], z_shape, axis=2)
     y_index = np.expand_dims(y_index, 0) # Add axis=0
     #x_index = np.tile(np.expand_dims(x_index, 0), (1, 1, z_shape)) # repeat along z
     x_index = np.repeat(x_index[:, :, np.newaxis], z_shape, axis=2)
     x_index = np.expand_dims(x_index, 0) # Add axis=0
     z_index = np.arange(z_shape)
     z_index = np.repeat(z_index, 2601)
     z_index = np.reshape(z_index, (z_shape, x_shape, y_shape))
     z_index = np.swapaxes(z_index, 0, 2)
     z_index = np.expand_dims(z_index, 0)
     sumtot = np.sum(image, axis=(1, 2, 3))
     x_ref = np.sum(x_index * image, axis = (1, 2, 3))/sumtot
     y_ref = np.sum(y_index * image, axis = (1, 2, 3))/sumtot
     z_ref = np.sum(z_index * image, axis = (1, 2, 3))/sumtot
     x_ref = np.expand_dims(x_ref, 1)
     y_ref = np.expand_dims(y_ref, 1)
     z_ref = np.expand_dims(z_ref, 1)
     sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z
     x_mid = np.sum(x_index * image, axis = (1, 2)) # sum events * x_index
     y_mid = np.sum(y_index * image, axis = (1, 2)) # sum events * y_index
     indexes = np.where(sumz > 0)
     x_mid[indexes] = x_mid[indexes]/sumz[indexes]
     y_mid[indexes] = y_mid[indexes]/sumz[indexes]
     #x_ref = np.expand_dims(x_mid[:, 0], 1) * np.ones_like(x_mid)
     #y_ref = np.expand_dims(y_mid[:, 0], 1) * np.ones_like(y_mid)
     z = np.arange(z_shape) # z indexes
     print((z - z_ref).shape)
     l = np.sqrt((x_mid-x_ref)**2.0 + (y_mid-y_ref)**2.0 + (z - z_ref)**2.0) # Magnitude of line connecting two points
     zproj = np.sqrt((x_mid-x_ref)**2.0 + + (z - z_ref)**2.0)
     indexes = np.where(l > 0)
     print(l.shape)
     print(zproj.shape)
     print(ang.shape)     
     ang[indexes] = (math.pi/2) + np.arctan(zproj[indexes]/l[indexes])
     #weighting
     
     return x_mid, y_mid, l, ang
                                                                                 
def Meas5_4(image, p1=0, p2=24): # Working version:p1 and p2 are not used. 3D angle with barycenter as reference point
     start = time.time()
     image = np.squeeze(image, axis=4)
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

     zproj = np.sqrt((x_mid-x_ref)**2.0  + (z - z_ref)**2.0)
     m = (y_mid-y_ref)/zproj
     #z_ref=np.squeeze(z_ref)
     #z_ref=z_ref.astype(int)
     z = z * np.ones_like(z_ref)
     print(z[0], z_ref[0])
     indexes = np.where(z<z_ref)
     m[indexes] = -1 * m[indexes]
     #for i in np.arange(image.shape[0]):
     #  m[i, :z_ref[i]] = -1 * m[i, :z_ref[i]]
     print(m[0])
     ang = (math.pi/2.0) - np.arctan(m)
     print(ang[0])
     print ('The time taken by Meas5_2 = {} seconds'.format(time.time()-start))
     return x_mid, y_mid, zproj, ang
                                                    

# Angle Measure using weighting by z for all events                                                                                                              
def ProcAngles2(meas):
   a = np.arange(meas.shape[1])
   print a
   avg = np.zeros(meas.shape[0])
   for i in np.arange(1, meas.shape[0]):
      avg[i] = np.sum( meas[i] * a)/ (0.5  * (meas.shape[1] * (meas.shape[1] - 1)))                                                                            
   return avg


def Meas2(events, yp1 = 3, yp2 = 21):
    images = tf.convert_to_tensor(events, dtype=tf.float32)
    x, y, l, m, m_meas = ecal_angle_3d_ver2(images)
    print x[0], y[0], m[0] 
    #m = np.squeeze(m)
    #m = np.mean(m)
    return x, y, l, m, m_meas

def Meas2_prev(events, yp1 = 3, yp2 = 21):
    images = tf.convert_to_tensor(events, dtype=tf.float32)
    x, y, l, m, m_meas = ecal_angle_3d_ver3(images)
    return x, y, l, m, m_meas
                         

def Meas3(events, yp1 = 3, yp2 = 21):
    event = np.sum(np.squeeze(events), axis=(1))
    maxy = np.argmax(event, axis=(0, 1))
    tan = (maxy[:,yp2]-maxy[:, yp1]) / np.float(yp2 - yp1)
    angle = math.atan(tan)
    return angle

def ecal_angle(image, p1, p2):
    a = K.sum(image, axis=(1, 4))
    b = K.argmax(a, axis=1)
    c = K.cast(b[:,p2] - b[:,p1], dtype='float32')/(p2 - p1)
    d = tf.atan(c)
    d = K.abs(d)
    d = K.expand_dims(d)
    return K.eval(d)

def ecal_angle_3d(image):
   start = time.time()
   image = K.squeeze(image, axis=4)
   x_shape= K.int_shape(image)[1]
   y_shape= K.int_shape(image)[2]
   z_shape= K.int_shape(image)[3]
   
   y_index = K.arange(y_shape)   # vector of indexes                                                                                                      
   y_index = K.cast(y_index, dtype='float32') # indexes to float                                                                                          
   y_index = K.tile(y_index, x_shape)         # repeat x shape times                                                                                      
   y_index = K.reshape(y_index, (y_shape, x_shape))  # reshape to make 2d                                                                                 
   x_index = K.transpose(y_index) # 2d x indexes                                                                                                          
   y_index = K.tile(K.expand_dims(y_index), (1, 1, z_shape)) # repeat along z                                                                             
   y_index = K.expand_dims(y_index, 0) # Add axis=0                                                                                                       
   x_index = K.tile(K.expand_dims(x_index), (1, 1, z_shape)) # repeat along z                                                                             
   x_index = K.expand_dims(x_index, 0) # Add axis=0                                                                                                       

   sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z                                                                                   
   x_mid = K.sum(x_index * image, axis = (1, 2)) # sum events * x_index                                                                                   
   y_mid = K.sum(y_index * image, axis = (1, 2)) # sum events * y_index                                                                                   

   x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum                                                
   y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum                                                

   x_ref = K.expand_dims(x_mid[:, 0], 1) * K.ones_like(x_mid)
   y_ref = K.expand_dims(y_mid[:, 0], 1) * K.ones_like(y_mid)
   z = K.arange(z_shape) # z indexes                                                                                                                      
   z = K.cast(z, dtype='float32') # cast to float                                                                                                         
   l = K.sqrt((x_mid-x_ref)**2.0 + (y_mid-y_ref)**2.0 + z**2.0) # Magnitude of line connecting two points
   zproj = K.sqrt((x_mid-x_ref)**2.0 + + z**2.0)
   ang = K.tf.where(K.equal(l, 0.0), K.zeros_like(l), tf.arctan(zproj/l))  # angle with z
   print ('The time taken by Keras meas1 = {} seconds'.format(time.time()-start))
   return K.eval(x_mid), K.eval(y_mid), K.eval(l), K.eval(ang)

def ecal_angle_3d_ver3(image):
   start = time.time()
   image = K.squeeze(image, axis=4)
   sumtot = K.sum(image, axis=(1, 2, 3))# sum of events
   x_shape= K.int_shape(image)[1]
   y_shape= K.int_shape(image)[2]
   z_shape= K.int_shape(image)[3]
   print(x_shape, y_shape, z_shape)
   x_ref = K.sum(K.sum(image, axis=(2, 3)) * (K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32') + 0.5), axis=1)
   y_ref = K.sum(K.sum(image, axis=(1, 3)) * (K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32') + 0.5), axis=1)
   z_ref = K.sum(K.sum(image, axis=(1, 2)) * (K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32') + 0.5), axis=1)

   x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) * K.cast(x_shape - 1, dtype='float32'), x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
   y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref)* K.cast(y_shape - 1, dtype='float32'), y_ref/sumtot)
   z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref)* K.cast(z_shape - 1, dtype='float32'), z_ref/sumtot)

   x_ref = K.expand_dims(x_ref, 1)
   y_ref = K.expand_dims(y_ref, 1)
   z_ref = K.expand_dims(z_ref, 1)
   
   sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z
   
   x = K.expand_dims(K.arange(x_shape), 0)
   x = K.cast(K.expand_dims(x, 2), dtype='float32') + 0.5
   y = K.expand_dims(K.arange(y_shape), 0)
   y = K.cast(K.expand_dims(y, 2), dtype='float32') + 0.5
   xsum = K.sum(image, axis=2)
   x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
   y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)

   x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
   y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum
   z = K.cast(K.arange(z_shape), dtype='float32') * K.ones_like(z_ref)
   zproj = K.sqrt((x_mid-x_ref)**2.0 + (z - z_ref)**2.0)
   m = (y_mid-y_ref)/zproj
   m = K.tf.where(K.tf.less(z, z_ref),  -1 * m, m)
   ang = (math.pi/2.0) - tf.atan(m)
   mang = K.mean(ang, axis=1)
   print ('The time taken by Keras 1 = {} seconds'.format(time.time()-start))
   return K.eval(x_mid), K.eval(y_mid), K.eval(zproj), K.eval(ang), K.eval(mang)
                                                                                                                           
                                                                     
def ecal_angle_3d_ver2(image):
   start = time.time()
   image = K.squeeze(image, axis=4)
   sumtot = K.sum(image, axis=(1, 2, 3))# sum of events
   amask = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
   masked_events = K.sum(amask)
   print '......Start'
   print K.int_shape(amask)
   print K.eval(amask[:10])
   print K.eval(masked_events)
   x_shape= K.int_shape(image)[1]
   y_shape= K.int_shape(image)[2]
   z_shape= K.int_shape(image)[3]
   print(x_shape, y_shape, z_shape)
   x_ref = K.sum(K.sum(image, axis=(2, 3)) * (K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32') + 0.5), axis=1)
   y_ref = K.sum(K.sum(image, axis=(1, 3)) * (K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32') + 0.5), axis=1)
   z_ref = K.sum(K.sum(image, axis=(1, 2)) * (K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32') + 0.5), axis=1)

   x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) * K.cast(x_shape - 1, dtype='float32'), x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
   y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref)* K.cast(y_shape - 1, dtype='float32'), y_ref/sumtot)
   z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref)* K.cast(z_shape - 1, dtype='float32'), z_ref/sumtot)
           
   x_ref = K.expand_dims(x_ref, 1)
   y_ref = K.expand_dims(y_ref, 1)
   z_ref = K.expand_dims(z_ref, 1)
             
   sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z
   zmask = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz) , K.ones_like(sumz))
   zunmasked_events = K.sum(zmask, axis=1)
   print('z mask')
   print(K.int_shape(zmask))
   print(K.eval(zmask[:5]))
   print (K.eval(zunmasked_events[:5]))
   print(np.sum(K.eval(zmask)))
   x = K.expand_dims(K.arange(x_shape), 0)
   x = K.cast(K.expand_dims(x, 2), dtype='float32') + 0.5
   y = K.expand_dims(K.arange(y_shape), 0)
   y = K.cast(K.expand_dims(y, 2), dtype='float32') + 0.5
   xsum = K.sum(image, axis=2)
   x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
   y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)

   x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
   y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum
   z = K.cast(K.arange(z_shape), dtype='float32') * K.ones_like(z_ref)
   #print(K.int_shape(x_ref))
   #print(K.int_shape(x_mid))
   #print(K.int_shape(z))
   #print(K.int_shape(z_ref))
   zproj = K.sqrt((x_mid-x_ref)**2.0 + (z - z_ref)**2.0)
   m = (y_mid-y_ref)/zproj
   #z_ref=K.squeeze(z_ref, 1)
   #z_ref=K.cast(z_ref, dtype='int32')
   print(K.eval(z[0]))
   print(K.eval(z_ref[0]))
   m = K.tf.where(K.tf.less(z, z_ref),  -1 * m, m)
   #m[:, :12] = -1 * m[:, :12]
   ang = (math.pi/2.0) - tf.atan(m)
   ang = ang * zmask
   mang = K.sum(ang, axis=1)/zunmasked_events
   print ("######################################")
   print(K.eval(m[0]))
   print(K.eval(ang[0]))
   print(K.eval(mang))
   print ("######################################")
   mang = K.tf.where(K.equal(amask, 0.), mang, 4 * K.ones_like(mang))
   print ('The time taken by Keras Meas5_2 = {} seconds'.format(time.time()-start))
   return K.eval(x_mid), K.eval(y_mid), K.eval(zproj), K.eval(ang), K.eval(mang)


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
