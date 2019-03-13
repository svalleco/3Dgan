import os
import sys
import h5py
import numpy as np
import math
import time
import glob
sys.path.insert(0,'../analysis')
import utils.GANutils as gan
import setGPU
import keras.backend as K
import tensorflow as tf

def GetAngleDataSorted(datafile):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   theta = np.array(f.get('theta'))
   eta = np.array(f.get('eta'))
   X=np.array(f.get('ECAL'))
   Y=np.array(f.get('energy'))
   X = X.astype(np.float32)
   Y = Y.astype(np.float32)
   theta = theta.astype(np.float32)
   eta = eta.astype(np.float32)
   indexes = np.where((Y >= 100.0) & (Y <=200.0))
   X = X[indexes]
   Y = Y[indexes]
   theta = theta[indexes]
   eta = eta[indexes]
   ecal = np.sum(X, axis=(1, 2, 3))
   return X, Y, theta, eta, ecal

def GetAngleDataProcessed(datafile):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   theta = np.array(f.get('theta'))
   eta = np.array(f.get('eta'))
   X=np.array(f.get('ECAL'))
   Y=np.array(f.get('energy'))
   #X = X.astype(np.float32)
   #Y = Y.astype(np.float32)
   #theta = theta.astype(np.float32)
   m_theta = np.array(f.get('mtheta'))
   #eta = eta.astype(np.float32)
   ecal = np.sum(X, axis=(1, 2, 3))
   bins = hist_count(X)
   return X, Y, theta, m_theta, eta, ecal, bins

def hist_count(x, p=1):
   x = np.expand_dims(x, axis=-1)
   bin1 = np.sum(np.where(x>(0.05**p) , 1, 0), axis=(1, 2, 3))
   bin2 = np.sum(np.where((x<(0.05**p)) & (x>(0.03**p)), 1, 0), axis=(1, 2, 3))
   bin3 = np.sum(np.where((x<(0.03**p)) & (x>(0.02**p)), 1, 0), axis=(1, 2, 3))
   bin4 = np.sum(np.where((x<(0.02**p)) & (x>(0.0125**p)), 1, 0), axis=(1, 2, 3))
   bin5 = np.sum(np.where((x<(0.0125**p)) & (x>(0.008**p)), 1, 0), axis=(1, 2, 3))
   bin6 = np.sum(np.where((x<(0.008**p)) & (x>(0.003**p)), 1, 0), axis=(1, 2, 3))
   bin7 = np.sum(np.where((x<(0.003**p)) & (x>0.), 1, 0), axis=(1, 2, 3))
   bin8 = np.sum(np.where(x==0, 1, 0), axis=(1, 2, 3))
   bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
   bins[np.where(bins==0)]=1
   return bins

def SaveData(data, filename, eventsperfile):
   X = data[0]
   Y = data[1]
   theta = data[2]
   mtheta = data[3]
   eta = data[4]
   ecal = data[5]
   bins = data[6]
   print('bin shape =', bins.shape)
   with h5py.File(filename ,'w') as outfile:
      outfile.create_dataset('ECAL',data=X[:eventsperfile])
      outfile.create_dataset('energy',data= Y[:eventsperfile])
      outfile.create_dataset('theta',data=theta[:eventsperfile])
      outfile.create_dataset('eta',data=eta[:eventsperfile])
      outfile.create_dataset('sum',data=ecal[:eventsperfile])
      outfile.create_dataset('bins',data=bins[:eventsperfile])
      outfile.create_dataset('mtheta',data=mtheta[:eventsperfile])
   print('Data is stored in {} file'.format(filename))
   out = [X[eventsperfile:], Y[eventsperfile:], theta[eventsperfile:], mtheta[eventsperfile:], eta[eventsperfile:], ecal[eventsperfile:], bins[eventsperfile:]]
   #if len(data)==6:
   #   out.append(mtheta[eventsperfile:])
   #out.append([eta[eventsperfile:], ecal[eventsperfile:]])
   return out

   
def GetAllData(datafiles, eventsperfile, filename, getdata=GetAngleDataSorted):
   num= 0
   for i, f in enumerate(datafiles):
       if i == 0:
         data = getdata(f)
       else:
         temp_data = getdata(f)
         new_data = []
         for item1, item2 in zip(data, temp_data):
           new_data.append(np.concatenate((item1, item2), axis=0))
         data = new_data
       print(data[0].shape)
       if data[0].shape[0] > eventsperfile:
           data = SaveData(data, filename+'_{:03d}.h5'.format(num), eventsperfile)
           print(data[0].shape)
           num+=1
   if data[0].shape[0] > 0:
      data = SaveData(data, filename+'_{:03d}.h5'.format(num), eventsperfile)
                 
def main():
   #datapath ='/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5'
   sortedpath = '/data/shared/gkhattak'
   datapath = sortedpath +'/EleMeasured3ThetaEscan/*.h5'
   Particles = ['Ele']
   eventsperfile = 5000
   MaxFiles = -1
   l1 = 100
   l2 = 200
   print ("Searching in :", datapath)
   Files =sorted( glob.glob(datapath))
   print ("Found {} files. ".format(len(Files)))
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

   for p in Particles:
     print('There are {} files for {}'.format(len(Samples[p]), p))
     pdir = os.path.join(sortedpath, '{}Measured4ThetaEscan'.format(p))
     gan.safe_mkdir(pdir)
     print ('Directory {} is created'.format(pdir))
     filename = os.path.join(pdir, '{}_VarAngleMeas_{:d}_{:d}'.format(p, l1, l2))
     GetAllData(Samples[p], eventsperfile, filename, GetAngleDataProcessed)

if __name__ == "__main__":
    main()
