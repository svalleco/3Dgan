import os
import h5py
import numpy as np
import math
import time
import glob
import GANutils as gan
import setGPU

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

def SaveData(data, filename, eventsperfile):
   X = data[0]
   Y = data[1]
   theta = data[2]
   eta = data[3]
   ecal = data[4]
   with h5py.File(filename ,'w') as outfile:
      outfile.create_dataset('ECAL',data=X[:eventsperfile])
      outfile.create_dataset('energy',data= Y[:eventsperfile])
      outfile.create_dataset('theta',data=theta[:eventsperfile])
      outfile.create_dataset('eta',data=eta[:eventsperfile])
      outfile.create_dataset('sum',data=ecal[:eventsperfile])
   print('Data is stored in {} file'.format(filename))
   return X[eventsperfile:], Y[eventsperfile:], theta[eventsperfile:], eta[eventsperfile:], ecal[eventsperfile:]

   
def GetAllData(datafiles, eventsperfile, filename):
   num= 0
   for i, f in enumerate(datafiles):
       if i == 0:
         data = GetAngleDataSorted(f)
       else:
         temp_data = GetAngleDataSorted(f)
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
   datapath ='/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5'
   sortedpath = '/bigdata/shared/gkhattak'
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
     pdir = os.path.join(sortedpath, '{}Escan'.format(p))
     gan.safe_mkdir(pdir)
     print ('Directory {} is created'.format(pdir))
     filename = os.path.join(pdir, '{}_VarAngleSorted_{:d}_{:d}'.format(p, l1, l2))
     GetAllData(Samples[p], eventsperfile, filename)

if __name__ == "__main__":
    main()
