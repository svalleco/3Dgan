import os
import h5py
import numpy as np
import math
import time
import glob
import GANutils as gan
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
   X = X.astype(np.float32)
   Y = Y.astype(np.float32)
   theta = theta.astype(np.float32)
   m_theta = measPython(X)
   eta = eta.astype(np.float32)
   ecal = np.sum(X, axis=(1, 2, 3))
   return X, Y, theta, m_theta, eta, ecal

def measKeras(events, yp1 = 3, yp2 = 21):
   images = tf.convert_to_tensor(events, dtype=tf.float32)
   m = ecal_angle_3d(images)
   return m

def ecal_angle_3d(image):
   start = time.time()
   x_shape= K.int_shape(image)[1]
   y_shape= K.int_shape(image)[2]
   z_shape= K.int_shape(image)[3]

   sumtot = K.sum(image, axis=(1, 2, 3))# sum of events

   amask = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
   masked_events = K.sum(amask) # counting zero sum events
   
   x_ref = K.sum(K.sum(image, axis=(2, 3)) * (K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32') + 0.5), axis=1)
   y_ref = K.sum(K.sum(image, axis=(1, 3)) * (K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32') + 0.5), axis=1)
   z_ref = K.sum(K.sum(image, axis=(1, 2)) * (K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32') + 0.5), axis=1)

   x_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref), x_ref/sumtot)# return max position if sumtot=0 and divide by sumtot otherwise
   y_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref), y_ref/sumtot)
   z_ref = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref), z_ref/sumtot)
         
   x_ref = K.expand_dims(x_ref, 1)
   y_ref = K.expand_dims(y_ref, 1)
   z_ref = K.expand_dims(z_ref, 1)

   sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z
   zmask = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz) , K.ones_like(sumz))
   zunmasked_events = K.sum(zmask, axis=1)
      
   x = K.expand_dims(K.arange(x_shape), 0)
   x = K.cast(K.expand_dims(x, 2), dtype='float32') + 0.5
   y = K.expand_dims(K.arange(y_shape), 0)
   y = K.cast(K.expand_dims(y, 2), dtype='float32') + 0.5

   x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
   y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)

   x_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
   y_mid = K.tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum
   z= (K.cast(K.arange(z_shape), dtype='float32') + 0.5) * K.ones_like(z_ref)
   zproj = K.sqrt((x_mid-x_ref)**2.0 + (z - z_ref)**2.0)
   m = (y_mid-y_ref)/zproj
   m = K.tf.where(K.tf.less(z, z_ref), -1 * m,  m)
   ang = (math.pi/2.0) - tf.atan(m)
   ang = ang * zmask
   ang = K.sum(ang, axis=1)/zunmasked_events
   ang = K.tf.where(K.equal(amask, 0.), ang, 100. * K.ones_like(ang))
         
   return K.eval(ang)
                                                                                                                              
def measPython(image): # Working version:p1 and p2 are not used. 3D angle with barycenter as reference point
   start = time.time()
   x_shape= image.shape[1]
   y_shape= image.shape[2]
   z_shape= image.shape[3]

   sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
   indexes = np.where(sumtot > 0)
   amask = np.ones_like(sumtot)
   amask[indexes] = 0
   
   #amask = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
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
      
   #ang = np.sum(ang, axis=1)/zunmasked_events #mean
   ang = ang * z # weighted by position
   sumz_tot = z * zmask
   ang = np.sum(ang, axis=1)/np.sum(sumz_tot, axis=1)
                 
   indexes = np.where(amask>0)
   ang[indexes] = 100.
   #ang = ang.reshape(-1, 1)
   print(ang.shape)
   print ('The time taken by Meas5_2 = {} seconds'.format(time.time()-start))
   return ang
                                                                                                                                                    
def SaveData(data, filename, eventsperfile):
   X = data[0]
   Y = data[1]
   theta = data[2]
   if len(data)==5:
      eta = data[3]
      ecal = data[4]
   elif len(data)==6:
      mtheta = data[3]
      eta = data[4]
      ecal = data[5]
   print(theta[:5])
   print(mtheta[:5])
   with h5py.File(filename ,'w') as outfile:
      outfile.create_dataset('ECAL',data=X[:eventsperfile])
      outfile.create_dataset('energy',data= Y[:eventsperfile])
      outfile.create_dataset('theta',data=theta[:eventsperfile])
      outfile.create_dataset('eta',data=eta[:eventsperfile])
      outfile.create_dataset('sum',data=ecal[:eventsperfile])
      if len(data)==6:
         outfile.create_dataset('mtheta',data=mtheta[:eventsperfile])
   print('Data is stored in {} file'.format(filename))
   out = [X[eventsperfile:], Y[eventsperfile:], theta[eventsperfile:], mtheta[eventsperfile:], eta[eventsperfile:], ecal[eventsperfile:]]
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
   datapath = sortedpath +'/EleEscan/*.h5'
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
     pdir = os.path.join(sortedpath, '{}Measured3ThetaEscan'.format(p))
     gan.safe_mkdir(pdir)
     print ('Directory {} is created'.format(pdir))
     filename = os.path.join(pdir, '{}_VarAngleMeas_{:d}_{:d}'.format(p, l1, l2))
     GetAllData(Samples[p], eventsperfile, filename, GetAngleDataProcessed)

if __name__ == "__main__":
    main()
