import numpy as np
import h5py

def temp_3Ddata():

   f = h5py.File("/afs/cern.ch/work/e/eorlova/Ele_Fixed100_total2.h5","r")
   data =f.get('ECAL')
   #dtag =f.get('TAG')
   xtr=np.array(data)
   labels = np.ones(xtr.shape[0])
   #tag=numpy.array(dtag)
   #xtr=xtr[...,numpy.newaxis]
   #xtr=numpy.rollaxis(xtr,4,1)
   print xtr.shape
   
   return xtr.reshape((xtr.shape[0], 25 * 25 * 25)).astype(np.float32), labels.astype(np.float32)

