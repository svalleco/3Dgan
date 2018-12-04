import numpy
import h5py

def lcd_3Ddata():

   f = h5py.File("EGshuffled.h5","r")
   data =f.get('ECAL')
   dtag =f.get('TAG')
   xtr=numpy.array(data)
   tag=numpy.array(dtag)
   #xtr=xtr[...,numpy.newaxis]
   #xtr=numpy.rollaxis(xtr,4,1)
   print xtr.shape
   
   return xtr, tag.astype(bool)
