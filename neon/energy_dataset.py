import numpy as np
import h5py

from neon.data import NervanaDataIterator

def temp_3Ddata():

   f = h5py.File("/afs/cern.ch/work/s/svalleco/public/Eshuffled100-200.h5","r")
   data = f.get('ECAL')
   #dtag =f.get('TAG')
   xtr = np.array(data)
   print (xtr.shape)
   #labels = np.ones(xtr.shape[0])
   labels =f.get('TAG')
   #tag=numpy.array(dtag)
   #xtr=xtr[...,numpy.newaxis]
   #xtr=numpy.rollaxis(xtr,4,1)
   #print xtr.shape
   
   return xtr.astype(np.float32), labels.astype(np.float32)
   #return xtr.reshape((xtr.shape[0], 25 * 25 * 25)).astype(np.float32), labels.astype(np.float32)

def get_output():
   f = h5py.File("/afs/cern.ch/work/s/svalleco/GAN/3Dgan_neon_test/neon/output_data.h5","r")
   data = f.get("dataset_1")
   x = np.array(data)
   return x.astype(np.float32)


class EnergyData(NervanaDataIterator):
    def __init__(self, X, Y, lshape):
        self.X = X 
        self.Y = Y
        self.shape = lshape
        self.start = 0
        self.ndata = X.shape[0]
        self.nfeatures =1*25*25*25
        self.nbatches = int(self.ndata/self.be.bsz)
        self.dev_X = (self.nfeatures, self.be.bsz)
        self.dev_Y = (self.be.bsz, self.be.bsz, self.be.bsz)  # 3 targets: real/fake, primaryE, sumEcal

    def reset(self):
        self.start = 0

    def __iter__(self):
        # 3. loop through minibatches in the dataset
        for index in range(self.start, self.ndata, self.be.bsz):
            # 3a. grab the right slice from the numpy arrays
            inputs = self.X[:index,:]
            sumE = ecal_train = np.sum(inputs, axis=(1)) 
            targets =(self.Y(:index), sumE)
            
            # The arrays X and Y data are in shape (batch_size, num_features),
            # but the iterator needs to return data with shape (num_features, batch_size).
            # here we transpose the data, and then store it as a contiguous array. 
            # numpy arrays need to be contiguous before being loaded onto the GPU.
            inputs = np.ascontiguousarray(inputs.T)
            targets = np.ascontiguousarray(targets.T)
                        
            # here we test your implementation
            # your slice has to have the same shape as the GPU tensors you allocated
            assert inputs.shape == self.dev_X.shape, \
                   "inputs has shape {}, but dev_X is {}".format(inputs.shape, self.dev_X.shape)
            assert targets.shape == self.dev_Y.shape, \
                   "targets has shape {}, but dev_Y is {}".format(targets.shape, self.dev_Y.shape)
            
            # 3b. transfer from numpy arrays to device
            # - use the GPU memory buffers allocated previously,
            #    and call the myTensorBuffer.set() function. 
            self.dev_X.set(inputs)
            self.dev_Y.set(targets)
            
            # 3c. yield a tuple of the device tensors.
            # the first should of shape (num_features, batch_size)
            # the second should of shape (4, batch_size)
            yield (self.dev_X, self.dev_Y)
