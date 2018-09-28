import sys
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import math
from keras import backend as K
import setGPU

# Computing log only when not zero
def logftn(x, base, f1, f2):
    select= np.where(x>0)
    x[select] = f1 * np.log10(x[select]) + f2
    return x

# Exponent
def expon(a):
    a = a - K.ones_like(a)
    a = K.log(10.0) * a
    a = K.exp(6.0 * a)
    return a

# Ecal sum Calculation
def get_ecal_sum(image):
    #image = np.squeeze(image, axis = 4)
    result = K.tf.where(K.equal(image, 0.0),  K.zeros_like(image), expon(image))
    result = K.sum(result, axis=(1, 2, 3))
    #ecal = K.expand_dims(sum)
    return K.eval(result)

def GetProcData(datafile, num_events):
    #get data for training                                                      
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=f.get('target')[:,1][:num_events]
    x=np.array(f.get('ECAL'))[:num_events]
    y=np.array(y)
    x[x < 1e-6] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

def PlotEcalSum(array, outfile, label, fig=1):
    plt.figure(fig)
    plt.hist(array, bins='auto', histtype='step', label=label)
    plt.legend()
    plt.xlabel('Sum of Ecal energies')
    print('Saving plot for Flat Ecal in {}.'.format(outfile))
    plt.savefig(outfile)

def main():
    datafile="/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5"
    num_events=1000
    outfile = 'ecal_sum_python'
    X, Y = GetProcData(datafile, num_events)
    PlotEcalSum(np.sum(X, axis=(1, 2, 3)), outfile + '1.pdf', label='G4')
    xlog = logftn(X, 10, 1.0/6.0, 1.0)
    ecal = get_ecal_sum(xlog)
    print(X.shape)
    print(xlog.shape)
    print(ecal.shape)
    PlotEcalSum(ecal, outfile + '2.pdf', label= 'out of lambda')
                   
if __name__ == '__main__':
    main()
