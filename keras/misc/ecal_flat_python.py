import sys
import os
import ROOT
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import math
from keras import backend as K
import ROOTutils  
# Computing log only when not zero
def logftn(x, base, f1, f2):
    select= np.where(x>0)
    x[select] = f1 * np.log10(x[select]) + f2
    return x

# Exponent
def expon(a):
    select= np.where(a>0)
    a[select] = np.exp(6.0 * (a[select] - 1.0) * np.log(10.0))
    return a

# Ecal sum Calculation
def get_log(image):
    image = np.squeeze(image, axis = 4)
    return expon(image)

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

def PlotEcalFlatlog(x, outfile, label, fig=1):
    x = x.flatten()
    bins = np.logspace(np.log10(1e-7),np.log10(1),50)
    plt.figure(fig)
    plt.hist(x, bins=bins, histtype='step', label=label)
    plt.legend()
    plt.xscale('log', nonposy='clip')
    plt.xlabel('Flattend Ecal energies')
    print('Saving plot for Flat Ecal in {}.'.format(outfile))
    plt.savefig(outfile)

def PlotEcalFlat(x, outfile, label, fig=1):
    x = x.flatten()
    bins = np.arange(0.01, 1, 0.01)
    plt.figure(fig)
    plt.hist(x, bins=bins, histtype='step', label=label)
    #plt.xscale('log', nonposy='clip')
    plt.legend()
    plt.xlabel('Log of G4')
    print('Saving plot for Flat Ecal in {}.'.format(outfile))
    plt.savefig(outfile)

def main():
    datafile="/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_1_1.h5"
    num_events=100
    outfile = 'flat_ecal_python'
    X, Y = GetProcData(datafile, num_events)
    PlotEcalFlatlog(X, outfile + '1.pdf', label='G4')
    xlog = logftn(X, 10, 1.0/6.0, 1.0)
    PlotEcalFlat(xlog, outfile + '_log.pdf', fig=2, label='log Ecal')
    x = get_log(xlog)
    PlotEcalFlatlog(x, outfile + '2.pdf', label= 'out of lambda')
                   
if __name__ == '__main__':
    main()
