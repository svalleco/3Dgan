import sys
import os

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from scipy.optimize import curve_fit
from scipy.stats import chisquare 
from scipy import stats
def ftn(energy,	A, B, k):
    e =	np.power(energy, k)
    return np.power((-B * e)/A, (1/k))

def plot_data(datafile, fig, plotfile):
   num_events= 1000

   if datafile== 0:
      d=h5py.File("/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5",'r')
      color='r'
   else:
      d=h5py.File("/bigdata/shared/LCD/NewV1/Pi0Escan/Pi0Escan_1_1.h5",'r')
      color='b'
   X=np.array(d.get('ECAL')[0:num_events])                                        
   e=d.get('target')[0:num_events]
   Y=(np.array(e[:,1]))
  #initialization of parameters
   X[X < 1e-6] = 0
   Y = Y/100
   Data = np.sum(X, axis=(1, 2, 3))
   plt.figure(fig)
   plt.scatter(Y, Data/Y, s=1, color=color)
   plt.gca().set_yscale('log')
   plt.savefig(plotfile)

def main():
   filename = 'pivsele_log.pdf'
   plot_data(0, 1, filename)
   plot_data(1, 1, filename)

if __name__ == "__main__":
    main()

