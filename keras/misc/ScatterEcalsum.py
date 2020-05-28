
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

data_file=1
num_events= 1000

if data_file== 0:
   d=h5py.File("ElectronEnergyFile.h5",'r')
   X=np.array(d.get('ECAL'))
   e=d.get('energy')
   Y=(np.array(e[:,0, 1]))
if data_file== 1:
    d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
    X=np.array(d.get('ECAL')[0:num_events])                                        
    e=d.get('target')[0:num_events]
    Y=(np.array(e[:,1]))
if data_file== 2:
   d=h5py.File("/afs/cern.ch/work/s/svalleco/public/Eshuffled100-200.h5",'r')
   X=np.array(d.get('ECAL'))
   e=d.get('energy')
   Y=(np.array(e[:,0, 1]))

# Initialization of parameters
X[X < 1e-6] = 0
Y = Y/100
Data = np.sum(X, axis=(1, 2, 3))
plt.figure(1)
plt.scatter(Y, Data, s=1)
z1 = np.polyfit(Y, Data, 2)
#print z
p1 = np.polyval(z1, Y)
#print Data.shape
#print p1.shape
a1, b1= chisquare(Data, f_exp=p1)
plt.scatter(Y, p1, color='r', s=1)
c1, d1 = stats.ks_2samp(Data, np.polyval(z1, Y))
#print a, b
#plt.show(1)
#plt.figure(2)
#plt.scatter(Y, Data, s=1)
z2 = np.polyfit(Y, Data, 3)
#print z
p2 = np.polyval(z2, Y)
a2, b2= chisquare(Data, f_exp=p2)
c2, d2 = stats.ks_2samp(Data, p2)
#print a, b
plt.scatter(Y, p2, color='g', s=1)
z3 = [0, 2, 0]
p3 = np.polyval(z3, Y)
a3, b3= chisquare(Data, f_exp=p3)
c3, d3 = stats.ks_2samp(Data, p3)
#print a, b
plt.scatter(Y, p3, color='y', s=1)
plt.savefig('data_fit2.pdf')
print '####################################################################'
print "Degree\t\tchisquare\t\t\tKalmagorov Smirnov"
print "%d\t%f\t%f\t%f\t%f" %(2, a1, b1, c1, d1)
print "%d\t%f\t%f\t%f\t%f" %(3, a2, b2, c2, d2)
print "%d\t%f\t%f\t%f\t%f" %(1, a3, b3, c3, d3)
print '####################################################################'
