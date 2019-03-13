import os
import h5py
import numpy as np
import math
import time
import glob
import sys
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')
from analysis.utils import GANutils as gan
from analysis.utils import ROOTutils as my
import setGPU
import keras.backend as K
import tensorflow as tf
import ROOT

def main():
   datapath = '/data/shared/gkhattak/EleMeasured3ThetaEscan/*.h5'
   outdir='results/bin_plots_scatter/'
   gan.safe_mkdir(outdir)
   Particles = ['Ele']
   datafiles = gan.GetDataFiles(datapath, Particles=Particles)
   maxval = 0
   count = 0
   limits_new= [0.08, 0.05, 0.035, 0.025, 0.018, 0.0122, 0.009, 0.006, 0.0025, 0]
   limits = [0.05, 0.03, 0.02, 0.0125, 0.008, 0.003, 0]
   for i, f in enumerate(datafiles[:1]):
      out = GetAngleData(f)
      bins=hist_count_new(out[0])
      if i==0:
         bin_data=[]
         for b in np.arange(bins.shape[1]):
            bin_data.append(bins[:, b])
         energies = out[1]
      else:
         for b in np.arange(bins.shape[1]):
            bin_data[b] =np.concatenate((bin_data[b], bins[:, b]), axis=0)
         energies = np.concatenate((energies, out[1]), axis=0)
   PlotBins(bin_data, energies, limits_new, outdir)

def PlotBins(bins, energies, limits, outdir):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    c1.SetGrid()
    p = [np.amin(energies), np.amax(energies)]
    profs=[]
    for i, b in enumerate(bins):
       label= "bin{}".format(i)
       maxb= np.amax(b)
       if i==len(bins)-1:
          minb = np.amin(b)
          maxb = maxb
       else:
          minb=0
          maxb= 1.1 * maxb
       profs.append(ROOT.TH2D(label, label, 100, p[0], p[1], 100, minb, maxb))
       prof=profs[i]
       prof.Sumw2()
       if i==0:
          blabel='greater than {} GeV'.format(limits[i]/50.)
       elif i==len(bins)-1:
          blabel='equal to {} GeV'.format(limits[i-1])
       elif i==len(bins)-2:
          blabel='from {} to {} GeV'.format(limits[i-1]/50., limits[i])
       else:
          blabel='from {} to {} GeV'.format(limits[i-1]/50., limits[i]/50.)
       prof.SetTitle("Bin Number {} for cell with deposition {}".format(i, blabel))
       prof.GetXaxis().SetTitle("Ep GeV")
       for j in np.arange(bins[i].shape[0]):
          prof.Fill(energies[j], bins[i][j])
       prof.GetYaxis().SetTitle("counts")
       prof.Draw('colz')
       c1.Update()
       c1.Print(os.path.join(outdir + label + '.pdf'))

def GetAngleData(datafile, datatype=['ECAL', 'energy']):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   out = []
   for d in datatype:
      out.append(np.array(f.get(d)))
   return out
                        
def hist_count(x):
      x=np.expand_dims(x, axis=4)
      #bin1 = np.sum(np.where(x> 0.2, 1, 0), axis=(1, 2, 3))
      #bin2 = np.sum(np.where((x<0.2) & (x>0.08) , 1, 0), axis=(1, 2, 3))
      bin3 = np.sum(np.where((x>0.05) , 1, 0), axis=(1, 2, 3))
      bin4 = np.sum(np.where((x<0.05) & (x>0.03), 1, 0), axis=(1, 2, 3))
      bin5 = np.sum(np.where((x<0.03) & (x>0.02), 1, 0), axis=(1, 2, 3))
      bin6 = np.sum(np.where((x<0.02) & (x>0.0125), 1, 0), axis=(1, 2, 3))
      bin7 = np.sum(np.where((x<0.0125) & (x>0.008), 1, 0), axis=(1, 2, 3))
      bin8 = np.sum(np.where((x<0.008) & (x>0.003), 1, 0), axis=(1, 2, 3))
      bin9 = np.sum(np.where((x<0.003) & (x>0.), 1, 0), axis=(1, 2, 3))
      bin10 = np.sum(np.where(x==0, 1, 0), axis=(1, 2, 3))
      return np.concatenate([bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10], axis=1)

# A histogram fucntion that counts cells in different bins
def hist_count_new(x, p=1):
   x=np.expand_dims(x, axis=4)
   bin1 = np.sum(np.where(x>(0.08**p) , 1, 0), axis=(1, 2, 3))
   bin2 = np.sum(np.where((x<(0.08**p)) & (x>(0.05**p)), 1, 0), axis=(1, 2, 3))
   bin3 = np.sum(np.where((x<(0.05**p)) & (x>(0.035**p)), 1, 0), axis=(1, 2, 3))
   bin4 = np.sum(np.where((x<(0.035**p)) & (x>(0.025**p)), 1, 0), axis=(1, 2, 3))
   bin5 = np.sum(np.where((x<(0.025**p)) & (x>(0.018**p)), 1, 0), axis=(1, 2, 3))
   bin6 = np.sum(np.where((x<(0.018**p)) & (x>(0.0122**p)), 1, 0), axis=(1, 2, 3))
   bin7 = np.sum(np.where((x<(0.0122**p)) & (x>(0.009**p)), 1, 0), axis=(1, 2, 3))
   bin8 = np.sum(np.where((x<(0.009**p)) & (x>(0.006**p)), 1, 0), axis=(1, 2, 3))
   bin9 = np.sum(np.where((x<(0.006**p)) & (x>(0.0025**p)), 1, 0), axis=(1, 2, 3))
   bin10 = np.sum(np.where((x<(0.0025**p)) & (x>0.), 1, 0), axis=(1, 2, 3))
   bin11 = np.sum(np.where(x==0, 1, 0), axis=(1, 2, 3))
   bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9, bin10, bin11], axis=1)
   bins[np.where(bins==0)]=1 # so that an empty bin will be assigned a count of 1 to avoid unstability
   return bins
                                                        

if __name__ == "__main__":
    main()
