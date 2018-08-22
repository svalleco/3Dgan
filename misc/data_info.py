
import sys
import os

import h5py
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
plt.switch_backend('Agg')      
                                                                                                                                                                                                  
def safe_mkdir(path):
    '''                                                                                                      Safe mkdir (i.e., don't create if already exists,                                                        and no violation of race conditions)                                                                     '''
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

def plot_max(array, index, out_file, num_fig, energy, bins):
     ## Plot the Histogram of Maximum energy deposition location on all axis                             
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= str(energy), normed=True)
   plt.legend()
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[0:index-1, 1], bins=bins, histtype='step', label=str(energy), normed=True)
   plt.legend()
   plt.xlabel('Position')

   plt.hist(array[0:index-1, 2], bins=bins, histtype='step', label=str(energy), normed=True)
   plt.legend(loc=1)
   plt.xlabel('Position')
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_energy(array, out_file, num_fig, energy):
   ### Plot Histogram of energy deposition along all three axis                                          
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(array[:, 0].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[:, 1].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()

   plt.subplot(223)
   plt.hist(array[:, 2].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()
   plt.savefig(out_file)

def plot_energy_hist(array, index, out_file, num_fig, energy):
   ### Plot total energy deposition cell by cell along x, y, z axis                                      
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index, 0].sum(axis = 0), label=str(energy))
   plt.legend()

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index, 1].sum(axis = 0), label=str(energy))
   plt.legend()

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index, 2].sum(axis = 0), label=str(energy))
   plt.legend()
   plt.savefig(out_file)

def plot_energy_mean(array, index, out_file, num_fig, energy):
   ### Plot total energy deposition cell by cell along x, y, z axis                                      
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index, 0].mean(axis = 0), label=str(energy))
   plt.legend()

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index, 1].mean(axis = 0), label=str(energy))
   plt.legend()

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index, 2].mean(axis = 0), label=str(energy))
   plt.legend()
   plt.savefig(out_file)

def read_data(data_file):
  if data_file== 0:
     d=h5py.File("ElectronEnergyFile.h5",'r')
     X=np.array(d.get('ECAL'))
     e=d.get('energy')
     Y=(np.array(e[:,0, 1]))
  if data_file== 1:
     d=h5py.File("/nfshome/gkhattak/Ele_v1_1_2.h5",'r')
     X=np.array(d.get('ECAL')[:10000])                                                  
     e=d.get('target')[:10000]
     Y=(np.array(e[:,1]))
  if data_file== 2:
     d=h5py.File("/afs/cern.ch/work/s/svalleco/public/Eshuffled100-200.h5",'r')
     X=np.array(d.get('ECAL'))
     e=d.get('energy')
     Y=(np.array(e[:,0, 1]))
  if data_file== 3:
     datapath = '/eos/project/d/dshep/LCD/V1/EleEscan'
     datafiles = sorted(glob.glob(datapath))
     print len(datafiles)
     dfile = datafiles[0]
     d=h5py.File(dfile,'r')
     X=np.array(d.get('ECAL'))                                                  
     Y=np.array(d.get('target'[:,1]))
     for f in range(1, 10):
       dfile = datafiles[f]
       d=h5py.File(dfile,'r')
       X=np.concatenate(X, d.get('ECAL'))  
       Y=np.concatenate(Y, np.array(d.get('target'[:,1])))
  if data_file== 4:
     d=h5py.File("/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5",'r')
     X=np.array(d.get('ECAL'))                                                  
     e=d.get('target')
     Y=(np.array(e[:,1]))
  if data_file== 5:
     d=h5py.File("/bigdata/shared/LCD/NewV1/Pi0Escan/Pi0Escan_1_1.h5",'r')
     X=np.array(d.get('ECAL'))
     e=d.get('target')
     Y=(np.array(e[:,1]))


  return(X,Y)


def plots(X, Y, num_events, outdir, d):
   # Initialization of parameters
   index50 = 0
   index100 = 0
   index150 = 0
   index200 = 0
   index300 = 0
   index400 = 0
   index500 = 0
   #Initialization of arrays
   events50 = np.zeros((num_events, 25, 25, 25))
   max_pos_50 = np.zeros((num_events, 3))
   events100 = np.zeros((num_events, 25, 25, 25))
   max_pos_100 = np.zeros((num_events, 3))
   events150 = np.zeros((num_events, 25, 25, 25))
   max_pos_150 = np.zeros((num_events, 3))
   events200 = np.zeros((num_events, 25, 25, 25))
   max_pos_200 = np.zeros((num_events, 3))
   events300 = np.zeros((num_events, 25, 25, 25))
   max_pos_300 = np.zeros((num_events, 3))
   events400 = np.zeros((num_events, 25, 25, 25))
   max_pos_400 = np.zeros((num_events, 3))
   events500 = np.zeros((num_events, 25, 25, 25))
   max_pos_500 = np.zeros((num_events, 3))
   sum_50 = np.zeros((num_events, 3, 25))
   sum_100 = np.zeros((num_events, 3, 25))
   sum_150 = np.zeros((num_events, 3, 25))
   sum_200 = np.zeros((num_events, 3, 25))
   sum_300 = np.zeros((num_events, 3, 25))
   sum_400 = np.zeros((num_events, 3, 25))
   sum_500 = np.zeros((num_events, 3, 25))


   size_data = int(X.shape[0])
   for i in range(size_data):
     if Y[i] > 45 and Y[i] > 55 and index50 < num_events:
        events50[index50] = X[i]
        index50 = index50 + 1
     elif Y[i] > 95 and Y[i] > 105 and index100 < num_events:
        events100[index100] = X[i]
        index100 = index100 + 1
     elif Y[i] > 145 and Y[i] > 155 and index150 < num_events:
        events150[index150] = X[i]
        index150 = index150 + 1
     elif Y[i] > 195 and Y[i] > 205 and index200 < num_events:
        events200[index200] = X[i]
        index200 = index200 + 1
     elif Y[i] > 295 and Y[i] > 305 and index300 < num_events:
        events300[index300] = X[i]
        index300 = index300 + 1
     elif Y[i] > 395 and Y[i] > 405 and index400 < num_events:
        events400[index400] = X[i]
        index400 = index400 + 1
     elif Y[i] > 495 and Y[i] > 505 and index500 < num_events:
       events500[index500] = X[i]
       index500 = index500 + 1

   for j in range(num_events):
     max_pos_50[j] = np.unravel_index(events50[j].argmax(), (25, 25, 25))
     max_pos_100[j] = np.unravel_index(events100[j].argmax(), (25, 25, 25))
     max_pos_150[j] = np.unravel_index(events150[j].argmax(), (25, 25, 25))
     max_pos_200[j] = np.unravel_index(events200[j].argmax(), (25, 25, 25))
     max_pos_300[j] = np.unravel_index(events300[j].argmax(), (25, 25, 25))
     max_pos_400[j] = np.unravel_index(events400[j].argmax(), (25, 25, 25))
     max_pos_500[j] = np.unravel_index(events500[j].argmax(), (25, 25, 25))
     sum_50[j, 0] = np.sum(events50[j], axis=(1,2))
     sum_50[j, 1] = np.sum(events50[j], axis=(0,2))
     sum_50[j, 2] = np.sum(events50[j], axis=(0,1))
     sum_100[j, 0] = np.sum(events100[j], axis=(1,2))
     sum_100[j, 1] = np.sum(events100[j], axis=(0,2))
     sum_100[j, 2] = np.sum(events100[j], axis=(0,1))
     sum_150[j, 0] = np.sum(events150[j], axis=(1,2))
     sum_150[j, 1] = np.sum(events150[j], axis=(0,2))
     sum_150[j, 2] = np.sum(events150[j], axis=(0,1))
     sum_200[j, 0] = np.sum(events200[j], axis=(1,2))
     sum_200[j, 1] = np.sum(events200[j], axis=(0,2))
     sum_200[j, 2] = np.sum(events200[j], axis=(0,1))
     sum_300[j, 0] = np.sum(events300[j], axis=(1,2))
     sum_300[j, 1] = np.sum(events300[j], axis=(0,2))
     sum_300[j, 2] = np.sum(events300[j], axis=(0,1))
     sum_400[j, 0] = np.sum(events400[j], axis=(1,2))
     sum_400[j, 1] = np.sum(events400[j], axis=(0,2))
     sum_400[j, 2] = np.sum(events400[j], axis=(0,1))
     sum_500[j, 0] = np.sum(events500[j], axis=(1,2))
     sum_500[j, 1] = np.sum(events500[j], axis=(0,2))
     sum_500[j, 2] = np.sum(events500[j], axis=(0,1))
   print(index50, index100, index150, index200, index300, index400, index500)
   #### Generate a table to screen
   print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t"
   print "50 \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(index50, np.amax(events50), str(np.unravel_index(events50.argmax(), (index50, 25, 25, 25))), np.mean(events50), np.amin(events50))
   print "100 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index100, np.amax(events100), str(np.unravel_index(events100.argmax(), (index100, 25, 25, 25))), np.mean(events100), np.amin(events100))
   print "150 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index150, np.amax(events150), str(np.unravel_index(events150.argmax(), (index150, 25, 25, 25))), np.mean(events150), np.amin(events150))
   print "200 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index200, np.amax(events200), str(np.unravel_index(events200.argmax(), (index200, 25, 25, 25))), np.mean(events200), np.amin(events200))
   print "300 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index300, np.amax(events300), str(np.unravel_index(events300.argmax(), (index300, 25, 25, 25))), np.mean(events300), np.amin(events300))
   print "400 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index400, np.amax(events400), str(np.unravel_index(events400.argmax(), (index400, 25, 25, 25))), np.mean(events400), np.amin(events400))
   print "500 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index500, np.amax(events500), str(np.unravel_index(events500.argmax(), (index500, 25, 25, 25))), np.mean(events500), np.amin(events500))
   
   safe_mkdir(outdir)
   bins = np.arange(0, 25, 1)
   outfile = os.path.join(outdir, 'Position_of_max.pdf')
   plot_max(max_pos_50, index50, outfile, 1, 50, bins)
   plot_max(max_pos_100, index100, outfile, 1, 100, bins)
   plot_max(max_pos_150, index150, outfile, 1, 150, bins)
   plot_max(max_pos_200, index200, outfile, 1, 200, bins)
   plot_max(max_pos_300, index300, outfile, 1, 300, bins)
   plot_max(max_pos_400, index400, outfile, 1, 400, bins)            
   plot_max(max_pos_500, index500, outfile, 1, 500, bins)
   
   plot_energy(sum_50, os.path.join(outdir, 'Flat_energy_50.pdf'), 2, 50)
   plot_energy(sum_100, os.path.join(outdir, 'Flat_energy_100.pdf'), 3, 100)
   plot_energy(sum_150, os.path.join(outdir, 'Flat_energy_150.pdf'), 4, 150)
   plot_energy(sum_200, os.path.join(outdir, 'Flat_energy_200.pdf'), 5, 200)
   plot_energy(sum_300, os.path.join(outdir, 'Flat_energy_300.pdf'), 6, 300)
   plot_energy(sum_400, os.path.join(outdir, 'Flat_energy_400.pdf'), 7, 400)
   plot_energy(sum_500, os.path.join(outdir, 'Flat_energy_500.pdf'), 8, 500)

   plot_energy_hist(sum_50, index50, os.path.join(outdir, 'hist_50.pdf'), 9, 50)
   plot_energy_hist(sum_100, index100, os.path.join(outdir, 'hist_100.pdf'), 10, 100)
   plot_energy_hist(sum_150, index150, os.path.join(outdir, 'hist_150.pdf'), 11, 150)
   plot_energy_hist(sum_200, index200, os.path.join(outdir, 'hist_200.pdf'), 12, 200)
   plot_energy_hist(sum_300, index300, os.path.join(outdir, 'hist_300.pdf'), 13, 300)
   plot_energy_hist(sum_400, index400, os.path.join(outdir, 'hist_400.pdf'), 14, 400)
   plot_energy_hist(sum_500, index500, os.path.join(outdir, 'hist_500.pdf'), 15, 500)

   plot_energy_hist(sum_50, index50, os.path.join(outdir, 'hist_all.pdf'), 16, 50)
   plot_energy_hist(sum_100, index100, os.path.join(outdir, 'hist_all.pdf'), 16, 100)
   plot_energy_hist(sum_150, index150, os.path.join(outdir, 'hist_all.pdf'), 16, 150)
   plot_energy_hist(sum_200, index200, os.path.join(outdir, 'hist_all.pdf'), 16, 200)
   plot_energy_hist(sum_300, index300, os.path.join(outdir, 'hist_all.pdf'), 16, 300)
   plot_energy_hist(sum_400, index400, os.path.join(outdir, 'hist_all.pdf'), 16, 400)
   plot_energy_hist(sum_500, index500, os.path.join(outdir, 'hist_all.pdf'), 16, 500)

   plot_energy_mean(sum_50, index50, os.path.join(outdir, 'hist_mean_all.pdf'), 17, 50)
   plot_energy_mean(sum_100, index100, os.path.join(outdir, 'hist_mean_all.pdf'), 17, 100)
   plot_energy_mean(sum_150, index150, os.path.join(outdir, 'hist_mean_all.pdf'), 17, 150)
   plot_energy_mean(sum_200, index200, os.path.join(outdir, 'hist_mean_all.pdf'), 17, 200)
   plot_energy_mean(sum_300, index300, os.path.join(outdir, 'hist_mean_all.pdf'), 17, 300)
   plot_energy_mean(sum_400, index400, os.path.join(outdir, 'hist_mean_all.pdf'), 17, 400)
   plot_energy_mean(sum_500, index500, os.path.join(outdir, 'hist_mean_all.pdf'), 17, 500)

   plt.figure(18)
   plt.title('Primary Energy')
   ebins=np.arange(0, 500, 10)
   plt.hist(Y, bins=ebins, histtype='step', label= str(d),normed=True)
   plt.savefig(os.path.join(outdir, 'Incoming_energy_histogram.pdf'))
            
   plt.figure(19)
   plt.title('Ecal Energy')
   plt.hist(np.sum(X, axis=(1, 2, 3)), bins='auto', histtype='step', normed=True, label= str(d)) 
   plt.savefig(os.path.join(outdir, 'ECAL_histogram.pdf'))
                                 
   plt.figure(18)
   plt.title('Total Ecal Energy')
   plt.hist(np.multiply(50, np.sum(X, axis=(1, 2, 3))), bins=ebins, histtype='step', label=str(d) , normed=True)
   plt.legend(loc=8)
   plt.savefig(os.path.join(outdir, 'Combined_histogram.pdf'))

def main():
   data_file= [4,5]
   outdir= 'pi0_info_plots'
   num_events = 100
   for d in data_file:
      X, Y = read_data(d)
      plots(X, Y, num_events, outdir, d)
if __name__ == "__main__":
    main()
