from os import path
#from ROOT import TCanvas, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import glob
import numpy.core.umath_tests as umath
import matplotlib.lines as mlines

#Architectures 
from ecalvegan import generator, discriminator
plt.switch_backend('Agg')

disc_weights="pionweights/params_discriminator_epoch_029.hdf5"
gen_weights= "pionweights/params_generator_epoch_029.hdf5"
#disc_weights = "lr008_disc.hdf5"
#gen_weights = "lr008_gen.hdf5"

plots_dir = "pion_ep30/"
latent = 200
num_data = 100000
num_events = 2000
m = 3
scale = 100

datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path                                         
#datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5'
sortedpath = 'sorted_*.hdf5'
Test = False

save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
read_data = False # True if previously sorted data is saved 
save_gen = False # True if saving generated data. 
read_gen = False # True if generated data is already saved and can be loaded
save_disc = False # True if discriminiator data is to be saved
read_disc = False # True if discriminated data is to be loaded

def plot_max(array, out_file, num_fig, plot_label, pos=0):
   ## Plot the Histogram of Maximum energy deposition location on all axis                             
   max_x = array[:,0]
   bins = np.arange(0, 25, 1)
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   label= plot_label + '\n{:.2f}'.format(np.mean(array[:,0]))+ '({:.2f})'.format(np.std(array[:,0])) 
   plt.hist(array[:,0], bins=bins, histtype='step', label= label, normed=1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   label= plot_label + '\n{:.2f}'.format(np.mean(array[:,1]))+ '({:.2f}'.format(np.std(array[:,1])) 
   plt.hist(array[:,1], bins=bins, histtype='step', label=label, normed=1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.xlabel('Position')

   plt.subplot(223)
   label= plot_label + '\n{:.2f}'.format(np.mean(array[:,2]))+ '({:.2f})'.format(np.std(array[:,2])) 
   plt.hist(array[:,2], bins=bins, histtype='step', label=label, normed=1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.xlabel('Position')
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_flat_energy(arrayx, arrayy, arrayz, out_file, num_fig, plot_label):
   ### Plot Histogram of energy flat distribution along all three axis                                                
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(arrayx.flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend()
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(arrayy.flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend()
   plt.xlabel('Energy')

   plt.subplot(223)
   plt.hist(arrayz.flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend()
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_energy(array, out_file, num_fig, plot_label, color='blue', pos=0):
   ### Plot Histogram of Primary energy                                                                                       
   plt.figure(num_fig)
   ebins=np.arange(0, 600, 5)
   label= plot_label + '\n{:.2f}'.format(np.mean(array))+ '({:.2f})'.format(np.std(array)) 
   plt.hist(array, bins=ebins, histtype='step', label=label, color=color, normed=1)
   plt.xticks([0, 10, 50, 100, 150, 200, 300, 400, 500, 600])
   plt.xlabel('Energy GeV')
   plt.ylabel('Events')
   pos = 0 if energy <= 300 else 2
   plt.legend(title='   Mean(std)', loc=pos)
   plt.savefig(out_file)
   
def plot_energy_axis_log(arrayx, arrayy, arrayz, out_file, num_fig, plot_label, pos=0):
   ### Plot total energy deposition cell by cell along x, y, z axis on log scale                                                   
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   sumx_array = np.mean(arrayx, axis=0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumx_array))+ '({:.2f})'.format(np.std(sumx_array))
   plt.semilogy(sumx_array, label=plot_label)
   plt.ylabel('ECAL Energy')
   plt.xlim(xmax=25)
   plt.legend(loc=2, fontsize='x-small')

   plt.subplot(222)
   plt.title('Y-axis')
   sumy_array =np.mean(arrayy, axis=0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumy_array))+ '({:.2f})'.format(np.std(sumy_array))
   plt.semilogy(sumy_array, label=plot_label)
   plt.legend(loc=2, fontsize='small')
   plt.xlabel('Position')
   plt.xlim(xmax=25)

   plt.subplot(223)
   sumz_array =np.mean(arrayz, axis=0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumz_array))+ '({:.2f})'.format(np.std(sumz_array))
   plt.semilogy( sumz_array, label=plot_label)
   plt.legend(loc=8, fontsize='small')
   plt.xlabel('Z axis Position')
   plt.ylabel('ECAL Energy')
   plt.xlim(xmax=25)
   plt.savefig(out_file)

def plot_energy_wt_hist(arrayx, arrayy, arrayz, out_file, num_fig, plot_label, pos=0):
   ### Plot total energy deposition cell by cell along x, y, z axis as histogram                                         
   bins = np.arange(0, 25, 1)
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   sumx_array = arrayx.sum(axis=0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumx_array))+ '({:.2f})'.format(np.std(sumx_array))
   plt.hist(bins, weights = sumx_array, label=plot_label, bins = bins, histtype='step', normed=1)
   plt.legend(loc=2, fontsize='xx-small')

   plt.subplot(222)
   plt.title('Y-axis')
   sumy_array = arrayy.sum(axis=0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumy_array))+ '({:.2f})'.format(np.std(sumy_array))
   plt.hist(bins, weights = sumy_array, label=plot_label, bins = bins, histtype='step', normed=1)
   plt.legend(loc=2, fontsize='small')
   plt.xlabel('Position')

   plt.subplot(223)
   sumz_array =arrayz.sum(axis=0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumz_array))+ '({:.2f})'.format(np.std(sumz_array))
   plt.hist(bins, weights = sumz_array, label=plot_label, bins = bins, histtype='step', normed=1)
   plt.legend(loc=8, fontsize='small')
   plt.xlabel('Z axis Position')
   plt.ylabel('ECAL Energy')

   plt.savefig(out_file)

def plot_energy_axis(arrayx, arrayy, arrayz, out_file, num_fig, plot_label, pos=0):
   ### Plot total energy deposition cell by cell along x, y, z axis                                                                                                                                                          
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   sumx_array = arrayx.sum(axis=0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumx_array))+ '({:.2f})'.format(np.std(sumx_array))
   plt.plot(sumx_array, label=plot_label)
   plt.ylabel('ECAL Energy')
   plt.legend(loc=2, fontsize='xx-small')

   plt.subplot(222)
   plt.title('Y-axis')
   sumy_array =arrayy.sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumy_array))+ '({:.2f})'.format(np.std(sumy_array))
   plt.plot(sumy_array, label=plot_label)
   plt.legend(loc=2, fontsize='small')
   plt.xlabel('Position')

   plt.subplot(223)
   sumz_array =arrayz.sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumz_array))+ '({:.2f})'.format(np.std(sumz_array))
   plt.plot( sumz_array, label=plot_label)
   plt.legend(loc=8, fontsize='small')
   plt.xlabel('Z axis Position')
   plt.ylabel('ECAL Energy')

   plt.savefig(out_file)

def plot_energy_mean(arrayx, arrayy, arrayz, out_file, num_fig, plot_label):
   ### Plot total energy deposition cell by cell along x, y, z axis                                   

   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(arrayx.mean(axis = 0), label=plot_label)
   plt.legend()
   plt.ylabel('Mean Energy')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(arrayy.mean(axis = 0), label=plot_label)
   plt.legend()
   plt.xlabel('Position')

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(arrayz.mean(axis = 0), label=plot_label)
   plt.xlabel('Position')
   plt.legend()
   plt.ylabel('Mean Energy')
   plt.savefig(out_file)

def plot_real(array, out_file, num_fig, plot_label):
   ## Plot the disc real/fake flag                                                                                    
   plt.figure(num_fig)
   bins = np.arange(0, 1, 0.01)
   plt.figure(num_fig)
   plt.title('Real/ Fake')
   label= plot_label + '\n{:.2f}'.format(np.mean(array))+ '({:.2f})'.format(np.std(array))
   plt.hist(array, bins=bins, histtype='step', label= label, normed=1)
   plt.legend()
   plt.ylabel('Events')
   plt.xlabel('Real/fake')
   plt.savefig(out_file)

def plot_error(error, out_file, num_fig, plot_label, pos=2):
   # plot error for discriminator
   plt.figure(num_fig)
   bins = np.linspace(-200, 200, 30)
   label= plot_label + '\n{:.2f}'.format(np.mean(np.absolute(error))) + '({:.2f})'.format(np.std(np.absolute(error)))
   plt.hist(error, bins=bins, histtype='step', label=label)
   plt.xlabel('error GeV')
   plt.ylabel('Number of events')
   plt.legend(title='                       Mean     ( std )', loc=pos)
   plt.savefig(out_file)

def plot_ecal_error(array, out_file, num_fig, plot_label, pos=2):
   # plot error in ecal sum with Ep                                                                                          
   plt.figure(num_fig)
   bins = np.linspace(-150, 150, 30)
   diffarray = array
   label= plot_label + '\n{:.2f}'.format(np.mean(np.absolute(diffarray))) + '({:.2f})'.format(np.std(np.absolute(diffarray)))
   plt.hist(diffarray, bins=bins, histtype='step', label=label)
   plt.xlabel('error GeV')
   plt.ylabel('Number of events')
   plt.legend(title='                       Mean     ( std )', loc=pos)
   plt.savefig(out_file)

def plot_ecal(array, out_file, num_fig, plot_label):
   # plot ecal sum                                                                                                    
   bins = np.arange(0, 600, 5)
   plt.figure(num_fig)
   ecal_array= np.multiply(50, array)
   label= plot_label + '\n{:.2f}'.format(np.mean(ecal_array))+ '({:.2f})'.format(np.std(ecal_array))
   plt.title('ECAL SUM')
   plt.xlabel('ECAL SUM GeV')
   plt.ylabel('Events')
   plt.hist(ecal_array, bins=bins, histtype='step', label=label, normed=1)
   pos = 0 if energy <= 300 else 2                                      
   plt.legend(loc=pos)
   plt.savefig(out_file)
   
def plot_hits(array, out_file, num_fig, plot_label):
   # plot ecal sum                                                                                                                                        
   #bins = np.arange(0, 600, 5)
   plt.figure(num_fig)
   ecal_array= np.multiply(50, array)
   hit_array = ecal_array>0.01
   hits = np.sum(hit_array, axis=(1, 2, 3))
   label= plot_label + '\n{:.2f}'.format(np.mean(hits))+ '({:.2f})'.format(np.std(hits))
   plt.title('Num Hits')
   plt.xlabel('ECAL Hits')
   plt.ylabel('Events')
   plt.hist(hits, bins='auto', histtype='step', label=label, normed=1)
   pos = 0 if energy <= 300 else 2
   plt.legend(loc=pos)
   plt.savefig(out_file)

def plot_1to2(array, out_file, num_fig, plot_label):
   # plot ecal sum                                              
   bins = np.arange(0, 5, 0.1)
   plt.figure(num_fig)
   ecal_array1= np.sum(array[:,:,:,0:12], axis=(1, 2, 3))
   ecal_array2= np.sum(array[:,:,:,13:24], axis=(1, 2, 3))
   ratio = ecal_array1/ ecal_array2
   label= plot_label + '\n{:.2f}'.format(np.mean(ratio))+ '({:.2f})'.format(np.std(ratio))
   plt.title('ECAL RATIO LAYER 1 to 2')
   plt.xlabel('Layer1/Layer2')
   plt.ylabel('Events')
   plt.hist(ratio, bins=bins , histtype='step', label=label, normed=1)
   pos = 0 #if energy <= 300 else 2
   plt.legend(loc=pos)
   plt.savefig(out_file)

def plot_1tototal(array, out_file, num_fig, plot_label):
                                                                                                                      
   #bins = np.linspace(0, 0.01, 50)
   plt.figure(num_fig)
   ecal_array1=np.sum(array[:, :, :, 0:12], axis=(1, 2, 3))
   ecal_total=np.sum(array[:, :, :, :], axis=(1, 2, 3))
   ratio = ecal_array1/ ecal_total
   label= plot_label + '\n{:.2f}'.format(np.mean(ratio))+ '({:.2f})'.format(np.std(ratio))
   plt.title('ECAL RATIO LAYER 1 to total')
   plt.xlabel('Half1/Total')
   plt.ylabel('Events')
   plt.hist(ratio, bins='auto', histtype='step', label=label, normed=1)
   pos = 0 #if energy <= 300 else 2                                                                                                                        
   plt.legend(loc=pos)
   plt.savefig(out_file)

def plot_moment(array, out_file, num_fig, plot_label, moment):
   # plot error                         
   plt.figure(num_fig)
   bins = np.linspace(0, 4, 30)
   label= plot_label + '\n{:.2f}'.format(np.mean(array))+ '({:.2f})'.format(np.std(array))
   plt.hist(array, bins='auto', histtype='step', label=label, normed=1)
   plt.xlabel('moment_' + str(moment + 1))
   plt.ylabel('Number of events')
   plt.legend(title='                       Mean     ( std )')
   plt.savefig(out_file)

def plot_perror_relative(array, out_file, num_fig, energy, color, bar=1):
   # plot Energy error divided by Ep                                                                             
   plt.figure(num_fig)
   data = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=6, label='Data')
   gan = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='None', markersize=6, label='GAN')                                                                    
   if bar==0:
      plt.scatter(energy, np.mean(array), color =color)
   else:
      plt.errorbar(energy, np.mean(array), yerr= np.std(array), fmt='o', color =color, elinewidth=1 , capsize=4)
   plt.ylabel('Ep - Er / Ep')
   plt.xlabel('Energy GeV')
   #plt.ylim(ymax = 0.2, ymin = -0.2)
   plt.legend( fontsize='x-small', handles = [data, gan])
   plt.savefig(out_file)

def plot_ecal_relative(array, out_file, num_fig,  energy, color, bar=1):
   # plot ecal error divided by Ep                                                                           
   plt.figure(num_fig)
   data = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=6, label='Data')
   gan = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='None', markersize=6, label='GAN')
   if bar==0:
      plt.scatter(energy, np.mean(array), color=color)
   else:
      plt.errorbar(energy, np.mean(array), yerr= np.std(array), fmt='o',  elinewidth=1 , capsize=4, color=color)
   plt.ylabel('Ep - Ecal / Ep')
   plt.xlabel('Energy GeV')
   #plt.ylim(ymax = 0.3, ymin = -0.3)
   plt.legend(handles = [data, gan], fontsize='x-small')
   plt.savefig(out_file)

def plot_ecal_ratio(array, out_file, num_fig, energy, color, bar=1):
   # plot Ecal/Ep  
   plt.figure(num_fig)
   data = mlines.Line2D([], [], color='tab:blue', marker='o', linestyle='None', markersize=6, label='Data')
   gan = mlines.Line2D([], [], color='tab:orange', marker='o', linestyle='None', markersize=6, label='GAN')
   if bar==0:
      plt.scatter(energy, np.mean(array), color =color)
   else:
      plt.errorbar(energy, np.mean(array), yerr= np.std(array), fmt='o', color =color)
   plt.ylabel('Ep/Ecal')
   plt.xlabel('Energy')
   #plt.ylim(ymax = 0.1, ymin = -1.5)
   plt.legend(handles = [data, gan], fontsize='x-small')
   plt.savefig(out_file)

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=200000, EventsperFile = 10000, Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))  
    print ("Found {} files. ".format(len(Files)))
    Filesused = int(math.ceil(nEvents/EventsperFile))
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
    out=[]
    for j in range(len(Fractions)):
        out.append([])
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName][:Filesused]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out

def get_data(datafile):
    #get data for training                                                                                                                                                                      
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=f.get('target')
    x=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    x[x < 1e-6] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    ecal = np.sum(x, axis=(1, 2, 3))
    return x, y, ecal

def sort(data, energies):
    X = data[0]
    Y = data[1]
    tolerance = 5
    srt = {}
    for energy in energies:
       indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
       if len(indexes) > num_events:
          indexes = indexes[:num_events]
       srt["events_act" + str(energy)] = X[indexes]
       srt["energy" + str(energy)] = Y[indexes]
    return srt

def save_sorted(srt, energies):
    for energy in energies:
       filename = "sorted_{:03d}.hdf5".format(energy)
       with h5py.File(filename ,'w') as outfile:
          outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
          outfile.create_dataset('Target',data=srt["energy" + str(energy)])
       print "Sorted data saved to ", filename

def save_generated(events, sampled_energies, energy):
    filename = "Gen_{:03d}.hdf5".format(energy)
    with h5py.File(filename ,'w') as outfile:
       outfile.create_dataset('ECAL',data=events)
       outfile.create_dataset('Target',data=sampled_energies)
    print "Generated data saved to ", filename

def save_discriminated(disc, energies):
    for energy in energies:
      filename = "Disc_{:03d}.hdf5".format(energy)
      with h5py.File(filename ,'w') as outfile:
        outfile.create_dataset('ISREAL_ACT',data=disc["isreal_act" + str(energy)])
        outfile.create_dataset('ISREAL_GAN',data=disc["isreal_gan" + str(energy)])
        outfile.create_dataset('AUX_ACT',data=disc["aux_act" + str(energy)])
        outfile.create_dataset('AUX_GAN',data=disc["aux_gan" + str(energy)])
        outfile.create_dataset('ECAL_ACT',data=disc["ecal_act" + str(energy)])
        outfile.create_dataset('ECAL_GAN',data=disc["ecal_gan" + str(energy)])
      print "Discriminated data saved to ", filename

def get_disc(energy):
    filename = "Disc_{:03d}.hdf5".format(energy)
    f=h5py.File(filename,'r')
    isreal_act = np.array(f.get('ISREAL_ACT'))
    isreal_gan = np.array(f.get('ISREAL_GAN'))
    aux_act = np.array(f.get('AUX_ACT'))
    aux_gan = np.array(f.get('AUX_GAN'))
    ecal_act = np.array(f.get('ECAL_ACT'))
    ecal_gan = np.array(f.get('ECAL_GAN'))
    print "Discriminated file ", filename, " is loaded"
    return isreal_act, aux_act, ecal_act, isreal_gan, aux_gan, ecal_gan

def load_sorted(sorted_path):
    sorted_files = sorted(glob.glob(sorted_path))
    energies = []
    srt = {}
    for f in sorted_files:
       energy = int(filter(str.isdigit, f)[:-1])
       energies.append(energy)
       srtfile = h5py.File(f,'r')
       srt["events_act" + str(energy)] = np.array(srtfile.get('ECAL'))
       srt["energy" + str(energy)] = np.array(srtfile.get('Target'))
       print "Loaded from file", f
    return energies, srt
 
def get_gen(energy):
    filename = "Gen_{:03d}.hdf5".format(energy)
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    print "Generated file ", filename, " is loaded"
    return generated_images

def generate(g, index, sampled_labels):
    noise = np.random.normal(0, 1, (index, latent))
    sampled_labels=np.expand_dims(sampled_labels, axis=1)
    gen_in = sampled_labels * noise
    generated_images = g.predict(gen_in, verbose=False, batch_size=50)
    return generated_images

def discriminate(d, images):
    isreal, aux_out, ecal_out = np.array(d.predict(images, verbose=False, batch_size=50))
    return isreal, aux_out, ecal_out

def get_max(images):
    index = images.shape[0]
    max_pos = np.zeros((index, 3)) 
    for i in range(index):
       max_p = images[i].argmax()
       max_loc = np.unravel_index(max_p, (25, 25, 25))
       max_pos[i] = max_loc
    return max_pos

def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

def get_moments(images, sumsx, sumsy, sumsz, totalE, m):
    ecal_size = 25
    totalE = np.squeeze(totalE)
    index = images.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = umath.inner1d(sumsx, moments) /totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = umath.inner1d(sumsy, moments) /totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = umath.inner1d(sumsz, moments)/totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

def safe_mkdir(path):
    '''                                                                                                            
    Safe mkdir (i.e., don't create if already exists,                                                              
    and no violation of race conditions)                                                                           
    '''
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

if __name__ == '__main__':
    
    var = {}
    if read_data:
       start = time.time()
       energies, var = load_sorted(sortedpath)
       sort_time = time.time()- start
       print "Events were loaded in {} seconds".format(sort_time)
    else:
       # Getting Data
       events_per_file = 10000
       energies = [50, 100, 200, 250, 300, 400, 500]
       Trainfiles, Testfiles = DivideFiles(datapath, nEvents=num_data, EventsperFile = events_per_file, datasetnames=["ECAL"], Particles =["Pi0"]) 
       if Test:
          data_files = Testfiles
       else:
          data_files = Trainfiles + Testfiles
       start = time.time()
       for index, dfile in enumerate(data_files):
          data = get_data(dfile)
          sorted_data = sort(data, energies)
          data = None
          if index==0:
             var.update(sorted_data)
          else:
             for key in var:
               var[key]= np.append(var[key], sorted_data[key], axis=0)
       data_time = time.time() - start
       print "{} events were loaded in {} seconds".format(num_data, data_time)
       if save_data:
          save_sorted(var, energies)        
    total = 0
    for energy in energies:
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
      total += var["index" + str(energy)]
      data_time = time.time() - start
    print "{} events were put in {} bins".format(total, len(energies))
    if read_gen:
      for energy in energies:
        var["events_gan" + str(energy)]= get_gen(energy)
    else:
      g = generator(latent)
      g.load_weights(gen_weights)
      start = time.time()
      for energy in energies:
        var["events_gan" + str(energy)] = generate(g, var["index" + str(energy)], var["energy" + str(energy)]/100)/scale
        if save_gen:
          save_generated(var["events_gan" + str(energy)], var["energy" + str(energy)], energy)
      gen_time = time.time() - start
      print "Generator took {} seconds to generate {} events".format(gen_time, total)
    if read_disc:
      for energy in energies:
        var["isreal_act" + str(energy)], var["aux_act" + str(energy)], var["ecal_act"+ str(energy)], var["isreal_gan" + str(energy)], var["aux_gan" + str(energy)], var["ecal_gan"+ str(energy)]= get_disc(energy)
    else: 
      d = discriminator()
      d.load_weights(disc_weights)
      start = time.time()
      for energy in energies:
        var["isreal_act" + str(energy)], var["aux_act" + str(energy)], var["ecal_act"+ str(energy)]= discriminate(d, var["events_act" + str(energy)])
        var["isreal_gan" + str(energy)], var["aux_gan" + str(energy)], var["ecal_gan"+ str(energy)]= discriminate(d, var["events_gan" + str(energy)])
      disc_time = time.time() - start
      print "Discriminator took {} seconds for {} data and generated events".format(disc_time, total)
      
      if save_disc:
        save_discriminated(var, energies)
    for energy in energies:
      print 'Calculations for ....', energy
      var["isreal_act" + str(energy)], var["aux_act" + str(energy)], var["ecal_act"+ str(energy)]= np.squeeze(var["isreal_act" + str(energy)]), np.squeeze(var["aux_act" + str(energy)]), np.squeeze(var["ecal_a\
ct"+ str(energy)])
      var["isreal_gan" + str(energy)], var["aux_gan" + str(energy)], var["ecal_gan"+ str(energy)]= np.squeeze(var["isreal_gan" + str(energy)]), np.squeeze(var["aux_gan" + str(energy)]), np.squeeze(var["ecal_g\
an"+ str(energy)])
      var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)])
      var["max_pos_gan" + str(energy)] = get_max(var["events_gan" + str(energy)])
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["events_act" + str(energy)], var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m)
      var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["events_gan" + str(energy)], var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m)
      var["Perror_act" + str(energy)] = (var["energy"+ str(energy)] - 100 * var["aux_act" + str(energy)])
      var["Perror_gan" + str(energy)] = (var["energy"+ str(energy)] - 100 * var["aux_gan" + str(energy)])
      var["Pnerror_act" + str(energy)] = var["Perror_act" + str(energy)]/var["energy"+ str(energy)]
      var["Pnerror_gan" + str(energy)] = var["Perror_gan" + str(energy)]/var["energy"+ str(energy)]
      var["Eerror_act" + str(energy)] = (var["energy"+ str(energy)] - 50 * var["ecal_act" + str(energy)])
      var["Eerror_gan" + str(energy)] = (var["energy"+ str(energy)] - 50 * var["ecal_gan" + str(energy)])
      var["Enerror_act" + str(energy)] = var["Eerror_act" + str(energy)]/var["energy"+ str(energy)]
      var["Enerror_gan" + str(energy)] = var["Eerror_gan" + str(energy)]/var["energy"+ str(energy)]
      var["Eratio_act" + str(energy)] = var["energy"+ str(energy)]/var["ecal_act" + str(energy)]
      var["Eratio_gan" + str(energy)] = var["energy"+ str(energy)]/ var["ecal_gan" + str(energy)]
    #### Generate Data table to screen                                                                                 
    print "Actual Data"
    print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2"
    for energy in energies:
      print "{}\t{}\t{:.4f}\t\t{}\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1]))

    #### Generate GAN table to screen                                                                                  
    print "Generated Data"
    print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2"
    for energy in energies:
      print "{}\t{}\t{:.4f}\t\t{}\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]), np.mean(var["max_pos_gan" + str(energy)], axis=0), np.mean(var["events_gan" + str(energy)]), np.mean(var["momentX_gan"+ str(energy)][:, 1]), np.mean(var["momentY_gan"+ str(energy)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)][:, 1]))

    ## Make folders for plots                                                                                          
    discdir = plots_dir + 'disc_outputs'
    safe_mkdir(discdir)
    actdir = plots_dir + 'Actual'
    safe_mkdir(actdir)
    gendir = plots_dir + 'Generated'
    safe_mkdir(gendir)
    comdir = plots_dir + 'Combined'
    safe_mkdir(comdir)
    mdir = plots_dir + 'Moments'
    safe_mkdir(mdir)
fig = 1
start = time.time()
## Make seperate plots
for energy in energies:
   maxfile = "Position_of_max_" + str(energy) + ".pdf"
   histfile = "hist_" + str(energy) + ".pdf"
   ecalfile = "ecal_" + str(energy) + ".pdf"
   energyfile = "energy_" + str(energy) + ".pdf"
   drealfile = "realfake_act" + str(energy) + ".pdf"
   grealfile = "realfake_gan" + str(energy) + ".pdf"
   derrorfile = "error_act" + str(energy) + ".pdf"
   gerrorfile = "error_gan" + str(energy) + ".pdf"
   dlabel = "Data " + str(energy)
   glabel = "Data " + str(energy)
   ecalerrorfile = "ecal_error" + str(energy) + ".pdf"
   plot_max(var["max_pos_act" + str(energy)], os.path.join(actdir, maxfile), fig, dlabel)
   fig+=1
   plot_energy_axis(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(actdir, histfile), fig, dlabel)
   fig+=1
   plot_ecal(var["ecal_act" + str(energy)], os.path.join(actdir, ecalfile), fig, dlabel)
   fig+=1
   plot_max(var["max_pos_gan" + str(energy)],  os.path.join(gendir, maxfile), fig, glabel)
   fig+=1
   plot_energy_axis(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(gendir, histfile), fig, glabel)
   fig+=1
   plot_ecal(var["ecal_gan" + str(energy)], os.path.join(gendir, ecalfile), fig, glabel)
   fig+=1
   #plot_ecal_error(var["Eerror_act" + str(energy)], os.path.join(actdir, ecalerrorfile), fig, dlabel)
   #fig+=1
   #plot_ecal_error(var["Eerror_gan" + str(energy)], os.path.join(gendir, ecalerrorfile), fig, glabel)
   #fig+=1
   plot_energy(np.multiply(100, var["aux_act" + str(energy)]), os.path.join(actdir, energyfile), fig, dlabel, 'green')
   fig+=1
   plot_real(var["isreal_act" + str(energy)], os.path.join(discdir, drealfile), fig, glabel)
   fig+=1
   plot_error(var["Perror_act" + str(energy)], os.path.join(discdir, derrorfile), fig, dlabel, pos=0)
   fig+=1
   plot_energy(np.multiply(100, var["aux_gan" + str(energy)]), os.path.join(gendir, energyfile), fig, glabel, 'green')
   fig+=1
   plot_real(var["isreal_gan" + str(energy)], os.path.join(discdir, grealfile), fig, glabel)
   fig+=1
   plot_error(var["Perror_gan" + str(energy)],  os.path.join(discdir, gerrorfile), fig, glabel, pos=0)
   fig+=1
plt.close("all")   
## Make combined plots
for energy in energies:

   flatfile = "Flat_energy_all.pdf"
   histallfile = "hist_all.pdf"
   meanallfile = "hist_mean_all.pdf"
   Efile = "error_normalized.pdf"
   Ecalfile = "ecal_normalized.pdf"
   Ecalratiofile = "ecal_ratio.pdf"
   dlabel = "Data " + str(energy)
   glabel = "GAN " + str(energy)
   plot_flat_energy(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(actdir, flatfile), fig, dlabel)
   plot_energy_wt_hist(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(actdir, "w_" + histallfile), fig+1, dlabel)
   plot_energy_axis(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(actdir, histallfile), fig+2, dlabel)
   plot_energy_mean(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(actdir, meanallfile), fig+3, dlabel)
   plot_flat_energy(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(gendir, flatfile), fig + 4, glabel)
   plot_energy_wt_hist(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(gendir, histallfile), fig+5, glabel)
   plot_energy_axis(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(gendir, "w_" + histallfile), fig+6, glabel)
   plot_energy_mean(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(gendir, meanallfile), fig+7, glabel)
   plot_perror_relative(var["Pnerror_act" + str(energy)], os.path.join(actdir, Efile), fig + 8, energy, 'tab:blue', 1)
   plot_perror_relative(var["Pnerror_gan" + str(energy)], os.path.join(gendir, Efile), fig + 9, energy, 'tab:orange', 1)
   plot_perror_relative(var["Pnerror_act" + str(energy)], os.path.join(comdir, Efile), fig + 10, energy, 'tab:blue')
   plot_perror_relative(var["Pnerror_gan" + str(energy)], os.path.join(comdir, Efile), fig + 10, energy, 'tab:orange')
   plot_ecal_relative(var["Enerror_act" + str(energy)], os.path.join(actdir, Ecalfile), fig+11, energy, 'tab:blue', 1)
   plot_ecal_relative(var["Enerror_gan" + str(energy)], os.path.join(gendir, Ecalfile), fig+12, energy, 'tab:orange', 1)
   plot_ecal_relative(var["Enerror_act" + str(energy)], os.path.join(comdir, Ecalfile), fig+13, energy, 'tab:blue')
   plot_ecal_relative(var["Enerror_gan" + str(energy)], os.path.join(comdir, Ecalfile), fig+13, energy, 'tab:orange')
   plot_ecal_ratio(var["Eratio_act" + str(energy)], os.path.join(actdir, Ecalratiofile), fig+14, energy, 'tab:blue')
   plot_ecal_ratio(var["Eratio_gan" + str(energy)], os.path.join(gendir, Ecalratiofile), fig+15, energy, 'tab:orange')

plt.close("all")
fig+=16
## Make superimposed plots                                                                                                                                                                                       
for energy in energies:
   maxfile = "Position_of_max_" + str(energy) + ".pdf"
   histfile = "hist_" + str(energy) + ".pdf"
   ecalfile = "ecal_" + str(energy) + ".pdf"
   energyfile = "energy_" + str(energy) + ".pdf"
   energy2Dfile = "energy2D" + str(energy) + ".pdf"
   errorfile = "error_" + str(energy) + ".pdf"
   realfile = "realfake_" + str(energy) + ".pdf"
   dlabel = "Data " + str(energy)
   glabel = "GAN " + str(energy)
   xfile = "xmoment" + str(energy) + "_"
   yfile = "ymoment" + str(energy) + "_"
   zfile = "zmoment" + str(energy) + "_"
   ratio12file= "RatioL1to2_" + str(energy)+ ".pdf"
   ratio1allfile= "RatioL1tototal_" + str(energy)+ ".pdf"
   hitsfile= "Hits_" + str(energy) + ".pdf"
   plot_max(var["max_pos_act" + str(energy)], os.path.join(comdir, maxfile), fig, dlabel)
   plot_max(var["max_pos_gan" + str(energy)], os.path.join(comdir, maxfile), fig, glabel)
   fig+=1
   plot_energy_axis_log(var["sumsx_act" + str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(comdir, "log_" + histfile), fig, dlabel)
   plot_energy_axis_log(var["sumsx_gan" + str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(comdir, "log_" + histfile), fig, glabel)
   fig+=1
   plot_energy_axis(var["sumsx_act" + str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(comdir, histfile), fig, dlabel)
   plot_energy_axis(var["sumsx_gan" + str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(comdir, histfile), fig, glabel)
   fig+=1
   plot_energy_wt_hist(var["sumsx_act" + str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], os.path.join(comdir, "wt_" + histfile), fig, dlabel)
   plot_energy_wt_hist(var["sumsx_gan" + str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], os.path.join(comdir, "wt_" + histfile), fig, glabel)
   fig+=1
   plot_ecal(var["ecal_act" + str(energy)], os.path.join(comdir, ecalfile), fig, dlabel)
   plot_ecal(var["ecal_gan" + str(energy)], os.path.join(comdir, ecalfile), fig, glabel)
   fig+=1
   plot_energy(np.multiply(100, var["aux_act" + str(energy)]), os.path.join(comdir, energyfile), fig, dlabel, 'blue')
   plot_energy(np.multiply(100, var["aux_gan" + str(energy)]), os.path.join(comdir, energyfile), fig, glabel, 'green')
   fig+=1
   plot_real(var["isreal_act" + str(energy)], os.path.join(discdir, realfile), fig, dlabel)
   plot_real(var["isreal_gan" + str(energy)], os.path.join(discdir, realfile), fig, glabel)
   fig+=1
   plot_error(var["Perror_act" + str(energy)], os.path.join(discdir, errorfile), fig, dlabel, pos=0)
   plot_error(var["Perror_gan" + str(energy)], os.path.join(discdir, errorfile), fig, glabel, pos=0)
   fig+=1
   plot_1to2(var["events_act" + str(energy)], os.path.join(comdir, ratio12file), fig, dlabel)
   plot_1to2(var["events_gan" + str(energy)], os.path.join(comdir, ratio12file), fig, glabel)
   fig+=1
   plot_1tototal(var["events_act" + str(energy)], os.path.join(comdir, ratio1allfile), fig, dlabel)
   plot_1tototal(var["events_gan" + str(energy)], os.path.join(comdir, ratio1allfile), fig, glabel)
   fig+=1
   plot_hits(var["events_act" + str(energy)], os.path.join(comdir, hitsfile), fig, dlabel)
   plot_hits(var["events_gan" + str(energy)], os.path.join(comdir, hitsfile), fig, glabel)
   fig+=1
   for mmt in range(m):
      mfile=xfile + str(mmt) + ".pdf"
      plot_moment(var["momentX_act" + str(energy)][:, mmt], os.path.join(mdir, mfile), fig, dlabel, mmt)
      plot_moment(var["momentX_gan" + str(energy)][:, mmt], os.path.join(mdir, mfile), fig, glabel, mmt)
      fig+=1
      mfile=yfile + str(mmt) + ".pdf"
      plot_moment(var["momentY_act" + str(energy)][:, mmt], os.path.join(mdir, mfile), fig, dlabel, mmt)
      plot_moment(var["momentY_gan" + str(energy)][:, mmt], os.path.join(mdir, mfile), fig, glabel, mmt)
      fig+=1
      mfile=zfile + str(mmt) + ".pdf"
      plot_moment(var["momentZ_act" + str(energy)][:, mmt], os.path.join(mdir, mfile), fig, dlabel, mmt)
      plot_moment(var["momentZ_gan" + str(energy)][:, mmt], os.path.join(mdir, mfile), fig, glabel, mmt)
      fig+=1

plt.close("all")
plot_time = time.time() - start
print '{} Plots are generated in {} seconds'.format(fig, plot_time)
print 'Plots are saved in ', plots_dir
