from os import path
from ROOT import TCanvas, TGraph, gStyle, TProfile, TH3D
import os
import h5py
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import math
import time
import glob
import numpy.core.umath_tests as umath
#import matplotlib.lines as mlines

#Architectures 
from ecalvegan import generator, discriminator
#plt.switch_backend('Agg')

disc_weights="params_discriminator_epoch_067.hdf5"
gen_weights= "params_generator_epoch_067.hdf5"
#disc_weights = "rootfit_disc.hdf5"
#gen_weights = "rootfit_gen.hdf5"

plots_dir = "lrep68_plots/"
latent = 200
num_data = 100000
num_events = 2000
m = 3
scale = 1

#datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path                                         
datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5'
sortedpath = 'sorted_*.hdf5'
Test = False

save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
read_data = True # True if previously sorted data is saved 
save_gen = False # True if saving generated data. 
read_gen = True # True if generated data is already saved and can be loaded
save_disc = False # True if discriminiator data is to be saved
read_disc = True # True if discriminated data is to be loaded

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
       Trainfiles, Testfiles = DivideFiles(datapath, nEvents=num_data, EventsperFile = events_per_file, datasetnames=["ECAL"], Particles =["Ele"]) 
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

    for energy in energies:
        Ecal_data = TH3D("Data","E distribution", 25,0,25,25,0,25,25,0,25) 
        Ecal_gan = TH3D("GAN","E distribution", 25,0,25,25,0,25,25,0,25)
        for event in range(var["index" + str(energy)]):
           for x in range(25):
              for y in range(25):
                 for z in range(25):
                    Ecal_data.Fill(x, y, z, var["events_act"+ str(energy)][event][x, y, z])
