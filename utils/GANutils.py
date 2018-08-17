import os
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath

# Flip a random bit of an array
def BitFlip(x, prob=0.05):
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

# Find a Fit for Expected Ecal sum based on particle type, Ecal scaling and fit parameters. 
def GetEcalFit(sampled_energies, particle='Ele', mod=0, xscale=1):
    if mod==0:
       return np.multiply(2, sampled_energies)
    elif mod==1:
       if particle == 'Ele':
         root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale
       elif particle == 'Pi0':
         root_fit = [0.0085, -0.094, 2.051]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale
# From Adlkit. Function to return list of Train and Test files based on available events, train-test fraction, Dataset and Particles
def DivideFiles(FileSearch="/data/LCD/*/*.h5", Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    Files =sorted( glob.glob(FileSearch))
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
      Sample=Samples[SampleName]
      NFiles=len(Sample)
      for j,Frac in enumerate(Fractions):
         EndI=int(SampleI[i]+ round(NFiles*Frac))
         out[j]+=Sample[SampleI[i]:EndI]
         SampleI[i]=EndI
    return out
                                                                                                                                                                                                                                    
# Modified from Adlkit to use only specified number of events. Function to return list of Train and Test files based on available events, train-test fraction, Dataset and Particles     
def DivideFilesSafe(FileSearch="/data/LCD/*/*.h5", nEvents=800000, EventsperFile = 10000, Fractions=[.25,.75],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
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

#Reading data from a file
def GetData(datafile):
    #get data for training
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    y=f.get('target')
    x=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    x[x < 1e-6] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

#Sorting events in Energy bins from each file. The bin 0 can be used for Uniform energy. This bin should be filled from first file only using flag=True and the flag should be False for rest so that it is not continously appended
def sort(data, energies, flag=False, num_events=2000):
    X = data[0]
    Y = data[1]
    tolerance = 5
    srt = {}
    for energy in energies:
       if energy == 0 and flag:
          srt["events_act" + str(energy)] = X[:10000] # More events in random bin
          srt["energy" + str(energy)] = Y[:10000]
          print( srt["events_act" + str(energy)].shape)
       else:
          indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
          if len(indexes) > num_events:
             indexes = indexes[:num_events]
          srt["events_act" + str(energy)] = X[indexes]
          srt["energy" + str(energy)] = Y[indexes]
    return srt

#Save sorted data to a directory
def save_sorted(srt, energies, srtdir):
    safe_mkdir(srtdir)
    for energy in energies:
       srtfile = os.path.join(srtdir, "events_{:03d}.h5".format(energy))
       with h5py.File(srtfile ,'w') as outfile:
          outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
          outfile.create_dataset('Target',data=srt["energy" + str(energy)])
       print( "Sorted data saved to {}".format(srtfile))

#Save generated data to a directory
def save_generated(events, sampled_energies, energy, gendir):
    safe_mkdir(gendir)
    filename = os.path.join(gendir,"Gen_{:03d}.hdf5".format(energy))
    with h5py.File(filename ,'w') as outfile:
       outfile.create_dataset('ECAL',data=events)
       outfile.create_dataset('Target',data=sampled_energies)
    print ("Generated data saved to ", filename)

#Save discriminated data to a directory
def save_discriminated(disc, energy, discdir):
    safe_mkdir(discdir)
    filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
    with h5py.File(filename ,'w') as outfile:
      outfile.create_dataset('ISREAL_ACT',data=disc["isreal_act" + str(energy)])
      outfile.create_dataset('ISREAL_GAN',data=disc["isreal_gan" + str(energy)])
      outfile.create_dataset('AUX_ACT',data=disc["aux_act" + str(energy)])
      outfile.create_dataset('AUX_GAN',data=disc["aux_gan" + str(energy)])
      outfile.create_dataset('ECAL_ACT',data=disc["ecal_act" + str(energy)])
      outfile.create_dataset('ECAL_GAN',data=disc["ecal_gan" + str(energy)])
    print( "Discriminated data saved to ", filename)

#Load discriminator results    
def get_disc(energy, discdir):
    filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    isreal_act = np.array(f.get('ISREAL_ACT'))
    isreal_gan = np.array(f.get('ISREAL_GAN'))
    aux_act = np.array(f.get('AUX_ACT'))
    aux_gan = np.array(f.get('AUX_GAN'))
    ecal_act = np.array(f.get('ECAL_ACT'))
    ecal_gan = np.array(f.get('ECAL_GAN'))
    print ("Discriminated file ", filename, " is loaded")
    return isreal_act, aux_act, ecal_act, isreal_gan, aux_gan, ecal_gan

#Load Sorted data
def load_sorted(sorted_path, energies):
    sorted_files = sorted(glob.glob(sorted_path))
    #energies = []
    srt = {}
    for f in sorted_files:
       energy = int(filter(str.isdigit, f)[:-1])
       if energy in energies:
          srtfile = h5py.File(f,'r')
          srt["events_act" + str(energy)] = np.array(srtfile.get('ECAL'))
          srt["energy" + str(energy)] = np.array(srtfile.get('Target'))
          print ("Loaded from file", f)
    return srt

#Load Generated data
def get_gen(energy, gendir):
    filename = os.path.join(gendir, "Gen_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    print ("Generated file ", filename, " is loaded")
    return generated_images

#Get generated events
def generate(g, index, sampled_labels, latent=200):
    noise = np.random.normal(0, 1, (index, latent))
    sampled_labels=np.expand_dims(sampled_labels, axis=1)
    gen_in = sampled_labels * noise
    generated_images = g.predict(gen_in, verbose=False, batch_size=50)
    return generated_images

#Get discrimintor result
def discriminate(d, images):
    isreal, aux_out, ecal_out = np.array(d.predict(images, verbose=False, batch_size=50))
    return isreal, aux_out, ecal_out

#Get max of images
def get_max(images, axis=3):
    index = images.shape[0]
    x= images.shape[1]
    y= images.shape[2]
    z= images.shape[3]
    max_pos = np.zeros((index, axis))
    for i in range(index):
       max_p = images[i].argmax()
       max_loc = np.unravel_index(max_p, (x, y, z))
       max_pos[i] = max_loc
    return max_pos

#Get sums along x, y, z axis for 3D image
def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

#Get Moments
def get_moments(images, sumsx, sumsy, sumsz, totalE, m, ecalx=25, ecaly=25, ecalz=25):
    totalE = np.squeeze(totalE)
    index = images.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecalx), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = umath.inner1d(sumsx, moments) /totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecaly), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = umath.inner1d(sumsy, moments) /totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecalz), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = umath.inner1d(sumsz, moments)/totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

# Create subdirectory only if it does not exist
def safe_mkdir(path):
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

# Make all computations for analysis
def perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, scales, flags, latent, events_per_file=10000, particle='Ele'):
    sortedpath = os.path.join(sortdir, 'events_*.h5')
    Test = flags[0]
    save_data = flags[1]
    read_data = flags[2]
    save_gen = flags[3]
    read_gen = flags[4]
    save_disc = flags[5]
    read_disc =  flags[6]
    var= {}
    if read_data: # Read from sorted dir                                                                                                                                                                           
       start = time.time()
       var = load_sorted(sortedpath, energies)
       sort_time = time.time()- start
       print ("Events were loaded in {} seconds".format(sort_time))
    else:
       # Getting Data                                                                                                                                                                                              
       events_per_file = 10000
       Filesused = int(math.ceil(num_data/events_per_file))
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
       Trainfiles = Trainfiles[: Filesused]
       Testfiles = Testfiles[: Filesused]
       print (Trainfiles)
       print (Testfiles)
       if Test:
          data_files = Testfiles
       else:
          data_files = Trainfiles
       start = time.time()
       for index, dfile in enumerate(data_files):
          data = get_data(dfile)
          if index==0:
             var = sort(data, energies, True, num_events)
          else:
             sorted_data = sort(data, energies, False, num_events)
             for key in var:
                var[key]= np.append(var[key], sorted_data[key], axis=0)
       data = None
       data_time = time.time() - start

       print ("{} events were loaded in {} seconds".format(num_data, data_time))
       if save_data:
          save_sorted(var, energies, sortdir)
    total = 0
    for energy in energies:
    #calculations for data events                                                                                                                                                                                 
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
      total += var["index" + str(energy)]
      var["ecal_act"+ str(energy)]=np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))
      var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)])
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      #var["ecal_act"+ str(energy)]= np.sum(var["events_act" + str(energy)], axis=(1, 2, 3))
                                                                                                                  
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["events_act" + str(energy)], var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m)

    data_time = time.time() - start
    print ("{} events were put in {} bins".format(total, len(energies)))
    #### Generate Data table to screen                                                                                                                                                                             
    print ("Actual Data")
    print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")
    for energy in energies:
       print ("{}\t{}\t{:.4f}\t\t{}\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1])))

    for gen_weights, disc_weights, scale, i in zip(gweights, dweights, scales, np.arange(len(gweights))):
       gendir = gendirs + '/n_' + str(i)
       discdir = discdirs + '/n_' + str(i)
       for energy in energies:
          var["events_gan" + str(energy)]={}
          var["isreal_act" + str(energy)]={}
          var["isreal_gan" + str(energy)]={}
          var["aux_act" + str(energy)]={}
          var["aux_gan" + str(energy)]={}
          var["ecal_act" + str(energy)]={}
          var["ecal_gan" + str(energy)]={}
          var["max_pos_gan" + str(energy)]={}
          var["sumsx_gan"+ str(energy)]={}
          var["sumsy_gan"+ str(energy)]={}
          var["sumsz_gan"+ str(energy)]={}
          var["momentX_gan" + str(energy)]={}
          var["momentY_gan" + str(energy)]={}
          var["momentZ_gan" + str(energy)]={}
          if read_gen:
             var["events_gan" + str(energy)]['n_'+ str(i)]= get_gen(energy, gendir)
          else:
             g.load_weights(gen_weights)
             start = time.time()
             var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, var["index" + str(energy)], var["energy" + str(energy)]/100, latent)
             if save_gen:
                save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], var["energy" + str(energy)], energy, gendir)
             gen_time = time.time() - start
             var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, var["index" + str(energy)], var["energy" + str(energy)]/100, latent)
             if save_gen:
                save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], var["energy" + str(energy)], energy, gendir)
             gen_time = time.time() - start
             print ("Generator took {} seconds to generate {} events".format(gen_time, var["index" +str(energy)]))
          if read_disc:
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)], var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= get_disc(energy, discdir)
          else:
             d.load_weights(disc_weights)
             start = time.time()
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= discriminate(d, var["events_act" + str(energy)] * scale)
             var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= discriminate(d, var["events_gan" + str(energy)]['n_'+ str(i)] )
             disc_time = time.time() - start
             print ("Discriminator took {} seconds for {} data and generated events".format(disc_time, var["index" +str(energy)]))

             if save_disc:
               discout = {}
               for key in var:
                  if key in ["isreal_act" + str(energy), "aux_act" + str(energy), "isreal_gan" + str(energy), "aux_gan" + str(energy), "ecal_act"+ str(energy), "ecal_gan"+ str(energy)]:
                     discout[key]=var[key]['n_'+ str(i)]
               for key in discout:
                   print (key)
               gan.save_discriminated(discout, energy, discdir)
          print ('Calculations for ....', energy)
          var["events_gan" + str(energy)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)]/scale
          var["isreal_act" + str(energy)]['n_'+ str(i)] = np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)])
          print(var["isreal_act" + str(energy)]['n_'+ str(i)].shape)
          var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["ecal_act"+ str(energy)]['n_'+ str(i)]/scale)

          var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["ecal_gan"+ str(energy)]['n_'+ str(i)]/scale)
          var["max_pos_gan" + str(energy)]['n_'+ str(i)] = get_max(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)] = get_sums(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["momentX_gan" + str(energy)]['n_'+ str(i)], var["momentY_gan" + str(energy)]['n_'+ str(i)], var["momentZ_gan" + str(energy)]['n_'+ str(i)] = get_moments(var["events_gan" + str(energy)]['n_'+ str(i)], var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)], m)

       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))

       #### Generate GAN table to screen                                                                                                                                                                          
 
       print ("Generated Data")
       print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")

       for energy in energies:
          print ("{}\t{}\t{:.4f}\t\t{}\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["max_pos_gan" + str(energy)]['n_'+ str(i)], axis=0), np.mean(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["momentX_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentY_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)]['n_'+ str(i)][:, 1])))
    return var
