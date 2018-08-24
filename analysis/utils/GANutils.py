##### Common functions #################
import os
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath

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

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=800000, EventsperFile = 10000, Fractions=[.25,.75],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
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

def GetAngleData(datafile, thresh=1e-6, angtype='eta', offset=0.0):
    #get data for training                                                                                        
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))
    Y=np.array(f.get('energy'))
    ang = np.array(f.get(angtype))
    ang = ang + offset
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, ang, ecal

def get_sorted_angle(datafiles, energies, flag=False, num_events1=10000, num_events2=2000, tolerance1=5, tolerance2=0.05, Data=GetAngleData, angtype='theta', thresh=1e-6, offset=0.0):
    srt = {}
    for index, datafile in enumerate(datafiles):
       data = Data(datafile, thresh = thresh, angtype=angtype, offset= offset)
       X = data[0]
       sumx = np.sum(np.squeeze(X), axis=(1, 2, 3))
       indexes= np.where(sumx>0)
       X=X[indexes]
       Y = data[1]
       Y=Y[indexes]
       angle = data[2]
       angle=angle[indexes]
       for energy in energies:
           if index== 0:
              if energy == 0:
                 srt["events_act" + str(energy)] = X # More events in random bin                                                                                                                                  
                 srt["energy" + str(energy)] = Y
                 srt["angle" + str(energy)] = angle
                 if srt["events_act" + str(energy)].shape[0] > num_events1:
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                    srt["angle" + str(energy)]= srt["theta" + str(energy)][:num_events1]
                    print('For {} energy {} events were found in first file'.format(energy, srt["events_act" + str(energy)].shape[0]))
                    flag=False
              else:
                 indexes = np.where((Y > energy - tolerance1 ) & ( Y < energy + tolerance1))
                 srt["events_act" + str(energy)] = X[indexes]
                 srt["energy" + str(energy)] = Y[indexes]
                 srt["angle" + str(energy)]= angle[indexes]
                 print('For {} energy {} events were found in first file'.format(energy, srt["events_act" + str(energy)].shape[0]))
           else:
              if energy == 0:
                 if flag:
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X, axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y, axis=0)
                    srt["angle" + str(energy)]=np.append(srt["angle" + str(energy)], angle, axis=0)
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                       srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                       srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                       srt["angle" + str(energy)]=srt["angle" + str(energy)][:num_events1]
                       flag=False
              else:
                 if srt["events_act" + str(energy)].shape[0] < num_events2:
                    indexes = np.where((Y > energy - tolerance1 ) & ( Y < energy + tolerance1))
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X[indexes], axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y[indexes], axis=0)
                    srt["angle" + str(energy)]=np.append(srt["angle" + str(energy)], angle[indexes], axis=0)
                 else:
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events2]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events2]
                    srt["angle" + str(energy)]=srt["angle" + str(energy)][:num_events2]
              print('For {} energy {} events were loaded'.format(energy, srt["events_act" + str(energy)].shape[0]))
    return srt

def save_sorted(srt, energies, srtdir):
    safe_mkdir(srtdir)
    for energy in energies:
       srtfile = os.path.join(srtdir, "events_{:03d}.h5".format(energy))
       with h5py.File(srtfile ,'w') as outfile:
          outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
          outfile.create_dataset('Target',data=srt["energy" + str(energy)])
          outfile.create_dataset('Angle',data=srt["angle" + str(energy)])
       print "Sorted data saved to {}".format(srtfile)

def save_generated(events, sampled_energies, sampled_angle, energy, gendir):
    safe_mkdir(gendir)
    filename = os.path.join(gendir,"Gen_{:03d}.hdf5".format(energy))
    with h5py.File(filename ,'w') as outfile:
       outfile.create_dataset('ECAL',data=events)
       outfile.create_dataset('Target',data=sampled_energies)
       outfile.create_dataset('Angle',data=sampled_angle)
    print "Generated data saved to ", filename

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
      outfile.create_dataset('ANGLE_ACT',data=disc["angle_act" + str(energy)])
      outfile.create_dataset('ANGLE_GAN',data=disc["angle_gan" + str(energy)])
    print "Discriminated data saved to ", filename

def get_disc(energy, discdir):
    filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    isreal_act = np.array(f.get('ISREAL_ACT'))
    isreal_gan = np.array(f.get('ISREAL_GAN'))
    aux_act = np.array(f.get('AUX_ACT'))
    aux_gan = np.array(f.get('AUX_GAN'))
    ecal_act = np.array(f.get('ECAL_ACT'))
    ecal_gan = np.array(f.get('ECAL_GAN'))
    angle_act = np.array(f.get('ANGLE_ACT'))
    angle_gan = np.array(f.get('ANGLE_GAN'))
    print "Discriminated file ", filename, " is loaded"
    return isreal_act, aux_act, angle_act, ecal_act, isreal_gan, aux_gan, angle_gan, ecal_gan

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
          srt["angle" + str(energy)] = np.array(srtfile.get('Angle'))
          print "Loaded from file", f
    return srt

def get_gen(energy, gendir):
    filename = os.path.join(gendir, "Gen_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    print "Generated file ", filename, " is loaded"
    return generated_images

def generate(g, index, sampled_labels, sampled_angle, latent=256):
    noise = np.random.normal(0, 1, (index, latent-1))
    sampled_labels=np.expand_dims(sampled_labels, axis=1)
    noise = sampled_labels * noise
    sampled = np.concatenate((sampled_angle.reshape(-1, 1), noise), axis=1)
    generated_images = g.predict(sampled, verbose=False, batch_size=50)
    generated_images[generated_images < 1e-4]= 0
    return generated_images

def discriminate(d, images):
    isreal, aux_out, angle_out, ecal_out = np.array(d.predict(images, verbose=False, batch_size=50))
    return isreal, aux_out, angle_out, ecal_out

def get_max(images):
    index = images.shape[0]
    print images[0].shape
    x=images.shape[1]
    y=images.shape[2]
    z=images.shape[3]
    max_pos = np.zeros((index, 3))
    for i in range(index):
       max_p = images[i].argmax()
       max_loc = np.unravel_index(max_p, (x, y, z))
       max_pos[i] = max_loc
    return max_pos

def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

def get_moments(sumsx, sumsy, sumsz, totalE, m, x=51, y=51, z=25):
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    totalE = np.squeeze(totalE)
    index = sumsx.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(x), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = np.divide(umath.inner1d(sumsx, moments) ,totalE)
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(y), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = np.divide(umath.inner1d(sumsy, moments), totalE)
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(z), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = np.divide(umath.inner1d(sumsz, moments), totalE)
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)                                                                                                                          
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

def perform_calculations_angle(g, d, gweights, dweights, energies, angles, aindexes, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, xscales, angscales, flags, latent, events_per_file=10000, particle='Ele', Data=GetAngleData, angtype='eta', thresh=1e-6, offset=0.0):
    sortedpath = os.path.join(sortdir, 'events_*.h5')
    print flags
    Test = flags[0]
    save_data = flags[1]
    read_data = flags[2]
    save_gen = flags[3]
    read_gen = flags[4]
    save_disc = flags[5]
    read_disc =  flags[6]
    var= {}
    num_events1= 10000
    num_events2 = num_events
    tolerance2 = 0.1
    if read_data: # Read from sorted dir
       start = time.time()
       var = load_sorted(sortedpath, energies)
       sort_time = time.time()- start
       print "Events were loaded in {} seconds".format(sort_time)
    else:
       # Getting Data                                                                                                                                                                                           
       events_per_file = 10000
       Filesused = int(math.ceil(num_data/events_per_file))
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
       Trainfiles = Trainfiles[: Filesused]
       Testfiles = Testfiles[: Filesused]
       print Trainfiles
       print Testfiles
       if Test:
          data_files = Testfiles
       else:
          data_files = Trainfiles
       start = time.time()
       var = get_sorted_angle(data_files, energies, True, num_events1, num_events2, Data=Data, angtype=angtype, thresh=thresh, offset=offset)
       data_time = time.time() - start
       print "{} events were loaded in {} seconds".format(num_data, data_time)
       if save_data:
          save_sorted(var, energies, sortdir)
    total = 0
    for energy in energies:
      x = var["events_act"+ str(energy)].shape[1]
      y =var["events_act"+ str(energy)].shape[2]
      z =var["events_act"+ str(energy)].shape[3]
      #calculations for data events
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
      total += var["index" + str(energy)]
      var["ecal_act"+ str(energy)]=np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))
      var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)])
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
      print ('angle varies from {} to {}'.format(np.amax(var["angle" + str(energy)]), np.amin(var["angle" + str(energy)])))
      for a, index in zip(angles, aindexes):
         indexes = np.where(((var["angle" + str(energy)]) > a - tolerance2) & ((var["angle" + str(energy)]) < a + tolerance2))
         print('from {} to {}'.format(a - tolerance2,  a + tolerance2))
         print(var["angle" + str(energy)][:5])
        
         var["events_act" + str(energy) + "ang_" + str(index)] = var["events_act" + str(energy)][indexes]
         var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)][indexes]
         var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)][indexes]
         var["sumsx_act"+ str(energy) + "ang_" + str(index)] = var["sumsx_act"+ str(energy)][indexes]
         var["sumsy_act"+ str(energy) + "ang_" + str(index)] = var["sumsy_act"+ str(energy)][indexes]
         var["sumsz_act"+ str(energy) + "ang_" + str(index)] = var["sumsz_act"+ str(energy)][indexes]
         print ('{} for angle bin {} total events were {}'.format(index, a, var["events_act" + str(energy) + "ang_" + str(index)].shape[0]))
    data_time = time.time() - start
    print "{} events were put in {} bins".format(total, len(energies))
    #### Generate Data table to screen                                                                                                                                                                             
    print "Actual Data"
    print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2"
    for energy in energies:
       print "{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1]))

    for gen_weights, disc_weights, scale, ascale, i in zip(gweights, dweights, xscales, angscales, np.arange(len(gweights))):
       gendir = gendirs + '/n_' + str(i)
       discdir = discdirs + '/n_' + str(i)
       for energy in energies:
          var["events_gan" + str(energy)]={}
          var["isreal_act" + str(energy)]={}
          var["isreal_gan" + str(energy)]={}
          var["aux_act" + str(energy)]={}
          var["aux_gan" + str(energy)]={}
          var["angle_act" + str(energy)]={}
          var["angle_gan" + str(energy)]={}
          var["ecal_act" + str(energy)]={}
          var["ecal_gan" + str(energy)]={}
          var["max_pos_gan" + str(energy)]={}
          var["sumsx_gan"+ str(energy)]={}
          var["sumsy_gan"+ str(energy)]={}
          var["sumsz_gan"+ str(energy)]={}
          var["momentX_gan" + str(energy)]={}
          var["momentY_gan" + str(energy)]={}
          var["momentZ_gan" + str(energy)]={}
          for index in aindexes:
            var["events_gan" + str(energy) + "ang_" + str(index)]={}
            var["isreal_act" + str(energy) + "ang_" + str(index)]={}
            var["isreal_gan" + str(energy) + "ang_" + str(index)]={}
            var["aux_act" + str(energy)+ "ang_" + str(index)]={}
            var["aux_gan" + str(energy)+ "ang_" + str(index)]={}
            var["angle_act" + str(energy)+ "ang_" + str(index)]={}
            var["angle_gan" + str(energy)+ "ang_" + str(index)]={}
            var["ecal_act" + str(energy)+ "ang_" + str(index)]={}
            var["ecal_gan" + str(energy)+ "ang_" + str(index)]={}
            var["sumsx_gan"+ str(energy)+ "ang_" + str(index)]={}
            var["sumsy_gan"+ str(energy)+ "ang_" + str(index)]={}
            var["sumsz_gan"+ str(energy)+ "ang_" + str(index)]={}

          if read_gen:
             var["events_gan" + str(energy)]['n_'+ str(i)]= get_gen(energy, gendir)
          else:
             g.load_weights(gen_weights)
             start = time.time()
             var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, var["index" + str(energy)], var["energy" + str(energy)]/100, (var["angle"+ str(energy)]) * ascale, latent)
             if save_gen:
                save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], var["energy" + str(energy)], var["angle"+ str(energy)], energy, gendir)
             gen_time = time.time() - start
             print "Generator took {} seconds to generate {} events".format(gen_time, var["index" +str(energy)])
          if read_disc:
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["angle_act"+ str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)], var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["angle_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= get_disc(energy, discdir)
          else:
             d.load_weights(disc_weights)
             start = time.time()
             var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["angle_act"+ str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= discriminate(d, var["events_act" + str(energy)] * scale)
             var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["angle_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= discriminate(d, var["events_gan" + str(energy)]['n_'+ str(i)] )
             disc_time = time.time() - start
             print "Discriminator took {} seconds for {} data and generated events".format(disc_time, var["index" +str(energy)])

             if save_disc:
               discout = {}
               for key in var:
                  if key in ["isreal_act" + str(energy), "aux_act" + str(energy), "isreal_gan" + str(energy), "aux_gan" + str(energy), "ecal_act"+ str(energy), "ecal_gan"+ str(energy), "angle_act"+ str(energy), "angle_gan"+ str(energy)]:
                     discout[key]=var[key]['n_'+ str(i)]
               save_discriminated(discout, energy, discdir)
          print 'Calculations for ....', energy
          var["events_gan" + str(energy)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)]/scale
          #var["isreal_act" + str(energy)]['n_'+ str(i)] = np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)])
          var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["angle_act"+ str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_act" + str(energy)]['n_'+ str(i)]), np.squeeze((var["angle_act"+ str(energy)]['n_'+ str(i)]))/ascale, np.squeeze(var["ecal_act"+ str(energy)]['n_'+ str(i)]/scale)
          var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["angle_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["angle_gan"+ str(energy)]['n_'+ str(i)] )/ascale, np.squeeze(var["ecal_gan"+ str(energy)]['n_'+ str(i)]/scale)
          var["max_pos_gan" + str(energy)]['n_'+ str(i)] = get_max(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)] = get_sums(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["momentX_gan" + str(energy)]['n_'+ str(i)], var["momentY_gan" + str(energy)]['n_'+ str(i)], var["momentZ_gan" + str(energy)]['n_'+ str(i)] = get_moments(var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)], m, x=x, y=y, z=z)
          for a, index in zip(angles, aindexes):
             indexes = np.where(((var["angle" + str(energy)]) > a - tolerance2) & ((var["angle" + str(energy)]) < a + tolerance2))
             var["events_gan" + str(energy) + "ang_" + str(index)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["sumsx_gan"+ str(energy)+ "ang_" + str(index)]['n_'+ str(i)] = var["sumsx_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["sumsy_gan"+ str(energy)+ "ang_" + str(index)]['n_'+ str(i)] =var["sumsy_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["sumsz_gan"+ str(energy)+ "ang_" + str(index)]['n_'+ str(i)] =var["sumsz_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["isreal_act" + str(energy)+ "ang_" + str(index)]['n_'+ str(i)] = var["isreal_act" + str(energy)]['n_'+ str(i)][indexes]
             var["isreal_gan" + str(energy)+ "ang_" + str(index)]['n_'+ str(i)] = var["isreal_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["aux_act" + str(energy)+ "ang_" + str(index)]['n_'+ str(i)] = var["aux_act" + str(energy)]['n_'+ str(i)][indexes]
             var["aux_gan" + str(energy)+ "ang_" + str(index)]['n_'+ str(i)] = var["aux_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["angle_act" + str(energy)+ "ang_" + str(index)]['n_'+ str(i)] = var["angle_act" + str(energy)]['n_'+ str(i)][indexes]
             var["angle_gan" + str(energy)+ "ang_" + str(index)]['n_'+ str(i)] = var["angle_gan" + str(energy)]['n_'+ str(i)][indexes]
       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))

       #### Generate GAN table to screen                                                                                                                                                                          
 
       print "Generated Data"
       print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2"

       for energy in energies:
          print "{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["max_pos_gan" + str(energy)]['n_'+ str(i)], axis=0), np.mean(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["momentX_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentY_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)]['n_'+ str(i)][:, 1]))
    return var
