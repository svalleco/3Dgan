from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath

def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=400000, EventsperFile = 10000, Fractions=[.5,.5],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
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

def sort(data, energies, flag=False, num_events=2000):
    X = data[0]
    Y = data[1]
    tolerance = 5
    srt = {}
    for energy in energies:
       if energy == 0 and flag:
          srt["events_act" + str(energy)] = X[:2000]
          srt["energy" + str(energy)] = Y[:2000]
          print srt["events_act" + str(energy)].shape
       else:
          indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
          if len(indexes) > num_events:
             indexes = indexes[:num_events]
          srt["events_act" + str(energy)] = X[indexes]
          srt["energy" + str(energy)] = Y[indexes]
    return srt

def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

def save_sorted(srt, energies, srtdir):
    safe_mkdir(srtdir)
    for energy in energies:
       srtfile = os.path.join(srtdir, "events_{:03d}.h5".format(energy))
       with h5py.File(srtfile ,'w') as outfile:
          outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
          outfile.create_dataset('Target',data=srt["energy" + str(energy)])
       print "Sorted data saved to {}".format(srtfile)

def save_generated(events, sampled_energies, energy, gendir):
    safe_mkdir(gendir)
    filename = os.path.join(gendir,"Gen_{:03d}.hdf5".format(energy))
    with h5py.File(filename ,'w') as outfile:
       outfile.create_dataset('ECAL',data=events)
       outfile.create_dataset('Target',data=sampled_energies)
    print "Generated data saved to ", filename

def save_discriminated(disc, energies, discdir):
    safe_mkdir(discdir)
    for energy in energies:
      filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
      with h5py.File(filename ,'w') as outfile:
        outfile.create_dataset('ISREAL_ACT',data=disc["isreal_act" + str(energy)])
        outfile.create_dataset('ISREAL_GAN',data=disc["isreal_gan" + str(energy)])
        outfile.create_dataset('AUX_ACT',data=disc["aux_act" + str(energy)])
        outfile.create_dataset('AUX_GAN',data=disc["aux_gan" + str(energy)])
        outfile.create_dataset('ECAL_ACT',data=disc["ecal_act" + str(energy)])
        outfile.create_dataset('ECAL_GAN',data=disc["ecal_gan" + str(energy)])
      print "Discriminated data saved to ", filename


def main():
   #Architectures 
   from EcalEnergyGan import generator, discriminator
   gen_weights1 = 'veganweights/params_generator_epoch_029.hdf5'# 1 gpu  
   #gen_weights1 = 'generator_1gpu_042.hdf5'# 1 gpu  
  # gen_weights2 = 'generator_2gpu_023.hdf5'# 2 gpu                                                            
  # gen_weights3 = '4gpu_gen.hdf5'          # 4 gpu                                                            
  # gen_weights4 = 'generator_8gpu_005.hdf5'# 8 gpu                                                            
  # gen_weights5 = 'generator_16gpu_002.hdf5'# 16 gpu          

   disc_weights1 = 'veganweights/params_discriminator_epoch_029.hdf5'# 1 gpu      
  # disc_weights2 = 'discriminator_2gpu_023.hdf5'# 2 gpu                                                           
  # disc_weights3 = '4gpu_disc.hdf5'             # 4 gpu                                                           
  # disc_weights4 = 'discriminator_8gpu_005.hdf5'# 8 gpu                                                   
  # disc_weights5 = 'discriminator_16gpu_002.hdf5'# 16 gpu   

   plots_dir = "distributed_Test/"
   latent = 200
   num_data = 100000
   num_events = 2000
   m = 3
   energies=[0, 50, 100, 200, 250, 300, 400, 500]
   #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path                                         
   datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5'
   sortdir = 'SortedData'
   gendir = 'Gen'
   discdir = 'Disc'
   Test = True
   save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
   read_data = False # True if loading previously sorted data  
   save_gen = False # True if saving generated data. 
   read_gen = False # True if generated data is already saved and can be loaded
   save_disc = False # True if discriminiator data is to be saved
   read_disc =  False # True if discriminated data is to be loaded from previously saved file

   flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   #dweights = [disc_weights1, disc_weights2, disc_weights3, disc_weights4, disc_weights5]
   #gweights = [gen_weights1, gen_weights2, gen_weights3, gen_weights4, gen_weights5]
   dweights = [disc_weights1]
   gweights = [gen_weights1]
   #scales = [1, 1, 1, 1, 1]
   scales = [1]
   d = discriminator()
   g = generator(latent)
   var= perform_calculations(g, d, gweights, dweights, energies, datapath, sortdir, gendir, discdir, num_data, num_events, m, scales, flags, latent)
   get_plots(var, plots_dir, energies, m, len(gweights))



def get_data(datafile):
    #get data for training                                                                                                                                                                      
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=f.get('target')
    x=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    x[x < 1e-6] = 0
    x = x[:,12,:,:]
    x = np.expand_dims(x, axis=-1)
    x = np.moveaxis(x, -1, 1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

def get_gen(energy, gendir):
    filename = os.path.join(gendir, "Gen_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    print "Generated file ", filename, " is loaded"
    return generated_images

def generate(g, index, sampled_labels, latent=200):
    noise = np.random.normal(0, 1, (index, latent))
    sampled_labels=np.expand_dims(sampled_labels, axis=1)
    gen_in = sampled_labels * noise
    generated_images = g.predict(gen_in, verbose=False, batch_size=50)
    return generated_images

def discriminate(d, images):
    isreal, aux_out, ecal_out = np.array(d.predict(images, verbose=False, batch_size=50))
    return isreal, aux_out, ecal_out


def fill_profile(prof, x, y):
   for i in range(len(y)):
      prof.Fill(y[i], x[i])


def plot_ecal_ratio_profile(ecal1, ecal2, y, out_file):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 520)
   Gprof = ROOT.TProfile("Gprof", "Gprof", 100, 0, 520)
   c1.SetGrid()

   Eprof.SetStats(kFALSE)
   Gprof.SetStats(kFALSE)

   fill_profile(Eprof, ecal1/y, y)
   fill_profile(Gprof, ecal2/y, y)

   Eprof.SetTitle("Ratio of Ecal and Ep")
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("Ecal/Ep")
   Eprof.GetYaxis().SetRangeUser(0, 0.03)
   Eprof.Draw()
   Eprof.SetLineColor(4)
   Gprof.SetLineColor(2)
   Gprof.Draw('sames')
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"Data","l")
   legend.AddEntry(Gprof, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def perform_calculations(g, d, gweights, dweights, energies, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, scales, flags, latent, events_per_file=10000):
    sortedpath = os.path.join(sortdir, 'events_*.h5')
    Test = flags[0]
    save_data = flags[1]
    read_data = flags[2]
    save_gen = flags[3]
    read_gen = flags[4]
    save_disc = flags[5]
    read_disc =  flags[6]
    var={}
#    for gen_weights, disc_weights, i in zip(gweights, dweights, np.arange(len(gweights))):
    if read_data: # Read from sorted dir
       start = time.time()
       energies, var = load_sorted(sortedpath)
       sort_time = time.time()- start
       print "Events were loaded in {} seconds".format(sort_time)
    else:
       # Getting Data
       events_per_file = 10000
       Filesused = int(math.ceil(num_data/events_per_file))
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =["Ele"])
       Trainfiles = Trainfiles[: Filesused]
       Testfiles = Testfiles[: Filesused]
       print Trainfiles
       print Testfiles
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
       print "{} events were loaded in {} seconds".format(num_data, data_time)
       if save_data:
          save_sorted(var, energies, sortdir)
    total = 0
    for energy in energies:
    #calculations for data events
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
      total += var["index" + str(energy)]
      var["ecal_act"+ str(energy)]= np.sum(var["events_act" + str(energy)], axis=(2, 3))

    data_time = time.time() - start
    print "{} events were put in {} bins".format(total, len(energies))
    #### Generate Data table to screen                                                                                                                                                                          
    print "Actual Data"
    print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomentz2"

    for gen_weights, disc_weights, scale, i in zip(gweights, dweights, scales, np.arange(len(gweights))):
       var['n_'+ str(i)]={}
       gendir = gendirs + '/n_' + str(i)
       discdir = discdirs + '/n_' + str(i)
       if read_gen:
         for energy in energies:
           var['n_'+ str(i)]["events_gan" + str(energy)]= get_gen(energy, gendir)
       else:
         g.load_weights(gen_weights)
         start = time.time()
         for energy in energies:
            var['n_'+ str(i)]["events_gan" + str(energy)] = generate(g, var["index" + str(energy)], var["energy" + str(energy)]/100, latent)
            if save_gen:
              save_generated(var['n_'+ str(i)]["events_gan" + str(energy)], var["energy" + str(energy)], energy, gendir)
         gen_time = time.time() - start
         print "Generator took {} seconds to generate {} events".format(gen_time, total)
       if read_disc:
         for energy in energies:
           var['n_'+ str(i)]["isreal_act" + str(energy)], var['n_'+ str(i)]["aux_act" + str(energy)], var['n_'+ str(i)]["ecal_act"+ str(energy)], var['n_'+ str(i)]["isreal_gan" + str(energy)], var['n_'+ str(i)]["aux_gan" + str(energy)], var['n_'+ str(i)]["ecal_gan"+ str(energy)]= get_disc(energy, discdir)
       else:
         d.load_weights(disc_weights)
         start = time.time()
         for energy in energies:
           var['n_'+ str(i)]["isreal_act" + str(energy)], var['n_'+ str(i)]["aux_act" + str(energy)], var['n_'+ str(i)]["ecal_act"+ str(energy)]= discriminate(d, var["events_act" + str(energy)] * scale)
           var['n_'+ str(i)]["isreal_gan" + str(energy)], var['n_'+ str(i)]["aux_gan" + str(energy)], var['n_'+ str(i)]["ecal_gan"+ str(energy)]= discriminate(d, var['n_'+ str(i)]["events_gan" + str(energy)] )
         disc_time = time.time() - start
         print "Discriminator took {} seconds for {} data and generated events".format(disc_time, total)

         if save_disc:
           save_discriminated(var['n_'+ str(i)], energies, discdir)

       for energy in energies:
         print 'Calculations for ....', energy
         var['n_'+ str(i)]["events_gan" + str(energy)] = var['n_'+ str(i)]["events_gan" + str(energy)]/scale
         var['n_'+ str(i)]["isreal_act" + str(energy)], var['n_'+ str(i)]["aux_act" + str(energy)], var['n_'+ str(i)]["ecal_act"+ str(energy)]= np.squeeze(var['n_'+ str(i)]["isreal_act" + str(energy)]), np.squeeze(var['n_'+ str(i)]["aux_act" + str(energy)]), np.squeeze(var['n_'+ str(i)]["ecal_act"+ str(energy)]/scale)
         var['n_'+ str(i)]["isreal_gan" + str(energy)], var['n_'+ str(i)]["aux_gan" + str(energy)], var['n_'+ str(i)]["ecal_gan"+ str(energy)]= np.squeeze(var['n_'+ str(i)]["isreal_gan" + str(energy)]), np.squeeze(var['n_'+ str(i)]["aux_gan" + str(energy)]), np.squeeze(var['n_'+ str(i)]["ecal_gan"+ str(energy)]/scale)

       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))

       #### Generate GAN table to screen                                                                                  
       print "Generated Data"
       print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomentz2"
    return var

def get_plots(var, plots_dir, energies, m, n):
    
    #actdir = plots_dir + 'Actual'
    #safe_mkdir(actdir)
    alldir = plots_dir + 'All'
    safe_mkdir(alldir)

    for i in np.arange(n):
      ## Make folders for plots                                                                                          
      #discdir = plots_dir + 'disc_outputs'+ 'plots_' + str(i) + '/'
      #safe_mkdir(discdir)
      #gendir = plots_dir + 'Generated' + 'plots_' + str(i) + '/'
      #safe_mkdir(gendir)
      #comdir = plots_dir + 'Combined' + 'plots_' + str(i) + '/'
      #safe_mkdir(comdir)
      #mdir = plots_dir + 'Moments' + 'plots_' + str(i) + '/'
      #safe_mkdir(mdir)
      #start = time.time()
   
      #for energy in energies:
      #   maxfile = "Position_of_max_" + str(energy) + ".pdf"
      #   histfile = "hist_" + str(energy) + ".pdf#"
      #   histlfile = "hist_log" + str(energy) + ".pdf"
      #   ecalfile = "ecal_" + str(energy) + ".pdf"
      #   energyfile = "energy_" + str(energy) + ".pdf"
      #   realfile = "realfake_" + str(energy) + ".pdf"
      #   momentfile = "moment" + str(energy) + ".pdf"
      #   ecalerrorfile = "ecal_error" + str(energy) + ".pdf"
      #allfile = 'All_energies.pdf'
      #allecalrelativefile = 'All_ecal_relative.pdf'
      #allerrorfile = 'All_relative_auxerror.pdf'
      allecalfile = 'All_ecal.pdf'
       
         #plot_max(var["max_pos_act" + str(energy)], var['n_' + str(i)]["max_pos_gan" + str(energy)], os.path.join(actdir, maxfile), os.path.join(gendir, maxfile), os.path.join(comdir, maxfile), energy)
         #plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsz_act"+ str(energy)], var['n_' + str(i)]["sumsx_gan"+ str(energy)], var['n_' + str(i)]["sumsz_gan"+ str(energy)], os.path.join(actdir, histfile), os.path.join(gendir, histfile), os.path.join(comdir, histfile),os.path.join(alldir, "hist_" + str(energy)), i, energy)
         #plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsz_act"+ str(energy)], var['n_' + str(i)]["sumsx_gan"+ str(energy)], var['n_' + str(i)]["sumsz_gan"+ str(energy)], os.path.join(actdir, histlfile), os.path.join(gendir, histlfile), os.path.join(comdir, histlfile), os.path.join(alldir, "histl_" + str(energy) ), i, energy, log=1)
         #plot_ecal_hist(var["ecal_act" + str(energy)], var['n_' + str(i)]["ecal_gan" + str(energy)], os.path.join(discdir, ecalfile), energy)
         #plot_ecal_flatten_hist(var["events_act" + str(energy)], var['n_' + str(i)]["events_gan" + str(energy)], os.path.join(comdir, 'flat' + ecalfile), energy)
         #plot_ecal_hits_hist(var["events_act" + str(energy)], var['n_' + str(i)]["events_gan" + str(energy)], os.path.join(comdir, 'hits' + ecalfile), energy)
         #plot_primary_hist(var['n_' + str(i)]["aux_act" + str(energy)] * 100, var['n_' + str(i)]["aux_gan" + str(energy)] * 100, os.path.join(discdir, energyfile), energy)
         #plot_realfake_hist(var['n_' + str(i)]["isreal_act" + str(energy)], var['n_' + str(i)]["isreal_gan" + str(energy)], os.path.join(discdir, realfile), energy)
         #plot_primary_error_hist(var['n_' + str(i)]["aux_act" + str(energy)], var['n_' + str(i)]["aux_gan" + str(energy)], var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile), energy)
         #for mmt in range(m):                                                                                            
      #plot_energy_hist_root_all(var["sumsz_act50"], var["sumsz_act100"], var["sumsz_act400"], var["sumsz_act500"], var['n_' + str(i)]["sumsz_gan50"], var['n_' + str(i)]["sumsz_gan100"], var['n_' + str(i)]["sumsz_gan400"], var['n_' + str(i)]["sumsz_gan500"], 50, 100, 400, 500, os.path.join(comdir, allfile))
      plot_ecal_ratio_profile(var['n_' + str(i)]["ecal_act0"], var['n_' + str(i)]["ecal_gan0"], var["energy0"], os.path.join(alldir, allecalfile))
      #plot_ecal_relative_profile(var['n_' + str(i)]["ecal_act0"], var['n_' + str(i)]["ecal_gan0"], var["energy0"], os.path.join(comdir, allecalrelativefile))
      #plot_aux_relative_profile(var['n_' + str(i)]["aux_act0"], var['n_' + str(i)]["aux_gan0"], var["energy0"], os.path.join(comdir, allerrorfile))
      print 'Plots are saved in ', plots_dir
      plot_time= time.time()- start
      print 'Plots are generated in {} seconds'.format(plot_time)
if __name__ == "__main__":
    main()

