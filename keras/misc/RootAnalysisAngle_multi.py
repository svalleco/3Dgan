#!/usr/bin/env python
# -*- coding: utf-8 -*-
## This script loads weights into architectures for generator and discriminator. Different Physics quantities are then calculated and plotted for a 100-200 GeV events from LCD variable
import sys
sys.path.insert(0,'../')
from analysis.utils.GANutils import perform_calculations_angle, perform_calculations_multi   # to calculate different Physics quantities
from analysis.utils.GANutils import * 
from analysis.utils.RootPlotsGAN import get_plots_angle, get_plots_multi         # to make plots with ROOT
import os
import h5py
import numpy as np
import math
import argparse

#from memory_profiler import profile
if os.environ.get('HOSTNAME') == 'tlab-gpu-oldeeptector.cern.ch': # Here a check for host can be used        
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech                                                                                
except:
    pass

sys.path.insert(0,'../')

def main():
   parser = get_parser()
   params = parser.parse_args()

   datapath =params.datapath
   latent = params.latentsize
   particle= params.particle
   angtype= params.angtype
   plotdir= params.outdir + '/'
   sortdir= params.sortdir
   gendir= params.gendir
   discdir= params.discdir
   nbEvents= params.nbEvents
   binevents= params.binevents
   moments= params.moments
   addloss= params.addloss
   angloss= params.angloss
   concat= params.concat
   cell= params.cell
   corr= params.corr
   test= params.test
   stest= params.stest
   save_data= params.save_data
   read_data= params.read_data
   save_gen= params.save_gen
   read_gen= params.read_gen
   save_disc= params.save_disc
   read_disc= params.read_disc
   ifpdf= params.ifpdf
   grid = params.grid
   leg= 1 if stest else params.leg
   statbox= params.statbox
   mono= params.mono
   labels = params.labels if isinstance(params.labels, list) else [params.labels]
   gweights= params.gweights if isinstance(params.gweights, list) else [params.gweights]
   dweights= params.dweights if isinstance(params.dweights, list) else [params.dweights]
   xscales= params.xscales 
   ascales= params.ascales 
   yscale= params.yscale
   xpowers = params.xpower 
   thresh = params.thresh
   dformat = params.dformat
   ang = params.ang
   
   #Architecture 
   if ang:
     
     from AngleArch3dGAN_old import generator, discriminator
     from AngleArch3dGAN_newarch2 import generator2, discriminator2

     dscale=50.
     if not xscales:
       xscales=1.
     if not xpowers:
       xpowers = 0.85
     if not latent:
       latent = 256
     if not ascales:
       ascales = 1
       
     if datapath=='reduced':
       datapath = "/storage/group/gpu/bigdata/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV
       events_per_file = 5000
       energies = [0, 110, 150, 190]
     elif datapath=='full':
       datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
       events_per_file = 10000
       energies = [0, 50, 100, 200, 250, 300, 400, 500]
       #energies =[0, 160, 200, 250, 290]

   else:
     from EcalEnergyGan import generator, discriminator
     dscale=1
     if not xscales:
       xscales=100.
     if not xpowers:
       xpowers = 1.
     if not latent:
       latent =200
     if not ascales:
       ascales = 1

     if datapath=='full':
       datapath ='/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5'
     events_per_file = 10000
     energies = [0, 50, 100, 200, 250, 300, 400, 500]  
   if tlab: 
     #Weights
     dweights=["/gkhattak/weights/3dgan_weights_gan_training/params_discriminator_epoch_059.hdf5"]
     gweights= ["/gkhattak/weights/3dgan_weights_gan_training/params_generator_epoch_059.hdf5"]
     datapath = '/eos/user/g/gkhattak/VarAngleData/*Measured3ThetaEscan/*.h5'
     events_per_file = 5000
   xscales = xscales if isinstance(xscales, list) else [xscales]*len(gweights)
   ascales = ascales if isinstance(ascales, list) else [ascales]*len(gweights)
   xpowers = xpowers if isinstance(xpowers, list) else [xpowers]*len(gweights)
   angles = [62, 90, 118]
   flags =[test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]

   if ang:
     d = discriminator(xpowers[0], dformat=dformat)
     g = generator(latent, dformat=dformat)
   
     d2 = discriminator2(xpowers[0])
     g2 = generator2(latent)

     var= perform_calculations_angle(g, d, g2, d2, gweights, dweights, energies, angles, 
                datapath, sortdir, gendir, discdir, nbEvents, binevents, moments, xscales, xpowers,
                ascales, dscale, flags, latent, particle, events_per_file=events_per_file, thresh=thresh, angtype=angtype, offset=0.0,
                angloss=angloss, addloss=addloss, concat=concat#, Data=GetDataAngle2 
                , pre =taking_power, post =inv_power  # Adding other preprocessing, Default is simple scaling                 
     )

     get_plots_angle(var, labels, plotdir, energies, angles, angtype, moments, 
               len(gweights), ifpdf=ifpdf, grid=grid, stest=stest, angloss=angloss, 
                addloss=addloss, cell=cell, corr=corr, leg=leg, statbox=statbox, mono=mono)
   else:
     d = discriminator( dformat=dformat)
     g = generator(latent,  dformat=dformat)
     var= perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendir, discdir, num_data=nbEvents
         , num_events=binevents, m=moments, scales=xscales, thresh=thresh, flags=flags, latent=latent, particle=particle, dformat=dformat)
     get_plots_multi(var, labels, plotdir, energies, moments, len(gweights), cell=cell, corr=corr,
                     ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)

def sqrt(n, scale=1):
   return np.sqrt(n * scale)

def square(n, scale=1):
   return np.square(n)/scale
        
def taking_power(n, scale=1.0, power=1.0):
   return(np.power(n * scale, power))

def inv_power(n, scale=1.0, power=1.0):
   return(np.power(n, 1.0/power))/scale

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, help='size of random N(0, 1) latent space to sample')    #parser.add_argument('--model', action='store', default=AngleArch3dgan, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='full', help='HDF5 files to train from.')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle used.')
    parser.add_argument('--outdir', action='store', type=str, default='results/3dgan_Analysis/', help='Directory to store the analysis plots.')
    parser.add_argument('--sortdir', action='store', type=str, default='SortedData', help='Directory to store sorted data.')
    parser.add_argument('--gendir', action='store', type=str, default='Gen', help='Directory to store the generated images.')
    parser.add_argument('--discdir', action='store', type=str, default='Disc', help='Directory to store the discriminator outputs.')
    parser.add_argument('--nbEvents', action='store', type=int, default=100000, help='Max limit for events used for Testing')
    parser.add_argument('--eventsperfile', action='store', type=int, default=5000, help='Number of events in a file')
    parser.add_argument('--binevents', action='store', type=int, default=10000, help='Number of events in each bin')
    parser.add_argument('--moments', action='store', type=int, default=3, help='Number of moments to compare')
    parser.add_argument('--addloss', action='store', type=int, default=1, help='If using bin count loss')
    parser.add_argument('--angloss', action='store', type=int, default=1, help='Number of loss terms related to angle')
    parser.add_argument('--concat', action='store', type=int, default=2, help='Modes related to combining conditions with latent 0)not cancatenated.. 1)concatenate angle...3) concatenate energy and angle')
    parser.add_argument('--cell', action='store', type=int, default=0, help='Whether to plot cell energies..0)Not plotted...1)Only for bin with uniform spectrum.....2)For all energy bins')
    parser.add_argument('--corr', action='store', type=int, default=2, help='Plot correlation plots..0)Not plotted...1)detailed features ...2) reduced features.. 3) reduced features for each energy bin')
    parser.add_argument('--test', action='store', type=int, default=1,  help='Use Test data')
    parser.add_argument('--stest', action='store', type=int, default=0, help='Statistics test for shower profiles')
    parser.add_argument('--save_data', default=False, action='store_true', help='Save sorted data')
    parser.add_argument('--read_data', default=False, action='store_true', help='Get saved and sorted data')
    parser.add_argument('--save_gen', default=False, action='store_true', help='Save generated images')
    parser.add_argument('--read_gen', default=False, action='store_true', help='Get saved generated images')
    parser.add_argument('--save_disc', default=False, action='store_true', help='Save discriminator output')
    parser.add_argument('--read_disc', default=False, action='store_true', help='Get discriminator output')
    parser.add_argument('--ifpdf', default=1, type=int, action='store', help='Whether generate pdf plots or .C plots')
    parser.add_argument('--grid', default=0, type=int, action='store', help='set grid')
    parser.add_argument('--leg', default=0, type=int, action='store', help='add legends')
    parser.add_argument('--statbox', default=0, type=int, action='store', help='add statboxes')
    parser.add_argument('--mono',  default=False, action='store_true', help='changing line style as well as color for comparison')
    parser.add_argument('--labels',  default='', type=str, nargs='+', help='id for particular weights when comparing multiple training')
    parser.add_argument('--gweights', action='store', type=str, nargs='+', default=['../weights/3dgan_weights_wt_aux/params_generator_epoch_035.hdf5'], help='comma delimited list for paths to Generator weights.')
    parser.add_argument('--dweights', action='store', type=str, nargs='+', default=['../weights/3dgan_weights_wt_aux/params_discriminator_epoch_035.hdf5'], help='comma delimited list for paths to Discriminator weights')
    parser.add_argument('--xscales', action='store', type=int, nargs='+', help='Multiplication factors for cell energies')
    parser.add_argument('--ascales', action='store', type=int, nargs='+', help='Multiplication factors for angles')
    parser.add_argument('--yscale', action='store', default=100., help='Division Factor for Primary Energy.')
    parser.add_argument('--xpower', action='store', help='Power of cell energies')
    parser.add_argument('--thresh', action='store', default=0, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
    parser.add_argument('--ang', action='store', default=1, type=int, help='if variable angle')
    return parser
   
# If using reduced Ecal 25x25x25 then use the following function as argument to perform_calculations_angle, Data=GetAngleDataEta_reduced
def GetAngleDataEta_reduced(datafile, thresh=1e-6):
    #get data for training                                                                                        
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))[:, 13:38, 13:38, :]
    Y=np.array(f.get('energy'))
    eta = np.array(f.get('eta')) + 0.6
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, eta, ecal
#get data for training                                                                                                       
def GetDataAngle2(datafile, xscale =1, xpower=1, yscale = 1, angscale=1, angtype='theta', offset=0.0, thresh=1e-4, daxis=-1):
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))* xscale
    Y=np.array(f.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where((ecal > 10.0) & (Y > 150) & (Y < 350))
    print('From {} events {} passed'.format(Y.shape[0], indexes[0].shape[0]))
    X=X[indexes]
    Y=Y[indexes]
    ecal = ecal[indexes]
    if angtype in f:
      ang = np.array(f.get(angtype))[indexes]
    else:
      ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal

def perform_calculations_angle(g1, d1, g2, d2, gweights, dweights, energies, angles, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, xscales, xpowers, angscales, dscale, flags, latent, particle='Ele', Data=GetAngleData, events_per_file=5000, angtype='theta', thresh=1e-6, offset=0.0, angloss=1, addloss=0, concat=1, pre=preproc, post=postproc, tolerance2 = 0.1, num_events1=10000):
    sortedpath = os.path.join(sortdir, 'events_*.h5')
    print( flags)
    # assign values to flags that decide if data is to be read from dataset or pre binned data
    # Also if saved generated and discriminated data is to be used
    
    Test = flags[0]
    save_data = flags[1]
    read_data = flags[2]
    save_gen = flags[3]
    read_gen = flags[4]
    save_disc = flags[5]
    read_disc =  flags[6]
    var= {}
    num_events1= num_events1
    num_events2 = num_events
    ang =1

    # Read from sorted dir with binned data
    if read_data: 
       start = time.time()
       var = load_sorted(sortedpath, energies, ang) # returning a dict with sorted data
       print( "Events were loaded in {} seconds".format(time.time()- start))

    # If reading from unsorted data. The data will be read and sorted in bins   
    else:
       Filesused = int(math.ceil(num_data/events_per_file)) # num_data is number of events to be used from unsorted data/ events in each file
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle]) # get test and train files
       Trainfiles = Trainfiles[: Filesused] # The number of files to read is limited by Fileused
       Testfiles = Testfiles[: Filesused]
       print (Trainfiles)
       print (Testfiles)
       if Test:
          data_files = Testfiles  # Test data will be read in test mode
       else:
          data_files = Trainfiles  # else train data will be used
       start = time.time()
       var = get_sorted_angle(data_files, energies, True, num_events1, num_events2, Data=Data, angtype=angtype, thresh=thresh*dscale, offset=offset) # returning a dict with sorted data. 
       print ("{} events were loaded in {} seconds".format(num_data, time.time() - start))
       
       # If saving the binned data. This will only run if reading from data directly
       if save_data:
          save_sorted(var, energies, sortdir, ang) # saving sorted data in a directory

    total = 0

    # For each energy bin
    for energy in energies:
      # Getting dimensions of ecal images  
      x = var["events_act"+ str(energy)].shape[1]
      y =var["events_act"+ str(energy)].shape[2]
      z =var["events_act"+ str(energy)].shape[3]

      # scaling to GeV
      if not dscale==1: var["events_act"+ str(energy)]= var["events_act"+ str(energy)]/dscale
      #calculations for data events
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0] # number of events in bin
      total += var["index" + str(energy)] # total events 
      ecal =np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))# sum actual events for moment calculations
      var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)]) # get position of maximum deposition
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)]) # get sums along different axis
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)],
                                                                var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], ecal, m, x=x, y=y, z=z) # calculate moments
      for index, a in enumerate(angles):
         indexes = np.where(((var["angle" + str(energy)]) > np.radians(a) - tolerance2) & ((var["angle" + str(energy)]) < np.radians(a) + tolerance2)) # all events with angle within a bin
         # angle bins are added to dict
         var["events_act" + str(energy) + "ang_" + str(a)] = var["events_act" + str(energy)][indexes]
         var["energy" + str(energy) + "ang_" + str(a)] = var["energy" + str(energy)][indexes]
         var["angle" + str(energy) + "ang_" + str(a)] = var["angle" + str(energy)][indexes]
         var["sumsx_act"+ str(energy) + "ang_" + str(a)] = var["sumsx_act"+ str(energy)][indexes]
         var["sumsy_act"+ str(energy) + "ang_" + str(a)] = var["sumsy_act"+ str(energy)][indexes]
         var["sumsz_act"+ str(energy) + "ang_" + str(a)] = var["sumsz_act"+ str(energy)][indexes]
         var["momentX_act"+ str(energy) + "ang_" + str(a)] = var["momentX_act"+ str(energy)][indexes]
         var["momentY_act"+ str(energy) + "ang_" + str(a)] = var["momentY_act"+ str(energy)][indexes]
         var["momentZ_act"+ str(energy) + "ang_" + str(a)] = var["momentZ_act"+ str(energy)][indexes]
                           
         print ('{} for angle bin {} total events were {}'.format(index, a, var["events_act" + str(energy) + "ang_" + str(a)].shape[0]))

    print ("{} events were put in {} bins".format(total, len(energies)))
    #### Generate Data table to screen                                                                                                                                                                             
    print ("Actual Data")
    print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")
    for energy in energies:
       print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1])))

    for energy in energies:
       # creating dicts for all GAN quantities 
       var["events_gan" + str(energy)]={}
       var["isreal_act" + str(energy)]={}
       var["isreal_gan" + str(energy)]={}
       var["aux_act" + str(energy)]={}
       var["aux_gan" + str(energy)]={}
       var["angle_act" + str(energy)]={}
       var["angle_gan" + str(energy)]={}
       if angloss==2:
          var["angle2_act" + str(energy)]={}
          var["angle2_gan" + str(energy)]={}
       if addloss:
          var["addloss_act" + str(energy)]={}
          var["addloss_gan" + str(energy)]={}
                   
       var["ecal_act" + str(energy)]={}
       var["ecal_gan" + str(energy)]={}
       var["max_pos_gan" + str(energy)]={}
       var["sumsx_gan"+ str(energy)]={}
       var["sumsy_gan"+ str(energy)]={}
       var["sumsz_gan"+ str(energy)]={}
       var["momentX_gan" + str(energy)]={}
       var["momentY_gan" + str(energy)]={}
       var["momentZ_gan" + str(energy)]={}
       for index, a in enumerate(angles):
          var["events_gan" + str(energy) + "ang_" + str(a)]={}
          var["isreal_act" + str(energy) + "ang_" + str(a)]={}
          var["isreal_gan" + str(energy) + "ang_" + str(a)]={}
          var["aux_act" + str(energy)+ "ang_" + str(a)]={}
          var["aux_gan" + str(energy)+ "ang_" + str(a)]={}
          var["angle_act" + str(energy)+ "ang_" + str(a)]={}
          var["angle_gan" + str(energy)+ "ang_" + str(a)]={}
          if angloss==2:
            var["angle2_act" + str(energy)+ "ang_" + str(a)]={}
            var["angle2_gan" + str(energy)+ "ang_" + str(a)]={}
          if addloss:
            var["addloss_act" + str(energy)+ "ang_" + str(a)]={}
            var["addloss_gan" + str(energy)+ "ang_" + str(a)]={}
                      
          var["ecal_act" + str(energy)+ "ang_" + str(a)]={}
          var["ecal_gan" + str(energy)+ "ang_" + str(a)]={}
          var["sumsx_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["sumsy_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["sumsz_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["momentX_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["momentY_gan"+ str(energy)+ "ang_" + str(a)]={}
          var["momentZ_gan"+ str(energy)+ "ang_" + str(a)]={}
                              

       for gen_weights, disc_weights, scale, power, ascale, i in zip(gweights, dweights, xscales, xpowers, angscales, np.arange(len(gweights))):
          gendir = gendirs + '/n_' + str(i)
          discdir = discdirs + '/n_' + str(i)
                            
          if read_gen:
             var["events_gan" + str(energy)]['n_'+ str(i)]= get_gen(energy, gendir)
          else:
             if i==0:
               g=g1
             else:
               g=g2
             g.load_weights(gen_weights)
             start = time.time()
             var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100, (var["angle"+ str(energy)]) * ascale], latent, concat)
             var["events_gan" + str(energy)]['n_'+ str(i)] = post(var["events_gan" + str(energy)]['n_'+ str(i)], scale, power)
             var["events_gan" + str(energy)]['n_'+ str(i)][var["events_gan" + str(energy)]['n_'+ str(i)]< thresh * dscale] = 0

             if save_gen:
                save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], [var["energy" + str(energy)], var["angle"+ str(energy)]], energy, gendir)
             var["events_gan" + str(energy)]['n_'+ str(i)] = pre (var["events_gan" + str(energy)]['n_'+ str(i)], scale, power)
             gen_time = time.time() - start
             print( "Generator took {} seconds to generate {} events".format(gen_time, var["index" +str(energy)]))
          if read_disc:
             disc_out = get_disc(energy, discdir, angloss, addloss, ang)
             var["isreal_act" + str(energy)]['n_'+ str(i)] = disc_out[0]
             var["aux_act" + str(energy)]['n_'+ str(i)] = disc_out[1]
             var["ecal_act"+ str(energy)]['n_'+ str(i)] = disc_out[2]
             var["isreal_gan" + str(energy)]['n_'+ str(i)] = disc_out[3]
             var["aux_gan" + str(energy)]['n_'+ str(i)] = disc_out[4]
             var["ecal_gan"+ str(energy)]['n_'+ str(i)] = disc_out[5]
             var["angle_act"+ str(energy)]['n_'+ str(i)] = disc_out[6]
             var["angle_gan"+ str(energy)]['n_'+ str(i)] = disc_out[7]
             if angloss==2:
                var["angle2_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[8])
                var["angle2_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[9])
             else:
                if addloss:
                    var["addloss_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[8])
                    var["addloss_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out[9])
          else:
             if i==0:
               d=d1
             else:
               d=d2

             d.load_weights(disc_weights)
             start = time.time()
             disc_out_act = discriminate(d, pre(dscale * var["events_act" + str(energy)], scale, power))
             disc_out_gan =discriminate(d, var["events_gan" + str(energy)]['n_'+ str(i)])
             var["isreal_act" + str(energy)]['n_'+ str(i)]= np.array(disc_out_act[0])
             var["isreal_gan" + str(energy)]['n_'+ str(i)]= np.array(disc_out_gan[0])
             var["aux_act" + str(energy)]['n_'+ str(i)] = np.array(disc_out_act[1])
             var["aux_gan" + str(energy)]['n_'+ str(i)]= np.array(disc_out_gan[1])
             var["angle_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[2])
             var["angle_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[2])
             if angloss==2:
                 var["angle2_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[3])
                 var["angle2_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[3])
                 var["ecal_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[4])
                 var["ecal_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[4])
             else:
                 var["ecal_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[3])
                 var["ecal_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[3])
             if addloss:
                 var["addloss_act"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_act[4])
                 var["addloss_gan"+ str(energy)]['n_'+ str(i)] = np.array(disc_out_gan[4])
                                
             disc_time = time.time() - start
             print ("Discriminator took {} seconds for {} data and generated events".format(disc_time, var["index" +str(energy)]))

             if save_disc:
               discout = {}
               for key in var:
                  if key in ["isreal_act" + str(energy), "aux_act" + str(energy), "isreal_gan" + str(energy), "aux_gan" + str(energy), "ecal_act"+ str(energy), "ecal_gan"+ str(energy), "angle2_act"+ str(energy), "angle2_gan"+ str(energy), "angle_act"+ str(energy), "angle_gan"+ str(energy), "addloss_act"+ str(energy), "addloss_gan"+ str(energy)]:
                     discout[key]=var[key]['n_'+ str(i)]
               save_discriminated(discout, energy, discdir, angloss, addloss, ang)
          print ('Calculations for ....', energy)
          var["events_gan" + str(energy)]['n_'+ str(i)] = post(var["events_gan" + str(energy)]['n_'+ str(i)], scale, power)/dscale
          #var["events_gan" + str(energy)]['n_'+ str(i)][var["events_gan" + str(energy)]['n_'+ str(i)]< thresh] = 0
          var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["angle_act"+ str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_act" + str(energy)]['n_'+ str(i)]), np.squeeze((var["angle_act"+ str(energy)]['n_'+ str(i)]))/ascale, np.squeeze(var["ecal_act"+ str(energy)]['n_'+ str(i)]/(dscale * scale))
          var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["angle_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["angle_gan"+ str(energy)]['n_'+ str(i)] )/ascale, np.squeeze(var["ecal_gan"+ str(energy)]['n_'+ str(i)]/(dscale * scale))
          if angloss==2:
              var["angle2_act"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["angle2_act"+ str(energy)]['n_'+ str(i)]))/ascale
              var["angle2_gan"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["angle2_gan"+ str(energy)]['n_'+ str(i)]))/ascale
          if addloss:
              var["addloss_act"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["addloss_act"+ str(energy)]['n_'+ str(i)]))
              var["addloss_gan"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["addloss_gan"+ str(energy)]['n_'+ str(i)]))
                            
          var["max_pos_gan" + str(energy)]['n_'+ str(i)] = get_max(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)] = get_sums(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["momentX_gan" + str(energy)]['n_'+ str(i)], var["momentY_gan" + str(energy)]['n_'+ str(i)], var["momentZ_gan" + str(energy)]['n_'+ str(i)] = get_moments(var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)], m, x=x, y=y, z=z)
          for index, a in enumerate(angles):
             indexes = np.where(((var["angle" + str(energy)]) > np.radians(a) - tolerance2) & ((var["angle" + str(energy)]) < np.radians(a) + tolerance2))
             var["events_gan" + str(energy) + "ang_" + str(a)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["sumsx_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["sumsx_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["sumsy_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["sumsy_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["sumsz_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["sumsz_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["isreal_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["isreal_act" + str(energy)]['n_'+ str(i)][indexes]
             var["isreal_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["isreal_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["aux_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["aux_act" + str(energy)]['n_'+ str(i)][indexes]
             var["aux_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["aux_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["ecal_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["ecal_act" + str(energy)]['n_'+ str(i)][indexes]
             var["ecal_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["ecal_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["angle_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle_act" + str(energy)]['n_'+ str(i)][indexes]
             var["angle_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle_gan" + str(energy)]['n_'+ str(i)][indexes]
             var["momentX_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["momentX_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["momentY_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["momentY_gan"+ str(energy)]['n_'+ str(i)][indexes]
             var["momentZ_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["momentZ_gan"+ str(energy)]['n_'+ str(i)][indexes]
                                       
             if angloss==2:
               var["angle2_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle2_act" + str(energy)]['n_'+ str(i)][indexes]
               var["angle2_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle2_gan" + str(energy)]['n_'+ str(i)][indexes]
             if addloss:
               var["addloss_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["addloss_act" + str(energy)]['n_'+ str(i)][indexes]
               var["addloss_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["addloss_gan" + str(energy)]['n_'+ str(i)][indexes]
       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))
    for i in np.arange(len(gweights)):
      #### Generate GAN table to screen                                                                                                       
      print( "Generated Data for {}".format(i))
      print( "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")

      for energy in energies:
         print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["max_pos_gan" + str(energy)]['n_'+ str(i)], axis=0), np.mean(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["momentX_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentY_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)]['n_'+ str(i)][:, 1])))
    return var

if __name__ == "__main__":
    main()
