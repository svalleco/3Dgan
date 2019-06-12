from __future__ import print_function
from os import path
import ROOT
import h5py
import numpy as np
import keras.backend as K
import tensorflow as tf
#import tensorflow.python.ops.image_ops_impl as image
import time
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../analysis')
from utils.GANutils import *
from utils.RootPlotsGAN import * 
import utils.ROOTutils as roo
import ROOT.TSpectrum2 as sp
from skimage import measure
import math
from AngleArch3dGAN import generator, discriminator
try:
      import setGPU
except:
      pass

def main():
    latent = 256  #latent space
    power=0.85    #power for cell energies used in training
    thresh =0.0   #threshold used
    numdata = 100000
    energies=[0, 110, 150, 190]
    concat = 2
    angles = [62, 90, 118]
    outdir = 'results/Outliers_gan_training_peaks/' # dir for results
    safe_mkdir(outdir)
    datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path
    g = generator(latent)       # build generator
    d = discriminator()
    gen_weight1= "../weights/3dgan_weights_gan_training/params_generator_epoch_089.hdf5" # weights for generator
    disc_weight1= "../weights/3dgan_weights_gan_training/params_discriminator_epoch_089.hdf5" # weights for discriminator
    sortdir = 'SortedData'
    gendir = 'Gen'
    discdir = 'Disc'
    labels=['']
    sigma = 1
    peak_thresh = 0.3
    var = perform_calculations_angle(g, d, gweights=[gen_weight1], dweights=[disc_weight1], energies=energies, angles=angles, 
               datapath=datapath, sortdir=sortdir, gendirs=gendir, discdirs=discdir, num_data=numdata, num_events=5000, m=2, xscales=[1], xpowers=[power], angscales=[1], 
               dscale=50., flags=[True, False, False, False, False, False, False], latent=latent, particle='Ele', angtype='theta', thresh=thresh, addloss=1, concat=concat, 
                pre =taking_power, post =inv_power )
    
    get_plots_angle(var, labels, outdir, energies, angles, angtype='mtheta', m=2, n=1, ifpdf=True, cell=0, sigma=sigma, thresh=peak_thresh)

def taking_power(n, scale=1.0, power=1.0):
   return(np.power(n * scale, power))

def inv_power(n, scale=1.0, power=1.0):
   return(np.power(n, 1.0/power))/scale


##################################### Get plots for variable angle #####################################################################                                                                                                                                                 

def get_plots_angle(var, labels, plots_dir, energies, angles, angtype, m, n, ifpdf=True, stest=True, angloss=1, addloss=0, cell=0, corr=0, grid=True, leg=True, statbox=True, mono=False, sigma=1, thresh=0.2):
   actdir = plots_dir + 'Actual'
   safe_mkdir(actdir)
   discdir = plots_dir + 'disc_outputs'
   safe_mkdir(discdir)
   gendir = plots_dir + 'Generated'
   safe_mkdir(gendir)
   comdir = plots_dir + 'Combined'
   safe_mkdir(comdir)
   mdir = plots_dir + 'Moments'
   safe_mkdir(mdir)
   eventdir = plots_dir + 'Outliers'
   safe_mkdir(eventdir)
   start = time.time()
   plots = 0
   nout = 20 # outliers plotted
   ang=1
   opt="colz"
   for energy in energies:
      x=var["events_act" + str(energy)].shape[1]
      y=var["events_act" + str(energy)].shape[2]
      z=var["events_act" + str(energy)].shape[3]
      maxfile = "Position_of_max_" + str(energy)
      maxlfile = "Position_of_max_" + str(energy)
      histfile = "hist_" + str(energy)
      histlfile = "hist_log" + str(energy)
      ecalfile = "ecal_" + str(energy)
      energyfile = "energy_" + str(energy)
      realfile = "realfake_" + str(energy)
      momentfile = "moment" + str(energy)
      auxfile = "Auxilliary_"+ str(energy)
      ecalerrorfile = "ecal_error" + str(energy)
      angfile = "angle_"+ str(energy)
      aerrorfile = "error_"
      allfile = 'All_energies'
      allecalfile = 'All_ecal'
      allecalrelativefile = 'All_ecal_relative'#.pdf'
      allauxrelativefile = 'All_aux_relative'#.pdf'
      allerrorfile = 'All_relative_auxerror'#.pdf'                                                                                                                                                                                                                                        
      correlationfile = 'Corr'
      if 0 in energies:
         pmin = np.amin(var["energy" + str(energy)])
         pmax = np.amax(var["energy" + str(energy)])
         p = [int(pmin), int(pmax)]
      else:
         p = [100, 200]

      if energy==0:
         #plot_ecal_ratio_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)],
         #                           var["energy" + str(energy)], labels, os.path.join(comdir, allecalfile),
         #                            p, ifpdf=ifpdf, stest=stest, grid=grid, leg=leg, statbox=statbox, mono=mono)
         #plots+=1
         #plot_ecal_relative_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)],
         #                           var["energy" + str(energy)], labels, os.path.join(comdir, allecalrelativefile),
         #                           p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         #plots+=1
         #plot_aux_relative_profile(var["aux_act" + str(energy)], var["aux_gan"+ str(energy)],
         #                          var["energy"+ str(energy)], os.path.join(comdir, allauxrelativefile),
         #                          labels, p, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         #plots+=1
         if corr==1:
           plot_correlation(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],
                           var["sumsz_act"+ str(energy)], var["momentX_act" + str(energy)],
                           var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)],
                           var["ecal_act" + str(energy)],  var["sumsx_gan"+ str(energy)],
                           var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],
                           var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)],
                           var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)],
                           var["energy" + str(energy)], var["events_act" + str(energy)],
                            var["events_gan" + str(energy)], os.path.join(comdir, correlationfile), labels, leg=leg)
         elif corr>1:
           plot_correlation_small(var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)], var["ecal_act" + str(energy)],  var["momentX_gan" + str(energy)],
                                  var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)], var["energy" + str(energy)], var["events_act" + str(energy)],
                                  var["events_gan" + str(energy)], os.path.join(comdir, correlationfile+ "small"), labels, leg=leg)
           plots+=1

         if cell:
           plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                  os.path.join(comdir, 'flat' + 'log' + ecalfile), energy, labels, p=p,
                                  log=1, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
           plots+=1
           plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                  os.path.join(comdir, 'flat' + ecalfile), energy, labels, p=p,
                                  ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
           plots+=1
         #plot_sparsity(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'spartsity'), energy, labels,
         #              threshmin=-13, threshmax=1, logy=0, min_max=0, ifpdf=ifpdf, mono=mono,
         #              leg=leg, grid=grid, statbox=statbox)
         #plots+=1
      plot_ecal_hist(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)],
                     os.path.join(discdir, ecalfile), energy, labels, p, stest=stest,
                     ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      if cell>1:
         plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                os.path.join(comdir, 'flat' + 'log' + ecalfile), energy, labels, p=p,
                                log=1, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
         plots+=1
         plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                os.path.join(comdir, 'flat' + ecalfile), energy, labels, p=p,
                                ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)

      plots+=1
      plot_ecal_hits_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                os.path.join(comdir, 'hits' + ecalfile), energy, labels, p, stest=stest,
                                ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_aux_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)] ,
                    os.path.join(discdir, energyfile), energy, labels, p,
                    ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)],
               x, y, z, os.path.join(actdir, maxfile), os.path.join(gendir, maxfile),
               os.path.join(comdir, maxfile), energy, labels, p=p,
               stest=stest, ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)],
               x, y, z, os.path.join(actdir, maxlfile),
               os.path.join(gendir, maxlfile), os.path.join(comdir, 'log' + maxlfile),
               energy, labels, log=1, p=p, stest=stest, ifpdf=ifpdf, grid=grid, leg=leg, mono=mono)
      plots+=1
      plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],
                               var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)],
                               var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],
                               x, y, z, os.path.join(actdir, histfile), os.path.join(gendir, histfile),
                               os.path.join(comdir, histfile), energy, labels, p=p, stest=stest,
                               ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],
                            var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)],
                            var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],
                            x, y, z, os.path.join(actdir, histlfile), os.path.join(gendir, histlfile),
                            os.path.join(comdir, histlfile), energy, labels, log=1, p=p, stest=stest,
                            ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      plot_realfake_hist(var["isreal_act" + str(energy)], var["isreal_gan" + str(energy)],
                         os.path.join(discdir, realfile), energy, labels, p,
                         ifpdf=ifpdf, grid=grid, leg=leg, statbox=statbox, mono=mono)
      plots+=1
      #plot_primary_error_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)],
      #                   var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile),
      #                   energy, labels, p, grid=grid, ifpdf=ifpdf, statbox=statbox, mono=mono)

      #plots+=1
      #plot_angle_2Dhist(var["angle_act" + str(energy)], var["angle_gan" + str(energy)],  var["angle" + str(energy)],
      #                  os.path.join(discdir, angfile + "ang_2D") , angtype, labels, p,
      #                  ifpdf=ifpdf, grid=grid, leg=leg)
      #plots+=1
      if angloss==2:
         plot_angle_2Dhist(var["angle2_act" + str(energy)], var["angle2_gan" + str(energy)],  var["angle" + str(energy)],
                           os.path.join(discdir, angfile + "ang2_2D") , angtype,
                           labels, p, ifpdf=ifpdf, grid=grid, leg=leg)
         plots+=1
      for mmt in range(m):
         plot_moment(var["momentX_act" + str(energy)], var["momentX_gan" + str(energy)],
                     os.path.join(mdir, 'x' + str(mmt + 1) + momentfile), 'x', energy, mmt,
                     labels, p, ifpdf=ifpdf, grid=grid, leg=leg, mono=mono, stest=stest)
         plots+=1
         plot_moment(var["momentY_act" + str(energy)], var["momentY_gan" + str(energy)],
                     os.path.join(mdir, 'y' + str(mmt + 1) + momentfile), 'y', energy, mmt,
                     labels, p, ifpdf=ifpdf, grid=grid, leg=leg, mono=mono, stest=stest)
         plots+=1
         plot_moment(var["momentZ_act" + str(energy)], var["momentZ_gan" + str(energy)],
                     os.path.join(mdir, 'z' + str(mmt + 1) + momentfile), 'z', energy, mmt,
                     labels, p, ifpdf=ifpdf, grid=grid, leg=leg, mono=mono, stest=stest)
         plots+=1
   
      geventdir =  eventdir + '/GAN_outliers/{}GeV/'.format(energy) 
      safe_mkdir(geventdir)
      deventdir =  eventdir + '/G4_outliers/{}GeV/'.format(energy)
      safe_mkdir(deventdir)
      index_act = np.argsort(var["isreal_act" + str(energy)]['n_0'])
      index_act = index_act[-nout:]
      index_gan = np.argsort(var["isreal_gan" + str(energy)]['n_0'])
      index_gan = index_gan[:nout]
      for n in np.arange(len(index_act)):
          PlotEventPeaks(var["events_act" + str(energy)][index_act][n], var["energy_act" + str(energy)][index_act][n],
                         var["ang_act" + str(energy)][index_act][n], os.path.join(deventdir, 'Event{}.pdf'.format(n)), n, opt=opt, label='G4', sigma=sigma, thresh=thresh)
          PlotEventPeaks(var["events_gan" + str(energy)]['n_0'][index_gan][n], var["energy_gan" + str(energy)][index_gan][n],
                         var["ang_gan" + str(energy)][index_gan][n], os.path.join(geventdir, 'Event{}.pdf'.format(n)), n, opt=opt, label='GAN', sigma=sigma, thresh=thresh)
   print ('Plots are saved in ', plots_dir)
   plot_time= time.time()- start
   print ('{} Plots are generated in {} seconds'.format(plots, plot_time))
                                           

def PlotEventPeaks(event, energy, theta, out_file, n, opt="", unit='degrees', label="", sigma=1, thresh=0.2):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) #make                                              
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   s = sp()
   ang1 = MeasPython(np.moveaxis(event, 3, 0))
   ang2 = MeasPython(np.moveaxis(event, 3, 0), mod=2)
   if unit == 'degrees':
      ang1= np.degrees(ang1)
      ang2= np.degrees(ang2)
      theta = np.degrees(theta)
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   leg.SetTextSize(0.05)
   leg.SetHeader("#splitline{Weighted Histograms for energies}{deposited in x, y and z planes}", "C")
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}'.format(energy, theta), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}'.format(energy, theta), '', x, 0, x, y, 0, y)
   hx.SetStats(0)
   hy.SetStats(0)
   hz.SetStats(0)
   #ROOT.gPad.SetLogz()                                                                                              
   ROOT.gStyle.SetPalette(1)
   event = np.expand_dims(event, axis=0)
   my.FillHist2D_wt(hx, np.sum(event, axis=1))
   my.FillHist2D_wt(hy, np.sum(event, axis=2))
   my.FillHist2D_wt(hz, np.sum(event, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hx)                                                                                                  
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   #my.stat_pos(hy)                                                                                                  
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   nfound = s.Search(hz, sigma, "col", thresh)
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Single {} event with {} peaks'.format(label, nfound),"l")
   leg.AddEntry(hy, 'Energy Input = {:.2f} GeV'.format(energy),"l")
   leg.AddEntry(hz, 'Theta Input  = {:.2f} {}'.format(theta, unit),"l")
   #leg.AddEntry(hz, 'Computed Theta (mean)     = {:.2f} {}'.format(ang1[0], unit),"l")                              
   #leg.AddEntry(hz, 'Computed Theta (weighted) = {:.2f} {}'.format(ang2[0], unit),"l")                              
   leg.Draw()
   #my.stat_pos(hz)                                                                                                  
   canvas.Update()
   canvas.Print(out_file)


def perform_calculations_angle(g, d, gweights, dweights, energies, angles, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, xscales, xpowers, angscales,
                               dscale, flags, latent, particle='Ele', Data=GetAngleData, events_per_file=5000, angtype='theta', thresh=1e-6, offset=0.0, angloss=1,
                               addloss=0, concat=1, pre=preproc, post=postproc, tolerance2 = 0.1):
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
    num_events1= 10000
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
       var = get_sorted_angle(data_files, energies, True, num_events1, num_events2, Data=Data, angtype=angtype, thresh=thresh, offset=offset) # returning a dict with sorted data.                                                                                                                                                             
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
      print ("{} events were put in {} bins".format(total, len(energies)))
    #### Generate Data table to screen
    
    print ("Actual Data")
    print ("Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")
    for energy in energies:
       print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]),
            np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]),
            np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1])))

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
             g.load_weights(gen_weights)
             start = time.time()
             var["events_gan" + str(energy)]['n_'+ str(i)] = generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100, (var["angle"+ str(energy)]) * ascale], latent, concat)
             if save_gen:
                save_generated(var["events_gan" + str(energy)]['n_'+ str(i)], [var["energy" + str(energy)], var["angle"+ str(energy)]], energy, gendir)
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
          var["events_gan" + str(energy)]['n_'+ str(i)][var["events_gan" + str(energy)]['n_'+ str(i)]< thresh] = 0
          var["isreal_act" + str(energy)]['n_'+ str(i)], var["aux_act" + str(energy)]['n_'+ str(i)], var["angle_act"+ str(energy)]['n_'+ str(i)], var["ecal_act"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_act" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_act" + str(energy)]['n_'+ str(i)]), np.squeeze((var["angle_act"+ str(energy)]['n_'+ str(i)]))/ascale, np.squeeze(var["ecal_act"+ str(energy)]['n_'+ str(i)]/(dscale * scale))
          var["isreal_gan" + str(energy)]['n_'+ str(i)], var["aux_gan" + str(energy)]['n_'+ str(i)], var["angle_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)]= np.squeeze(var["isreal_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["aux_gan" + str(energy)]['n_'+ str(i)]), np.squeeze(var["angle_gan"+ str(energy)]['n_'+ str(i)] )/ascale, np.squeeze(var["ecal_gan"+ str(energy)]['n_'+ str(i)]/(dscale * scale))
          if angloss==2:
              var["angle2_act"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["angle2_act"+ str(energy)]['n_'+ str(i)]))/ascale
              var["angle2_gan"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["angle2_gan"+ str(energy)]['n_'+ str(i)]))/ascale
          if addloss:
              var["addloss_act"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["addloss_act"+ str(energy)]['n_'+ str(i)]))
              var["addloss_gan"+ str(energy)]['n_'+ str(i)]=np.squeeze((var["addloss_gan"+ str(energy)]['n_'+ str(i)]))
          ## selecting only few events
          indexes_act = np.where(var["isreal_act" + str(energy)]['n_'+ str(i)] > 0.6)
          indexes_gan = np.where(var["isreal_gan" + str(energy)]['n_'+ str(i)] < 0.4)
 
          # filter g4 events
          var["events_act" + str(energy)] = var["events_act" + str(energy)][indexes_act]
          print('{} g4 events are selected'.format( var["events_act" + str(energy)].shape[0]))
          var["max_pos_act" + str(energy)]= var["max_pos_act" + str(energy)][indexes_act]
          var["sumsx_act"+ str(energy)] = var["sumsx_act"+ str(energy)][indexes_act]
          var["sumsy_act"+ str(energy)] = var["sumsy_act"+ str(energy)][indexes_act]
          var["sumsz_act"+ str(energy)] = var["sumsz_act"+ str(energy)][indexes_act]
          var["momentX_act" + str(energy)] = var["momentX_act" + str(energy)][indexes_act]
          var["momentY_act" + str(energy)] = var["momentY_act" + str(energy)][indexes_act]
          var["momentZ_act" + str(energy)] = var["momentZ_act" + str(energy)][indexes_act]
          var["isreal_act" + str(energy)]['n_'+ str(i)] = var["isreal_act" + str(energy)]['n_'+ str(i)][indexes_act]
          var["ecal_act"+ str(energy)]['n_'+ str(i)] = var["ecal_act"+ str(energy)]['n_'+ str(i)][indexes_act]
          var["aux_act" + str(energy)]['n_'+ str(i)] = var["aux_act" + str(energy)]['n_'+ str(i)][indexes_act]
          var["addloss_act"+ str(energy)]['n_'+ str(i)] = var["addloss_act"+ str(energy)]['n_'+ str(i)][indexes_act]
          var["angle_act"+ str(energy)]['n_'+ str(i)] = var["angle_act"+ str(energy)]['n_'+ str(i)][indexes_act]
          var["energy_act" + str(energy)] =  var["energy" + str(energy)][indexes_act]
          var["ang_act"+ str(energy)] = var["angle"+ str(energy)][indexes_act]
          # filter gan events
          var["events_gan" + str(energy)]['n_'+ str(i)]= var["events_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
          print('{} gan events are selected'.format( var["events_gan" + str(energy)]['n_'+ str(i)].shape[0]))
          var["energy_gan" + str(energy)]=  var["energy" + str(energy)][indexes_gan]
          var["ang_gan"+ str(energy)] = var["angle"+ str(energy)][indexes_gan]
          var["isreal_gan" + str(energy)]['n_'+ str(i)] = var["isreal_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
          var["ecal_gan"+ str(energy)]['n_'+ str(i)] = var["ecal_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
          var["aux_gan" + str(energy)]['n_'+ str(i)] = var["aux_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
          var["addloss_gan"+ str(energy)]['n_'+ str(i)] = var["addloss_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
          var["angle_gan"+ str(energy)]['n_'+ str(i)] = var["angle_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
                                                  
          var["max_pos_gan" + str(energy)]['n_'+ str(i)] = get_max(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)] = get_sums(var["events_gan" + str(energy)]['n_'+ str(i)])
          var["momentX_gan" + str(energy)]['n_'+ str(i)], var["momentY_gan" + str(energy)]['n_'+ str(i)], var["momentZ_gan" + str(energy)]['n_'+ str(i)] = get_moments(var["sumsx_gan"+ str(energy)]['n_'+ str(i)], var["sumsy_gan"+ str(energy)]['n_'+ str(i)], var["sumsz_gan"+ str(energy)]['n_'+ str(i)], var["ecal_gan"+ str(energy)]['n_'+ str(i)], m, x=x, y=y, z=z)
          for index, a in enumerate(angles):
             indexes_act = np.where(((var["ang_act" + str(energy)]) > np.radians(a) - tolerance2) & ((var["ang_act" + str(energy)]) < np.radians(a) + tolerance2))
             indexes_gan = np.where(((var["ang_gan" + str(energy)]) > np.radians(a) - tolerance2) & ((var["ang_gan" + str(energy)]) < np.radians(a) + tolerance2))
             var["events_gan" + str(energy) + "ang_" + str(a)]['n_'+ str(i)] = var["events_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
             var["sumsx_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["sumsx_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
             var["sumsy_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["sumsy_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
             var["sumsz_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["sumsz_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
             var["isreal_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["isreal_act" + str(energy)]['n_'+ str(i)][indexes_act]
             var["isreal_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["isreal_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
             var["aux_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["aux_act" + str(energy)]['n_'+ str(i)][indexes_act]
             var["aux_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["aux_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
             var["ecal_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["ecal_act" + str(energy)]['n_'+ str(i)][indexes_act]
             var["ecal_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["ecal_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
             var["angle_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle_act" + str(energy)]['n_'+ str(i)][indexes_act]
             var["angle_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
             var["momentX_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["momentX_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
             var["momentY_gan"+ str(energy)+ "ang_" + str(a)]['n_'+ str(i)] =var["momentY_gan"+ str(energy)]['n_'+ str(i)][indexes_gan]
             

             if angloss==2:
               var["angle2_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle2_act" + str(energy)]['n_'+ str(i)][indexes_act]
               var["angle2_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["angle2_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
             if addloss:
               var["addloss_act" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["addloss_act" + str(energy)]['n_'+ str(i)][indexes_act]
               var["addloss_gan" + str(energy)+ "ang_" + str(a)]['n_'+ str(i)] = var["addloss_gan" + str(energy)]['n_'+ str(i)][indexes_gan]
       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))
    for i in np.arange(len(gweights)):
      #### Generate GAN table to screen                                                                                                                                 
      print( "Generated Data for {}".format(i))
      print( "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2")

      for energy in energies:
         print ("{}\t{}\t{:.4f}\t\t{}\t\t\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["max_pos_gan" + str(energy)]['n_'+ str(i)], axis=0), np.mean(var["events_gan" + str(energy)]['n_'+ str(i)]), np.mean(var["momentX_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentY_gan"+ str(energy)]['n_'+ str(i)][:, 1]), np.mean(var["momentZ_gan"+ str(energy)]['n_'+ str(i)][:, 1])))

      index_act = np.where(var["isreal_act" + str(energy)]['n_'+ str(i)] > 0.6)
      print(len(index_act))
      
    return var
                               

if __name__ == "__main__":
    main()
      
