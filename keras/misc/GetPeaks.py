from __future__ import print_function
from os import path
import ROOT
import h5py
import os
## setting seed ###
from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(10)
import random
random.seed(10)
os.environ['PYTHONHASHSEED'] = '0'
##################

import numpy as np
import keras.backend as K
import tensorflow as tf
#import tensorflow.python.ops.image_ops_impl as image 
import time
import sys
sys.path.insert(0,'../keras')
sys.path.insert(0,'../keras/analysis')
import utils.GANutils as gan
import utils.ROOTutils as roo
import ROOT.TSpectrum2 as sp2
import ROOT.TSpectrum as sp
import math
from CheckOutliers import PlotEventPeaks
from AngleArch3dGAN import generator, discriminator
try:
  import setGPU
except:
  pass

def main():
  latent = 256  #latent space
  power=0.85    #power for cell energies used in training
  thresh =0.0   #threshold used
  get_shuffled= True # whether to make plots for shuffled
  labels =["G4", "GAN"] # labels
  plotsdir = 'results/peaks2_test_data' # dir for results
  gan.safe_mkdir(plotsdir) 
  #datapath = "/bigdata/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path
  datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech 
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  #energies =[110, 150, 190] 
  energies = [400]
  angles=[62, 118]
  L=1e-6
  concat=2
  dscale =50.
  g = generator(latent)       # build generator
  gen_weight1= "../keras/weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[-10:], energies, num_events1=10000, num_events2=5000, thresh=thresh) # load data in a dict
  sigma = 1
  peak_threshold=0.3
  # for each energy bin
  plist_g4=[]
  plist_gan=[]
  elist_g4=[]
  elist_gan=[]
  for energy in energies:
     for a in angles:
       filename = path.join(plotsdir, "peaks_{}GeV_{}degree.pdf".format(energy, a)) # file name
       if a==0: 
            indexes = np.where(sorted_data["angle" + str(energy)] > 0)
       else:
            indexes = np.where(((sorted_data["angle" + str(energy)]) > math.radians(a) - 0.1) & ((sorted_data["angle" + str(energy)]) < math.radians(a) + 0.1))
       sorted_data["events_act" + str(energy) + "ang_" + str(a)] = sorted_data["events_act" + str(energy)][indexes]
       sorted_data["energy" + str(energy) + "ang_" + str(a)] = sorted_data["energy" + str(energy)][indexes]
       sorted_data["angle" + str(energy) + "ang_" + str(a)] = sorted_data["angle" + str(energy)][indexes]
       
       index= sorted_data["events_act" + str(energy)+ "ang_" + str(a)].shape[0]  # number of events in bin
       # generate images
       generated_images = gan.generate(g, index, [sorted_data["energy" + str(energy)+ "ang_" + str(a)]/100.
                                    , sorted_data["angle" + str(energy)+ "ang_" + str(a)]]
                                       , latent, concat=concat)
       # post processing
       generated_images = np.power(generated_images, 1./power)
       peaks_g4 = FindPeaks(sorted_data["events_act" + str(energy) + "ang_" + str(a)],  sigma, peak_threshold)
       peaks_gan = FindPeaks(generated_images,  sigma, peak_threshold)
       generated_images = generated_images/dscale
       sorted_data["events_act" + str(energy) + "ang_" + str(a)] = sorted_data["events_act" + str(energy) + "ang_" + str(a)]/dscale
       #Draw1d(peaks_g4, peaks_gan, filename, labels)
       
       indexes_gan = np.where(peaks_gan > 1)
       events_gan = generated_images[indexes_gan]
       n = events_gan.shape[0]
       plist_gan.append((n * 100.0)/index)
       if n>0:
         elist_gan.append((plist_gan[-1] * np.sqrt(n))/n)
       else:
         elist_gan.append(0.)    
       print('fraction gan', plist_gan[-1])
       print('error gan', elist_gan[-1])
       print("For energy {} GeV and angle {} degrees there are {}/{} events with multi peaks for GAN".format(energy, a, n, index))
       indexes_act = np.where(peaks_g4 > 1)
       events_act = sorted_data["events_act" + str(energy) + "ang_" + str(a)][indexes_act]
       m = events_act.shape[0]
       plist_g4.append((m * 100.0)/index)
       if m>0:
         elist_g4.append((plist_g4[-1] * np.sqrt(m))/m)
       else:
         elist_g4.append(0.)
       print('fraction g4', plist_g4[-1])
       print('error g4', elist_g4[-1])                                                                                                                                                                                                                                             
       print("For energy {} GeV and angle {} degrees there are {}/{} events with multi peaks from G4".format(energy, a, m, index))  
       
       edir = plotsdir + '/events_{}GeV_{}degree/'.format(energy, a)
       gan.safe_mkdir(edir)
       if n > 0:
         e1 = sorted_data["energy" + str(energy)+ "ang_" + str(a)][indexes_gan]
         ang1 = sorted_data["angle" + str(energy) + "ang_" + str(a)][indexes_gan]
         p1 = peaks_gan[indexes_gan]
         gdir = edir + 'GAN'
         gan.safe_mkdir(gdir)
         for i in np.arange(min(n, 10)):
           PlotEventPeaks(events_gan[i], e1[i], ang1[i], path.join(gdir, 'Event{}.pdf'.format(i)), i, opt='colz', label='GAN'.format(p1[i]), sigma=sigma, thresh=peak_threshold)
       if m > 0:
         e2 = sorted_data["energy" + str(energy)+ "ang_" + str(a)][indexes_act]
         ang2 = sorted_data["angle" + str(energy) + "ang_" + str(a)][indexes_act]
         p2 = peaks_g4[indexes_act]
         ddir =edir + 'G4'
         gan.safe_mkdir(ddir)
         for i in np.arange(min(m, 10)):
            PlotEventPeaks(events_act[i], e2[i], ang2[i], path.join(ddir, 'Event{}.pdf'.format(i)), i, opt='colz', label='G4'.format(p2[i]), sigma=sigma, thresh=peak_threshold)
       
  DrawGraph2(plist_g4, plist_gan, elist_g4, elist_gan, energies, angles, path.join(plotsdir, 'peak_percent.pdf'), labels=['G4', 'GAN']) 
                              
def FindPeaks(events, sigma, thresh):
   n = events.shape[0]
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
   s = sp2()
   hz = []
   peaks =[]
   for i, event in enumerate(events):
     hz.append(ROOT.TH2F('z_{}'.format(i), '', x, 0, x, y, 0, y))
     event = np.expand_dims(event, axis=0)
     roo.FillHist2D_wt(hz[i], np.sum(event, axis=3))
     nfound = s.Search(hz[i], sigma, "col", thresh)
     peaks.append(nfound)
   return(np.array(peaks))

def FindPeaks1D(events, sigma, thresh):
  n = events.shape[0]
  x = events.shape[1]
  y = events.shape[2]
  z = events.shape[3]
  s = sp()
  hz = []
  peaks =[]
  print(events.shape)
  for i, event in enumerate(events):
    hz.append(ROOT.TH1F('z_{}'.format(i), '', z, 0, z))
    event = np.sum(event, axis=(0, 1))
    roo.fill_hist(hz[i], event)
    nfound = s.Search(hz[i], sigma, "col", thresh)
    peaks.append(nfound)
  return(np.array(peaks))
                                                      
 
# Plot for MSCN
def Draw1d(array1, array2, filename, labels):
  c=ROOT.TCanvas("c" ,"" ,200 ,10 ,700 ,500)
  c.SetGrid()
  min1=np.amin(array1)
  min2=np.amin(array2)
  max1=np.amax(array1)
  max2=np.amax(array2)
  mean1=np.mean(array1)
  mean2=np.mean(array2)
  n = array1.shape[0]
  fr_g4 = np.sum(np.where(array1>1, 1., 0.))/n
  fr_gan = np.sum(np.where(array2>1, 1., 0.))/n
  ROOT.gStyle.SetOptStat(111111)
  leg=ROOT.TLegend(.1, .1, .3, .3)
  hist1 = ROOT.TH1D(labels[0], "Number of Peaks for {} events;Peaks;normalized count".format(n), 12, -1, 5)
  hist2 = ROOT.TH1D(labels[1], "Number of Peaks for {} events;Peaks;normalized count".format(n), 12, -1, 5)
  hist1.Sumw2()
  hist2.Sumw2()
  roo.fill_hist(hist1, array1)
  roo.fill_hist(hist2, array2)
  roo.normalize(hist1, 1)
  roo.normalize(hist2, 1)
  leg.AddEntry(hist1, "{} ({:.4f} %)".format(labels[0], fr_g4 * 100), "l")
  leg.AddEntry(hist2, "{} ({:.4f} %)".format(labels[1], fr_gan * 100), "l")
  hist1.GetYaxis().SetRangeUser(0, 1.2)
  hist1.Draw()
  hist1.Draw('hist')
  hist2.Draw('sames')
  hist2.Draw('hist sames')
  c.Update()
  roo.stat_pos(hist2)
  hist1.SetLineColor(2)
  hist2.SetLineColor(4)
  leg.Draw()
  
  c.Update()
  c.Print(filename)
  print (' The plot is saved in.....{}'.format(filename))
                  
def DrawGraph(list1, list2, elist1, elist2,energies, angles, filename, labels):
  c=ROOT.TCanvas("c" ,"" ,200 ,10 ,700 ,500)
  ROOT.gStyle.SetOptStat(111111)
  leg=ROOT.TLegend(.7, .6, .9, .9)
  mg=ROOT.TMultiGraph()
  n = len(energies)
  m = len(angles)
  color=2
  peaks_g4 = []
  peaks_gan = []
  ex_g4 = []
  ex_gan = []
  ey_g4 = []
  ey_gan = []

  for j in np.arange(m):
    peaks_g4.append(np.zeros(n))
    peaks_gan.append(np.zeros(n))
    ex_g4.append(np.zeros(n))
    ex_gan.append(np.zeros(n))
    ey_g4.append(np.zeros(n))
    ey_gan.append(np.zeros(n))
  e = np.zeros(n)
  for i in np.arange(n):
    e[i] = energies[i]
    for j in np.arange(m):
      peaks_g4[j][i] = list1[(i*(m-1)) + j]
      peaks_gan[j][i] = list2[(i*(m-1)) + j]
      ex_g4[j][i] = elist1[(i*(m-1)) + j]
      ex_gan[j][i] = elist2[(i*(m-1)) + j]
      ey_g4[j][i] = 0.5
      ey_gan[j][i] = 0.5

  graphs_g4 =[]
  graphs_gan =[]
  
  for i in np.arange(m):
    graphs_g4.append(ROOT.TGraphErrors(n, e, peaks_g4[i], ex_g4[i], ey_g4[i]))
    mg.Add(graphs_g4[i])
    graphs_g4[i].SetLineColor(color)
    graphs_g4[i].SetMarkerColor(color)
    graphs_g4[i].SetMarkerStyle(21)
    leg.AddEntry(graphs_g4[i], "{} {} degree".format(labels[0], angles[i]), "l")
    graphs_gan.append(ROOT.TGraphErrors(n, e, peaks_gan[i], ex_gan[i], ey_gan[i]))
    mg.Add(graphs_gan[i])
    graphs_gan[i].SetLineColor(color)
    graphs_gan[i].SetLineStyle(2)
    graphs_gan[i].SetMarkerColor(color)
    graphs_gan[i].SetMarkerStyle(21)
    leg.AddEntry(graphs_gan[i], "{} {} degree".format(labels[1], angles[i]), "l")
    color+=2        
  mg.SetTitle("Percentage of events with multiple peaks;Ep;percentage [%]")
  mg.GetYaxis().SetRangeUser(0.,10.)
  mg.Draw('ALP')
  c.Update()
  leg.Draw()
              
  c.Update()
  c.Print(filename)
  c.Print('peaks_graph.C')
  print (' The plot is saved in.....{}'.format(filename))

def DrawGraph2(list1, list2, elist1, elist2,energies, angles, filename, labels):
  c=ROOT.TCanvas("c" ,"" ,200 ,10 ,700 ,500)
  ROOT.gStyle.SetOptStat(111111)
  leg=ROOT.TLegend(.7, .6, .9, .9)
  mg=ROOT.TMultiGraph()
  n = len(energies)
  color=2
  peaks_g4 = np.zeros(n)
  peaks_gan = np.zeros(n)
  ex_g4 = np.zeros(n)
  ex_gan = np.zeros(n)
  ey_g4 = np.zeros(n)
  ey_gan = np.zeros(n)
  e = np.zeros(n)
  for i in np.arange(n):
    peaks_g4[i] = list1[i]
    peaks_gan[i] = list2[i]
    ex_g4[i] = elist1[i]
    ex_gan[i] = elist2[i]
    ey_g4[i] = 0.5
    ey_gan[i] = 0.5
    e[i]=energies[i]

  graphs_g4=ROOT.TGraphErrors(n, e, peaks_g4, ex_g4, ey_g4)
  mg.Add(graphs_g4)
  graphs_g4.SetLineColor(color)
  graphs_g4.SetMarkerColor(color)
  graphs_g4.SetMarkerStyle(21)
  leg.AddEntry(graphs_g4, "{} ".format(labels[0]), "l")
  graphs_gan =ROOT.TGraphErrors(n, e, peaks_gan, ex_gan, ey_gan)
  mg.Add(graphs_gan)
  graphs_gan.SetLineColor(color+2)
  graphs_gan.SetLineStyle(2)
  graphs_gan.SetMarkerColor(color)
  graphs_gan.SetMarkerStyle(21)
  leg.AddEntry(graphs_gan, "{}".format(labels[1]), "l")
  color+=2
  mg.SetTitle("Percentage of events with multiple peaks;Ep;percentage [%]")
  mg.GetYaxis().SetRangeUser(0.,10.)
  mg.Draw('ALP')
  c.Update()
  leg.Draw()

  c.Update()
  c.Print(filename)
  c.Print('peaks_graph.C')
  print (' The plot is saved in.....{}'.format(filename))
                                                                          

if __name__ == "__main__":
  main()
