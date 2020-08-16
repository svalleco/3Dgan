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
import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as roo
from skimage import measure
import math
import random
from AngleArch3dGAN import generator, discriminator
try:
  import setGPU
except:
  pass

def main():
  datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5"
  #datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"
  particle = 'Ele'
  data_files = gan.GetDataFiles(datapath, [particle]) # get list of files
  energies =[0, 50, 100, 200, 300, 400, 500]# energy bins
  angles=[62, 90, 118]

  latent = 256  #latent space
  power=0.85    #power for cell energies used in training
  thresh =0   #threshold used
  concat=2 # concat to latent
  dscale =50.0 # energies = GeV/dscale

  stest = True
  get_shuffled= True # whether to make plots for shuffled
  
  plotsdir = 'results/SSIM_gan_training_epsilon_2_500GeV_ep21_thesis/' # dir for results
  gan.safe_mkdir(plotsdir) 
  L=[1, 1e-2, 1e-4, 1e-6]

  g = generator(latent)       # build generator
  gen_weight= "../weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5" # weights for generator
  g.load_weights(gen_weight) # load weights
  sorted_data = gan.get_sorted_angle(data_files[-20:], energies, thresh=thresh) # load data in a dict
  
  ssim_dict={}
  for i, l in enumerate(L):
     # for each energy bin
     ssim_dict['L{}'.format(str(i))]={}
     for energy in energies:
       ssim_dict['L{}'.format(str(i))]['energy{}'.format(str(energy))]={}
       for a in angles:
          filename = path.join(plotsdir, "IQA_{}GeV_{}degree.pdf".format(energy, a)) # file name
          indexes = np.where(((sorted_data["angle" + str(energy)]) > math.radians(a) - 0.1) & ((sorted_data["angle" + str(energy)]) < math.radians(a) + 0.1))
          sorted_data["events_act" + str(energy) + "ang_" + str(a)] = sorted_data["events_act" + str(energy)][indexes]
          sorted_data["energy" + str(energy) + "ang_" + str(a)] = sorted_data["energy" + str(energy)][indexes]
          sorted_data["angle" + str(energy) + "ang_" + str(a)] = sorted_data["angle" + str(energy)][indexes]
          sorted_data["events_act" + str(energy) + "ang_" + str(a)] = sorted_data["events_act" + str(energy) + "ang_" + str(a)]/dscale
          index= sorted_data["events_act" + str(energy)+ "ang_" + str(a)].shape[0]  # number of events in bin
          # generate images
          generated_images = gan.generate(g, index, [sorted_data["energy" + str(energy)+ "ang_" + str(a)]/100.
                                    , sorted_data["angle" + str(energy)+ "ang_" + str(a)]]
                                       , latent, concat=concat)
          # post processing
          generated_images = np.power(generated_images, 1./power)
          generated_images = generated_images /dscale
          generated_images[generated_images < thresh]=0

          shuffled_data = sorted_data["events_act" + str(energy)+ "ang_" + str(a)]* 1.
          np.random.shuffle(shuffled_data)

          shuffled_gan = generated_images * 1.
          np.random.shuffle(shuffled_gan)
       
          ssim_array=SSIM(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], shuffled_gan, multichannel=True, data_range=l, gaussian_weights=True, use_sample_covariance=False, shuffle=get_shuffled)

          ssim_dict['L{}'.format(str(i))]['energy' + str(energy)]['angle' + str(a)]={}
          ssim_dict['L{}'.format(str(i))]['energy' + str(energy)]['angle' + str(a)]['G4XGAN']=ssim_array

          print('Energy={}    Angle={}'.format(energy, a))
          print('G4 x GAN: SSIM mean ={} std ={}'.format(np.mean(ssim_array), np.std(ssim_array)))

          ssim_g4=SSIM(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], shuffled_data, multichannel=True, data_range=l, gaussian_weights=True, use_sample_covariance=False, shuffle=get_shuffled)
          ssim_dict['L{}'.format(str(i))]['energy' + str(energy)]['angle' + str(a)]['G4XG4']=ssim_g4
       
          print('G4 x G4  : SSIM Data mean ={} std={}'.format(np.mean(ssim_g4), np.std(ssim_g4)))

          ssim_gan=SSIM(generated_images, shuffled_gan, multichannel=True, data_range=l, gaussian_weights=True, use_sample_covariance=False, shuffle=get_shuffled)
          ssim_dict['L{}'.format(str(i))]['energy' + str(energy)]['angle' + str(a)]['GANXGAN']=ssim_gan
          print('GAN X GAN: SSIM GAN  mean={} std={}'.format(np.mean(ssim_gan), np.std(ssim_gan)))
     for a in angles:
          DrawMulti(ssim_dict['L{}'.format(str(i))], energies[1:], a, l, path.join(plotsdir, "L{}_degree{}.pdf".format(i, a)))
          DrawScatter(ssim_dict['L{}'.format(str(i))]['energy0'], sorted_data["energy0" + "ang_" + str(a)], a, l, path.join(plotsdir, "L{}_degree{}_scatter.pdf".format(i, a)))
                                        

def SSIM(images1, images2, multichannel=True, data_range=1, gaussian_weights=True, use_sample_covariance=False, shuffle=True):
  ssim_val=[]
  images3 = images2 * 1.0
  np.random.shuffle(images3)
  for i in np.arange(images1.shape[0]):
     ssim_val.append(measure.compare_ssim(images1[i], images2[i], multichannel=multichannel, data_range=data_range, gaussian_weights=gaussian_weights, use_sample_covariance=use_sample_covariance))
  return np.array(ssim_val)

# Plot for MSCN
def DrawMulti(dict_L, energies, angle, L, filename, grid=0):
  c=ROOT.TCanvas("c" ,"" ,200 ,10 ,700 ,500)
  if grid: c.SetGrid()
  color = 2
  legend = ROOT.TLegend(.65, .65, .89, .89)
  legend.SetBorderSize(0)
  mg=ROOT.TMultiGraph()
  g_g4xgan = ROOT.TGraphErrors(len(energies))

  for i, energy in enumerate(energies):
      mean = np.mean(dict_L['energy' + str(energy)]['angle' + str(angle)]['G4XGAN'])
      g_g4xgan.SetPoint(i, energy, mean)
      g_g4xgan.SetPointError(i, 5, np.std(dict_L['energy' + str(energy)]['angle' + str(angle)]['G4XGAN']))
  g_g4xgan.SetLineColor(color)
  g_g4xgan.SetMarkerColor(color)
  g_g4xgan.SetMarkerStyle(21)
  
  mg.Add(g_g4xgan)
  legend.AddEntry(g_g4xgan, "MC vs. GAN", "l")

  color+=2
  g_g4xg4 = ROOT.TGraphErrors(len(energies))
  for i, energy in enumerate(energies):
      g_g4xg4.SetPoint(i, energy, np.mean(dict_L['energy' + str(energy)]['angle' + str(angle)]['G4XG4']))
      g_g4xg4.SetPointError(i, 5, np.std(dict_L['energy' + str(energy)]['angle' + str(angle)]['G4XG4']))
  g_g4xg4.SetLineColor(color)
  g_g4xg4.SetMarkerColor(color)
  g_g4xg4.SetMarkerStyle(21)
  mg.Add(g_g4xg4)
  legend.AddEntry(g_g4xg4, "MC vs. MC", "l")

  color+=2
  g_ganxgan = ROOT.TGraphErrors(len(energies))
  for i, energy in enumerate(energies):
      g_ganxgan.SetPoint(i, energy, np.mean(dict_L['energy' + str(energy)]['angle' + str(angle)]['GANXGAN']))
      g_ganxgan.SetPointError(i, 5, np.std(dict_L['energy' + str(energy)]['angle' + str(angle)]['GANXGAN']))
  g_ganxgan.SetLineColor(color)
  g_ganxgan.SetMarkerColor(color)
  g_ganxgan.SetMarkerStyle(21)
  mg.Add(g_ganxgan)
  legend.AddEntry(g_ganxgan, "GAN vs. GAN", "l")

  mg.SetTitle("SSIM: L = {}   #theta = {}#circ;Ep [GeV];SSIM".format(L, angle))
  if L == 1:
    mg.GetYaxis().SetRangeUser(0.9,  1.1)
  elif L==0.01:
    mg.GetYaxis().SetRangeUser(0.8,  1)
  else:
    mg.GetYaxis().SetRangeUser(0.,  1)
  mg.GetYaxis().SetTitleSize(0.045)
  mg.GetXaxis().SetTitleSize(0.045)
  mg.Draw('ALP')
  c.Update()
  legend.Draw()
  c.Update()

  c.Print(filename)
  print (' The plot is saved in.....{}'.format(filename))

# Plot for MSCN                                                                                                                                                                                                                                                                           
def DrawScatter(dict_E, energy, angle, L, filename, labels="", grid=0):
  c=ROOT.TCanvas("c" ,"" ,200 ,10 ,700 ,500)
  if grid: c.SetGrid()
  color = 2
  legend = ROOT.TLegend(.7, .7, .89, .89)
  legend.SetBorderSize(0)
  mg=ROOT.TMultiGraph()
  g_g4xgan = ROOT.TGraph(energy.shape[0])

  for i in np.arange(energy.shape[0]):
    g_g4xgan.SetPoint(i, energy[i], dict_E['angle' + str(angle)]['G4XGAN'][i])
  g_g4xgan.SetMarkerColor(color)
  g_g4xgan.SetMarkerStyle(3)
  mg.Add(g_g4xgan)
  legend.AddEntry(g_g4xgan, "G4 x GAN", "p")
  color+=2
  g_g4xg4 = ROOT.TGraph(energy.shape[0])
  for i in np.arange(energy.shape[0]):
    g_g4xg4.SetPoint(i, energy[i], dict_E['angle' + str(angle)]['G4XG4'][i])
  g_g4xg4.SetMarkerColor(color)
  g_g4xg4.SetMarkerStyle(3)
  mg.Add(g_g4xg4)
  legend.AddEntry(g_g4xg4, "G4 x G4", "p")

  color+=2
  g_ganxgan = ROOT.TGraph(energy.shape[0])
  for i in np.arange(energy.shape[0]):
    g_ganxgan.SetPoint(i, energy[i], dict_E['angle' + str(angle)]['GANXGAN'][i])
  g_ganxgan.SetMarkerColor(color)
  g_ganxgan.SetMarkerStyle(3)
  mg.Add(g_ganxgan)
  legend.AddEntry(g_ganxgan, "GAN x GAN", "p")
  mg.SetTitle("SSIM: L={} Angle={} #circ;Ep [GeV];ssim".format(L, angle))
  mg.Draw('AP')
  c.Update()
  legend.Draw()
  c.Update()

  c.Print(filename)
  print (' The plot is saved in.....{}'.format(filename))
                  
if __name__ == "__main__":
  main()
