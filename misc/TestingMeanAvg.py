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
import utils.GANutils as gan
import utils.ROOTutils as roo
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
  get_shuffled= True # whether to make plots for shuffled
  labels =["G4", "GAN"] # labels
  plotsdir = 'results/TestMeanStd' # dir for results
  gan.safe_mkdir(plotsdir) 
  datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path     
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  energies =[0, 110, 150, 190]# energy bins
  #angles=[math.radians(x) for x in [62, 90, 118]]
  angles=[62, 90, 118]
  L=1e-2
  g = generator(latent)       # build generator
  gen_weight1= "../weights/3dgan_weights_bins_pow_p85/params_generator_epoch_059.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[24:], energies, thresh=thresh) # load data in a dict

  # for each energy bin
  for energy in energies:
     for a in angles:
       filename = path.join(plotsdir, "{}GeV_{}degree".format(energy, a)) # file name
       indexes = np.where(((sorted_data["angle" + str(energy)]) > math.radians(a) - 0.1) & ((sorted_data["angle" + str(energy)]) < math.radians(a) + 0.1))
       sorted_data["events_act" + str(energy) + "ang_" + str(a)] = sorted_data["events_act" + str(energy)][indexes]
       sorted_data["energy" + str(energy) + "ang_" + str(a)] = sorted_data["energy" + str(energy)][indexes]
       sorted_data["angle" + str(energy) + "ang_" + str(a)] = sorted_data["angle" + str(energy)][indexes]
       
       index= sorted_data["events_act" + str(energy)+ "ang_" + str(a)].shape[0]  # number of events in bin
       # generate images
       generated_images = gan.generate(g, index, [sorted_data["energy" + str(energy)+ "ang_" + str(a)]/100.
                                    , sorted_data["angle" + str(energy)+ "ang_" + str(a)]]
                                    , latent, concat=1)
       # post processing
       generated_images = np.power(generated_images, 1./power)
       generated_images = generated_images/50.
       sorted_data["events_act" + str(energy) + "ang_" + str(a)]= sorted_data["events_act" + str(energy) + "ang_" + str(a)]/50.

       # Get MSCN Coefficients
       means_g4, std_g4= Means_window(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], window_size=(3, 3, 5))
       means_gan, std_gan= Means_window(generated_images, window_size=(3, 3, 5))
       corr = Corr_window(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], generated_images, window_size=(3, 3, 5))
       #flattening
       m_g4= means_g4.flatten()
       m_gan=means_gan.flatten()
       s_g4= std_g4.flatten()
       s_gan=std_gan.flatten()
       corr = corr.flatten()
       Draw1d([m_g4, m_gan], a, energy, filename + '_Mean.pdf', labels, 'mean')
       Draw1d([s_g4, s_gan], a, energy, filename + '_std.pdf', labels, 'std')
       ssim_l = 2 * m_g4 * m_gan/(m_g4**2 + m_gan**2)
       ssim_c = 2 * s_g4 * s_gan/(s_g4**2 + s_gan**2)
       ssim_corr = np.where((s_g4>0) & (s_gan>0), (4 * corr * m_g4 * m_gan)/((m_g4**2 + m_gan**2)*(s_g4**2 + s_gan**2)), 0)
       print('ssim corr', ssim_corr.shape)
       print('max', np.amax(ssim_corr))
       print('min', np.amin(ssim_corr))
       Draw1d([ssim_l, ssim_c], a, energy, filename + '_ssim.pdf', ["luminosity", "contrast"], 'index', ylog=1)
       Draw1d([corr], a, energy, filename + '_corr.pdf', ["corr"], 'correlation')
       Draw1d([ssim_corr], a, energy, filename + '_ssim_corr.pdf', ["structural similarity"], 'index', xlog=0)
              
# mean of non zero
def mean_sparse(x):
  count = np.sum(np.where(x!=0, 1., 0.), axis=(1, 2, 3))
  sum = np.sum(x, axis=(1, 2, 3))
  mean = np.zeros(count.shape)
  indexes = np.where(count > 0)
  mean[indexes]=np.divide(sum[indexes], count[indexes])
  return mean, count

# std of non zero
def std_sparse(x, mean, count, debug=0):
  mean = mean.reshape((mean.shape[0], 1, 1, 1))
  std = x - mean
  std[x==0]=0.
  std= np.square(std)
  std = np.sum(std, axis=(1, 2, 3))
  indexes = np.where(count > 0)
  std[indexes] = np.divide(std[indexes], count[indexes])
  std = np.sqrt(std)
  return std

# MSCN considering only non zero points
def Means_window(images, window_size=(3, 3, 5)):
  images = np.squeeze(images)
  x_shape= images.shape[1]
  y_shape= images.shape[2]
  z_shape= images.shape[3]
  means=[]
  stds=[]  
  for x in np.arange(0, x_shape, window_size[0]):
    for y in np.arange(0, y_shape, window_size[1]):
      for z in np.arange(0, z_shape, window_size[2]):
        image_slice=images[:, x:x+window_size[0], y:y+window_size[0], z:z+window_size[1]]
        mean = np.mean(image_slice, axis=(1, 2, 3))
        std = np.std(image_slice, axis=(1, 2, 3))
        
        means.append(mean)
        stds.append(std)
  return np.array(means), np.array(stds)

# MSCN considering only non zero points
def Corr_window(images1, images2, window_size=(3, 3, 5)):
    images1 = np.squeeze(images1)
    images2 = np.squeeze(images2)
    x_shape= images1.shape[1]
    y_shape= images1.shape[2]
    z_shape= images1.shape[3]
    num_events= images1.shape[0]
    corr=[]
    for n in np.arange(num_events):
      for x in np.arange(0, x_shape, window_size[0]):
        for y in np.arange(0, y_shape, window_size[1]):
          for z in np.arange(0, z_shape, window_size[2]):
            image_slice1=images1[n, x:x+window_size[0], y:y+window_size[0], z:z+window_size[1]]
            image_slice2=images2[n, x:x+window_size[0], y:y+window_size[0], z:z+window_size[1]]
            c = np.corrcoef(image_slice1.flatten(), image_slice2.flatten())
            corr.append(c[0, 1])
    return np.array(corr)
                                                                        
# Plot for MSCN
def Draw1d(arrays, angle, energy, filename, labels, metric="", ylog=0, xlog=1):
  c=ROOT.TCanvas("c" ,"" ,200 ,10 ,700 ,500)
  c.SetGrid()
  if ylog:
    ROOT.gPad.SetLogy()
  if xlog:
    ROOT.gPad.SetLogx()
  ROOT.gStyle.SetOptStat(111111)
  leg=ROOT.TLegend(.8, .1, .9, .3)
  l = "{} for G4 vs. GAN for {}GeV and {} degrees".format(metric, energy, angle)
  lx = "{}".format(metric)
  hists=[]
  col=2
  
  for i, array in enumerate(arrays):
    if xlog:
      min = -8
      max = 1
    else:
      if i==0:
        min = 0.9 * np.amin(array)
        max = 1.1 * np.amax(array)
    hists.append(ROOT.TH1D(labels[0] + str(i), "", 100, min, max))
    hists[i].Sumw2()
    roo.BinLogX(hists[i])
    roo.fill_hist(hists[i], array.flatten())
    roo.normalize(hists[i], 1)
    leg.AddEntry(hists[i], "{}".format(labels[i]), "l")
    if i==0:
      hists[i].GetYaxis().SetTitle('normalized entries')
      hists[i].GetYaxis().CenterTitle()
      hists[i].GetXaxis().SetTitle(lx)
      hists[i].SetTitle(l)
      hists[i].Draw()
      hists[i].Draw('hist')
      hists[i].Draw('sames')
    else:
      hists[i].Draw('hist sames')
      c.Update()
      if i>0: roo.stat_pos(hists[i])
    hists[i].SetLineColor(col)
    col+=2
    c.Update()
  leg.Draw()
  c.Update()
  c.Print(filename)
  print (' The plot is saved in.....{}'.format(filename))
                  
if __name__ == "__main__":
  main()
