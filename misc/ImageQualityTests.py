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
  plotsdir = 'results/IQA_p85_ep27' # dir for results
  gan.safe_mkdir(plotsdir) 
  datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path     
  data_files = gan.GetDataFiles(datapath, ['Ele']) # get list of files
  energies =[0, 110, 150, 190]# energy bins
  #angles=[math.radians(x) for x in [62, 90, 118]]
  angles=[62, 90, 118]
  g = generator(latent)       # build generator
  gen_weight1= "../weights/3dgan_weights_bins_pow_p85/params_generator_epoch_027.hdf5" # weights for generator
  g.load_weights(gen_weight1) # load weights
  sorted_data = gan.get_sorted_angle(data_files[24:], energies, thresh=thresh) # load data in a dict

  # for each energy bin
  for energy in energies:
     for a in angles:
       filename = path.join(plotsdir, "IQA_{}GeV_{}degree.pdf".format(energy, a)) # file name
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

       # Get MSCN Coefficients
       mscn_g4= MSCN_sparse(sorted_data["events_act" + str(energy)+ "ang_" + str(a)])
       mscn_gan= MSCN_sparse(generated_images)

       #flattening
       mscn_g4= mscn_g4.flatten()
       mscn_gan=mscn_gan.flatten()

       #removing zeros
       mscn_g4= mscn_g4[mscn_g4!=0]
       mscn_gan= mscn_gan[mscn_gan!=0]

       #find other metrics
       data_range= np.amax(sorted_data["events_act" + str(energy)+ "ang_" + str(a)])
       ms_ssim=measure.compare_ssim(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], generated_images, multichannel=True, data_range=data_range, gaussian_weights=True, use_sample_covariance=False)
       print('Energy={}'.format(energy))
       print('SSIM={}'.format(ms_ssim))
       psnr = measure.compare_psnr(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], generated_images, data_range=data_range)
       print('PSNR={}'.format(psnr))
       #make plot
       Draw1d(mscn_g4, mscn_gan, filename, labels, [ms_ssim, psnr])

       if get_shuffled:
          # repeat for shuffled data
          filename = path.join(plotsdir, "IQA_Data_shuffled{}GeV_{}degree.pdf".format(energy, a))
          shuffled_data = sorted_data["events_act" + str(energy)+ "ang_" + str(a)]* 1.
          np.random.shuffle(shuffled_data)
          mscn_2 = MSCN_sparse(shuffled_data)
          mscn_2 = mscn_2.flatten()
          mscn_2 = mscn_2[mscn_2!=0]

          ms_ssim=measure.compare_ssim(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], shuffled_data, multichannel=True, data_range=data_range, gaussian_weights=True, use_sample_covariance=False)
          print('SSIM Data shuffled ={}'.format(ms_ssim))
          psnr = measure.compare_psnr(sorted_data["events_act" + str(energy)+ "ang_" + str(a)], shuffled_data, data_range=data_range)
          print('PSNR Data shuffled ={}'.format(psnr))
          Draw1d(mscn_g4, mscn_2, filename, ['G4', 'G4 shuffled'], [ms_ssim, psnr])

          # repeat for shuffled GAN
          filename = path.join(plotsdir, "IQA_GAN_shuffled{}GeV_{}degree.pdf".format(energy, a))
          shuffled_gan= generated_images * 1.
          np.random.shuffle(shuffled_gan)
          mscn_2 = MSCN_sparse(shuffled_gan)
          mscn_2 = mscn_2.flatten()
          mscn_2 = mscn_2[mscn_2!=0]

          ms_ssim=measure.compare_ssim(generated_images, shuffled_gan, multichannel=True, data_range=data_range, gaussian_weights=True, use_sample_covariance=False)
          print('SSIM GAN shuffled ={}'.format(ms_ssim))
          psnr = measure.compare_psnr(generated_images, shuffled_gan, data_range=data_range)
          print('PSNR GAN shuffled ={}'.format(psnr))
          Draw1d(mscn_gan, mscn_2, filename, ['GAN', 'GAN shuffled'], [ms_ssim, psnr])
                              

# MSCN original
def MSCN(images):
  images = np.squeeze(images)
  x_shape= images.shape[1]
  y_shape= images.shape[2]
  z_shape= images.shape[3]
  mscn = np.zeros_like(images)
  for x in np.arange(0, x_shape, 3):
    for y in np.arange(0, y_shape, 3):
      for z in np.arange(0, z_shape, 5):
        mean = np.mean(images[:, x:x+3, y:y+3, z:z+5], axis=(1, 2, 3))
        std = np.std(images[:, x:x+3, y:y+3, z:z+5], axis=(1, 2, 3))
        slice_shape = images[:, x:x+3, y:y+3, z:z+5].shape
        mean = mean.reshape((mean.shape[0], 1, 1, 1))
        std = std.reshape((mean.shape[0], 1, 1, 1))
        mscn[:, x:x+3, y:y+3, z:z+5]= (images[:, x:x+3, y:y+3, z:z+5] - mean)/(std + 1e-7)
  return mscn

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
def MSCN_sparse(images):
  images = np.squeeze(images)
  x_shape= images.shape[1]
  y_shape= images.shape[2]
  z_shape= images.shape[3]
  mscn=np.zeros_like(images)
  for x in np.arange(0, x_shape, 3):
    for y in np.arange(0, y_shape, 3):
      for z in np.arange(0, z_shape, 5):
        image_slice=images[:, x:x+3, y:y+3, z:z+5]
        mean, count = mean_sparse(image_slice)
        std = std_sparse(image_slice, mean, count)
        mean = mean.reshape((mean.shape[0], 1, 1, 1))
        std = std.reshape((mean.shape))
        indexes=np.where(image_slice ==0)
        image_slice= (image_slice-mean)/(std+1e-7)
        image_slice[indexes]=0
        mscn[:, x:x+3, y:y+3, z:z+5]=image_slice
  return mscn

def MS_SSIM(images1, images2):
  #images1 = np.squeeze(images1)
  #images2 = np.squeeze(images2)
  image_tensor1=K.tf.convert_to_tensor(images1)
  image_tensor2=K.tf.convert_to_tensor(images2)
  ssim_val = ssim_skimage(image_tensor1, image_tensor2)
  return K.eval(ssim_val)

# Plot for MSCN
def Draw1d(array1, array2, filename, labels, metrics=[]):
  c=ROOT.TCanvas("c" ,"" ,200 ,10 ,700 ,500)
  c.SetGrid()
  min1=np.amin(array1)
  min2=np.amin(array2)
  max1=np.amax(array1)
  max2=np.amax(array2)
  mean1=np.mean(array1)
  mean2=np.mean(array2)
  ROOT.gStyle.SetOptStat(111111)
  leg=ROOT.TLegend(.1, .1, .4, .3)
  hist1 = ROOT.TH1D(labels[0], "MSCN Co efficients for Ecal and Ep;MSCN Co efficient;normalized count", 100, -7., 5.)
  hist2 = ROOT.TH1D(labels[1], "MSCN Co efficients for Ecal and Ep;MSCN Co efficient;normalized count", 100, -7., 5.)
  roo.fill_hist(hist1, array1)
  roo.fill_hist(hist2, array2)
  roo.normalize(hist1, 1)
  roo.normalize(hist2, 1)
  leg.AddEntry(hist1, "{}".format(labels[0]), "l")
  leg.AddEntry(hist2, "{} ".format(labels[1]), "l")
  if len(metrics)==2:
    leg.AddEntry(hist2, "ssim={:.4f} psnr={:.4f}".format(metrics[0], metrics[1]), "l")
  hist1.Sumw2()
  hist2.Sumw2()
  hist1.GetYaxis().CenterTitle()
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
                  
if __name__ == "__main__":
  main()
