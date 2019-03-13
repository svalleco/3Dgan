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
from AngleArch3dGAN import generator, discriminator
try:
  import setGPU
except:
  pass

def main():
  num_events =1000
  latent = 256
  power=0.75
  labels =["G4", "GAN"]
  filename = 'results/IQA_hist_p85.pdf'
  datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5" # Data path     

  g = generator(latent)
  gen_weight1= "../weights/3dgan_weights_bins_pow_p85/params_generator_epoch_059.hdf5"
  g.load_weights(gen_weight1)
  data_files = gan.GetDataFiles(datapath, ['Ele'])
  images, y, ang = gan.GetAngleData(data_files[0], thresh=0., angtype='theta')
  images, y, ang = images[:num_events], y[:num_events], ang[:num_events]
  y = y/100.
  
  #np.random.shuffle(y)
  generated_images = gan.generate(g, num_events, [y, ang], latent, concat=1)
  generated_images = np.power(generated_images, 1./power)

  mscn_g4= MSCN_sparse(images)
  mscn_gan= MSCN_sparse(generated_images)

  #flattening
  mscn_g4= mscn_g4.flatten()
  mscn_gan=mscn_gan.flatten()
  #removing zeros
  mscn_g4= mscn_g4[mscn_g4!=0]
  mscn_gan= mscn_gan[mscn_gan!=0]
  data_range= np.amax(images)

  ms_ssim=measure.compare_ssim(images, generated_images, multichannel=True, data_range=data_range, gaussian_weights=True, use_sample_covariance=False)
  print('SSIM={}'.format(ms_ssim))
  psnr = measure.compare_psnr(images, generated_images, data_range=data_range)
  print('PSNR={}'.format(psnr))
  Draw1d(mscn_g4, mscn_gan, filename, labels, [ms_ssim, psnr])

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
                  

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data, z_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1, -size//4 + 1:size//4 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    
    z_data = np.expand_dims(y_data, axis=-1)
    z_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    z = tf.constant(z_data, dtype=tf.float32)
    
    g = tf.exp(-((x**2 + y**2 )/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=5, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    print(window.shape)
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv3d(img1, window, strides=[1,1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv3d(img2, window, strides=[1,1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv3d(img1*img1, window, strides=[1,1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv3d(img2*img2, window, strides=[1,1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv3d(img1*img2, window, strides=[1,1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=2):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    print(K.int_shape(img1))
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool3d(img1, [1,2,2,2,1], [1,2,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool3d(img2, [1,2,2,2,1], [1,2,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

if __name__ == "__main__":
  main()
