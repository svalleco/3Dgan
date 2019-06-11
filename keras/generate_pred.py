from os import path
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from array import array
import time
import keras.backend as K
from EcalEnergyGan import generator, discriminator
K.set_image_data_format('channels_last')
# This functions loads data from a file and also does any pre processing
def GetData(datafile, xscale =1, yscale = 100, dimensions = 3):
    #get data for training
    with h5py.File(datafile,'r') as f:

      X=np.array(f.get('ECAL'))  # 10000, 25, 25, 25

      Y=f.get('target')
      Y=(np.array(Y[:,1]))

      X[X < 1e-6] = 0
      X = np.expand_dims(X, axis=-1)  # 10000, 25, 25, 25, 1
      X = X.astype(np.float32)
      if dimensions == 2:
          X = np.sum(X, axis=(1))
          X = xscale * X
      X = np.moveaxis(X, -1, 1)   # 10000, 1, 25, 25, 25 for GAN training

      Y = np.expand_dims(Y, axis=-1)
      Y = Y.astype(np.float32)
      Y = Y/yscale
      Y = np.moveaxis(Y, -1, 1)

      # ecal is a scalar of the total energy collected by the EM-Calorimeter
      ecal = np.sum(X, axis=(2, 3, 4))  # get total energy by summing alone (x, y, z), shape becomes (10000, 1)
      # X: ECAL
      # Y: Target
      print('X shape', X.shape)
      return X, Y, ecal

num_events=1000
latent = 200
gm = generator(latent)


#Get Actual Data
X, Y, Data = GetData("/eos/user/g/gkhattak/FixedAngleData/EleEscan_1_1.h5")#("/home/maxwell/Works/3Dgan/data/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_2_9.h5")
X = X[:num_events]
Y = Y[:num_events]
Data = Data[:num_events]
print(Y.shape)
print(Y[:10])
print(Data[:10])
color =2

gweight1= '/gkhattak/hvd_weights/generator_params_generator_epoch_004.hdf5'#'generator_params_generator_epoch_017.hdf5'
gweights = [gweight1]
scales=[1]
for i, gweight in enumerate(gweights):
  #Generate events
  gm.load_weights(gweight)
  noise = np.random.normal(0, 1, (num_events, latent))
  generator_in = np.multiply(Y, noise)
  start = time.time()
  generated_images = gm.predict(generator_in, verbose=False, batch_size=16)
  print("Time {}s".format(time.time() - start))
  with h5py.File('gen_imgs.h5', 'w') as h5f:
      h5f.create_dataset('gen_imgs', data=generated_images)
  GData = np.sum(generated_images, axis=(1, 2, 3))/scales[i]
  print(generated_images.shape, GData.shape)
  print(GData[:10])
