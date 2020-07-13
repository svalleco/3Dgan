import sys
from os import path
import h5py
import ROOT
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
import numpy as np
from array import array
import time
import keras.backend as K
from EcalEnergyGan import generator, discriminator
from EcalEnergyTrain_hvd import GetData

num_events=1000
latent = 200
keras_dformat='channels_last'
K.set_image_data_format(keras_dformat)
gm = generator(latent, keras_dformat)

gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
c.SetGrid()
gStyle.SetOptStat(0)
Eprof = TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 500)

gweight1='/gkhattak/hvd_weights/generator_params_generator_epoch_004.hdf5'
gweights = [gweight1]
label = ['4n_16w_bs8']
scales = [1]
filename = 'ecal_ratio_multi.pdf'

X, Y, Data = GetData("/eos/user/g/gkhattak/FixedAngleData/EleEscan_1_1.h5")
X = X[:num_events]
Y = Y[:num_events]
Data = Data[:num_events]

for j in np.arange(num_events):
  Eprof.Fill(100 * Y[j], Data[j]/(Y[j]* 100))
Eprof.SetTitle("Ratio of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep")
Eprof.GetYaxis().SetTitle("Ecal/Ep")
Eprof.Draw()
Eprof.GetYaxis().SetRangeUser(0, 0.03)
color =2
Eprof.SetLineColor(color)
legend = TLegend(0.8, 0.8, 0.9, 0.9)
legend.AddEntry(Eprof, "Data", "l")
Gprof = []

import time 
for i, gweight in enumerate(gweights):
  Gprof.append( TProfile("Gprof" +str(i), "Gprof" + str(i), 100, 0, 500))
  gm.load_weights(gweight)
  noise = np.random.normal(0, 1, (num_events, latent))
  generator_in = np.multiply(np.reshape(Y, (-1, 1)), noise)
  start = time.time()
  generated_images = gm.predict(generator_in, verbose=False, batch_size=100)
  print("Time {}s".format(time.time() - start))
  GData = np.sum(generated_images, axis=(1, 2, 3))/scales[i]
  for j in range(num_events):
    Gprof[i].Fill(Y[j] * 100, GData[j]/(Y[j]* 100))
  color = color + 2
  Gprof[i].SetLineColor(color)
  Gprof[i].Draw('sames')
  c.Modified()
  legend.AddEntry(Gprof[i], label[i], "l")
  legend.Draw()
  c.Update()

c.Print(filename)
print (' The plot is saved in.....{}'.format(filename))
