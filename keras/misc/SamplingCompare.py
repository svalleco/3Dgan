from os import path
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
import h5py
import numpy as np
from array import array
import time
import sys
sys.path.insert(0,'../')
from EcalEnergyGan import generator, discriminator
import keras.backend as K

dforamt = 'channels_last'
K.set_image_data_format(dformat)

#gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
c.SetGrid()
gStyle.SetOptStat(0)
Eprof = TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 500)
num_events=1000
latent = 200
g = generator(latent)
#gweight = 'gen_rootfit_2p1p1_ep33.hdf5'
gweight1 = '/afs/cern.ch/work/g/gkhattak/params_generator_epoch_041.hdf5' # 1 gpu
#gweight2 = '/afs/cern.ch/work/g/gkhattak/3DGan_weights/result_8260L_omp23_8k_128n_llr/generator_params_generator_epoch_025.hdf5'
gweight3 = '/afs/cern.ch/work/g/gkhattak/3DGan_weights/result_8260L_omp23_4k_200n_nowarm_low_0.001/generator_params_generator_epoch_099.hdf5'
gweight4 = '/afs/cern.ch/work/g/gkhattak/3DGan_weights/results_8276_omp27_8k_256n/generator_params_generator_epoch_024.hdf5'
g.load_weights(gweight1)
gweights = [gweight1, gweight3, gweight4]
label = ['Node 1', 'Node 200', 'Node 256']
scales = [100, 1, 1]
color = [4, 2, 3, 6, 7, 8]
filename = 'ecal_ratio_surf_sara.pdf'
#Get Actual Data
d = h5py.File("/eos/user/g/gkhattak/FixedAngleData/EleEscan_1_1.h5")
X=np.array(d.get('ECAL')[0:num_events], np.float64)
Y=np.array(d.get('target')[0:num_events][:,1], np.float64)
X[X < 1e-6] = 0
Y = Y
Data = np.sum(X, axis=(1, 2, 3))

for j in np.arange(num_events):
      Eprof.Fill(Y[j], Data[j]/Y[j])
Eprof.SetTitle("Ratio of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep")
Eprof.GetYaxis().SetTitle("Ecal/Ep")
Eprof.Draw()
Eprof.Fit('pol6')
c.Update()
Eprof.GetFunction("pol6").SetLineColor(color[0])
c.Update()
Eprof.SetStats(0)
Eprof.GetYaxis().SetRangeUser(0, 0.04)
Eprof.SetLineColor(color[0])
legend = TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry(Eprof, "Data", "l")
Gprof = []
for i, gweight in enumerate(gweights):
   Gprof.append( TProfile("Gprof" +str(i), "Gprof" + str(i), 100, 0, 500))
   #Generate events
   print(gweight)
   g.load_weights(gweight)
   noise = np.random.normal(0, 1, (num_events, latent))
   generator_in = np.multiply(np.reshape(Y/100, (-1, 1)), noise)
   generated_images = g.predict(generator_in, verbose=False, batch_size=100)
   GData = np.sum(generated_images, axis=(1, 2, 3))/scales[i]
   print np.amax(GData)
   for j in range(num_events):
      Gprof[i].Fill(Y[j], GData[j]/Y[j])
   Gprof[i].SetLineColor(color[i + 1])
   Gprof[i].Draw('sames')
   c.Modified()
   legend.AddEntry(Gprof[i], label[i], "l")
   legend.Draw()
   c.Update()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)
