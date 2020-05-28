from os import path
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from array import array
import time
import ROOT
import setGPU
from EcalEnergyGan import generator, discriminator

#gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
c.SetGrid()
gStyle.SetOptStat(0)
#c.SetLogx ()
hist_g4 = ROOT.TH2F("g4", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 400, 100, 500, 100, 0, 0.04)
hist_g4.Sumw2()
num_events=1000
latent = 200
g = generator(latent)
#gweight = 'gen_rootfit_2p1p1_ep33.hdf5'
gweight1 = 'params_generator_epoch_041.hdf5' # 1 gpu
gweight2 = 'params_generator_epoch_005.hdf5' # 8 gpu
gweight3 = 'params_generator_epoch_002.hdf5'# 16 gpu
g.load_weights(gweight1)
gweights = [gweight1, gweight2, gweight3]
label = ['1 gpu','8 gpu', '16 gpu']
scales = [100, 1, 1]
filename = 'ecal_ratio_stats_500.pdf'
#Get Actual Data
d=h5py.File("/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5")
#d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
X=np.array(d.get('ECAL')[0:num_events], np.float64)                             
Y=np.array(d.get('target')[0:num_events][:,1], np.float64)
X[X < 1e-6] = 0
Y = Y
Data = np.sum(X, axis=(1, 2, 3))

for j in np.arange(num_events):
   hist_g4.Fill(Y[j], Data[j]/Y[j])
hist_g4.SetTitle("Ratio of Ecal and Ep")
hist_g4.GetXaxis().SetTitle("Ep")
hist_g4.GetYaxis().SetTitle("Ecal/Ep")
hist_g4.Draw()
#hist_g4.GetYaxis().SetRangeUser(0, 0.03)
color =2
hist_g4.SetMarkerColor(color)
legend = TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry(hist_g4, "Geant4", "p")
hist_gan = []
hist_g4.Draw()
c.Update()
for i, gweight in enumerate(gweights):
#for i in np.arange(1):
#   gweight=gweights[i]                                                                                                                                                                     
   hist_gan.append( ROOT.TH2F("Gan" +str(i), "Gan" + str(i), 400, 100, 500, 100, 0, 0.04))
   #Gprof[i].SetStates(0)
   #Generate events
   g.load_weights(gweight)
   noise = np.random.normal(0, 1, (num_events, latent))
   generator_in = np.multiply(np.reshape(Y/100, (-1, 1)), noise)
   generated_images = g.predict(generator_in, verbose=False, batch_size=100)
   GData = np.sum(generated_images, axis=(1, 2, 3))/scales[i]
   hist_gan[i].Sumw2()
   print GData.shape
   for j in range(num_events):
      hist_gan[i].Fill(Y[j], GData[j]/Y[j])
   color = color + 2
   chi2 = hist_gan[0].Chi2Test(hist_gan[i], "UU")
   k = hist_gan[0].KolmogorovTest(hist_gan[i], "UU")
   hist_gan[i].SetMarkerColor(color)
   hist_gan[i].Draw('sames')
   c.Modified()
   legend.AddEntry(hist_gan[i], '{} chi2 = {} K = {} '.format(label[i], chi2, k) , "p")
   legend.Draw()
   c.Update()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)
# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
