import sys
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats, TH2F
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from array import array
import time
sys.path.insert(0,'../')
from AngleArch3dGAN import generator, discriminator
try:
      import setGPU
except:
      pass

sys.path.insert(0,'../')
import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import analysis.utils.RootPlotsGAN as pl

weightdir = '../weights/'
#gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
#c.SetGrid()
gStyle.SetOptStat(0)
p =[350, 400]
bins = 10
filename = 'results/ecal_ratio_stest_10000events_rang8_2h.pdf'
#c.SetLogx ()
Eprof = TH2F("Eprof", "Sampling fraction vs. Ep;Ep [GeV];#S_{f}", bins, p[0], p[1], 10, 0.018, 0.022)
num_events=10000
latent = 256
dscale  = 50.0
power = 0.85
thresh = 0
g = generator(latent)
#gweight = 'gen_rootfit_2p1p1_ep33.hd_gan_trainingf5'
gweight1 = weightdir + '3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5'
#gweight2 = weightdir + 'surfsara_2_500GeV/params_generator_epoch_068.hdf5'
#gweight3 = weightdir + 'surfsara_128n/params_generator_epoch_193.hdf5' 
#gweight4 = weightdir + 'surfsara_256n/params_generator_epoch_139.hdf5'
gweights = [gweight1]#, gweight2]#, gweight3, gweight4]
label = ['1 node', '80 node', '128 node', '256 node']
scales = [1, 1, 1, 1]
color = [2, 4, 6, 7, 8]

#Get Actual Data
datapath = "/storage/group/gpu/bigdata/gkhattak/ProcessedVarAngle/*scan/*scan*.h5"
#datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"
datafiles = gan.GetDataFiles(datapath, ['Ele'])
d=h5py.File(datafiles[0], 'r')

X=np.array(d.get('ECAL')[0:num_events], np.float32)
Y=np.array(d.get('energy')[0:num_events], np.float32)
theta=np.array(d.get('mtheta')[0:num_events], np.float32)
X[X < thresh] = 0
Data = np.sum(X, axis=(1, 2, 3))/dscale
num_events = X.shape[0]
print(num_events)
for j in np.arange(X.shape[0]):
      Eprof.Fill(Y[j], Data[j]/Y[j])
Eprof.SetTitle("Ratio of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep [GeV]")
Eprof.GetYaxis().SetTitle("Ecal/Ep")
Eprof.Draw()
#Eprof.Fit('pol6')
#c.Update()
#Eprof.GetFunction("pol6").SetLineColor(color[0])
c.Update()
Eprof.SetStats(0)
Eprof.GetYaxis().SetRangeUser(0.01, 0.03)
Eprof.SetLineColor(color[0])
legend = TLegend(0.5, 0.5, 0.89, 0.89)
legend.AddEntry(Eprof, "G4", "l")
legend.SetBorderSize(0)

Gprof = []
for i, gweight in enumerate(gweights):
   Gprof.append( TH2F("Gprof" +str(i), "Gprof" + str(i), bins, p[0], p[1], 10, 0.018, 0.022))
   #Generate events
   g.load_weights(gweight)
   generated_images = np.power(gan.generate(g, num_events, [Y/100., theta], latent=latent, concat=2), 1./power)
   generated_images[generated_images < thresh] =0
   GData = np.sum(generated_images, axis=(1, 2, 3))/(dscale)
   print GData.shape
   for j in range(num_events):
      Gprof[i].Fill(Y[j], GData[j]/Y[j])
   Gprof[i].SetLineColor(color[i + 1])
   Gprof[i].Draw('sames')
   c.Modified()
   k1 = Eprof.KolmogorovTest(Gprof[i])
   chi21 = Eprof.Chi2Test(Gprof[i])
   legend.AddEntry(Gprof[i], label[i], "l")
   legend.AddEntry(Gprof[i], 'k={}'.format(k1), "l")
   legend.AddEntry(Gprof[i], 'chi2={}'.format(chi21), "l")
   legend.Draw()
   c.Update()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)

                                                                  
