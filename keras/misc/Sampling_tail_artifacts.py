import sys
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
#from ROOT import gROOT, gBenchmark
import ROOT
import h5py
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from array import array
import time
sys.path.insert(0,'../keras')
from AngleArch3dGAN import generator, discriminator

try:
      import setGPU
except:
      pass

import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import analysis.utils.RootPlotsGAN as pl

weightdir = '../keras/weights/'
#gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
#c.SetGrid()
gStyle.SetOptStat(0)
p =[0, 500]
#c.SetLogx ()
Eprof = TProfile("Eprof", "Sampling fraction vs. Ep;Ep [GeV];#S_{f}", 100, p[0], p[1])
num_events=5000
latent = 256
dscale  = 50.0
power = 0.85
thresh = 0
g = generator(latent)
#gweight = 'gen_rootfit_2p1p1_ep33.hd_gan_trainingf5'
gweight1 = weightdir + '3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5'
gweight2 = weightdir + 'surfsara_2_500GeV/params_generator_epoch_068.hdf5'
#gweight3 = weightdir + 'surfsara_128n/params_generator_epoch_193.hdf5' 
#gweight4 = weightdir + 'surfsara_256n/params_generator_epoch_139.hdf5'
gweights = [gweight1, gweight2]#, gweight3, gweight4]
label = ['1 node', '80 node', '128 node', '256 node']
scales = [1, 1, 1, 1]
color = [2, 4, 6, 7, 8]
filename = 'results/ecal_ratio_gen_outliers.pdf'
filename2 = 'results/theta_outliers.pdf'
#Get Actual Data
datapath = "/storage/group/gpu/bigdata/gkhattak/ProcessedVarAngle/*scan/*scan*.h5"
#datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"
datafiles = gan.GetDataFiles(datapath, ['Ele'])
d=h5py.File(datafiles[0], 'r')

X=np.array(d.get('ECAL')[0:num_events], np.float32)
Y=np.array(d.get('energy')[0:num_events], np.float32)
theta=np.array(d.get('mtheta')[0:num_events], np.float32)
num_events = X.shape[0]
X[X < thresh] = 0
Data = np.sum(X, axis=(1, 2, 3))/dscale

for j in np.arange(X.shape[0]):
      Eprof.Fill(Y[j], Data[j]/Y[j])
Eprof.SetTitle("Ratio of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep [GeV]")
Eprof.GetYaxis().SetTitle("Ecal/Ep")
Eprof.Draw()
Eprof.Fit('pol6')
c.Update()
Eprof.GetFunction("pol6").SetLineColor(color[0])
c.Update()
Eprof.SetStats(0)
Eprof.GetYaxis().SetRangeUser(0., 0.1)
Eprof.SetLineColor(color[0])
legend = TLegend(0.7, 0.7, 0.89, 0.89)
legend.AddEntry(Eprof, "G4", "l")
legend.SetBorderSize(0)

Gprof = []
oprof =[]
col=1
for i, gweight in enumerate(gweights):
   Gprof.append( TProfile("Gprof" +str(i), "Gprof" + str(i), 100, p[0], p[1]))
   oprof.append( TProfile("oprof" +str(i), "oprof" + str(i), 100, p[0], p[1]))
   #Generate events
   g.load_weights(gweight)
   generated_images = np.power(gan.generate(g, num_events, [Y/100., theta], latent=latent, concat=2), 1./power)/dscale
   generated_images[generated_images < thresh] =0
   sliced_images = np.squeeze(generated_images[:, 0:10, 0:10, :])
   sliced_images2 = np.squeeze(generated_images[:, -10:, -10:, :])
   sliced_images[sliced_images > 3e-3] = 1
   sliced_images2[sliced_images2 > 3e-3] = 1
   sum_sliced = np.sum(sliced_images, axis =(1, 2, 3))
   sum_sliced2 = np.sum(sliced_images2, axis =(1, 2, 3))
   GData = np.sum(generated_images, axis=(1, 2, 3))
   sliced_indexes = np.where(sum_sliced > 2)
   sliced_indexes2 = np.where(sum_sliced2 > 2)
   outliers = generated_images[sliced_indexes]
   outliers2 = generated_images[sliced_indexes2]
   theta_outliers = theta[sliced_indexes]
   theta_outliers2 = theta[sliced_indexes2]
   theta_outliers = np.concatenate((theta_outliers, theta_outliers2), axis=0)
   sum_outliers = np.sum(outliers, axis =(1, 2, 3))
   sum_outliers2 = np.sum(outliers2, axis =(1, 2, 3))
   Y_outliers = Y[sliced_indexes]
   Y_outliers2 = Y[sliced_indexes2]
   print(Y_outliers.shape, Y_outliers2.shape)
   sum_outliers = np.concatenate((sum_outliers, sum_outliers2), axis=0)
   Y_outliers = np.concatenate((Y_outliers, Y_outliers2), axis=0)

   for j in range(num_events):
      Gprof[i].Fill(Y[j], GData[j]/Y[j])
   Gprof[i].SetLineColor(color[col])
   col+=1
   Gprof[i].Draw('sames')
   c.Modified()
   k = Eprof.KolmogorovTest(Gprof[i])
   c2 = Eprof.Chi2Test(Gprof[i])
   legend.AddEntry(Gprof[i], label[i], "l")
   
   if outliers.shape[0] > 0:   
      for j in range(outliers.shape[0]):
        oprof[i].Fill(Y[j], sum_outliers[j]/Y_outliers[j])
      oprof[i].SetLineColor(color[col])
      col+=1
      oprof[i].Draw('sames')
      c.Modified()
      legend.AddEntry(oprof[i], label[i]+' outliers', "l")
   legend.Draw()
   c.Update()

c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)

tmin = np.amin(theta_outliers)
tmax = np.amax(theta_outliers)
print(tmax)
hist = ROOT.TH1F('angle', 'angle', 100, np.degrees(tmin) -10, np.degrees(tmax) + 10)
r.fill_hist(hist, np.degrees(theta_outliers))
hist.SetTitle("Theta for outliers")
hist.GetXaxis().SetTitle("Theta [degress]")
hist.GetYaxis().SetTitle("entries")
hist.Draw()
c.Update()
c.Print(filename2)
print ' The plot is saved in.....{}'.format(filename2)



                                                                  
