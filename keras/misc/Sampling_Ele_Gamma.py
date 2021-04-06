import sys
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
import ROOT
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from array import array
import time
sys.path.insert(0,'../')
from AngleArch3dGAN import generator

try:
      import setGPU
except:
      pass

import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import analysis.utils.RootPlotsGAN as pl

def GetAngleData(datafile, numevents, ftn=0, scale=1, angtype='theta'):
   #get data for training                                                                                                                                                                                  
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents]) * scale
   if ftn!=0:
      x = ftn(x)
   ang = np.array(f.get(angtype)[:numevents])
   return x, y, ang


weightdir = '../weights/'
#gStyle.SetOptStat(0)
#gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
#c.SetGrid()
gStyle.SetOptStat(0)
p =[100, 400]
#c.SetLogx ()
numdata=10000
latent = 256
dscale  = 50.0
power = 0.85
thresh = 0
particles =['e^{-}', 'GAN e^{-}', '#gamma', 'GAN #gamma']
filename = 'results/sampling_Ele_Gamma_wtfit_rd.pdf'
color =[2, 3, 4, 6]
#Get Actual Data
datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_1_1.h5"
#datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/ChPiEscan/ChPiEscan_RandomAngle_1_1.h5"
#datapath3 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/Pi0Escan/Pi0Escan_RandomAngle_1_1.h5"
datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/GammaEscan/GammaEscan_RandomAngle_1_1.h5"

x, y, ang=GetAngleData(datapath, numdata)

print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
x = x/dscale

x2, y2, ang2=GetAngleData(datapath2, numdata)
x2 = x2/dscale
"""
x3, y3, ang3=GetAngleData(datapath3, numdata)
x3 = x3/dscale

x4, y4, ang4=GetAngleData(datapath4, numdata)
x4 = x4/dscale
"""
g = generator(latent)

gweight = weightdir + '3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5'
gweight2 = weightdir + '3dgan_weights__gamma/params_generator_epoch_020.hdf5'

g.load_weights(gweight)
x3 = np.power(gan.generate(g, numdata, [y/100., ang], latent=latent, concat=2), 1./power)
x3[x3 < thresh] =0
x3 = x3/dscale

g.load_weights(gweight2)
x4 = np.power(gan.generate(g, numdata, [y2/100., ang2], latent=latent, concat=2), 1./power)
x4[x4 < thresh] =0
x4 = x4/dscale

legend = ROOT.TLegend(.7, .55, .89, .89)
legend.SetBorderSize(0)
Eprofs=[]
i=0
for X, Y in zip([x, x3, x2, x4], [y, y, y2, y2]):
  Eprofs.append(TProfile("Eprof"+str(i), "Sampling fraction vs. Ep;Ep [GeV];#S_{f}", 20, p[0], p[1])) 
  Data = np.sum(X, axis=(1, 2, 3))
  Eprof= Eprofs[i]
  Eprof.SetStats(0)
  for j in np.arange(X.shape[0]):
    Eprof.Fill(Y[j], Data[j]/Y[j])
  if i==0: 
    Eprof.SetTitle("Ratio of Ecal and Ep")
    Eprof.GetXaxis().SetTitle("Ep [GeV]")
    Eprof.GetYaxis().SetTitle("Ecal/Ep")
    Eprof.Draw()
    Eprof.GetYaxis().SetRangeUser(0.015, 0.03)
  else:
    Eprof.Draw('sames')
  Eprof.SetLineColor(color[i])
  #Eprof.Fit('pol6')
  c.Update()
  #Eprof.GetFunction("pol6").SetLineColor(color[i])
  c.Update()
  legend.AddEntry(Eprof, particles[i] ,"l")
  i+=1
k1 = Eprofs[0].KolmogorovTest(Eprofs[1], 'UU NORM')
k2 = Eprofs[2].KolmogorovTest(Eprofs[3], 'UU NORM')
legend.AddEntry(Eprofs[1], particles[1] + ' (k={:.4f})'.format(k1) ,"l")
legend.AddEntry(Eprofs[3], particles[3] + ' (k={:.4f})'.format(k2) ,"l")
legend.Draw()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)

                                                                  
