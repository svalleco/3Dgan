import sys
import os
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
gStyle.SetTitleSize(0., "t1")
#gStyle.SetGridLength(20,"x")
#gStyle.SetOptFit (1111) # superimpose fit results
gStyle.SetOptStat(0)
p =[0, 500]
s = 5
numdata=10000
latent = 256
dscale  = 50.0
power = 0.85
thresh = 0
bins = int((p[1]-p[0])/s)
particles =['e^{-}', 'GAN e^{-}', 'Chpi', 'GAN Chpi']
outdir = 'results/moments_chpi/'
gan.safe_mkdir(outdir)
filename = os.path.join(outdir, 'moment')
color =[2, 3, 4, 6]
kopt = 'WW'
ratio = 1

#Get Actual Data
datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_1_1.h5"
datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/ChPiEscan/ChPiEscan_RandomAngle_1_1.h5"
#datapath3 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/Pi0Escan/Pi0Escan_RandomAngle_1_1.h5"
#datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/GammaEscan/GammaEscan_RandomAngle_1_1.h5"

x, y, ang=GetAngleData(datapath, numdata)
ecal = np.sum(x, axis=(1, 2, 3))
indexes = np.where(ecal > 10.0)
x=x[indexes]
y=y[indexes]
ang = ang[indexes]
num1 = x.shape[0]
print(num1)
print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
x = x/dscale

x2, y2, ang2=GetAngleData(datapath2, numdata)
ecal = np.sum(x2, axis=(1, 2, 3))
indexes = np.where(ecal > 10.0)
x2=x2[indexes]
y2=y2[indexes]
ang2 = ang2[indexes]
num2 = x2.shape[0]
print(num2)
print('The angle data varies from {} to {}'.format(np.amin(x2[x2>0]), np.amax(x2)))
x2 = x2/dscale
num = min(num1, num2)


g = generator(latent)

gweight = weightdir + '3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5'
gweight2 = weightdir + '3dgan_weights__gamma/params_generator_epoch_020.hdf5'
gweight3 = weightdir + '3dgan_weights_gan_training_Ch_pion/params_generator_epoch_189.hdf5'

g.load_weights(gweight2)
x3 = np.power(gan.generate(g, num, [y[:num]/100., ang[:num]], latent=latent, concat=2), 1./power)
x3[x3 < thresh] =0
x3 = x3/dscale
x3 = np.squeeze(x3)

g.load_weights(gweight3)
x4 = np.power(gan.generate(g, num, [y2[:num]/100., ang2[:num]], latent=latent, concat=2), 1./power)
x4[x4 < thresh] =0
x4 = x4/dscale
x4 = np.squeeze(x4)


c = ROOT.TCanvas("c", "canvas",800, 600)

legend = ROOT.TLegend(.7, .11, .89, .3)
legend.SetBorderSize(0)
Eprofs=[]
Eprofs2=[]
title = 'Mz1'
i=0
for X, Y in zip([x[:num], x3[:num], x2[:num], x4[:num]], [y[:num], y[:num], y2[:num], y2[:num]]):
  Eprofs.append(TProfile("Eprof"+str(i), "Eprof"+str(i), bins, p[0], p[1]))
  Eprofs2.append(TProfile("Eprof2"+str(i), "Eprof2"+str(i), bins, p[0], p[1], 's')) 
  ecal = np.sum(X, axis=(1, 2, 3))
  sumx, sumy, sumz = gan.get_sums(X)
  mx, my, mz = gan.get_moments(sumx, sumy, sumz, ecal, m=2, x=51, y=51, z=25)
  Eprof= Eprofs[i]
  Eprof.Sumw2()
  Eprof.SetStats(0)
  for j in np.arange(X.shape[0]):
    Eprof.Fill(Y[j], mz[j][0])
    Eprofs2[i].Fill(Y[j], mz[j][0])
    
  if i==0: 
    Eprof.SetTitle(title + " vs E_{P}")
    Eprof.GetXaxis().SetTitle("E_{p} [GeV]")
    Eprof.GetYaxis().SetTitle(title)
    Eprof.Draw()
    Eprofs2[i].Draw('sames E4')
  else:
    Eprof.Draw('sames')
    Eprofs2[i].Draw('sames E4')
  Eprof.SetLineColor(color[i])
  Eprofs2[i].SetFillColorAlpha(color[i], 0.35)
  c.Update()
      
  if (i%2!=0):
    k = Eprof.KolmogorovTest(Eprofs[i-1], kopt)
    legend.AddEntry(Eprof, particles[i]+' k={}'.format(k) ,"l")
  else:
    legend.AddEntry(Eprof, particles[i] ,"l")
  i+=1

legend.Draw()
c.Update()
Eprofs[0].GetXaxis().SetRangeUser(p[0], p[1])
Eprofs[0].GetYaxis().SetRangeUser(0, 25)
c.Update()

c.Print(filename+'.pdf')
print ' The plot is saved in.....{}.pdf'.format(filename)

                                                                  
