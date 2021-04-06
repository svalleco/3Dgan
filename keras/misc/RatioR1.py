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
import analysis.utils.ROOTutils as rt
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
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,500 ,500) #make nice
#c.SetGrid()
gStyle.SetOptStat(0)
#c.SetLogx ()
numdata=10000
latent = 256
dscale  = 50.0
power = 0.85
thresh = 0
particles =['e^{-}', 'GAN e^{-}', '#gamma', 'GAN #gamma']
outdir = 'results/r1_Ele_Gamma_zoom_50bin/'
gan.safe_mkdir(outdir)
filename = outdir + 'ratio1'
color =[2, 3, 4, 6]
#Get Actual Data
datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_1_1.h5"
#datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/ChPiEscan/ChPiEscan_RandomAngle_1_1.h5"
#datapath3 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/Pi0Escan/Pi0Escan_RandomAngle_1_1.h5"
datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/GammaEscan/GammaEscan_RandomAngle_1_1.h5"

x, y, ang=GetAngleData(datapath, numdata)
ecal = np.sum(x, axis=(1, 2, 3))
indexes = np.where(ecal > 10.0)
x=x[indexes]
y=y[indexes]
ang = ang[indexes]
num1 = x.shape[0]
print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
x = x/dscale

x2, y2, ang2=GetAngleData(datapath2, numdata)
ecal2 = np.sum(x2, axis=(1, 2, 3))
indexes2 = np.where(ecal2 > 10.0)
x2=x2[indexes2]
y2=y2[indexes2]
ang2 = ang2[indexes2]

x2 = x2/dscale
num2 = x2.shape[0]
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
x3 = np.power(gan.generate(g, num1, [y/100., ang], latent=latent, concat=2), 1./power)
x3[x3 < thresh] =0
x3 = x3/dscale
x3 = np.squeeze(x3)

g.load_weights(gweight2)
x4 = np.power(gan.generate(g, num2, [y2/100., ang2], latent=latent, concat=2), 1./power)
x4[x4 < thresh] =0
x4 = x4/dscale
x4 = np.squeeze(x4)

legend = ROOT.TLegend(.7, .55, .89, .89)
legend.SetBorderSize(0)
hists=[]
i=0
for X, Y in zip([x, x3, x2, x4], [y, y, y2, y2]):
  hists.append(ROOT.TH1F("hist"+str(i), "ratio 1 ;ratio 1; entries", 20, 0, 0.1))
  Data1 = np.sum(X[:, :, :, :8], axis=(1, 2, 3)) 
  Data = np.sum(X, axis=(1, 2, 3))
  r1 = Data1/Data

  h = hists[i]
  h.Sumw2()
  h.SetStats(0)
  for j in np.arange(X.shape[0]):
    h.Fill(Data1[j]/Data[j])
  if i==0: 
    h.SetTitle("Ratio r1")
    
    h.GetYaxis().SetRangeUser(0., 0.1)
    h.Draw()
    h.Draw('sames hist')
    #h.SetMaximum(1)
  else:
    h.Draw('sames')
    h.Draw('sames hist')
  c.Update()

  h.SetLineColor(color[i])
  c.Update()
  rt.normalize(h, mod=1)
  c.Update()
  if i%2==0:
    legend.AddEntry(h, particles[i] ,"l")
  else:
    k = hists[i].KolmogorovTest(hists[i-1], 'UU norm')
    legend.AddEntry(h, particles[i] + ' (k={:.4f})'.format(k) ,"l")
  i+=1

#legend.Draw()
c.Update()
c.Print(filename+'.pdf')
print ' The plot is saved in.....{}.pdf'.format(filename)

                                                                  
