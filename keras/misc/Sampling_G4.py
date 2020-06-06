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
sys.path.insert(0,'../keras')
from AngleArch3dGAN import generator, discriminator

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


weightdir = '../keras/weights/'
#gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
#c.SetGrid()
gStyle.SetOptStat(0)
p =[0, 500]
#c.SetLogx ()
numdata=10000
latent = 256
dscale  = 50.0
power = 0.85
thresh = 0
particles =['e^{-}', '#pi', '#pi^{0}', '#gamma']
filename = 'results/sampling_G4.pdf'
color =[2, 3, 4, 6]
#Get Actual Data
datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_1_1.h5"
datapath2 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/ChPiEscan/ChPiEscan_RandomAngle_1_1.h5"
datapath3 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/Pi0Escan/Pi0Escan_RandomAngle_1_1.h5"
datapath4 = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/GammaEscan/GammaEscan_RandomAngle_1_1.h5"

x, y, ang=GetAngleData(datapath, numdata)
print(datapath, numdata)
print(x.shape)
print(y[:10])
print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
x = x/dscale

x2, y2, ang2=GetAngleData(datapath2, numdata)
x2 = x2/dscale

x3, y3, ang3=GetAngleData(datapath3, numdata)
x3 = x3/dscale

x4, y4, ang4=GetAngleData(datapath4, numdata)
x4 = x4/dscale
legend = ROOT.TLegend(.8, .65, .89, .89)
legend.SetBorderSize(0)
Eprofs=[]
i=0
for X, Y in zip([x, x2, x3, x4], [y, y2, y3, y4]):
  Eprofs.append(TProfile("Eprof"+str(i), "Sampling fraction vs. Ep;Ep [GeV];#S_{f}", 100, p[0], p[1])) 
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
    Eprof.GetYaxis().SetRangeUser(0.0, 0.03)
  else:
    Eprof.Draw('sames')
  Eprof.SetLineColor(color[i])
  Eprof.Fit('pol6')
  c.Update()
  Eprof.GetFunction("pol6").SetLineColor(color[i])
  c.Update()
  legend.AddEntry(Eprof, particles[i] ,"l")
  i+=1
legend.Draw()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)

                                                                  
