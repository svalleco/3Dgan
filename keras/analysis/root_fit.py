from os import path
from ROOT import TCanvas, TGraph, gStyle
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from array import array

gStyle.SetOptFit (111) # superimpose fit results
c1=TCanvas("c1" ,"Data" ,200 ,10 ,700 ,500) #make nice
c1.SetGrid ()

num_events=10000
d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
X=np.array(d.get('ECAL')[0:num_events], np.float32)                             
Y=np.array(d.get('target')[0:num_events][:,1], np.float32)
# Initialization of parameters
X[X < 1e-6] = 0
Y = Y
Data = np.sum(X, axis=(1, 2, 3))
a = np.ndarray( [num_events] )
b = np.ndarray( [num_events] )
for i in range(len(a)):
   a[i] = Data[i]
   b[i] = Y[i]
gr  = TGraph( num_events, b, a)
gr.SetTitle("Ecal versus Ep ; Primary Energy GeV ;Total Energy Deposited in Ecal GeV/50")
gr.Draw('ATP')
gr.Fit("pol2")
c1.Update()
c1.Print("pol2Fit.pdf")
gr.Fit("pol3")
c1.Update()
c1.Print("pol3Fit.pdf")
gr.Fit("pol4")
c1.Update()
c1.Print("pol4Fit.pdf")
gr.Fit("gaus")
c1.Update()
c1.Print("guassFit.pdf")
gr.Fit("expo")
c1.Update()
c1.Print("expoFit.pdf")

# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
