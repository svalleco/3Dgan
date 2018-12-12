from os import path
from ROOT import TCanvas, TGraph, gStyle, TMultiGraph
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

mg=TMultiGraph();

num_weights=30
total = np.zeros((num_weights))
energy = np.zeros((num_weights))
position = np.zeros((num_weights))
epoch = np.zeros((num_weights))
d = np.arange(num_weights)
a, b, c=np.loadtxt("resultfile.txt", unpack=True)
for i in range(num_weights):
   total[i]= a[i]
   energy[i]= b[i]
   position[i]= c[i]
   epoch[i] = d[i]
   
gt  = TGraph( num_weights , epoch, total )
mg.Add(gt)
ge = TGraph( num_weights , epoch, energy )
mg.Add(ge)
gp = TGraph( num_weights , epoch, position )
mg.Add(gp)
mg.SetTitle("Optimization function;Epoch;Score")
mg.Draw('ALP')
gt.Fit("pol1")
ge.Fit("pol1")
gp.Fit("pol1")
c1.Update()
c1.Print("Linfit.pdf")

gt.Fit("pol2")
ge.Fit("pol2")
gp.Fit("pol2")
c1.Update()
c1.Print("pol2fit.pdf")

# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
