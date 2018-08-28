from os import path
from ROOT import TCanvas, TGraph, gStyle, TProfile, TPaveStats, TLegend
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from array import array

gStyle.SetOptFit (1111) # superimpose fit results
c1=TCanvas("c1" ,"Data" ,200 ,10 ,700 ,500) #make nice
c1.SetGrid()
#c1.SetLogx ()
Eprof = TProfile("Eprof", "Ecal/Ep versus Ep;Ep GeV; Ecal/Ep", 100, 0, 5, 0, 3)
num_events=10000
d=h5py.File("/eos/project/d/dshep/LCD/DDHEP/EleEscan_RandomAngle_1_MERGED/EleEscan_RandomAngle_1_1.h5")
#d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
X=np.array(d.get('ECAL')[0:num_events], np.float64)                             
Y=np.array(d.get('energy')[0:num_events], np.float64)

print(X.shape)
print(Y.shape)
print(np.amax(X))
print(np.amax(Y))
print(np.amin(X))
print(np.amin(Y))

# Initialization of parameters
X[X < 1e-6] = 0
Y = Y/100
Data = np.sum(X, axis=(1, 2, 3))
a = np.ndarray( [num_events] )
b = np.ndarray( [num_events] )
for i in range(len(a)):
   a[i] = Data[i]/Y[i]
   b[i] = Y[i]
   Eprof.Fill(b[i], a[i])

Eprof.Draw()
c1.Update()

#Eprof.Fit("pol2")
#c1.Update()
#c1.Print("prof2Fit.pdf")
#Eprof.Fit("pol3")
#c1.Update()
#c1.Print("prof3Fit.pdf")
Eprof.Fit("pol4")
c1.Update()

sb1= Eprof.GetListOfFunctions().FindObject("stats")
sb1.SetX1NDC(.1)
sb1.SetX2NDC(.3)
sb1.SetY1NDC(.4)
sb1.SetY2NDC(.1)
c1.Modified()

c1.Print("VARangle4Fit.pdf")
#Eprof.Fit("gaus")
#c1.Print("profguassFit.pdf")
#Eprof.Fit("expo")

#c1.Update()
#c1.Print("profexpoFit.pdf")

# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
