from os import path
from ROOT import TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from array import array
import time

from ecalvegan import generator, discriminator

gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
c1=TCanvas("c1" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500) #make nice
c1.SetGrid()
c1.SetLogx ()
#Eprof = TProfile("Eprof", "Eprof", 100, 0, 5, 0, 3)
#Gprof = TProfile("Gprof", "Gprof", 100, 0, 5, 0, 3)

mg=TMultiGraph()

num_events=1000
latent = 200
g = generator(latent)
#d = discriminator()

gweight = 'generator8p2p1_049.hdf5'
dweight = 'discriminator8p2p1_049.hdf5'
g.load_weights(gweight)
#d.load_weights(dweight)

start = time.time()
#Get Actual Data
#d=h5py.File("/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_1_1.h5")
d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
X=np.array(d.get('ECAL')[0:num_events], np.float64)                             
Y=np.array(d.get('target')[0:num_events][:,1], np.float64)
X[X < 1e-6] = 0
Y = Y/100
Data = np.sum(X, axis=(1, 2, 3))
load_time = time.time()- start
print(' {} events loaded in {:.4f} seconds'.format(num_events, load_time))

start = time.time()
#Generate events
noise = np.random.normal(0, 1, (num_events, latent))
generator_in = np.multiply(np.reshape(Y, (-1, 1)), noise)
generated_images = g.predict(generator_in, verbose=False, batch_size=100)
gen_time = time.time() - start
print(' {} events generated in {:.4f} seconds'.format(num_events, gen_time))
GData = np.sum(generated_images, axis=(1, 2, 3))
print GData.shape

# Initialization of parameters
a = np.ndarray( [num_events] )
b = np.ndarray( [num_events] )
c = np.ndarray( [num_events] )

for i in range(len(a)):
   a[i] = Data[i]/Y[i]
   b[i] = Y[i]
   c[i] = GData[i]/Y[i]
   #Eprof.Fill(b[i], a[i])
   #Gprof.Fill(b[i], c[i])
#Eprof.Draw()
#Gprof.Draw()

Eprof = TGraph(num_events, b, a)
#Eprof.SetMarkerSize(1)
Eprof.SetMarkerStyle(3)
Eprof.SetMarkerColor(2)
mg.Add(Eprof)
Gprof = TGraph(num_events, b, c)
#Gprof.SetMarkerSize(1)
Gprof.SetMarkerStyle(3)
Gprof.SetMarkerColor(4)
mg.Add(Gprof)
mg.SetTitle("Ratio of Ecal and Ep;Ep;Ecal/Ep")
mg.Draw('APE')
c1.Update()
Eprof.Fit("pol4")
c1.Update()
Gprof.Fit("pol4")
Gprof.GetFunction("pol4").SetLineColor(4)
c1.Update()

sb2=TPaveStats()
sb= Gprof.GetListOfFunctions().FindObject("stats")
sb2.SetX1NDC(.2)
sb2.SetX2NDC(.4)
sb2.SetY1NDC(.7)
sb2.SetY2NDC(.9)
sb2.SetTextColor(4)
c1.Modified()
c1.Print("Genfit.pdf")

# request user action before ending (and deleting graphics window)
raw_input('Press <ret> to end -> ')
