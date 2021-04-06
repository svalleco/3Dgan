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
p =[150, 350]
s = 5
numdata=10000
latent = 256
dscale  = 50.0
power = 0.85
thresh = 0
bins = int((p[1]-p[0])/s)
particles =['e^{-}', 'GAN e^{-}', '#gamma', 'GAN #gamma']
outdir = 'results/ratio_Ele_Gamma_150_350/'
gan.safe_mkdir(outdir)
filename = os.path.join(outdir, 'Ecal_')
color =[2, 3, 4, 6]
kopt = 'WW'
ratio = 1

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

if ratio:
   title = 'SF'
   filename = filename + 'ratio'
else:
   title ='E_{sum}'
   filename = filename + 'sum'
c = ROOT.TCanvas("c", "canvas",800, 350)

pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 0.5, 1.0)
pad1.SetBottomMargin(0)
pad1.SetLeftMargin(0.12)
pad1.Draw()
#pad1.SetGridx()
pad1.cd()
if ratio:
   legend = ROOT.TLegend(.6, .55, .89, .89)
else:
   legend = ROOT.TLegend(.15, .55, .4, .89)
legend.SetBorderSize(0)
Eprofs=[]
Temps = []
i=0
for X, Y in zip([x, x3, x2, x4], [y, y, y2, y2]):
  Eprofs.append(TProfile("Eprof"+str(i), "Eprof"+str(i), bins, p[0], p[1])) 
  Temps.append(TProfile("Temp"+str(i), "temp"+str(i), bins, p[0], p[1], 's'))
  Data = np.sum(X, axis=(1, 2, 3))
  
  Eprof= Eprofs[i]
  Eprof.Sumw2()
  Eprof.SetStats(0)
  for j in np.arange(X.shape[0]):
    if ratio:
      Eprof.Fill(Y[j], Data[j]/Y[j])
      Temps[i].Fill(Y[j], Data[j]/Y[j])
    else:
      Eprof.Fill(Y[j], Data[j])
      Temps[i].Fill(Y[j], Data[j])
  if i==0: 
    Eprof.SetTitle(title + " vs E_{P}")
    #Eprof.GetXaxis().SetTitle("E_{p} [GeV]")
    Eprof.GetYaxis().SetTitle("Ecal")
    Eprof.Draw()
  else:
    Eprof.Draw('sames')
  
  Eprof.SetLineColor(color[i])
  #Eprof.Fit('pol6')
  c.Update()
  #Eprof.GetFunction("pol6").SetLineColor(color[i])
  
  if (i%2!=0):
      k = Eprof.KolmogorovTest(Eprofs[i-1], kopt)
      print(i, i-1, k)
      legend.AddEntry(Eprof, particles[i]+' k={}'.format(k) ,"l")
  else:
    legend.AddEntry(Eprof, particles[i] ,"l")
  i+=1

legend.Draw()
c.Update()
Eprofs[0].GetXaxis().SetRangeUser(p[0], p[1])
if ratio:
   Eprofs[0].GetYaxis().SetRangeUser(0.017, 0.025)
else:
   Eprofs[0].GetYaxis().SetRangeUser(0, 10)
Eprofs[0].GetYaxis().SetTitle(title)
Eprofs[0].GetYaxis().SetLabelSize(0.025)
#Eprofs[0].GetXaxis().SetLabelSize(0.025)
#ax = ROOT.TGaxis( -5, 20, -5, 220, 20,220,510,"");
#   axis->SetLabelFont(43); // Absolute font size in pixel (precision 3)
#   axis->SetLabelSize(15);
#   axis->Draw();
c.Update()
c.cd()
pad2 = ROOT.TPad("pad2", "pad2", 0, 0.05, 0.5, 0.3)
pad2.SetTopMargin(0)
pad2.SetBottomMargin(0.2)
pad2.SetLeftMargin(0.12)
#pad2.SetGridx()
pad2.Draw()
pad2.cd()

m1 = ROOT.TMultiGraph()
graphs = []
for i, Eprof in enumerate(Eprofs):
   graphs.append(ROOT.TGraphErrors())
   if (i%2==0):
      pmean =[]
   for b in np.arange(1, bins+1):
      mean = Eprof.GetBinContent(b)
      if (i%2==0):
         pmean.append(mean)
         graphs[i].SetPoint(b-1, p[0]+((b-0.5) * (p[1]-p[0])/bins), 0)
         if mean>0:
            graphs[i].SetPointError(b-1, (p[1]-p[0])/(2*bins), ((Eprof.GetBinError(b))/mean))
         else:
            graphs[i].SetPointError(b-1, (p[1]-p[0])/(2*bins), 0)
      else:
        if pmean[b-1] > 0:
          graphs[i].SetPoint(b-1, p[0] + ((b-0.5) * (p[1]-p[0])/bins), (pmean[b-1]-mean)/pmean[b-1])
        else:
          graphs[i].SetPoint(b-1, p[0]+((b-0.5) * (p[1]-p[0])/bins), 0)
        if mean > 0:
          print(b, mean, pmean[b-1], np.sqrt((((Eprofs[i-1].GetBinError(b))**2 + (Eprofs[i].GetBinError(b))**2))))#/(pmean[b-1]-mean)**2) + (((Eprofs[i-1].GetBinError(b))/mean)**2)))
          graphs[i].SetPointError(b-1, (p[1]-p[0])/(2*bins), np.sqrt((((Eprofs[i-1].GetBinError(b)/pmean[b-1])**2)) + ((Eprofs[i].GetBinError(b)/mean)**2)))#/(pmean[b-1]-mean)**2) + (((Eprofs[i-1].GetBinError(b))/mean)**2)))
        else:
          graphs[i].SetPointError(b-1, (p[1]-p[0])/(2*bins), 0)
   graphs[i].SetLineColor(color[i])
   m1.Add(graphs[i])
m1.Draw('ALPE')

m1.SetTitle(' ;E_{P} [GeV]; relative error')
#m1.SetTitleSize(0)
m1.GetYaxis().SetTitleOffset(0.8)
m1.GetYaxis().CenterTitle(1)
m1.GetYaxis().SetTitleSize(0.1)
m1.GetXaxis().SetTitleSize(0.1)
m1.GetXaxis().SetRangeUser(p[0], p[1])
m1.GetYaxis().SetRangeUser(-0.1, 0.1)
m1.GetYaxis().SetLabelSize(0.08)
m1.GetXaxis().SetLabelSize(0.08)
#m1.GetYaxis().SetNdivisions(305)
m1.Draw('ALPE')
c.Update()

c.Update()
c.cd()

pad3 = ROOT.TPad("pad1", "pad1", 0.5, 0.3, 1.0, 1.0)
pad3.SetBottomMargin(0)
pad3.SetLeftMargin(0.12)
pad3.Draw()
#pad3.SetGridx()
pad3.cd()

m2 = ROOT.TMultiGraph()
graphstd = []
for i, t in enumerate(Temps):
   graphstd.append(ROOT.TGraphErrors())     
   for b in np.arange(1, bins+1):
      std = t.GetBinError(b)
      mean = Eprofs[i].GetBinContent(b)
      print(b, std, mean, t.GetBinEntries(b))
      if mean > 0:
         graphstd[i].SetPoint(b-1, p[0]+((b-0.5) * (p[1]-p[0])/bins), std/mean)
      else:
         graphstd[i].SetPoint(b-1, p[0]+((b-0.5) * (p[1]-p[0])/bins), 0)
      if t.GetBinEntries(b) > 0:
         graphstd[i].SetPointError(b-1, (p[1]-p[0])/(2*bins), std/(mean * np.sqrt(t.GetBinEntries(b))))
      else:
         graphstd[i].SetPointError(b-1, (p[1]-p[0])/(2*bins), 0)
   m2.Add(graphstd[i])
   graphstd[i].SetLineColor(color[i])
m2.Draw('ALPE')
m2.SetTitle(title + ';abc;#sigma')
m2.GetXaxis().SetTitleSize(0.)
m2.GetYaxis().SetTitleOffset(1.)
m2.GetYaxis().CenterTitle(1)

m2.GetYaxis().SetTitleSize(0.03)
m2.GetXaxis().SetRangeUser(p[0], p[1])
m2.GetYaxis().SetRangeUser(0, 0.25)
m2.GetYaxis().SetLabelSize(0.025)
m2.GetXaxis().SetLabelSize(0.)
#m1.GetYaxis().SetNdivisions(305)
m2.Draw('ALPE')

c.Update()
c.cd()

pad4 = ROOT.TPad("pad2", "pad2", 0.5, 0.05, 1, 0.3)
pad4.SetTopMargin(0)
pad4.SetBottomMargin(0.2)
pad4.SetLeftMargin(0.12)
#pad4.SetGridx()
pad4.Draw()
pad4.cd()

m3 = ROOT.TMultiGraph()
diffstd = []

for i, t in enumerate(graphstd):
   diffstd.append(ROOT.TGraphErrors())
   if (i%2!=0):
      y1 = graphstd[i-1].GetY()
      y2 = graphstd[i].GetY()
      for pt in np.arange(bins):
        e1 = np.float(graphstd[i-1].GetErrorY(pt))
        e2 = np.float(graphstd[i].GetErrorY(pt))
        
        diffstd[i].SetPoint(pt, p[0]+((pt+0.5) * (p[1]-p[0])/bins), (y1[pt]-y2[pt])/y1[pt])
        diffstd[i].SetPointError(pt, (p[1]-p[0])/(2*bins), ((np.divide(e1, y1[pt])) + np.divide(e2, y2[pt])))
   else:
      for pt in np.arange(bins):
        diffstd[i].SetPoint(pt, p[0]+((pt+0.5) * (p[1]-p[0])/bins), 0)
        diffstd[i].SetPointError(pt, (p[1]-p[0])/(2*bins), 0)
   m3.Add(diffstd[i])
   diffstd[i].SetLineColor(color[i])
m3.Draw('ALPE')
m3.SetTitle(' ;E_{P} [GeV]; #sigma error')
#m3.SetTitleSize(0)
m3.GetYaxis().SetTitleOffset(0.8)
m3.GetYaxis().CenterTitle(1)

m3.GetYaxis().SetTitleSize(0.1)
m3.GetXaxis().SetTitleSize(0.1)
m3.GetXaxis().SetRangeUser(p[0], p[1])
m3.GetYaxis().SetRangeUser(-0.5, 1.5)
m3.GetYaxis().SetLabelSize(0.08)
m3.GetXaxis().SetLabelSize(0.08)
#m3.GetYaxis().SetNdivisions(305)                                                                                                                                                                          
m3.Draw('ALPE')

c.Update()

c.Print(filename+'.pdf')
print ' The plot is saved in.....{}.pdf'.format(filename)

                                                                  
