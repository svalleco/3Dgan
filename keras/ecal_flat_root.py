import sys
import os
import ROOT
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import math
from keras import backend as K
import ROOTutils as my
  
# Computing log only when not zero
def logftn(x, base, f1, f2):
    select= np.where(x>0)
    x[select] = f1 * np.log10(x[select]) + f2
    return x

# Exponent
def expon(a):
    select= np.where(a>0)
    a[select] = np.exp(6.0 * (a[select] - 1.0) * np.log(10.0))
    return a

# Ecal sum Calculation
def get_log(image):
    image = np.squeeze(image, axis = 4)
    return expon(image)

def GetProcData(datafile, num_events):
    #get data for training                                                      
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=f.get('target')[:,1][:num_events]
    x=np.array(f.get('ECAL'))[:num_events]
    y=np.array(y)
    x[x < 1e-6] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

def PlotEcalFlatlog(x, X, outfile, ifpdf=True):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    c1.SetGrid()
    color =2
    ROOT.gPad.SetLogx()
    hd = ROOT.TH1F("Geant4", "", 100, -6, 0)
    my.BinLogX(hd)
    hd.GetXaxis().SetTitle("Ecal GeV")
    my.fill_hist(hd, x.flatten())
    hd.Draw()
    hd.SetLineColor(color)
    legend = ROOT.TLegend(.5, .8, .6, .9)
    legend.AddEntry(hd,"Ecal Energy","l")
    hg = ROOT.TH1F("FromLambda", "", 100, -6, 0)
    color+=2
    my.BinLogX(hg)
    my.fill_hist(hg, X.flatten())
    hg.SetLineColor(color)
    hg.Draw('sames')
    c1.Update()
    legend.AddEntry(hg, "From Lambda", "l")
    legend.Draw()
    c1.Modified()
    if ifpdf:
      c1.Print(outfile + 'Log.pdf')
    else:
      c1.Print(outfile + 'Log.C')

def PlotEcalFlat(x, outfile, ifpdf=True):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make 
    c1.SetGrid()
    hd = ROOT.TH1F("Lambda", "", 100, 0.0001, 1)
    my.fill_hist(hd, x.flatten())
    hd.Draw()
    c1.Update()
    legend = ROOT.TLegend(.5, .8, .6, .9)
    legend.AddEntry(hd,"From Lambda","l")
    c1.Update()
    if ifpdf:
      c1.Print(outfile + 'Flat.pdf')
    else:
      c1.Print(outfile + 'Flat.C')


def main():
    datafile="/eos/project/d/dshep/LCD/V1/EleEscan/EleEscan_1_1.h5"
    num_events=100
    outfile = 'ecal_root'
    X, Y = GetProcData(datafile, num_events)
    xlog = logftn(X, 10, 1.0/6.0, 1.0)
    PlotEcalFlat(xlog, outfile)
    x = get_log(xlog)
    PlotEcalFlatlog(X, x, outfile)
                   
if __name__ == '__main__':
    main()
