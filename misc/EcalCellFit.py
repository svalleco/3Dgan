# This script allows to fit the cell energies
from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import math
import time
import glob
import sys
import numpy.core.umath_tests as umath
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/analysis')
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')
sys.path.insert(0,'/nfshome/gkhattak/keras/architectures_tested/')
import utils.GANutils as gan
import utils.ROOTutils as r
import setGPU

def main():
    datafile = "/data/shared/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"
    numevents= 1000
    thresh = 0
    dirname = 'results/ecal_flat_rd_range_fit'
    gan.safe_mkdir(dirname)
    fits = ['landau','gaus', 'expon']
    #log= False #take log of data
    x, y =GetData(datafile, numevents, thresh=thresh)
    #plot_ecal_flatten(x, os.path.join(dirname, 'ecal_flat'), fits, 'G4', norm=1, logy=0, logx=0)
    #plot_ecal_flatten(x, os.path.join(dirname, 'ecal_flat_logy'), fits, 'G4', norm=1, logy=1, logx=0)
    #plot_ecal_flatten(x, os.path.join(dirname, 'ecal_flat_logx'), fits, 'G4', norm=1, logy=0, logx=1)
    #plot_ecal_flatten(x, os.path.join(dirname, 'ecal_flat_logx_logy'), fits, 'G4', norm=1, logy=1, logx=1)
    #plot_ecal_flatten(x, os.path.join(dirname, 'ecal_flat_logbins'), fits, 'G4', norm=1, logy=1, logx=1, logbin=1)
    plot_ecal_flatten_fit(x, dirname, fits, "Cell", logy=0, norm=0, ifpdf=True, log=False)
    print('Result is saved in {}'.format(dirname))
    
def GetData(datafile, numevents, scale=1, thresh=1e-6):
    #get data for training
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=np.array(f.get('energy')[:numevents])
    x=np.array(f.get('ECAL')[:numevents])
    x[x<thresh] = 0
    x = x * scale
    return x, y

def plot_ecal_flatten_fit(event, out_dir, fits, label, logy=0, norm=0, ifpdf=True, log=False):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    c1.SetGrid()
    #if not log:
    ROOT.gPad.SetLogx()
    ROOT.gStyle.SetOptFit(1011)
    inv_pois = ROOT.TF1("inv_pois", "-1 *[0]*TMath::Poisson(x,[1])", -12, 3)
    pois = ROOT.TF1("pois", "[0]*TMath::Poisson(x,[1])", -12, 3)
    expon = ROOT.TF1("pois", "[0]*TMath::Power(x,[1]) + [3]")    
    title = "Cell energy deposits for 100-200 GeV"
    if logy:
        ROOT.gPad.SetLogy()
        title = title + " (logy) "
    hd= ROOT.TH1F(label, "", 100, -8, -2)
    r.BinLogX(hd)
    
    data = event.flatten()
    print('The mean of data is {} and std is {}'.format(np.mean(data), np.std(data)))
    #indexes = np.where(data>0)
    #data = data[indexes]
    if log:
       data= np.log(data)
    print(np.amin(data), np.amax(data))
    r.fill_hist(hd, data)
    if norm:
        r.normalize(hd, norm-1)
    average = hd.GetMean()
    inv_pois.SetParameter(0, average)
    inv_pois.SetParameter(1, np.sqrt(average))
    pois.SetParameter(0, average)
    pois.SetParameter(1, np.sqrt(average))
    hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV/50")
    hd.GetYaxis().SetTitle("Count")
    hd.GetYaxis().CenterTitle()
    hd.Draw()
    hd.Draw('sames hist')
    c1.Update()
    for fit in fits:
      hd.SetTitle(title + ' {} fit'.format(fit))
      hd.Fit(fit)
      c1.Update()
      c1.Modified()
      c1.Update()
      if ifpdf:
         c1.Print(out_dir + '/fit_{}.pdf'.format(fit))
      else:
         c1.Print(out_dir + '/fit_{}.C'.format(fit))

def plot_ecal_flatten(event, out_file, fits, label, logy=0,  logx=0, norm=0, ifpdf=True, logbin=0):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    c1.SetGrid()
    title = "Cell energy deposits for 100-200 GeV"
    if logx:
       ROOT.gPad.SetLogx()
       title = title + " (logy) "
    title = "Cell energy deposits for 100-200 GeV"
    if logy:
        ROOT.gPad.SetLogy()
        title = title + " (logy) "
    data = event.flatten()
    if logbin: 
      hd= ROOT.TH1F(label, "", 100, -10, -2)
      r.BinLogX(hd)
    else:
      hd= ROOT.TH1F(label, "", 100, 1e-10, 1e-2)
    indexes = np.where(data>0)
    data = data[indexes]
    print('The mean of data is {} and std is {}'.format(np.mean(data), np.std(data)))
    print('The min of data is {} and max is {}'.format(np.amin(data), np.amax(data)))
            
    r.fill_hist(hd, data)
    if norm:
        r.normalize(hd, norm-1)
    average = hd.GetMean()
    hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV/50")
    hd.GetYaxis().SetTitle("Count")
    hd.GetYaxis().CenterTitle()
    hd.SetTitle(title)
    hd.Draw()
    hd.Draw('sames hist')
    c1.Update()
    if ifpdf:
       c1.Print(out_file + '.pdf')
    else:
       c1.Print(out_file + '.C')

if __name__ == "__main__":
    main()                                                                                                                                                                                          
