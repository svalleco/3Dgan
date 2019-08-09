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
   datapath = "/bigdata/shared/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"
   outdir = 'results/data_bins_spectrum_tprof'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath) #, ftn=np.sqrt)
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))
   labels = ['G4 Var. Angle']
   
   bins=[0.05, 0.03, 0.02, 0.0125, 0.008, 0.003]
   scale = 50.
   bins.extend([0, -0.003])
   bins = sorted(bins)
   bins.extend([np.amax(x)])
 
   edges = np.zeros(len(bins))
   for i, bin in enumerate(bins):
     edges[i] = bins[i]/scale
   p=[int(np.amin(y)), int(np.amax(y))]
   #plot_ecal_hist_bins([x/scale], edges, outfile, y, labels, logy=0, logx=0, norm=2, p=p, ifpdf=True)   
   #plot_ecal_hist_bins([x/scale], edges, outfile+'_ylog', y, labels, logy=1, logx=0, norm=2, p=p, ifpdf=True)
   #plot_ecal_hist_bins([x/scale], edges[3:], outfile+'_xylog', y, labels, logy=1, logx=1, norm=2, p=p, ifpdf=True)
   #plot_ecal_hist_bins([x/scale], edges[3:], outfile+'_xlog', y, labels, logy=0, logx=1, norm=2, p=p, ifpdf=True)
   plot_ecal_prof_bins([x], outfile+'_prof', y, labels, scale=scale, logy=0, logx=0, norm=2, p=p, ifpdf=True)
   print('Histogram is saved in ', outfile)
         
def GetAngleData(datafile, numevents=1000, ftn=0, scale=1, angtype='theta'):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents]) * scale
   if ftn!=0:
      x = ftn(x)
   ang = np.array(f.get(angtype)[:numevents])
   return x, y, ang

def hist_count(x, p=1.0, daxis=(0, 1, 2), limits=[0.05, 0.03, 0.02, 0.0125, 0.008, 0.003]):
    limits=np.array(limits) # bin boundaries used
    limits= np.power(limits, p)
    bin1 = np.sum(np.where(x>(limits[0]) , 1, 0), axis=daxis)
    bin2 = np.sum(np.where((x<(limits[0])) & (x>(limits[1])), 1, 0), axis=daxis)
    bin3 = np.sum(np.where((x<(limits[1])) & (x>(limits[2])), 1, 0), axis=daxis)
    bin4 = np.sum(np.where((x<(limits[2])) & (x>(limits[3])), 1, 0), axis=daxis)
    bin5 = np.sum(np.where((x<(limits[3])) & (x>(limits[4])), 1, 0), axis=daxis)
    bin6 = np.sum(np.where((x<(limits[4])) & (x>(limits[5])), 1, 0), axis=daxis)
    bin7 = np.sum(np.where((x<(limits[5])) & (x>0.), 1, 0), axis=daxis)
    bin8 = np.sum(np.where(x==0, 1, 0), axis=daxis)
    bins = [bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8]
    return bins

def plot_ecal_flatten_hist_bins(events, bins, out_file, energy, labels, logy=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
      
   title = "Cell energy {:d}-{:d} GeV Primary".format(p[0], p[1])
   legend = ROOT.TLegend(.1, .6, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -8, 1))
      hd = hds[i]
      hd.SetStats(0)
      #r.BinLogX(hd)
      data = event.flatten()
      r.fill_hist(hd, data)
      lines=[]
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:                  
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV")
        hd.GetYaxis().SetTitle("Count")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        hd.Draw('sames hist')
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
      for ln, b in enumerate(bins):
        lines.append(ROOT.TLine(b, 0, b, 0.06))
        lines[ln].Draw('sames')
      color+=2
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()
      
   legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')

def plot_ecal_hist_bins(events, bin_edges, out_file, energy, labels, logy=0, logx=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                              
   c1.SetGrid()
   if logx:
     ROOT.gPad.SetLogx()
   title = "Cell energy {:d}-{:d} GeV Primary".format(p[0], p[1])
   legend = ROOT.TLegend(.1, .8, .3, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", len(bin_edges)-1, bin_edges))
      hd = hds[i]
      #hd.SetStats(0)
      #hd.Sumw2()
      for e in event:
        data = e.flatten()
        r.fill_hist(hd, data)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(color)
      if i ==0:
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Ecal Single cell depositions GeV")
        hd.GetYaxis().SetTitle("Count")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        hd.Draw('sames hist')
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
      color+=2
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()

   legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')

def plot_ecal_prof_bins(events, out_file, energy, labels, scale=1, bins = [0.05, 0.03, 0.02, 0.0125, 0.008, 0.003], logy=0, logx=0, norm=0, ifpdf=True, p=[2, 500]):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   xticks = sorted([10] + bins + [0])
   xticks = [_/scale for _ in xticks]
   print(xticks)                                                                                                                                                              
   c1.SetGrid()
   if logx:
     ROOT.gPad.SetLogx()
   title = "Cell energy {:d}-{:d} GeV Primary".format(p[0], p[1])
   legend = ROOT.TLegend(.4, .8, .6, .9)
   color =2
   if logy:
      ROOT.gPad.SetLogy()
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TProfile(label, "", len(bins) + 1, 0, len(bins)+ 1, "S"))
      hd = hds[i]
      hd.SetStats(0)
      for e in event:
         counts = hist_count(e, limits=bins)
         counts = np.flip(counts, 0)
         for c in np.arange(len(counts)-1):
           hd.Fill(c, counts[c + 1])
      hd.SetLineColor(color)
      if i ==0:
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Single cell depositions Bins [GeV]")
        for c in np.arange(1, len(counts)+ 1):
          hd.GetXaxis().ChangeLabel(c, -1, -1, -1, -1, -1, str(xticks[c-1]))
        hd.GetYaxis().SetTitle("Counts")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        #hd.Draw('sames hist')                                                                                                                                                                                        
      else:
        hd.Draw('sames')
        #hd.Draw('sames hist')                                                                                                                                                                                        
      color+=2
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()

   legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')

if __name__ == "__main__":
   main()
