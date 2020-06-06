#!/usr/bin/env python
# -*- coding: utf-8 -*-   

# This file tests the training results with the optimization function

from __future__ import print_function
import sys
import h5py
import os
import numpy as np
import glob
import numpy.core.umath_tests as umath
import time
import math
import ROOT

if '.cern.ch' in os.environ.get('HOSTNAME'): # Here a check for host can be used
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech
except:
    pass
sys.path.insert(0,'../keras/')
import analysis.utils.GANutils as gan

def main():
    result=[]
    resultfile = 'result_log.txt'
    file = open(resultfile)
    plotdir = 'obj_13layers'
    gan.safe_mkdir(plotdir)
    for line in file:
      fields = line.strip().split()
      fields = [float(_) for _ in fields]
      print(fields)
      result.append(fields)
    file.close
    epochs = np.arange(len(result))
    print(epochs)
    print('The result file {} is read.'.format(resultfile))
    PlotResultsRoot(result, plotdir, epochs, ang=0)

#Plots results in a root file
def PlotResultsRoot(result, resultdir, epochs, start=0, end=60, fits="", plotfile='obj_result', ang=1):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    legend = ROOT.TLegend(.5, .6, .9, .9)
    
    legend.SetTextSize(0.028)
    mg=ROOT.TMultiGraph()
    color1 = 2
    color2 = 8
    color3 = 4
    color4 = 6
    color5 = 7
    num = len(result)
    total = np.zeros((num))
    energy_e = np.zeros((num))
    mmt_e = np.zeros((num))
    ang_e = np.zeros((num))
    sf_e  = np.zeros((num))
    epoch = np.zeros((num))
      
    mine = 100
    minm = 100
    mint = 100
    mina = 100
    mins = 100
    for i, item in enumerate(result):
      epoch[i] = epochs[i]  
      total[i]=item[0]
      mmt_e[i]=item[1]
      energy_e[i]=item[2]
      sf_e[i]=item[3]
      if item[0]< mint:
         mint = item[0]
         mint_n = epoch[i]
      if item[1]< minm:
         minm = item[1]
         minm_n = epoch[i]
      if item[2]< mine:
         mine = item[2]
         mine_n = epoch[i]
      if item[3]< mins:
         mins = item[3]
         mins_n = epoch[i]
      if ang:
         ang_e[i]=item[4]
         if item[4]< mina:
           mina = item[4]
           mina_n = epoch[i]
                               
    gt  = ROOT.TGraph( num- start , epoch[start:], total[start:] )
    gt.SetLineColor(color1)
    mg.Add(gt)
    legend.AddEntry(gt, "Total error min = {:.4f} (epoch {})".format(mint, mint_n), "l")
    ge = ROOT.TGraph( num- start , epoch[start:], energy_e[start:] )
    ge.SetLineColor(color2)
    legend.AddEntry(ge, "Energy error min = {:.4f} (epoch {})".format(mine, mine_n), "l")
    mg.Add(ge)
    gm = ROOT.TGraph( num- start , epoch[start:], mmt_e[start:])
    gm.SetLineColor(color3)
    mg.Add(gm)
    legend.AddEntry(gm, "Moment error  = {:.4f} (epoch {})".format(minm, minm_n), "l")
    c1.Update()
    gs = ROOT.TGraph( num- start , epoch[start:], sf_e[start:])
    gs.SetLineColor(color4)
    mg.Add(gs)
    legend.AddEntry(gs, "Sampling Fraction error  = {:.4f} (epoch {})".format(mins, mins_n), "l")
    c1.Update()
                    
    if ang:
      ga = ROOT.TGraph( num- start , epoch[start:], ang_e[start:])
      ga.SetLineColor(color5)
      mg.Add(ga)
      legend.AddEntry(ga, "Angle error  = {:4f} (epoch {})".format(mina, mina_n), "l")
      c1.Update()
                    
    mg.SetTitle("Optimization function: Mean Relative Error on shower shapes, moment and sampling fraction;Epochs;Error")
    mg.Draw('ALP')
    mg.GetYaxis().SetRangeUser(0, 1.2 * np.amax(total))
    c1.Update()
    #legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, plotfile + '.pdf'))
    c1.Print(os.path.join(resultdir, plotfile + '.C'))

    fits = ['pol1', 'pol2', 'expo']
    for i, fit in enumerate(fits):
      mg.SetTitle("Optimization function: Mean Relative Error on shower sahpes, moments and sampling fraction({} fit);Epochs;Error".format(fit))  
      gt.Fit(fit)
      gt.GetFunction(fit).SetLineColor(color1)
      gt.GetFunction(fit).SetLineStyle(2)
    
      ge.Fit(fit)
      ge.GetFunction(fit).SetLineColor(color2)
      ge.GetFunction(fit).SetLineStyle(2)
            
      gm.Fit(fit)
      gm.GetFunction(fit).SetLineColor(color3)
      gm.GetFunction(fit).SetLineStyle(2)

      gs.Fit(fit)
      gs.GetFunction(fit).SetLineColor(color4)
      gs.GetFunction(fit).SetLineStyle(2)

      if i == 0:
        legend.AddEntry(gt.GetFunction(fit), 'Total fit', "l")
        legend.AddEntry(ge.GetFunction(fit), 'Energy fit', "l")
        legend.AddEntry(gm.GetFunction(fit), 'Moment fit', "l")  
        legend.AddEntry(gs.GetFunction(fit), 'S. Fr. fit', "l")
      #legend.Draw()
      c1.Update()
      c1.Print(os.path.join(resultdir, plotfile + '_{}.pdf'.format(fit)))
      c1.Print(os.path.join(resultdir, plotfile + '_{}.C'.format(fit)))
    print ('The plot is saved to {}'.format(resultdir))

if __name__ == "__main__":
    main()
