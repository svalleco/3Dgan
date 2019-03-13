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
from skimage import measure
import setGPU # if caltech
sys.path.insert(0,'../')
import analysis.utils.GANutils as gan

def main():
    # All of the following needs to be adjusted
    from AngleArch3dGAN import generator # architecture
    datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*VarAngleMeas_*.h5" # path to data
    genpath = "../weights/3dgan_weights_bins_pow_p85/params_generator*.hdf5"# path to weights
    plotsdir = 'results/iqa_epochs' # plot directory
    particle = "Ele" 
    scale = 1.0
    power = 0.85
    threshold = 0
    ang = 1
    g= generator(latent_size=256)
    start = 0
    stop = 5
    gen_weights=[]
    disc_weights=[]
    fits = ['pol2', 'pol3']
        
    gan.safe_mkdir(plotsdir)
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)

    epoch = []
    for i in np.arange(len(gen_weights)):
      name = os.path.basename(gen_weights[i])
      num = int(filter(str.isdigit, name)[:-1])
      epoch.append(num)
    
    print("{} weights are found".format(len(gen_weights)))

    post = inv_power
    pre = taking_power
    result= GetResult(g, gen_weights, datapath, power=power, scale=scale, post=post, pre=pre)
    PlotResultsRoot(result, os.path.join(plotsdir, 'IQA_metrics.pdf'), start, epoch, fits)

def taking_power(n, power, scale=1.):
    return np.power(n * scale, power)

def inv_power(n, power, scale=1.):
    return np.power(n, 1./power)/scale

def GetResult(g, genweights, datapath, num_events=1000, latent=256, power=1., scale=1., post=None, pre=None):
    data_files= gan.GetDataFiles(datapath, ['Ele'])
    images, y, ang = gan.GetAngleData(data_files[0], thresh=0., angtype='theta')
    images, y, ang = images[:num_events], y[:num_events], ang[:num_events]
    y=y/100.
    res={}
    metrics=['ssim', 'psnr']
    data_range = np.amax(images)
    L= 1e-6
    for m in metrics:
       res[m] = [] 
    for gw in genweights:
       print('Loading weights from {}'.format(gw))
       g.load_weights(gw)
       generated_images=gan.generate(g, num_events, [y, ang], latent, concat=1)
       #if post!=None:
       generated_images=np.power(generated_images, 1./power)

       ssim = measure.compare_ssim(images, generated_images, multichannel=True, data_range=data_range, gaussian_weights=True, use_sample_covariance=False)
       psnr = measure.compare_psnr(images, generated_images, data_range=data_range)
       print('SSIM ={} PSNR ={}'.format(ssim, psnr))
       res['ssim'].append(ssim)
       res['psnr'].append(psnr)

    return res
                          
#Plots results in a root file
def PlotResultsRoot(result, resultfile, start=0, epoch=[], fits=[]):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    c1.Divide(1, 2)
    c1.SetGrid ()
    c1.cd(1)
    color1=2
    color2=4
    max_ssim=0
    max_psnr=0
    epoch_ssim=0
    epoch_psnr=0
    num = len(result['ssim'])
    epochs = np.zeros((num))
    ssim = np.zeros((num))
    psnr = np.zeros((num))
    for i in np.arange(num):
       epochs[i]=epoch[i]
       ssim[i] = result['ssim'][i]
       psnr[i] = result['psnr'][i]
       if ssim[i]>max_ssim:
           max_ssim=ssim[i]
           epoch_ssim=int(epochs[i])
       if psnr[i] >max_psnr:
           max_psnr=psnr[i]
           epoch_psnr=int(epochs[i])
       
    g_ssim  = ROOT.TGraph(num-start , epochs[start:], ssim[start:] )
    legend1 = ROOT.TLegend(.6, .1, .9, .3)
    g_ssim.SetLineColor(color1)
    legend1.AddEntry(g_ssim, "SSIM max={:.4f} for {:d} epoch".format(max_ssim, epoch_ssim), "l")
    g_ssim.SetTitle("Image Quality Metrics;Epochs;SSIM")
    g_ssim.GetYaxis().SetTitleSize(0.055)
    g_ssim.GetXaxis().SetTitleSize(0.055)
    g_ssim.Draw()
    legend1.Draw()
    c1.Update()
    
    c1.cd(2)
    g_psnr = ROOT.TGraph( num- start , epochs[start:], psnr[start:] )
    g_psnr.SetLineColor(color2)
    legend2 = ROOT.TLegend(.6, .1, .9, .3)
    legend2.AddEntry(g_psnr, "PSNR max={:.4f} for {:d} epoch".format(max_psnr, epoch_psnr), "l")
    g_psnr.Draw()
    g_psnr.SetTitle(";Epochs;PSNR")
    g_psnr.GetYaxis().SetTitleSize(0.055)
    g_psnr.GetXaxis().SetTitleSize(0.055)
        
    legend2.Draw()
    c1.Update()
    
    s=2
    for fit in fits:
      c1.cd(1)
      g_ssim.Fit(fit)
      g_ssim.GetFunction(fit).SetLineColor(color1)
      g_ssim.GetFunction(fit).SetLineStyle(s)
      legend1.AddEntry(g_ssim.GetFunction(fit), fit, "l")
      legend1.Draw()
      c1.Update()

      c1.cd(2)
      g_psnr.Fit(fit)
      g_psnr.GetFunction(fit).SetLineColor(color2)
      g_psnr.GetFunction(fit).SetLineStyle(s)
      legend2.AddEntry(g_psnr.GetFunction(fit), fit, "l")
      legend2.Draw()
      c1.Update()
      s+=1
      
    c1.Print(resultfile)

if __name__ == "__main__":
   main()
