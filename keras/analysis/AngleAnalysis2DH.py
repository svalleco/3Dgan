from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import sys
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
import utils.GANutils as gan
import utils.ROOTutils as r
import setGPU #if Caltech

sys.path.insert(0,'/nfshome/gkhattak/3Dgan')

def main():
   genweight = "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_50weight_withoutsqrt/params_generator_epoch_059.hdf5"
   from AngleArch3dGAN import generator
   latent = 256
   g=generator(latent)
   g.load_weights(genweight)

   num_events = 1000 # Number of events for each bin
   num=10 # random events generated
   f = 57.325
   thetamin = 60/f
   thetamax = 120/f
   energies=[100, 150, 200]
   thetas = [62, 90, 118]
   ang = 1
   xscale = 1
   post = scale
   plotsdir = 'results/without_sqrt_genplots_ep59'
   gan.safe_mkdir(plotsdir)
   opt="colz"
   events = {}
   for energy in energies:
     events[str(energy)] = {} 
     for t in thetas:
         sampled_energies=energy/100 * np.ones((num_events))  # scale energy
         sampled_thetas = (np.float(t)/f )* np.ones((num_events)) # get radian
         events[str(energy)][str(t)] = gan.generate(g, num_events, sampled_energies, sampled_thetas) # generate
         events[str(energy)][str(t)] = post(events[str(energy)][str(t)], xscale)
         PlotAngleCut(events[str(energy)][str(t)], t, os.path.join(plotsdir, 'Theta{}_GeV{}.pdf'.format(t, energy)), opt=opt)
     plot_energy_hist_gen(events[str(energy)], os.path.join(plotsdir, 'Hist_GeV{}.pdf'.format(energy)), energy, thetas)
   for t in thetas:
      sampled_energies=np.random.uniform(1, 2, size=(num_events))
      sampled_thetas = (np.float(t) /f )* np.ones((num_events))
      events = gan.generate(g, num_events, sampled_energies, sampled_thetas)
      events = post(events, xscale)
      theta_dir = plotsdir+ '/theta_{}'.format(t)
      gan.safe_mkdir(theta_dir)
      for n in np.arange(num):
          PlotEvent(events[n], sampled_energies[n], sampled_thetas[n], os.path.join(theta_dir, 'Event{}.pdf'.format(n)), n, f, opt=opt)
            
      PlotAngleCut(events, t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)), opt=opt)

   sampled_energies=np.random.uniform(1, 2, size=(num))
   sampled_thetas = np.random.uniform(thetamin, thetamax, size=(num))
   events = gan.generate(g, num, sampled_energies, sampled_thetas)
   events = post(events, xscale)
   for n in np.arange(num):
     PlotEvent(events[n], sampled_energies[n], sampled_thetas[n], os.path.join(plotsdir, 'Event{}.pdf'.format(n)), n, f, opt=opt)
   print('Plots are saved in {}'.format(plotsdir))

def square(n, xscale):
   return np.square(n)/xscale

def scale(n, xscale):
   return n / xscale

def plot_energy_hist_gen(events, out_file, energy, thetas, log=0, ifC=False):
   canvas = TCanvas("canvas" ,"abc" ,200 ,10 ,700 ,500) #make  
   canvas.SetGrid()
   label = "Weighted Histograms for {} GeV".format(energy)
   canvas.Divide(2,2)
   color = 2
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.05)
   print (len(events))
   hx=[]
   hy=[]
   hz=[]
   thetas = list(reversed(thetas))
   for i, theta in enumerate(thetas):
      event = events[str(theta)]
      num = event.shape[0]
      sumx, sumy, sumz=gan.get_sums(event)
      x=sumx.shape[1]
      y=sumy.shape[1]
      z=sumz.shape[1]
      hx.append(ROOT.TH1F('GANx{:d}theta_{:d}GeV'.format(theta, energy), '', x, 0, x))
      hy.append(ROOT.TH1F('GANy{:d}theta_{:d}GeV'.format(theta, energy), '', y, 0, y))
      hz.append(ROOT.TH1F('GANz{:d}theta_{:d}GeV'.format(theta, energy), '', z, 0, z))
      hx[i].SetLineColor(color)
      hy[i].SetLineColor(color)
      hz[i].SetLineColor(color)
      hx[i].GetXaxis().SetTitle("X axis")
      hy[i].GetXaxis().SetTitle("Y axis")
      hz[i].GetXaxis().SetTitle("Z axis")
      canvas.cd(1)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hx[i], sumx)
   
      if i ==0:
         hx[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hx[i].DrawNormalized('sames hist')
      canvas.cd(2)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hy[i], sumy)
      if i==0:
         hy[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hy[i].DrawNormalized('sames hist')
      canvas.cd(3)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hz[i], sumz)
      if i==0:
         hz[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hz[i].DrawNormalized('sames hist')
                  
      canvas.cd(4)
      leg.AddEntry(hx[i], '{}theta {}events'.format(theta, num),"l")
      leg.SetHeader(label, 'C')
      canvas.Update()
      color+= 1
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file + '.pdf')
   if ifC:
      canvas.Print(out_file + '.C')

def MeasPython1(image, mod=0):
    x_shape= image.shape[1]
    y_shape= image.shape[2]
    z_shape= image.shape[3]
            
    sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
    indexes = np.where(sumtot > 0)
    amask = np.ones_like(sumtot)
    amask[indexes] = 0
    #amask = K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
    masked_events = np.sum(amask) # counting zero sum events

    x_ref = np.sum(np.sum(image, axis=(2, 3)) * np.expand_dims(np.arange(x_shape) + 0.5, axis=0), axis=1)
    y_ref = np.sum(np.sum(image, axis=(1, 3)) * np.expand_dims(np.arange(y_shape) + 0.5, axis=0), axis=1)
    z_ref = np.sum(np.sum(image, axis=(1, 2)) * np.expand_dims(np.arange(z_shape) + 0.5, axis=0), axis=1)

    x_ref[indexes] = x_ref[indexes]/sumtot[indexes]
    y_ref[indexes] = y_ref[indexes]/sumtot[indexes]
    z_ref[indexes] = z_ref[indexes]/sumtot[indexes]
                         
    sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z

    x = np.expand_dims(np.arange(x_shape) + 0.5, axis=0)
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(np.arange(y_shape) + 0.5, axis=0)
    y = np.expand_dims(y, axis=2)
    x_mid = np.sum(np.sum(image, axis=2) * x, axis=1)
    y_mid = np.sum(np.sum(image, axis=1) * y, axis=1)
    indexes = np.where(sumz > 0)

    zmask = np.zeros_like(sumz)
    zmask[indexes] = 1
    zunmasked_events = np.sum(zmask, axis=1)

    x_mid[indexes] = x_mid[indexes]/sumz[indexes]
    y_mid[indexes] = y_mid[indexes]/sumz[indexes]
    z = np.arange(z_shape) + 0.5# z indexes
    x_ref = np.expand_dims(x_ref, 1)
    y_ref = np.expand_dims(y_ref, 1)
    z_ref = np.expand_dims(z_ref, 1)

    zproj = np.sqrt((x_mid-x_ref)**2.0  + (z - z_ref)**2.0)
    m = (y_mid-y_ref)/zproj
    z = z * np.ones_like(z_ref)
    indexes = np.where(z<z_ref)
    m[indexes] = -1 * m[indexes]
    ang = (math.pi/2.0) - np.arctan(m)
    ang = ang * zmask
    if mod==0:
       ang = np.sum(ang, axis=1)/zunmasked_events
    if mod==1:
       wang = ang * sumz
       sumz_tot = sumz * zmask
       ang = np.sum(wang, axis=1)/np.sum(sumz_tot, axis=1)
              
    indexes = np.where(amask>0)
    ang[indexes] = 100.
                                                       #ang = ang.reshape(-1, 1)
    return ang
                                                             

def PlotEvent(event, energy, theta, out_file, n, factor, opt=""):
   canvas = TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) #make 
   canvas.Divide(2,2)
   event[event<1e-4]=0
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   print(event.shape)
   print(np.expand_dims(event, axis=0).shape)
   ang1 = MeasPython1(np.moveaxis(event, 3, 0))
   ang2 = MeasPython1(np.moveaxis(event, 3, 0), mod=1)
  
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   leg.SetTextSize(0.04)
   leg.SetHeader("#splitline{Weighted Histograms for energies deposited in}{x, y and z planes}", 'C')
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}degree'.format(100 * energy, factor * theta), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}degree'.format(100 * energy, factor * theta), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}degree'.format(100 * energy, factor * theta), '', x, 0, x, y, 0, y)
   gPad.SetLogz()
   event = np.expand_dims(event, axis=0)
   FillHist2D_wt(hx, np.sum(event, axis=1))
   FillHist2D_wt(hy, np.sum(event, axis=2))
   FillHist2D_wt(hz, np.sum(event, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Energy Input = {:.2f} GeV'.format(100 * energy),"l")
   leg.AddEntry(hy, 'Theta Input  = {:.2f} Degree'.format(theta * factor),"l")
   print (ang1.shape)
   leg.AddEntry(hz, 'Computed Theta (mean)     = {:.2f} Degree'.format(ang1[0] * factor),"l")
   leg.AddEntry(hz, 'Computed Theta (weighted) = {:.2f} Degree'.format(ang2[0]* factor),"l")
   leg.Draw()
   r.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotAngleCut(events, ang, out_file, opt=""):
   canvas = TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) 
   canvas.Divide(2,2)
   n = events.shape[0]
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
   gStyle.SetPalette(1)
   gPad.SetLogz()
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.05)
   hx = ROOT.TH2F('X{} Degree'.format(str(ang)), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('Y{} Degree'.format(str(ang)), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('Z{} Degree'.format(str(ang)), '', x, 0, x, y, 0, y)
   FillHist2D_wt(hx, np.sum(events, axis=1))
   FillHist2D_wt(hy, np.sum(events, axis=2))
   FillHist2D_wt(hz, np.sum(events, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hy.GetYaxis().CenterTitle()
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hz.GetYaxis().CenterTitle()
   canvas.cd(4)
   leg.SetHeader("#splitline{Weighted Histograms for energies}{deposited in x, y, z planes}", 'C')
   leg.AddEntry(hx, "{} Theta and {} events".format(ang, n), 'l')
   leg.Draw()
   canvas.Update()
   r.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def FillHist2D_wt(hist, array):
   array= np.squeeze(array, axis=3)
   dim1 = array.shape[0]
   dim2 = array.shape[1]
   dim3 = array.shape[2]
   bin1 = np.arange(dim1)
   bin2 = np.arange(dim2)
   bin3 = np.arange(dim3)
   count = 0
   for j in bin2:
     for k in bin3:
        for i in bin1:
            hist.Fill(j, k, array[i, j, k])
            count+=1

   #hist.Sumw2()

if __name__ == "__main__":
    main()


   
