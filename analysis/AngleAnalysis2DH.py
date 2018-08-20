from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
import GANutilsANG_concat as gan
import ROOTutils as r
import setGPU #if Caltech

def main():
   genweight = "3d_angleweights_1loss/params_generator_epoch_046.hdf5"
   from EcalCondGanAngle_3d_1loss import generator
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

   plotsdir = 'genplots_ep047'
   gan.safe_mkdir(plotsdir)
   opt="colz"
   events = {}
   for energy in energies:
     events[str(energy)] = {} 
     for t in thetas:
         sampled_energies=energy/100 * np.ones((num_events))  # scale energy
         sampled_thetas = (np.float(t)/f )* np.ones((num_events)) # get radian
         events[str(energy)][str(t)] = gan.generate(g, num_events, sampled_energies, sampled_thetas) # generate list of events
         PlotAngleCut(events[str(energy)][str(t)], t, os.path.join(plotsdir, 'Theta{}_GeV{}.pdf'.format(t, energy)), opt=opt)
     plot_energy_hist_gen(events[str(energy)], os.path.join(plotsdir, 'Hist_GeV{}.pdf'.format(energy)), energy, thetas)
   for t in thetas:
      sampled_energies=np.random.uniform(1, 2, size=(num_events))
      sampled_thetas = (np.float(t) /f )* np.ones((num_events))
      events = gan.generate(g, num_events, sampled_energies, sampled_thetas)
      PlotAngleCut(events, t, os.path.join(plotsdir, 'Theta_cut{}.pdf'.format(t)), opt=opt)

   sampled_energies=np.random.uniform(1, 2, size=(num))
   sampled_thetas = np.random.uniform(thetamin, thetamax, size=(num))
   events = gan.generate(g, num, sampled_energies, sampled_thetas)

   for n in np.arange(num):
     PlotEvent(events[n], sampled_energies[n], sampled_thetas[n], os.path.join(plotsdir, 'Event{}.pdf'.format(n)), n, f, opt=opt)
   print('Plots are saved in {}'.format(plotsdir))
     
def plot_energy_hist_gen(events, out_file, energy, thetas, log=0, ifC=False):
   canvas = TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make  
   canvas.SetGrid()
   canvas.Divide(2,2)
   color = 2
   leg = ROOT.TLegend(0.1,0.6,0.7,0.9)
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
      canvas.cd(1)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hx[i], sumx)
      if i ==0:
         hx[i].DrawNormalized('hist')
      else:
         hx[i].DrawNormalized('sames hist')
      canvas.cd(2)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hy[i], sumy)
      if i==0:
         hy[i].DrawNormalized('hist')
      else:
         hy[i].DrawNormalized('sames hist')
      canvas.cd(3)
      if log:
         gPad.SetLogy()
      r.fill_hist_wt(hz[i], sumz)
      if i==0:
         hz[i].DrawNormalized('hist')
      else:
         hz[i].DrawNormalized('sames hist')
                  
      canvas.cd(4)
      leg.AddEntry(hx[i], '{}theta {}events'.format(theta, num),"l")
      canvas.Update()
      color+= 1
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file + '.pdf')
   if ifC:
      canvas.Print(out_file + '.C')

def Meas3(event, p1=5, p2=15):
   z_shape = event.shape[2]
   x = np.zeros(z_shape) # shape = (z)
   y = np.zeros(z_shape)
   ang = np.zeros(z_shape)
   sumz = np.zeros(z_shape) 
   for j in np.arange(p2 + 1): # Looping over z
      sumz[j] = np.sum(event[:, :, j])
      x[j] = 0
      y[j] = 0
      for k in np.arange(event.shape[0]):  # Looping over x
         for l in np.arange(event.shape[1]): # Looping over y
            x[j] = x[j] + event[k, l, j] * k
            y[j] = y[j] + event[k, l, j] * l
      if sumz[j] > 0:                         # check for zero sum
         x[j] = x[j]/sumz[j]
         y[j] = y[j]/sumz[j]
         if j >p1:
           zproj = np.sqrt((x[j] - x[p1])**2 + (j-p1)**2) 
           ang[j] = np.arctan((y[j] - y[p1])/zproj)
           ang[j] = math.pi/2 - ang[j]
   result = ProcAngle1(ang[p1:p2+1], sumz[p1:p2+1])
   return result

def ProcAngle1(meas, sumz):
   sumtot = np.sum(sumz)
   avg = np.sum( meas * sumz)/ sumtot
   return avg

def ProcAngle2(meas, sumz):
   l = meas.shape[0]
   index = np.arange(1, l+1)
   sumtot = l * (l+ 1) * 0.5
   avg = np.sum( meas * index)/ sumtot
   return avg
      
def Meas1(event, yp1 = 3, yp2 = 21):
   event = np.sum(event , axis=(0))
   y = np.arange(event.shape[1])
   maxy = np.argmax(event, axis=0)
   p = np.polyfit(y[yp1:yp2], maxy[yp1:yp2], 1)
   angle = math.pi/2 - math.atan(p[0])
   return angle

def Meas2(event, yp1 = 3, yp2 = 21):
    event = np.sum(event, axis=(0))
    y = np.arange(event.shape[1])
    maxy = np.argmax(event, axis=0)
    tan = (maxy[yp2]-maxy[yp1]) / np.float(yp2 - yp1)
    angle = math.pi/2 - math.atan(tan)
    return angle

def PlotEvent(event, energy, theta, out_file, n, factor, opt=""):
   canvas = TCanvas("canvas" ,"GAN 2D Hist" ,200 ,10 ,700 ,500) #make 
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   ang1 = Meas1(event)
   ang2 = Meas2(event)
   ang3 = Meas3(event)
   leg = ROOT.TLegend(0.1,0.6,0.7,0.9)
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}degree'.format(100 * energy, factor * theta), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}degree'.format(100 * energy, factor * theta), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}degree'.format(100 * energy, factor * theta), '', x, 0, x, y, 0, y)
   gStyle.SetPalette(1)
   gPad.SetLogz()
   event = np.expand_dims(event, axis=0)
   FillHist2D_wt(hx, np.sum(event, axis=1))
   FillHist2D_wt(hy, np.sum(event, axis=2))
   FillHist2D_wt(hz, np.sum(event, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Sampled Energy={:.2f} GeV'.format(100 * energy),"l")
   leg.AddEntry(hy, 'Sampled Theta ={:.2f} Degree'.format(theta * factor),"l")
   leg.AddEntry(hz, 'Computed Theta={:.2f}/{:.2f}/{:.2f} Degree'.format(ang1 * factor, ang2 * factor, ang3 * factor),"l")
   leg.Draw()
   r.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotAngleCut(events, ang, out_file, opt=""):
   canvas = TCanvas("canvas" ,"GAN 2D Hist" ,200 ,10 ,700 ,500) 
   canvas.Divide(2,2)
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
   gStyle.SetPalette(1)
   gPad.SetLogz()
      
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
   canvas.Update()
   r.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   r.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
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


   
