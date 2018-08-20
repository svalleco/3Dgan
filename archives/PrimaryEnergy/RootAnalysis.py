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
import setGPU

def main():
   #Architectures 
   from EcalEnergyGan import generator, discriminator
   gen_weights1 = 'generator_1gpu_042.hdf5'# 1 gpu  
   gen_weights2 = 'generator_2gpu_023.hdf5'# 2 gpu                                                            
   gen_weights3 = '4gpu_gen.hdf5'          # 4 gpu                                                            
   gen_weights4 = 'generator_8gpu_005.hdf5'# 8 gpu                                                            
   gen_weights5 = 'generator_16gpu_002.hdf5'# 16 gpu          

   disc_weights1 = 'discriminator_1gpu_042.hdf5'# 1 gpu      
   disc_weights2 = 'discriminator_2gpu_023.hdf5'# 2 gpu                                                           
   disc_weights3 = '4gpu_disc.hdf5'             # 4 gpu                                                           
   disc_weights4 = 'discriminator_8gpu_005.hdf5'# 8 gpu                                                   
   disc_weights5 = 'discriminator_16gpu_002.hdf5'# 16 gpu   
   
   disc_weights="params_discriminator_epoch_041.hdf5"
   gen_weights= "params_generator_epoch_041.hdf5"

   plots_dir = "new_bestplots_ecal_log/"
   latent = 200
   num_data = 100000
   num_events = 2000
   m = 3
   energies=[0, 50, 100, 200, 250, 300, 400, 500]

   datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path                                         
   #datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5'
   sortdir = 'SortedData'
   gendir = 'Gen'  
   discdir = 'Disc' 
   Test = False
   save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
   read_data = False # True if loading previously sorted data  
   save_gen = False # True if saving generated data. 
   read_gen = False # True if generated data is already saved and can be loaded
   save_disc = False # True if discriminiator data is to be saved
   read_disc =  False # True if discriminated data is to be loaded from previously saved file
 
   flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   dweights = [disc_weights]
   gweights = [gen_weights]
   scales = [100]
   d = discriminator()
   g = generator(latent)
   var= perform_calculations(g, d, gweights, dweights, energies, datapath, sortdir, gendir, discdir, num_data, num_events, m, scales, flags, latent)
   get_plots(var, plots_dir, energies, m, len(gweights))

def BinLogX(h):
   axis = h.GetXaxis()
   bins = axis.GetNbins()
   From = axis.GetXmin()
   to = axis.GetXmax()
   width = (to - From) / bins
   new_bins = np.zeros(bins + 1)

   for i in np.arange(bins + 1):
     new_bins[i] = ROOT.TMath.Power(10, From + i * width)
   axis.Set(bins, new_bins)
   new_bins=None

def fill_hist(hist, array):
   [hist.Fill(_) for _ in array]
  
def fill_hist_wt(hist, weight):
   array= np.arange(0, 25, 1)
   for i in array:
     for j in np.arange(weight.shape[0]):
        hist.Fill(i, weight[j, i])

def get_hits(events, thresh):
   hit_array = events>thresh
   hits = np.sum(hit_array, axis=(1, 2, 3)) 
   return hits

def stat_pos(a, pos=0):
  if pos==0:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.1)
   sb1.SetX2NDC(.3)
  return sb1

def fill_profile(prof, x, y):
   for i in range(len(y)):
      prof.Fill(y[i], x[i])

def plot_ecal_ratio_profile(ecal1, ecal2, y, out_file):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 520)
   Gprof = ROOT.TProfile("Gprof", "Gprof", 100, 0, 520)
   c1.SetGrid()
   
   Eprof.SetStats(kFALSE)
   Gprof.SetStats(kFALSE)

   fill_profile(Eprof, ecal1/y, y)
   fill_profile(Gprof, ecal2/y, y)

   Eprof.SetTitle("Ratio of Ecal and Ep")
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("Ecal/Ep")
   Eprof.GetYaxis().SetRangeUser(0, 0.03)
   Eprof.Draw()
   Eprof.SetLineColor(4)
   Gprof.SetLineColor(2)
   Gprof.Draw('sames')
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"Data","l")
   legend.AddEntry(Gprof, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_ecal_relative_profile(ecal1, ecal2, y, out_file):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                 
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 520)
   Gprof = ROOT.TProfile("Gprof", "Gprof", 100, 0, 520)
   c1.SetGrid()

   Eprof.SetStats(kFALSE)
   Gprof.SetStats(kFALSE)
   size = len(y)
   fill_profile(Eprof, (y - 50 * ecal1)/y, y)
   fill_profile(Gprof, (y - 50 * ecal2)/y, y)

   Eprof.SetTitle("Ep - Ecal / Ep")
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("(Ep- Ecal)/Ep")
   Eprof.GetYaxis().SetRangeUser(-0.4, 0.3)                                                                                                                                                                            
   Eprof.Draw()
   Eprof.SetLineColor(4)
   Gprof.SetLineColor(2)
   Gprof.Draw('sames')
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"Data","l")
   legend.AddEntry(Gprof, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_aux_relative_profile(aux1, aux2, y, out_file):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                 
   Eprof = ROOT.TProfile("Eprof", "", 100, 0, 520)
   Gprof = ROOT.TProfile("Gprof", "Gprof", 100, 0, 520)
   c1.SetGrid()

   Eprof.SetStats(kFALSE)
   Gprof.SetStats(kFALSE)

   fill_profile(Eprof, (y - 100 *aux1)/y, y)
   fill_profile(Gprof, (y - 100 *aux2)/y, y)

   Eprof.SetTitle("Relative Error for Primary Energy")
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("Ep - aux/Ep")
   Eprof.GetYaxis().SetRangeUser(-0.2, 0.2)
   Eprof.Draw()
   Eprof.SetLineColor(4)
   Gprof.SetLineColor(2)
   Gprof.Draw('sames')
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"Data","l")
   legend.AddEntry(Gprof, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_ecal_hist(ecal1, ecal2, out_file, energy):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                 
   hd = ROOT.TH1F("DATA", "", 100, 0, 12)
   hg = ROOT.TH1F("GAN", "GAN", 100, 0, 12)
   c1.SetGrid()

   size = len(ecal2)
   fill_hist(hd, ecal1)
   fill_hist(hg, ecal2)
   if energy == 0:
      hd.SetTitle("Ecal Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Ecal Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")
   #Eprof.GetYaxis().SetTitle("Ecal/Ep")
   #Eprof.GetYaxis().SetRangeUser(0, 2)                                                                                                                                                                            
   hd.Draw()
   hd.SetLineColor(4)
   hd.SetLineColor(2)
   hg.Draw('sames')
   c1.Update()
   stat_pos(hg)
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(hd,"Data","l")
   legend.AddEntry(hg, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_ecal_flatten_hist(event1, event2, out_file, energy):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                
   hd = ROOT.TH1F("DATA", "", 100, -8, 0)
   hg = ROOT.TH1F("GAN", "GAN", 100, -8, 0)
   BinLogX(hd)
   BinLogX(hg)
   c1.SetGrid()
   gPad.SetLogy()
   gPad.SetLogx()
   #event1[event1 < 1e-6] = 0
   #event2[event2 < 1e-6] = 0
   fill_hist(hd, event1.flatten())
   fill_hist(hg, event2.flatten())
   if energy == 0:
      hd.SetTitle("Ecal Flat Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Ecal Flat Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")
   #Eprof.GetYaxis().SetTitle("Ecal/Ep")                                                                                                                                                                          
   #Eprof.GetYaxis().SetRangeUser(0, 2)                                                                                                                                           
   hd.Draw()
   hd.SetLineColor(4)
   hd.SetLineColor(2)
   hg.Draw('sames')
   c1.Update()
   stat_pos(hg)
   c1.Update()
   legend = TLegend(.5, .8, .6, .9)
   legend.AddEntry(hd,"Data","l")
   legend.AddEntry(hg, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_ecal_hits_hist(event1, event2, out_file, energy):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                
   hd = ROOT.TH1F("DATA", "", 50, 0, 3000)
   hg = ROOT.TH1F("GAN", "GAN", 50, 0, 3000)
   c1.SetGrid()
 
   thresh = 0.01 # GeV
   fill_hist(hd, get_hits(event1 * 50, thresh))
   fill_hist(hg, get_hits(event2 * 50, thresh))
   if energy == 0:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for Uniform Spectrum".format(thresh))
   else:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for {} GeV".format(thresh, energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")

   #Eprof.GetYaxis().SetRangeUser(0, 2)                                                                                                                                                                            
   hd.Draw()
   hd.SetLineColor(4)
   hd.SetLineColor(2)
   hg.Draw('sames')
   c1.Update()
   stat_pos(hg)
   c1.Update()
   legend = TLegend(.7, .1, .9, .2)
   legend.AddEntry(hd,"Data","l")
   legend.AddEntry(hg, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_primary_hist(aux1, aux2, out_file, energy):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                
   hd = ROOT.TH1F("DATA", "", 100, 0, 600)
   hg = ROOT.TH1F("GAN", "GAN", 100, 0, 600)
   c1.SetGrid()

   fill_hist(hd, aux1)
   fill_hist(hg, aux2)
   if energy == 0:
      hd.SetTitle("Auxilliary Energy Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Auxilliary Energy  Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Primary GeV")
   #Eprof.GetYaxis().SetTitle("Ecal/Ep")                                                                             
   #Eprof.GetYaxis().SetRangeUser(0, 2)                                                                           
   hd.Draw()
   hd.SetLineColor(4)
   hd.SetLineColor(2)
   hg.Draw('sames')
   c1.Update()
   stat_pos(hg)
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(hd,"Data","l")
   legend.AddEntry(hg, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_primary_error_hist(aux1, aux2, y, out_file, energy):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                
   hd = ROOT.TH1F("DATA", "", 100, -0.2, 0.2)
   hg = ROOT.TH1F("GAN", "GAN", 100, -0.2, 0.2)
   c1.SetGrid()

   fill_hist(hd, (y - aux1*100)/y)
   fill_hist(hg, (y- aux2*100)/y)
   if energy == 0:
      hd.SetTitle("Auxilliary Energy Error Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Auxilliary Energy  Error Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Primary GeV")
   #Eprof.GetYaxis().SetTitle("Ecal/Ep")                                                                             
   #Eprof.GetYaxis().SetRangeUser(0, 2)                                                                           
   hd.Draw()
   hd.SetLineColor(4)
   hd.SetLineColor(2)
   hg.Draw('sames')
   c1.Update()
   stat_pos(hg)
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(hd,"Data","l")
   legend.AddEntry(hg, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_realfake_hist(array1, array2, out_file, energy):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                
   hd = ROOT.TH1F("DATA", "", 100, 0, 1)
   hg = ROOT.TH1F("GAN", "GAN", 100, 0, 1)
   c1.SetGrid()

   fill_hist(hd, array1)
   fill_hist(hg, array2)
   if energy == 0:
      hd.SetTitle("Real/Fake Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Real/Fake Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Real Fake probability")

   hd.Draw()
   hd.SetLineColor(4)
   hd.SetLineColor(2)
   hg.Draw('sames')
   c1.Update()
   stat_pos(hg)
   c1.Update()
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(hd,"Data","l")
   legend.AddEntry(hg, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def plot_max(array1, array2, out_file1, out_file2, out_file3, energy, log=0):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   
   c1.Divide(2,2)
   h1x = ROOT.TH1F('Datax' + str(energy), '', 25, 0, 25)
   h1y = ROOT.TH1F('Datay' + str(energy), '', 25, 0, 25)
   h1z = ROOT.TH1F('Dataz' + str(energy), '', 25, 0, 25)
   h1x.SetLineColor(9)
   h1y.SetLineColor(9)
   h1z.SetLineColor(9)
   c1.cd(1)
   if log:
      gPad.SetLogy()
   fill_hist(h1x, array1[:,0])
   h1x.Draw()
   c1.cd(2)
   if log:
      gPad.SetLogy()
   fill_hist(h1y, array1[:,1])
   h1y.Draw()
   c1.cd(3)
   if log:
      gPad.SetLogy()
   fill_hist(h1z, array1[:,2])
   h1z.Draw()
   c1.cd(4)
   c1.Update()
   c1.Print(out_file1)

   h2x = ROOT.TH1F('GANx' + str(energy), '', 25, 0, 25)
   h2y = ROOT.TH1F('GANy' + str(energy), '', 25, 0, 25)
   h2z = ROOT.TH1F('GANz' + str(energy), '', 25, 0, 25)
   h2x.SetLineColor(46)
   h2y.SetLineColor(46)  
   h2z.SetLineColor(46)
   
   c1.cd(1)
   fill_hist(h2x, array2[:,0])
   h2x.Draw()
   c1.Update()
   stat_pos(h2x)
   c1.cd(2)
   fill_hist(h2y, array2[:,1])
   h2y.Draw()
   c1.Update()
   stat_pos(h2y)
   c1.cd(3)
   fill_hist(h2z, array2[:,2])
   h2z.Draw()
   c1.Update()
   stat_pos(h2z)
   c1.Update()
   c1.Print(out_file2)
   
   c1.cd(1)
   h1x.Draw('sames')
   c1.cd(2)
   h1y.Draw('sames')
   c1.cd(3)
   h1z.Draw('sames')
   c1.Update()
   c1.cd(4)
   leg = ROOT.TLegend(0.1,0.7,0.48,0.9)
   leg.AddEntry(h1x,"Data","l")
   leg.AddEntry(h2x,"GAN","l")
   leg.Draw()
   c1.Update()
   c1.Print(out_file3)

def plot_energy_hist_root_all(array1z, array2z, array3z, array4z, arrayg1z, arrayg2z, arrayg3z, arrayg4z,  energy1, energy2, energy3, energy4, out_file, log=0):
   canvas = TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   bins = np.arange(0, 25, 1)
   canvas.Divide(2,2)
   h1z = ROOT.TH1F('Dataz' + str(energy1), '', 25, 0, 25)
   h2z = ROOT.TH1F('Dataz' + str(energy2), '', 25, 0, 25)
   h3z = ROOT.TH1F('Dataz' + str(energy3), '', 25, 0, 25)
   h4z = ROOT.TH1F('Dataz' + str(energy4), '', 25, 0, 25)
   
   hg1z = ROOT.TH1F('Datagz' + str(energy1), '', 25, 0, 25)
   hg2z = ROOT.TH1F('Datagz' + str(energy2), '', 25, 0, 25)
   hg3z = ROOT.TH1F('Datagz' + str(energy3), '', 25, 0, 25)
   hg4z = ROOT.TH1F('Datagz' + str(energy4), '', 25, 0, 25)

   h1z.Sumw2()
   h2z.Sumw2()
   h3z.Sumw2()
   h4z.Sumw2()

   hg1z.Sumw2()
   hg2z.Sumw2()
   hg3z.Sumw2()
   hg4z.Sumw2()

   h1z.SetLineColor(9)
   h2z.SetLineColor(9)
   h3z.SetLineColor(9)
   h4z.SetLineColor(9)
   
   hg1z.SetLineColor(46)
   hg2z.SetLineColor(46)
   hg3z.SetLineColor(46)
   hg4z.SetLineColor(46)

   h1z.SetStats(kFALSE)
   h2z.SetStats(kFALSE)
   h3z.SetStats(kFALSE)
   h4z.SetStats(kFALSE)

   hg1z.SetStats(kFALSE)
   hg2z.SetStats(kFALSE)
   hg3z.SetStats(kFALSE)
   hg4z.SetStats(kFALSE)

   h3z.GetXaxis().SetTitle("Position along z axis")
   h3z.GetXaxis().SetTitleSize(0.045)
   h4z.GetXaxis().SetTitle("Position along z axis")
   h4z.GetXaxis().SetTitleSize(0.045)
   h1z.GetYaxis().SetTitle("Ecal Energy")
   h1z.GetYaxis().SetTitleSize(0.045)
   h3z.GetYaxis().SetTitle("Ecal Energy")
   h3z.GetYaxis().SetTitleSize(0.041)
   
   canvas.cd(1)
   fill_hist_wt(h1z, array1z)
   h1z.Draw()
   canvas.cd(2)
   fill_hist_wt(h2z, array2z)
   h2z.Draw()
   canvas.cd(3)                                                                                                                                                                                                   
   fill_hist_wt(h3z, array3z)
   h3z.Draw()
   canvas.cd(4)
   fill_hist_wt(h4z, array4z)
   h4z.Draw()

   canvas.Update()
   canvas.cd(1)
   fill_hist_wt(hg1z, arrayg1z)
   hg1z.Draw('sames')
   leg1 = ROOT.TLegend(0.1,0.9,0.3,0.77)
   leg1.AddEntry(h1z,"Data " + str(energy1) + " GeV","l")
   leg1.AddEntry(hg1z,"GAN "+ str(energy1) + " GeV","l")
   leg1.Draw()
   canvas.Update()

   canvas.cd(2)
   fill_hist_wt(hg2z, arrayg2z)
   hg2z.Draw('sames')
   leg2 = ROOT.TLegend(0.1,0.9,0.3,0.77)
   leg2.AddEntry(h2z,"Data " + str(energy2) + " GeV","l")
   leg2.AddEntry(hg2z,"GAN "+ str(energy2) + " GeV","l")
   leg2.Draw()
   canvas.Update()

   canvas.cd(3)
   fill_hist_wt(hg3z, arrayg3z)
   hg3z.Draw('sames')
   leg3 = ROOT.TLegend(0.1,0.9,0.3,0.77)
   leg3.AddEntry(h3z,"Data " + str(energy3) + " GeV","l")
   leg3.AddEntry(hg3z,"GAN "+ str(energy3) + " GeV","l")
   leg3.Draw()
   canvas.Update()

   canvas.cd(4)
   fill_hist_wt(hg4z, arrayg4z)
   hg4z.Draw('sames')

   canvas.Update()

   leg4 = ROOT.TLegend(0.1,0.9,0.3,0.77)
   leg4.AddEntry(h4z,"Data " + str(energy4) + " GeV","l")
   leg4.AddEntry(hg4z,"GAN "+ str(energy4) + " GeV","l")
   leg4.Draw()
   canvas.Update()
   canvas.Print(out_file)

def plot_energy_hist_root2(array1x, array1y, array1z, array2x, array2y, array2z, out_file1, out_file2, out_file3, energy, log=0):
   canvas = TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   bins = np.arange(0, 25, 1)
   canvas.Divide(2,2)
   h1x = ROOT.TH1F('Datax' + str(energy), '', 25, 0, 25)
   h1y = ROOT.TH1F('Datay' + str(energy), '', 25, 0, 25)
   h1z = ROOT.TH1F('Dataz' + str(energy), '', 25, 0, 25)
   h1x.Sumw2()
   h1y.Sumw2()
   h1z.Sumw2()
   h1x.SetLineColor(9)
   h1y.SetLineColor(9)
   h1z.SetLineColor(9)
   h1x.SetStats(kFALSE)
   h1y.SetStats(kFALSE)
   h1z.SetStats(kFALSE)
   h1x.GetXaxis().SetTitle("Position along x axis")
   h1z.GetXaxis().SetTitle("Position along z axis")
   canvas.cd(1)
   gPad.SetLogy()
   fill_hist_wt(h1x, array1x)
   h1x.Draw()
   canvas.cd(2)
   if log:
      gPad.SetLogy()
   fill_hist_wt(h1z, array1z)
   h1z.Draw()
   canvas.Update()
   h2x = ROOT.TH1F('GANx' + str(energy), '', 25, 0, 25)
   h2y = ROOT.TH1F('GANy' + str(energy), '', 25, 0, 25)
   h2z = ROOT.TH1F('GANz' + str(energy), '', 25, 0, 25)
   h2x.SetLineColor(46)
   h2y.SetLineColor(46)
   h2z.SetLineColor(46)
   h2x.SetStats(kFALSE)
   h2y.SetStats(kFALSE)
   h2z.SetStats(kFALSE)

   canvas.cd(1)
   fill_hist_wt(h2x, array2x)
   h2x.Draw('sames')
   canvas.Update()
   leg1 = ROOT.TLegend(0.1,0.9,0.3,0.8)
   leg1.AddEntry(h1x,"Data " + str(energy),"l")
   leg1.AddEntry(h2x,"GAN " + str(energy),"l")
   leg1.Draw()

   canvas.cd(2)
   fill_hist_wt(h2z, array2z)
   h2z.Draw('sames')
   canvas.Update()

   leg = ROOT.TLegend(0.1,0.9,0.3,0.8)
   leg.AddEntry(h1z,"Data " + str(energy),"l")
   leg.AddEntry(h2z,"GAN " + str(energy),"l")
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file3)

def plot_energy_hist_root(array1x, array1y, array1z, array2x, array2y, array2z, out_file1, out_file2, out_file3, histfile, i,energy, log=0):
   canvas = TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   bins = np.arange(0, 25, 1)
   canvas.Divide(2,2)
   h1x = ROOT.TH1F('Datax' + str(energy), '', 25, 0, 25)
   h1y = ROOT.TH1F('Datay' + str(energy), '', 25, 0, 25)
   h1z = ROOT.TH1F('Dataz' + str(energy), '', 25, 0, 25)
   h1x.Sumw2()
   h1y.Sumw2()
   h1z.Sumw2()
   h1x.SetLineColor(9)
   h1y.SetLineColor(9)
   h1z.SetLineColor(9)
   
   canvas.cd(1)
   if log:
      gPad.SetLogy()
   fill_hist_wt(h1x, array1x)
   h1x.Draw()
   canvas.cd(2)
   if log:
      gPad.SetLogy()
   fill_hist_wt(h1y, array1y)
   h1y.Draw()
   canvas.cd(3)
   if log:
      gPad.SetLogy()
   fill_hist_wt(h1z, array1z)
   h1z.Draw()
   canvas.cd(4)
   canvas.Update()
   canvas.Print(out_file1)

   h2x = ROOT.TH1F('GANx' + str(energy), '', 25, 0, 25)
   h2y = ROOT.TH1F('GANy' + str(energy), '', 25, 0, 25)
   h2z = ROOT.TH1F('GANz' + str(energy), '', 25, 0, 25)
   h2x.SetLineColor(46)
   h2y.SetLineColor(46)
   h2z.SetLineColor(46)
   canvas.cd(1)
   fill_hist_wt(h2x, array2x)
   h2x.Draw()
   canvas.Update()
   stat_pos(h2x)
   canvas.cd(2)
   fill_hist_wt(h2y, array2y)
   h2y.Draw()
   canvas.Update()
   stat_pos(h2y)
   canvas.cd(3)
   fill_hist_wt(h2z, array2z)
   h2z.Draw()
   canvas.Update()
   stat_pos(h2z)
   canvas.Update()
   canvas.Print(out_file2)

   canvas.cd(1)
   h1x.Draw('sames')
   canvas.cd(2)
   h1y.Draw('sames')
   canvas.cd(3)
   h1z.Draw('sames')
   canvas.Update()
   canvas.cd(4)
   leg = ROOT.TLegend(0.1,0.8,0.4, 1.0)
   leg.AddEntry(h1x,"Data","l")
   leg.AddEntry(h2x,"GAN" ,"l")
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file3)
   if i == 0:
     h1x.SetStats(kFALSE)
     h1y.SetStats(kFALSE)
     h1z.SetStats(kFALSE)
     h2x.SetStats(kFALSE)
     h2y.SetStats(kFALSE)
     h2z.SetStats(kFALSE)
     canvas.Print(histfile + '.root')
   else:
     f = ROOT.TFile(histfile + '.root', 'read')
     c = f.Get("canvas")
     c.Draw()
     color= h2x.GetLineColor()
     color = color + i * 5
     gPad.SetLineColor(color)
     h2x.SetLineColor(color)
     h2y.SetLineColor(color)
     h2z.SetLineColor(color)
     h2x.SetStats(kFALSE)
     h2y.SetStats(kFALSE)
     h2z.SetStats(kFALSE)

     c.cd(1)
     h2x.Draw('sames')
     c.cd(2)
     h2y.Draw('sames')
     c.cd(3)
     h2z.Draw('sames')
     c.cd(4)  
     leg = ROOT.TLegend(0.1,0.8 - i * 0.1,0.4, 0.9 - i * 0.1)
     leg.AddEntry(h2x,"GAN"+ str(i) ,"l")
     leg.Draw()
     c.Update()
     c.Print(histfile + '.root')
     c.Print(histfile + '.pdf')

def plot_moment(array1, array2, out_file, dim, energy, moment):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                
   if moment==0:
     bins = 25
     maxbin = 25
     minbin = 0
   else:
     bins = 50
     maxbin = max(np.amax(array1), np.amax(array2))
     minbin = min(np.amin(array1), np.amin(array2))
   hd = ROOT.TH1F("DATA", "", bins, minbin, maxbin)
   hg = ROOT.TH1F("GAN", "GAN", bins, minbin, maxbin)
   c1.SetGrid()

   fill_hist(hd, array1)
   fill_hist(hg, array2)
   if energy == 0:
      hd.SetTitle("Histogram for {} {} Moment Uniform Spectrum".format(dim, moment + 1))
   else:
      hd.SetTitle("Histogram for {} {} Moment for {} GeV".format(dim, moment + 1, energy))
   hd.GetXaxis().SetTitle("Position along {} axis".format(dim))
   #Eprof.GetYaxis().SetRangeUser(0, 2)                                                                           
   hd.Draw()
   hd.SetLineColor(4)
   hd.SetLineColor(2)
   hg.Draw('sames')
   c1.Update()
   stat_pos(hg)
   c1.Update()
   legend = TLegend(.7, .1, .9, .2)
   legend.AddEntry(hd,"Data","l")
   legend.AddEntry(hg, "GAN", "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   c1.Print(out_file)

def DivideFiles(FileSearch="/data/LCD/*/*.h5", nEvents=400000, EventsperFile = 10000, Fractions=[.5,.5],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))  
    print ("Found {} files. ".format(len(Files)))
    Filesused = int(math.ceil(nEvents/EventsperFile))
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName][:Filesused]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out

def get_data(datafile):
    #get data for training                                                                                                                                                                      
    print 'Loading Data from .....', datafile
    f=h5py.File(datafile,'r')
    y=f.get('target')
    x=np.array(f.get('ECAL'))
    y=(np.array(y[:,1]))
    x[x < 1e-6] = 0
    x = np.expand_dims(x, axis=4)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

def sort(data, energies, flag=False, num_events=2000):
    X = data[0]
    Y = data[1]
    tolerance = 5
    srt = {}
    for energy in energies:
       if energy == 0 and flag:
          srt["events_act" + str(energy)] = X[:2000]
          srt["energy" + str(energy)] = Y[:2000]
          print srt["events_act" + str(energy)].shape
       else:
          indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
          if len(indexes) > num_events:
             indexes = indexes[:num_events]
          srt["events_act" + str(energy)] = X[indexes]
          srt["energy" + str(energy)] = Y[indexes]
    return srt

def save_sorted(srt, energies, srtdir):
    safe_mkdir(srtdir)
    for energy in energies:
       srtfile = os.path.join(srtdir, "events_{:03d}.h5".format(energy))
       with h5py.File(srtfile ,'w') as outfile:
          outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
          outfile.create_dataset('Target',data=srt["energy" + str(energy)])
       print "Sorted data saved to {}".format(srtfile)

def save_generated(events, sampled_energies, energy, gendir):
    safe_mkdir(gendir)
    filename = os.path.join(gendir,"Gen_{:03d}.hdf5".format(energy))
    with h5py.File(filename ,'w') as outfile:
       outfile.create_dataset('ECAL',data=events)
       outfile.create_dataset('Target',data=sampled_energies)
    print "Generated data saved to ", filename

def save_discriminated(disc, energies, discdir):
    safe_mkdir(discdir)
    for energy in energies:
      filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
      with h5py.File(filename ,'w') as outfile:
        outfile.create_dataset('ISREAL_ACT',data=disc["isreal_act" + str(energy)])
        outfile.create_dataset('ISREAL_GAN',data=disc["isreal_gan" + str(energy)])
        outfile.create_dataset('AUX_ACT',data=disc["aux_act" + str(energy)])
        outfile.create_dataset('AUX_GAN',data=disc["aux_gan" + str(energy)])
        outfile.create_dataset('ECAL_ACT',data=disc["ecal_act" + str(energy)])
        outfile.create_dataset('ECAL_GAN',data=disc["ecal_gan" + str(energy)])
      print "Discriminated data saved to ", filename

def get_disc(energy, discdir):
    filename = os.path.join(discdir, "Disc_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    isreal_act = np.array(f.get('ISREAL_ACT'))
    isreal_gan = np.array(f.get('ISREAL_GAN'))
    aux_act = np.array(f.get('AUX_ACT'))
    aux_gan = np.array(f.get('AUX_GAN'))
    ecal_act = np.array(f.get('ECAL_ACT'))
    ecal_gan = np.array(f.get('ECAL_GAN'))
    print "Discriminated file ", filename, " is loaded"
    return isreal_act, aux_act, ecal_act, isreal_gan, aux_gan, ecal_gan

def load_sorted(sorted_path):
    sorted_files = sorted(glob.glob(sorted_path))
    energies = []
    srt = {}
    for f in sorted_files:
       energy = int(filter(str.isdigit, f)[:-1])
       energies.append(energy)
       srtfile = h5py.File(f,'r')
       srt["events_act" + str(energy)] = np.array(srtfile.get('ECAL'))
       srt["energy" + str(energy)] = np.array(srtfile.get('Target'))
       print "Loaded from file", f
    return energies, srt
 
def get_gen(energy, gendir):
    filename = os.path.join(gendir, "Gen_{:03d}.hdf5".format(energy))
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    print "Generated file ", filename, " is loaded"
    return generated_images

def generate(g, index, sampled_labels, latent=200):
    noise = np.random.normal(0, 1, (index, latent))
    sampled_labels=np.expand_dims(sampled_labels, axis=1)
    gen_in = sampled_labels * noise
    generated_images = g.predict(gen_in, verbose=False, batch_size=50)
    return generated_images

def discriminate(d, images):
    isreal, aux_out, ecal_out = np.array(d.predict(images, verbose=False, batch_size=50))
    return isreal, aux_out, ecal_out

def get_max(images):
    index = images.shape[0]
    max_pos = np.zeros((index, 3)) 
    for i in range(index):
       max_p = images[i].argmax()
       max_loc = np.unravel_index(max_p, (25, 25, 25))
       max_pos[i] = max_loc
    return max_pos

def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

def get_moments(images, sumsx, sumsy, sumsz, totalE, m):
    ecal_size = 25
    totalE = np.squeeze(totalE)
    index = images.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      print(totalE.shape)
      ECAL_momentX = umath.inner1d(sumsx, moments) /totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = umath.inner1d(sumsy, moments) /totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = umath.inner1d(sumsz, moments)/totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

def safe_mkdir(path):
   #Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception

def perform_calculations(g, d, gweights, dweights, energies, datapath, sortdir, gendirs, discdirs, num_data, num_events, m, scales, flags, latent, events_per_file=10000):
    sortedpath = os.path.join(sortdir, 'events_*.h5')
    Test = flags[0]
    save_data = flags[1]  
    read_data = flags[2]                                                                                              
    save_gen = flags[3]                                                                                               
    read_gen = flags[4]                                                                                               
    save_disc = flags[5]                                                                                              
    read_disc =  flags[6]
    var={}     
#    for gen_weights, disc_weights, i in zip(gweights, dweights, np.arange(len(gweights))):
    if read_data: # Read from sorted dir
       start = time.time()
       energies, var = load_sorted(sortedpath)
       sort_time = time.time()- start
       print "Events were loaded in {} seconds".format(sort_time)
    else:
       # Getting Data
       events_per_file = 10000
       Filesused = int(math.ceil(num_data/events_per_file))
       Trainfiles, Testfiles = DivideFiles(datapath, datasetnames=["ECAL"], Particles =["Ele"])
       Trainfiles = Trainfiles[: Filesused]
       Testfiles = Testfiles[: Filesused]
       print Trainfiles
       print Testfiles
       if Test:
          data_files = Testfiles
       else:
          data_files = Trainfiles 
       start = time.time()
       for index, dfile in enumerate(data_files):
          data = get_data(dfile)
          if index==0:
             var = sort(data, energies, True, num_events)
          else:
             sorted_data = sort(data, energies, False, num_events)
             for key in var:
                var[key]= np.append(var[key], sorted_data[key], axis=0)
       data = None
       data_time = time.time() - start
       print "{} events were loaded in {} seconds".format(num_data, data_time)
       if save_data:
          save_sorted(var, energies, sortdir)        
    total = 0
    for energy in energies:
    #calculations for data events
      var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
      total += var["index" + str(energy)]
      var["max_pos_act" + str(energy)] = get_max(var["events_act" + str(energy)])
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      var["ecal_act"+ str(energy)]= np.sum(var["events_act" + str(energy)], axis=(1, 2, 3))
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["events_act" + str(energy)], var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m)
      
    data_time = time.time() - start
    print "{} events were put in {} bins".format(total, len(energies))
    #### Generate Data table to screen                                                                                                                                                                          
    print "Actual Data"
    print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2"
    for energy in energies:
       print "{}\t{}\t{:.4f}\t\t{}\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}" .format(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), np.mean(var["max_pos_act" + str(energy)], axis=0), np.mean(var["events_act" + str(energy)]), np.mean(var["momentX_act"+ str(energy)][:, 1]), np.mean(var["momentY_act"+ str(energy)][:, 1]), np.mean(var["momentZ_act"+ str(energy)][:, 1]))

    for gen_weights, disc_weights, scale, i in zip(gweights, dweights, scales, np.arange(len(gweights))):    
       var['n_'+ str(i)]={}
       gendir = gendirs + '/n_' + str(i)
       discdir = discdirs + '/n_' + str(i)
       if read_gen:
         for energy in energies:
           var['n_'+ str(i)]["events_gan" + str(energy)]= get_gen(energy, gendir)
       else:
         g.load_weights(gen_weights)
         start = time.time()
         for energy in energies:
            var['n_'+ str(i)]["events_gan" + str(energy)] = generate(g, var["index" + str(energy)], var["energy" + str(energy)]/100, latent)
            if save_gen:
              save_generated(var['n_'+ str(i)]["events_gan" + str(energy)], var["energy" + str(energy)], energy, gendir)
         gen_time = time.time() - start
         print "Generator took {} seconds to generate {} events".format(gen_time, total)
       if read_disc:
         for energy in energies:
           var['n_'+ str(i)]["isreal_act" + str(energy)], var['n_'+ str(i)]["aux_act" + str(energy)], var['n_'+ str(i)]["ecal_act"+ str(energy)], var['n_'+ str(i)]["isreal_gan" + str(energy)], var['n_'+ str(i)]["aux_gan" + str(energy)], var['n_'+ str(i)]["ecal_gan"+ str(energy)]= get_disc(energy, discdir)
       else: 
         d.load_weights(disc_weights)
         start = time.time()
         for energy in energies:
           var['n_'+ str(i)]["isreal_act" + str(energy)], var['n_'+ str(i)]["aux_act" + str(energy)], var['n_'+ str(i)]["ecal_act"+ str(energy)]= discriminate(d, var["events_act" + str(energy)] * scale)
           var['n_'+ str(i)]["isreal_gan" + str(energy)], var['n_'+ str(i)]["aux_gan" + str(energy)], var['n_'+ str(i)]["ecal_gan"+ str(energy)]= discriminate(d, var['n_'+ str(i)]["events_gan" + str(energy)] )
         disc_time = time.time() - start
         print "Discriminator took {} seconds for {} data and generated events".format(disc_time, total)
      
         if save_disc:
           save_discriminated(var['n_'+ str(i)], energies, discdir)
    
       for energy in energies:
         print 'Calculations for ....', energy
         var['n_'+ str(i)]["events_gan" + str(energy)] = var['n_'+ str(i)]["events_gan" + str(energy)]/scale
         var['n_'+ str(i)]["isreal_act" + str(energy)], var['n_'+ str(i)]["aux_act" + str(energy)], var['n_'+ str(i)]["ecal_act"+ str(energy)]= np.squeeze(var['n_'+ str(i)]["isreal_act" + str(energy)]), np.squeeze(var['n_'+ str(i)]["aux_act" + str(energy)]), np.squeeze(var['n_'+ str(i)]["ecal_act"+ str(energy)]/scale)
         var['n_'+ str(i)]["isreal_gan" + str(energy)], var['n_'+ str(i)]["aux_gan" + str(energy)], var['n_'+ str(i)]["ecal_gan"+ str(energy)]= np.squeeze(var['n_'+ str(i)]["isreal_gan" + str(energy)]), np.squeeze(var['n_'+ str(i)]["aux_gan" + str(energy)]), np.squeeze(var['n_'+ str(i)]["ecal_gan"+ str(energy)]/scale)
         var['n_'+ str(i)]["max_pos_gan" + str(energy)] = get_max(var['n_'+ str(i)]["events_gan" + str(energy)])
         var['n_'+ str(i)]["sumsx_gan"+ str(energy)], var['n_'+ str(i)]["sumsy_gan"+ str(energy)], var['n_'+ str(i)]["sumsz_gan"+ str(energy)] = get_sums(var['n_'+ str(i)]["events_gan" + str(energy)])
         var['n_'+ str(i)]["momentX_gan" + str(energy)], var['n_'+ str(i)]["momentY_gan" + str(energy)], var['n_'+ str(i)]["momentZ_gan" + str(energy)] = get_moments(var['n_'+ str(i)]["events_gan" + str(energy)], var['n_'+ str(i)]["sumsx_gan"+ str(energy)], var['n_'+ str(i)]["sumsy_gan"+ str(energy)], var['n_'+ str(i)]["sumsz_gan"+ str(energy)], var['n_'+ str(i)]["ecal_gan"+ str(energy)], m)
   
       print('For {} iteration:\nWith Generator weights.....{}\nWith Discriminator weights.....{}'.format(i, gen_weights, disc_weights))
   
       #### Generate GAN table to screen                                                                                  
       print "Generated Data"
       print "Energy\tEvents\tMaximum Value\t\t\tMaximum loc\t\t\tMean\t\tMomentx2\tMomenty2\tMomentz2"
       for energy in energies:
         print "{}\t{}\t{:.4f}\t\t{}\t{:.2f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(energy, var["index" +str(energy)], np.amax(var['n_'+ str(i)]["events_gan" + str(energy)]), np.mean(var['n_'+ str(i)]["max_pos_gan" + str(energy)], axis=0), np.mean(var['n_'+ str(i)]["events_gan" + str(energy)]), np.mean(var['n_'+ str(i)]["momentX_gan"+ str(energy)][:, 1]), np.mean(var['n_'+ str(i)]["momentY_gan"+ str(energy)][:, 1]), np.mean(var['n_'+ str(i)]["momentZ_gan"+ str(energy)][:, 1]))
    return var

def get_plots(var, plots_dir, energies, m, n):
    
    actdir = plots_dir + 'Actual'
    safe_mkdir(actdir)
    alldir = plots_dir + 'All'
    safe_mkdir(alldir)

    for i in np.arange(n):
      ## Make folders for plots                                                                                          
      discdir = plots_dir + 'disc_outputs'+ 'plots_' + str(i) + '/'
      safe_mkdir(discdir)
      gendir = plots_dir + 'Generated' + 'plots_' + str(i) + '/'
      safe_mkdir(gendir)
      comdir = plots_dir + 'Combined' + 'plots_' + str(i) + '/'
      safe_mkdir(comdir)
      mdir = plots_dir + 'Moments' + 'plots_' + str(i) + '/'
      safe_mkdir(mdir)
      start = time.time()
   
      for energy in energies:
         maxfile = "Position_of_max_" + str(energy) + ".pdf"
         maxlfile = "Position_of_max_" + str(energy) + "_log.pdf"
         histfile = "hist_" + str(energy) + ".pdf"
         histlfile = "hist_log" + str(energy) + ".pdf"
         ecalfile = "ecal_" + str(energy) + ".pdf"
         energyfile = "energy_" + str(energy) + ".pdf"
         realfile = "realfake_" + str(energy) + ".pdf"
         momentfile = "moment" + str(energy) + ".pdf"
         ecalerrorfile = "ecal_error" + str(energy) + ".pdf"
         allfile = 'All_energies.pdf'
         allecalfile = 'All_ecal.pdf'
         allecalrelativefile = 'All_ecal_relative.pdf'
         allerrorfile = 'All_relative_auxerror.pdf'
         start = time.time()
       
         plot_max(var["max_pos_act" + str(energy)], var['n_' + str(i)]["max_pos_gan" + str(energy)], os.path.join(actdir, maxfile), os.path.join(gendir, maxfile), os.path.join(comdir, maxfile), energy)
         plot_max(var["max_pos_act" + str(energy)], var['n_' + str(i)]["max_pos_gan" + str(energy)], os.path.join(actdir, maxlfile), os.path.join(gendir, maxlfile), os.path.join(comdir, maxlfile), energy, log=1)
         plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var['n_' + str(i)]["sumsx_gan"+ str(energy)], var['n_' + str(i)]["sumsy_gan"+ str(energy)], var['n_' + str(i)]["sumsz_gan"+ str(energy)], os.path.join(actdir, histfile), os.path.join(gendir, histfile), os.path.join(comdir, histfile),os.path.join(alldir, "hist_" + str(energy)), i, energy)
         plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var['n_' + str(i)]["sumsx_gan"+ str(energy)], var['n_' + str(i)]["sumsy_gan"+ str(energy)], var['n_' + str(i)]["sumsz_gan"+ str(energy)], os.path.join(actdir, histlfile), os.path.join(gendir, histlfile), os.path.join(comdir, histlfile), os.path.join(alldir, "histl_" + str(energy) ), i, energy, log=1)
         plot_ecal_hist(var["ecal_act" + str(energy)], var['n_' + str(i)]["ecal_gan" + str(energy)], os.path.join(discdir, ecalfile), energy)
         plot_ecal_flatten_hist(var["events_act" + str(energy)], var['n_' + str(i)]["events_gan" + str(energy)], os.path.join(comdir, 'flat' + ecalfile), energy)
         plot_ecal_hits_hist(var["events_act" + str(energy)], var['n_' + str(i)]["events_gan" + str(energy)], os.path.join(comdir, 'hits' + ecalfile), energy)
         plot_primary_hist(var['n_' + str(i)]["aux_act" + str(energy)] * 100, var['n_' + str(i)]["aux_gan" + str(energy)] * 100, os.path.join(discdir, energyfile), energy)
         plot_realfake_hist(var['n_' + str(i)]["isreal_act" + str(energy)], var['n_' + str(i)]["isreal_gan" + str(energy)], os.path.join(discdir, realfile), energy)
         plot_primary_error_hist(var['n_' + str(i)]["aux_act" + str(energy)], var['n_' + str(i)]["aux_gan" + str(energy)], var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile), energy)
         for mmt in range(m):                                                                                            
            plot_moment(var["momentX_act" + str(energy)][:, mmt], var['n_' + str(i)]["momentX_gan" + str(energy)][:, mmt], os.path.join(mdir, 'x' + str(mmt + 1) + momentfile), 'x', energy, mmt)           
            plot_moment(var["momentY_act" + str(energy)][:, mmt], var['n_' + str(i)]["momentY_gan" + str(energy)][:, mmt], os.path.join(mdir, 'y' + str(mmt + 1) + momentfile), 'y', energy, mmt)           
            plot_moment(var["momentZ_act" + str(energy)][:, mmt], var['n_' + str(i)]["momentZ_gan" + str(energy)][:, mmt], os.path.join(mdir, 'z' + str(mmt + 1) + momentfile), 'z', energy, mmt)           
      plot_energy_hist_root_all(var["sumsz_act50"], var["sumsz_act100"], var["sumsz_act400"], var["sumsz_act500"], var['n_' + str(i)]["sumsz_gan50"], var['n_' + str(i)]["sumsz_gan100"], var['n_' + str(i)]["sumsz_gan400"], var['n_' + str(i)]["sumsz_gan500"], 50, 100, 400, 500, os.path.join(comdir, allfile))
      plot_ecal_ratio_profile(var['n_' + str(i)]["ecal_act0"], var['n_' + str(i)]["ecal_gan0"], var["energy0"], os.path.join(comdir, allecalfile))
      plot_ecal_relative_profile(var['n_' + str(i)]["ecal_act0"], var['n_' + str(i)]["ecal_gan0"], var["energy0"], os.path.join(comdir, allecalrelativefile))
      plot_aux_relative_profile(var['n_' + str(i)]["aux_act0"], var['n_' + str(i)]["aux_gan0"], var["energy0"], os.path.join(comdir, allerrorfile))
      print 'Plots are saved in ', plots_dir
      plot_time= time.time()- start
      print 'Plots are generated in {} seconds'.format(plot_time)

if __name__ == "__main__":
    main()
