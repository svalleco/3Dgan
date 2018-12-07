from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os, sys
import h5py
import numpy as np
import math
import time
import glob
import numpy.core.umath_tests as umath
from utils.GANutils import perform_calculations_multi, safe_mkdir #Common functions from GANutils.py
import utils.ROOTutils as my
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
 
import ptvsd
 
def main():
  #  ptvsd.enable_attach(address=('128.141.213.77', 3000), redirect_output=True)
  #  ptvsd.wait_for_attach()

   #Architectures 
   sys.path.insert(1, os.path.join(sys.path[0], '..'))
   from EcalEnergyGan import generator, discriminator
   disc_weights="good_weights/params_discriminator_epoch_041.hdf5"
   gen_weights= "good_weights/params_generator_epoch_041.hdf5"

   import keras.backend as K
   K.set_image_dim_ordering('tf')

   plots_dir = "correlation_plots/"
   latent = 200
   num_data = 100000
   num_events = 2000
   m = 3
   energies=[0, 50, 100, 200, 250, 300, 400, 500]
   particle='Ele'
   #datapath = '/bigdata/shared/LCD/NewV1/*scan/*.h5' #Training data path caltech
   datapath = '/mnt/nfshead/ahhesam/data/*.h5' # Training data CERN EOS
   sortdir = 'SortedData'
   gendir = 'Gen'  
   discdir = 'Disc' 
   Test = True
   stest = False 
   save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
   read_data = False # True if loading previously sorted data  
   save_gen =  False # True if saving generated data. 
   read_gen = False # True if generated data is already saved and can be loaded
   save_disc = False # True if discriminiator data is to be saved
   read_disc =  False # True if discriminated data is to be loaded from previously saved file
   ifpdf = True # True if pdf are required. If false .C files will be generated
 
   flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   # Lists for different versions comparison. The weights, scales and labels will need to be provided for each version
   dweights = [disc_weights]
   gweights = [gen_weights]
   scales = [100]
   labels = ['']
   d = discriminator()
   g = generator(latent)
   var= perform_calculations_multi(g, d, gweights, dweights, energies, datapath, sortdir, gendir, discdir, num_data, num_events, m, scales, flags, latent, particle)
   get_plots_multi(var, labels, plots_dir, energies, m, len(gweights), ifpdf, stest)

# computes correlation of a set of features and returns Fisher's Transform and names of features
def get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio):
   array = np.hstack((sumx, sumy, sumz))
   names = ['']
   for i in range(sumx.shape[1]):
      names = names + ['sumx'  + str(i)]
   for i in range(sumy.shape[1]):
      names = names + ['sumy'  + str(i)]
   for i in range(sumz.shape[1]):
      names = names + ['sumz'  + str(i)]
   m = momentx.shape[1]
   for i in range(m):
      array = np.hstack((array, momentx[:, i].reshape(-1, 1), momenty[:, i].reshape(-1, 1), momentz[:, i].reshape(-1, 1)))
      names = names + ['momentx' + str(i), 'momenty' + str(i), 'momentz' + str(i)]
   array = np.hstack((array, ecal.reshape(-1, 1), energy.reshape(-1, 1)))
   array = np.hstack((array, hits.reshape(-1, 1), ratio.reshape(-1, 1)))
   names = names + ['ecal sum', 'p energy', 'hits', 'ratio1_total']
   cor= np.corrcoef(array, rowvar=False)
   fisher= np.tanh(cor)
   return np.flip(fisher, 0), names
   
#Fills a 2D TGraph object
def fill_graph2D(graph, array):
   x = array.shape[0]
   y = array.shape[1]
   N = 0
   for i in range(x):
      for j in range(y):
         graph.SetPoint(N, i, j, array[i, j])
         N+=1

#Fills a 2D TGraph object with only elements from lower triangular matrix of given array
def fill_graph2D_dia(graph, array):
   x = array.shape[0]
   y = array.shape[1]
   N = 0
   for i in range(x):
      for j in range(x-i):
         graph.SetPoint(N, i, j, array[i, j])
         N+=1

#Get the lower triangular matrix of given array
def get_dia(array):
   x = array.shape[0]
   y = array.shape[1]
   darray = np.zeros((x, y))
   for i in range(x):
      for j in range(x-i):
         darray[i, j]=array[i, j]
   return darray

# Compute and plot correlation
def plot_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, gsumx, gsumy, gsumz, gmomentx, gmomenty, gmomentz, gecal, energy, events1, events2, out_file, labels):
   ecal = ecal["n_0"]
   hits = my.get_hits(events1)
   actcorr = plot_corr_python(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, my.get_hits(events1), my.ratio1_total(events1), out_file, 'Data')
   for i, key in enumerate(gsumx):
     gcorr = plot_corr_python(gsumx[key], gsumy[key], gsumz[key], gmomentx[key], gmomenty[key], gmomentz[key], gecal[key], energy, my.get_hits(events2[key]), my.ratio1_total(events2[key]), out_file, 'GAN{}_{}'.format(labels[i], i), compare=True, gprev=actcorr)

#plot correlation using Python
def plot_corr_python(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio, out_file, label, compare=False, gprev=0):
   corr, names = get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio)
   corr_dia= get_dia(corr)
   print(corr_dia.shape)
   x = np.arange(corr_dia.shape[0]+ 1)
   y = np.arange(corr_dia.shape[1]+ 1)
   X, Y = np.meshgrid(x, y)
   print x.shape
   if compare:
     mse_corr = np.sum(np.square(gprev - corr))/((corr_dia.shape[0]*(corr_dia.shape[0]-1))+ 1)
     print 'mse_corr={}'.format(mse_corr)
     dlabel='{}  mse_corr/num_features = {:.4f} '.format(label, mse_corr)
   else:
     dlabel=label
   plt.figure()
   plt.pcolor(X, Y, corr_dia, label=dlabel)
   plt.xticks(x, names, rotation='vertical', fontsize=4)
   plt.yticks(x, names[::-1], fontsize=4)
   plt.margins(0.1)
   plt.legend()
   plt.savefig(out_file + '_python' + label + '.pdf')
   return corr_dia

#plot correlation using root
def plot_corr_root(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio, out_file, label, compare=False, gprev=0):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                
   c1.SetGrid()
   color = 2
   gact, names = get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio)
   gact= get_dia(gact)
   print(gact.shape[0], len(names))
   Egraph =ROOT.TGraph2D()
   fill_graph2D(Egraph, gact)
   Egraph.Draw('colz')
   ylen = len(names)
   c1.Update()
   Egraph.GetYaxis().SetLabelOffset(1)
   Egraph.GetYaxis().SetNdivisions(10* ylen)
   Egraph.GetXaxis().SetLabelOffset(1)
   Egraph.GetXaxis().SetNdivisions(10* ylen)
   ty = ROOT.TText()
   ty.SetTextAlign(32)
   ty.SetTextSize(0.011)
   ty.SetTextFont(72)
   tx = ROOT.TText()
   tx.SetTextAlign(32)
   tx.SetTextSize(0.011)
   tx.SetTextFont(72)
   tx.SetTextAngle(70)
   y = np.arange(ylen)
   for i in y:
      ty.DrawText(-0.42,y[i],names[i])
      tx.DrawText(y[i],-0.42,names[ylen-i-1])
   print(len(names))
   c1.Update()
   legend = TLegend(.6, .8, .9, .9)
   if compare:
     mse_corr = np.sum(np.square(gprev - gact))/ylen
     legend.SetHeader('{}  mse_corr/num_features = {:.4f} '.format(label, mse_corr))
   else:
     #legend.AddEntry(Egraph, label, "l")
     legend.SetHeader(label)
   legend.Draw()
   c1.Update()
   c1.Print(out_file + '_' + label + '.pdf')
   return gact      

# PLot ecal ratio
def plot_ecal_ratio_profile(ecal1, ecal2, y, labels, out_file, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   if y.shape[0]> ecal1["n_0"].shape[0]:
      y = y[:ecal1["n_0"].shape[0]]
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 520)
   Eprof.SetStats(kFALSE)
   Eprof.SetTitle("Ratio of Ecal and Ep")
   my.fill_profile(Eprof, ecal1["n_0"]/y, y)
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("Ecal/Ep")
   Eprof.GetYaxis().SetRangeUser(0, 0.03)
   Eprof.Draw()
   Eprof.SetLineColor(color)
   color+=1
   legend = TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"Geant4","l")
   Gprofs=[]
   for i, key in enumerate(ecal2):
      Gprofs.append(ROOT.TProfile("Gprof" + str(i), "Gprof" + str(i), 100, 0, 520))
      Gprof = Gprofs[i]
      Gprof.SetStats(kFALSE)
      my.fill_profile(Gprof, ecal2[key]/y, y)
      color +=1
      if color in [10, 18, 19]:
          color+=1
      Gprof.SetLineColor(color)
      Gprof.Draw('sames')
      c1.Update()
      legend.AddEntry(Gprof, "GAN" + labels[i], "l")
      legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_aux_relative_profile(aux1, aux2, y, out_file, labels, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = TLegend(.7, .8, .9, .9)
   Gprofs=[]
   Eprofs=[]
   for i, key in enumerate(aux1):
     Eprofs.append(ROOT.TProfile("Eprof" + str(i),"Eprof" + str(i), 100, 0, 520))
     Gprofs.append(ROOT.TProfile("Gprof" + str(i),"Gprof" + str(i), 100, 0, 520))
     Eprof= Eprofs[i]
     Gprof= Gprofs[i]
     if i== 0:
       Eprof.SetStats(kFALSE)
       Eprof.SetTitle("Relative Error for Primary Energy")
       Eprof.GetXaxis().SetTitle("Ep GeV")
       Eprof.GetYaxis().SetTitle("Ep - aux/Ep")
       Eprof.GetYaxis().SetRangeUser(-0.2, 0.2)
       my.fill_profile(Eprof, (y - 100 *aux1[key])/y, y)
       Eprof.Draw()
       Eprof.SetLineColor(color)
       legend.AddEntry(Eprof,"Geant4" + labels[i],"l")
       color+=2
     else:
       my.fill_profile(Eprof, (y - 100 *aux1[key])/y, y)
       Eprof.Draw('sames')
       legend.AddEntry(Eprof,"Geant4" + labels[i],"l")
       color+=1

     Gprof.SetStats(kFALSE)
     my.fill_profile(Gprof, (y - 100 *aux2[key])/y, y)
     Gprof.SetLineColor(color)
     color+=1
     Gprof.Draw('sames')
     c1.Update()
     legend.AddEntry(Gprof, "GAN" + labels[i], "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_hist(ecal1, ecal2, out_file, energy, labels, ifpdf=True, stest=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color=2
   hd = ROOT.TH1F("Geant4", "", 20, 0, 11)
   my.fill_hist(hd, ecal1['n_0'])
   if energy == 0:
      hd.SetTitle("Ecal Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Ecal Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")
   hd.Draw()
   hd.SetLineColor(color)
   color+=2
   legend = TLegend(.4, .1, .6, .2)
   legend.AddEntry(hd,"Geant4" ,"l")
   hgs=[]
   pos =0
   for i, key in enumerate(ecal2):
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 20, 0, 11))
      hg= hgs[i]  
      hg.SetLineColor(color)
      color+=1
      c1.Update()
      my.fill_hist(hg, ecal2[key])
      hg.Draw('sames')
      c1.Update()
      my.stat_pos(hg, pos)
      pos+=1
      c1.Update()
      if energy == 0:
         legend = TLegend(.4, .1, .7, .2)
         legend.AddEntry(hd,"Geant4" ,"l")
         if stest:
           ks = hd.KolmogorovTest(hg, 'UU NORM')
           ch2 = hd.Chi2Test(hg, 'UU NORM')
           glabel = "GAN {}  K={:.6f}  ch2={:.6f}".format(labels[i], ks, ch2)
         else:
           glabel = "GAN {}".format(labels[i])
         legend.AddEntry(hg, glabel , "l")
      else:
         legend = TLegend(.7, .1, .9, .2)
         legend.AddEntry(hd,"Geant4" ,"l")
         legend.AddEntry(hg, "GAN {}".format(labels[i]) , "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_flatten_hist(event1, event2, out_file, energy, labels, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color =2
   gPad.SetLogx()
   hd = ROOT.TH1F("Geant4", "", 100, -6, 0)
   my.BinLogX(hd)
   my.fill_hist(hd, event1.flatten())
   if energy == 0:
      hd.SetTitle("Ecal Flat Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Ecal Flat Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")
   hd.Draw()
   hd.SetLineColor(color)
   legend = TLegend(.5, .8, .6, .9)
   legend.AddEntry(hd,"Geant4","l")
   color+=2
   hgs=[]
   pos = 0
   for i, key in enumerate(event2):                                                                                                                                                          
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 100, -6, 0))
      hg = hgs[i]
      my.BinLogX(hg)
      my.fill_hist(hg, event2[key].flatten())
      hg.SetLineColor(color)
      color+=2
      hg.Draw('sames')
      c1.Update()
      my.stat_pos(hg, pos)
      pos+=1
      c1.Update()
      legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_hits_hist(event1, event2, out_file, energy, labels, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   thresh = 0.0002 # GeV 
   color = 2                                                                                                                                                                
   hd = ROOT.TH1F("Geant4", "", 50, 0, 3000)
   my.fill_hist(hd, my.get_hits(event1, thresh))
   if energy == 0:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for Uniform Spectrum".format(thresh))
   else:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for {} GeV".format(thresh, energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")
   hd.Draw()
   hd.SetLineColor(color)
   color+=2
   hgs=[]
   pos = 0
   legend = TLegend(.5, .7, .6, .9)
   legend.AddEntry(hd,"Geant4","l")
   for i, key in enumerate(event2):
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 3000))
      hg = hgs[i]
      my.fill_hist(hg, my.get_hits(event2[key], thresh))
      hg.SetLineColor(color)
      color+=1
      hg.Draw('sames')
      legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
      c1.Update()
      my.stat_pos(hg)
      pos+=1
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_aux_hist(aux1, aux2, out_file, energy, labels, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   c1.SetGrid()
   color = 2
   legend = TLegend(.7, .8, .9, .9)
   hps=[]
   hgs=[]
   for i, key in enumerate(aux1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 100, 0, 520))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 100, 0, 520))
     hp= hps[i]
     hg= hgs[i]
     if i== 0:
       hp.SetStats(kFALSE)
       hp.SetTitle(" Primary Energy")
       hp.GetXaxis().SetTitle("Ep GeV")
       my.fill_hist(hp, 100 *aux1[key])
       hp.Draw()
       hp.SetLineColor(color)
       c1.Update()
       legend.AddEntry("Geant4" + labels[i],"Geant4" + labels[i],"l")
       color+=2
     else:
       my.fill_hist(hp, 100 *aux1[key])
       hp.Draw('sames')
       c1.Update()
       legend.AddEntry(hp,"Geant4" + labels[i],"l")
       color+=1
     if energy == 0:
       hp.SetTitle("Aux Energy Histogram for Uniform Spectrum")
     else:
       hp.SetTitle("Aux Energy Histogram for {} GeV".format(energy) )

     hg.SetStats(kFALSE)
     my.fill_hist(hg, 100 *aux2[key])
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     c1.Update()
     legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_primary_error_hist(aux1, aux2, y, out_file, energy, labels, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = TLegend(.7, .8, .9, .9)
   hps=[]
   hgs=[]
   if y.shape[0]> aux1["n_0"].shape[0]:
      y = y[:aux1["n_0"].shape[0]]
   for i, key in enumerate(aux1):          
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 100, -0.2, 0.2))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 100, -0.2, 0.2))
     hp= hps[i]
     hg= hgs[i]
     if i== 0:
       hp.SetStats(kFALSE)
       if energy == 0:
          hp.SetTitle("Aux Energy Relative Error Histogram for Uniform Spectrum")
       else:
          hp.SetTitle("Aux Energy Relative Error Histogram for {} GeV".format(energy) )
       hp.GetXaxis().SetTitle("Primary GeV")
       hp.GetYaxis().SetTitle("Ep - aux/Ep")
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw()
       hp.SetLineColor(color)
       legend.AddEntry("Geant4" + labels[i],"Geant4" + labels[i],"l")
       color+=2
     else:
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw('sames')
       legend.AddEntry(hp,"Geant4" + labels[i],"l")
       color+=1
     hg.SetStats(kFALSE)
     my.fill_hist(hg,  (y - aux2[key]*100)/y)
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     c1.Update()
     legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_realfake_hist(array1, array2, out_file, energy, labels, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = TLegend(.7, .8, .9, .9)
   hps=[]
   hgs=[]
   for i, key in enumerate(array1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 100, 0, 1))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 100, 0, 1))
     hp= hps[i]
     hg= hgs[i]
     if i== 0:
       hp.SetStats(kFALSE)
       if energy == 0:
          hp.SetTitle("Real/Fake Histogram for Uniform Spectrum")
       else:
          hp.SetTitle("Real/Fake Histogram for {} GeV".format(energy) )

       hp.GetXaxis().SetTitle("Real/Fake")
       my.fill_hist(hp, array1[key])
       hp.Draw()
       hp.SetLineColor(color)
       legend.AddEntry("Geant4" + labels[i],"Geant4" + labels[i],"l")
       color+=2
     else:
       my.fill_hist(hp, array1[key])
       hp.Draw('sames')
       legend.AddEntry(hp,"Geant4" + labels[i],"l")
       color+=1
     hg.SetStats(kFALSE)
     my.fill_hist(hg,  array2[key])
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     c1.Update()
     legend.AddEntry(hg, "GAN" + labels[i], "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_max(array1, array2, x, y, z, out_file1, out_file2, out_file3, energy, labels, log=0, ifpdf=True, stest=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   c1.Divide(2,2)
   print array1.shape
   h1x = ROOT.TH1F('G4x' + str(energy), '', x, 0, x)
   h1y = ROOT.TH1F('G4y' + str(energy), '', y, 0, y)
   h1z = ROOT.TH1F('G4z' + str(energy), '', z, 0, z)
   h1x.SetLineColor(color)
   h1y.SetLineColor(color)
   h1z.SetLineColor(color)
   c1.cd(1)
   if log:
      gPad.SetLogy()
   my.fill_hist(h1x, array1[:,0])
   h1x.Draw()
   c1.cd(2)
   if log:
      gPad.SetLogy()
   my.fill_hist(h1y, array1[:,1])
   h1y.Draw()
   c1.cd(3)
   if log:
      gPad.SetLogy()
   my.fill_hist(h1z, array1[:,2])
   h1z.Draw()
   c1.cd(4)
   c1.Update()
   if ifpdf:
      c1.Print(out_file1 + '.pdf')
   else:
      c1.Print(out_file1 + '.C')
   h2xs=[]
   h2ys=[]
   h2zs=[]
   color+=2
   leg = ROOT.TLegend(0.1,0.6,0.7,0.9)
   for i, key in enumerate(array2):
      h2xs.append(ROOT.TH1F('GANx' + str(energy) + str(i), '', x, 0, x))
      h2ys.append(ROOT.TH1F('GANy' + str(energy) + str(i), '', y, 0, y))
      h2zs.append(ROOT.TH1F('GANz' + str(energy) + str(i), '', z, 0, z))
      h2x=h2xs[i]
      h2y=h2ys[i]
      h2z=h2zs[i]
      h2x.SetLineColor(color)
      h2y.SetLineColor(color)  
      h2z.SetLineColor(color)
      c1.cd(1)
      my.fill_hist(h2x, array2[key][:,0])
      if i==0:
         h2x.Draw()
      else:
         h2x.Draw('sames')
      c1.Update()
      if stest:
         ks = h1x.KolmogorovTest(h2x, "UU NORM")
         ch2 = h1x.Chi2Test(h2x, "UU NORM")
         glabel = "GAN {} X axis K = {} ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {}".format(labels[i])
      leg.AddEntry(h2x, glabel,"l")
      my.stat_pos(h2x)
      c1.cd(2)
      my.fill_hist(h2y, array2[key][:,1])
      if i==0:
         h2y.Draw()
      else:
         h2y.Draw('sames')
      c1.Update()
      my.stat_pos(h2y)
      if stest:
         ks = h1y.KolmogorovTest(h2y, "UU NORM")
         ch2 = h1y.Chi2Test(h2y, "UU NORM")
         glabel = "GAN {} Y axis K = {} ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {}".format(labels[i])
      leg.AddEntry(h2y, glabel,"l")
      c1.cd(3)
      my.fill_hist(h2z, array2[key][:,2])
      if i==0:
         h2z.Draw()
      else:
         h2z.Draw('sames')
      c1.Update()
      if stest:
         ks = h1z.KolmogorovTest(h2z, "UU NORM")
         ch2 = h1z.Chi2Test(h2z, "UU NORM")
         glabel = "GAN {} Z axis K = {} ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {}".format(labels[i])
      leg.AddEntry(h2z, glabel,"l")
      my.stat_pos(h2z)
      c1.Update()
   c1.cd(4)
   leg.Draw()
   if ifpdf:
      c1.Print(out_file2 + '.pdf')
   else:
      c1.Print(out_file2 + '.C')
   c1.cd(1)
   h1x.Draw('sames')
   c1.cd(2)
   h1y.Draw('sames')
   c1.cd(3)
   h1z.Draw('sames')
   c1.Update()
   c1.cd(4)
   leg.AddEntry(h1x,"Data","l")
   leg.Draw()
   c1.Update()
   if ifpdf:
      c1.Print(out_file3 + '.pdf')
   else:
      c1.Print(out_file3 + '.C')

def plot_energy_hist_root(array1x, array1y, array1z, array2x, array2y, array2z, x, y, z, out_file1, out_file2, out_file3, energy, labels, log=0, ifpdf=True, stest=True):
   canvas = TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   canvas.SetGrid()
   color = 2
   canvas.Divide(2,2)
   h1x = ROOT.TH1F('G4x' + str(energy), '', x, 0, x)
   h1y = ROOT.TH1F('G4y' + str(energy), '', y, 0, y)
   h1z = ROOT.TH1F('G4z' + str(energy), '', z, 0, z)
   h1x.Sumw2()
   h1y.Sumw2()
   h1z.Sumw2()
   h1x.SetLineColor(color)
   h1y.SetLineColor(color)
   h1z.SetLineColor(color)
   color+=2
   canvas.cd(1)
   if log:
      gPad.SetLogy()
   my.fill_hist_wt(h1x, array1x)
   h1x.Draw()
   canvas.cd(2)
   if log:
      gPad.SetLogy()
   my.fill_hist_wt(h1y, array1y)
   h1y.Draw()
   canvas.cd(3)
   if log:
      gPad.SetLogy()
   my.fill_hist_wt(h1z, array1z)
   h1z.Draw()
   canvas.cd(4)
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file1 + '.pdf')
   else:
      canvas.Print(out_file1 + '.C')
   leg = ROOT.TLegend(0.1,0.6,0.7,0.9)
   h2xs=[]
   h2ys=[]
   h2zs=[]
   for i, key in enumerate(array2x):
      h2xs.append(ROOT.TH1F('GANx' + str(energy)+ str(i), '', 25, 0, 25))
      h2ys.append(ROOT.TH1F('GANy' + str(energy)+ str(i), '', 25, 0, 25))
      h2zs.append(ROOT.TH1F('GANz' + str(energy)+ str(i), '', 25, 0, 25))
      h2x=h2xs[i]
      h2y=h2ys[i]
      h2z=h2zs[i]
      h2x.SetLineColor(color)
      h2y.SetLineColor(color)
      h2z.SetLineColor(color)
      canvas.cd(1)
      my.fill_hist_wt(h2x, array2x[key])
      h2x.Draw()
      canvas.Update()
      my.stat_pos(h2x)
      if stest:
         res=np.array
         ks= h1x.KolmogorovTest(h2x, 'WW')
         ch2 = h1x.Chi2Test(h2x, 'WW')
         glabel = "GAN {} X axis K= {}  ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {} ".format(labels[i])
      leg.AddEntry(h2x, glabel,"l")
      canvas.Update()
      canvas.cd(2)
      my.fill_hist_wt(h2y, array2y[key])
      h2y.Draw()
      canvas.Update()
      my.stat_pos(h2y)
      if stest:
         ks= h1y.KolmogorovTest(h2y, 'WW')
         ch2 = h1y.Chi2Test(h2y, 'WW')
         glabel = "GAN {} Y axis K= {}  ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {} ".format(labels[i])
      leg.AddEntry(h2y, glabel,"l")
      canvas.Update()
      canvas.cd(3)
      my.fill_hist_wt(h2z, array2z[key])
      h2z.Draw()
      canvas.Update()
      my.stat_pos(h2z)
      canvas.Update()
      if stest:
         ks= h1z.KolmogorovTest(h2z, 'WW')
         ch2 = h1z.Chi2Test(h2z, 'WW')
         glabel = "GAN {} Z axis K= {}  ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {} ".format(labels[i])
      leg.AddEntry(h2z, glabel,"l")
      canvas.Update()
   canvas.cd(4)
   leg.Draw()
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file2 + '.pdf')
   else:
      canvas.Print(out_file2 + '.C')
   canvas.cd(1)
   h1x.Draw('sames')
   canvas.cd(2)
   h1y.Draw('sames')
   canvas.cd(3)
   h1z.Draw('sames')
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(h1x,"Data","l")
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file3)
 
def plot_moment(array1, array2, out_file, dim, energy, m, labels, ifpdf=True):
   c1 = TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   array1= array1[:, m]
   if m==0:
     bins = 25
     maxbin = 25
     minbin = 0
   else:
     bins = 50
     maxbin = np.amax(array1)+ 5
     minbin = min(0, np.amin(array1))
   c1.SetGrid()
   color = 2
   legend = TLegend(.7, .1, .9, .2)
   hd = ROOT.TH1F("Geant4"+ dim + str(m), "", bins, minbin, maxbin)
   if energy == 0:
      hd.SetTitle("{} {} Moment Histogram for Uniform Spectrum".format(m+1, dim))
   else:
      hd.SetTitle("{} {} Moment Histogram for {} GeV".format(m+1, dim, energy) )
      hd.GetXaxis().SetTitle("{} Moment for {} axis".format(m+1, dim))
   my.fill_hist(hd, array1)
   hd.Draw()
   hd.SetLineColor(color)
   c1.Update()
   legend.AddEntry(hd,"Geant4","l")
   c1.Update()
   color+=2
   hgs=[]
   for i, key in enumerate(array2):
      hgs.append(ROOT.TH1F("GAN"+ dim + str(m), "", bins, minbin, maxbin))
      hg= hgs[i]
      my.fill_hist(hg, array2[key][:, m])
      hg.SetLineColor(color)
      color+=1
      hg.Draw('sames')
      c1.Update()
      legend.AddEntry(hg,"GAN"+ str(labels[i]),"l")
      my.stat_pos(hg)
      c1.Update()
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def get_plots_multi(var, labels, plots_dir, energies, m, n, ifpdf=True, stest=True):
    
    actdir = plots_dir + 'Actual'
    safe_mkdir(actdir)
    discdir = plots_dir + 'disc_outputs'
    safe_mkdir(discdir)
    gendir = plots_dir + 'Generated' 
    safe_mkdir(gendir)
    comdir = plots_dir + 'Combined' 
    safe_mkdir(comdir)
    mdir = plots_dir + 'Moments' 
    safe_mkdir(mdir)
    start = time.time()
    plots = 0
    for energy in energies:
       x=var["events_act" + str(energy)].shape[1]
       y=var["events_act" + str(energy)].shape[2]
       z=var["events_act" + str(energy)].shape[3]
       maxfile = "Position_of_max_" + str(energy)# + ".pdf"
       maxlfile = "Position_of_max_" + str(energy)# + "_log.pdf"
       histfile = "hist_" + str(energy)# + ".pdf"
       histlfile = "hist_log" + str(energy)# + ".pdf"
       ecalfile = "ecal_" + str(energy)# + ".pdf"
       energyfile = "energy_" + str(energy)# + ".pdf"
       realfile = "realfake_" + str(energy)# + ".pdf"
       momentfile = "moment" + str(energy)# + ".pdf"
       auxfile = "Auxilliary_"+ str(energy)# + ".pdf"
       ecalerrorfile = "ecal_error" + str(energy)# + ".pdf"
       allfile = 'All_energies'#.pdf'
       allecalfile = 'All_ecal'#.pdf'
       allecalrelativefile = 'All_ecal_relative'#.pdf'
       allauxrelativefile = 'All_aux_relative'#.pdf'
       allerrorfile = 'All_relative_auxerror'#.pdf'
       correlationfile = 'Corr'
       start = time.time()
       if energy==0:
          plot_ecal_ratio_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], var["energy" + str(energy)], labels, os.path.join(comdir, allecalfile))
          plots+=1
          plot_aux_relative_profile(var["aux_act" + str(energy)], var["aux_gan"+ str(energy)], var["energy"+ str(energy)], os.path.join(comdir, allauxrelativefile), labels)
          plots+=1
          plot_correlation(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)], var["ecal_act" + str(energy)],  var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)], var["energy" + str(energy)], var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, correlationfile), labels)
       plot_ecal_hist(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], os.path.join(discdir, ecalfile), energy, labels, stest=stest)
       plots+=1
      # plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'flat' + ecalfile), energy, labels, stest=stest)
      # plots+=1
       plot_ecal_hits_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], os.path.join(comdir, 'hits' + ecalfile), energy, labels)
       plots+=1
       plot_aux_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)] , os.path.join(discdir, energyfile), energy, labels)
       plots+=1
       plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)], x, y, z, os.path.join(actdir, maxfile), os.path.join(gendir, maxfile), os.path.join(comdir, maxfile), energy, labels, stest=stest)
       plots+=1
       plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)], x, y, z, os.path.join(actdir, maxlfile), os.path.join(gendir, maxlfile), os.path.join(comdir, 'log' + maxlfile), energy, labels, log=1, stest=stest)
       plots+=1
       plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], x, y, z, os.path.join(actdir, histfile), os.path.join(gendir, histfile), os.path.join(comdir, histfile), energy, labels, stest=stest)
       plots+=1
       plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], x, y, z, os.path.join(actdir, histlfile), os.path.join(gendir, histlfile), os.path.join(comdir, histlfile), energy, labels, log=1, stest=stest)
       plots+=1
       plot_realfake_hist(var["isreal_act" + str(energy)], var["isreal_gan" + str(energy)], os.path.join(discdir, realfile), energy, labels)
       plots+=1
       plot_primary_error_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)], var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile), energy, labels)
       plots+=1
       print len(var["momentX_gan" + str(energy)]['n_0'])       
       for mmt in range(m):                                                                                            
          plot_moment(var["momentX_act" + str(energy)], var["momentX_gan" + str(energy)], os.path.join(mdir, 'x' + str(mmt + 1) + momentfile), 'x', energy, mmt, labels)
          plots+=1           
          plot_moment(var["momentY_act" + str(energy)], var["momentY_gan" + str(energy)], os.path.join(mdir, 'y' + str(mmt + 1) + momentfile), 'y', energy, mmt, labels)           
          plots+=1
          plot_moment(var["momentZ_act" + str(energy)], var["momentZ_gan" + str(energy)], os.path.join(mdir, 'z' + str(mmt + 1) + momentfile), 'z', energy, mmt, labels)           
          plots+=1    
          
    print 'Plots are saved in ', plots_dir
    plot_time= time.time()- start
    print '{} Plots are generated in {} seconds'.format(plots, plot_time)
    
if __name__ == "__main__":
    main()
