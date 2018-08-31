from os import path
import ROOT
import numpy as np
import os
import math
import time
import numpy.core.umath_tests as umath
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import ROOTutils as my # common utility functions for root
from GANutils import safe_mkdir, get_sums

##################################### Plots used in detailed analysis ######################################################


# computes correlation of a set of features and returns Fisher's Transform and names of features
def get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio):
   x = sumx.shape[1]
   y = sumy.shape[1]
   z = sumz.shape[1]
   array = np.hstack((sumx[:, 2:x-2], sumy[:, 2:y-2], sumz[:, 2:z-2]))
   names = ['']
   for i in range(2, x-2):
      names = names + ['sumx'  + str(i)]
   for i in range(2, y-2):
      names = names + ['sumy'  + str(i)]
   for i in range(2, z-2):
      names = names + ['sumz'  + str(i)]
   m = momentx.shape[1]
   for i in range(m):
      array = np.hstack((array, momentx[:, i].reshape(-1, 1), momenty[:, i].reshape(-1, 1), momentz[:, i].reshape(-1, 1)))
      names = names + ['momentx' + str(i), 'momenty' + str(i), 'momentz' + str(i)]
   array = np.hstack((array, ecal.reshape(-1, 1), energy.reshape(-1, 1)))
   array = np.hstack((array, hits.reshape(-1, 1), ratio.reshape(-1, 1)))
   names = names + ['ecal sum', 'p energy', 'hits', 'ratio1_total']
   cor= np.corrcoef(array, rowvar=False)
   cor= get_dia(cor)
   fisher= np.arctanh(cor)
   return np.flip(fisher, axis=0), names

#Get the lower triangular matrix of given array
def get_dia(array):
   darray = np.zeros_like(array)
   for i in np.arange(array.shape[0]):
      for j in np.arange(i):
         darray[i, j]=array[i, j]
   return darray

# Compute and plot correlation
def plot_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, gsumx, gsumy, gsumz, gmomentx, gmomenty, gmomentz, gecal, energy, events1, events2, out_file, labels):
   ecal = ecal["n_0"]
   hits = my.get_hits(events1)
   actcorr = plot_corr_python(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, my.get_hits(events1), my.ratio1_total(events1), out_file, 'Data')
   for i, key in enumerate(gsumx):
     gcorr = plot_corr_python(gsumx[key], gsumy[key], gsumz[key], gmomentx[key], gmomenty[key], gmomentz[key], 
             gecal[key], energy, my.get_hits(events2[key]), my.ratio1_total(events2[key]), 
             out_file, 'GAN{}_{}'.format(labels[i], i), compare=True, gprev=actcorr)

#Fills a 2D TGraph object
def fill_graph2D(graph, array):
   x = array.shape[0]
   y = array.shape[1]
   N = 0
   for i in range(x):
      for j in range(y):
         graph.SetPoint(N, i, j, array[i, j])
         N+=1

#plot correlation using Python
def plot_corr_python(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio, out_file, label, compare=False, gprev=0):
   corr, names = get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio)
   x = np.arange(corr.shape[0]+ 1)
   y = np.arange(corr.shape[1]+ 1)
   X, Y = np.meshgrid(x, y)
   if compare:
     num_squares = corr.shape[0]*(corr.shape[0]-1)/2
     mse_corr = np.sum(np.square(gprev - corr))/num_squares
     print 'mse_corr={}'.format(mse_corr)
     dlabel='{} mse_corr = {:.4f} '.format(label, mse_corr)
   else:
     dlabel=label
   plt.figure()
   plt.pcolor(X, Y, corr, label=dlabel, vmin = -3, vmax = 4)
   plt.xticks(x, names, rotation='vertical', fontsize=4)
   plt.yticks(x, names[::-1], fontsize=4)
   plt.margins(0.1)
   plt.colorbar()
   plt.legend()
   plt.savefig(out_file + '_python' + label + '.pdf')
   return corr

#plot correlation using root
def plot_corr_root(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio, out_file, label, compare=False, stest=True, gprev=0):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                
   color = 2
   gact, names = get_correlation(sumx, sumy, sumz, momentx, momenty, momentz, ecal, energy, hits, ratio)
   Egraph =ROOT.TGraph2D()
   Ggraph =ROOT.TGraph2D()
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
   legend = ROOT.TLegend(.6, .8, .9, .9)
   if compare:
     num_squares = gact.shape[0]*(gact.shape[0]-1)/2
     mse_corr = np.sum(np.square(gprev - gact))/num_squares
     legend.SetHeader('{}  mse_corr = {:.4f}'.format(label, mse_corr))
   else:
     legend.SetHeader(label)
   legend.Draw()
   c1.Update()
   c1.Print(out_file + '_' + label + '.pdf')
   return gact

# PLot ecal ratio
def plot_ecal_ratio_profile(ecal1, ecal2, y, labels, out_file, p=[100, 200], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   if y.shape[0]> ecal1["n_0"].shape[0]:
      y = y[:ecal1["n_0"].shape[0]]
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep", 100, p[0], p[1])
   Eprof.SetStats(ROOT.kFALSE)
   Eprof.SetTitle("Ratio of Ecal and Ep for {}-{} GeV".format(p[0], p[1]))
   my.fill_profile(Eprof, ecal1["n_0"]/y, y)
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("50 x Ecal/Ep")
   Eprof.GetYaxis().SetRangeUser(0, 2.5)
   Eprof.Draw()
   Eprof.SetLineColor(color)
   color+=1
   legend = ROOT.TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"G4","l")
   Gprofs=[]
   for i, key in enumerate(ecal2):
      Gprofs.append(ROOT.TProfile("Gprof" + str(i), "Gprof" + str(i), 100, int(p[0]), int(p[1])))
      Gprof = Gprofs[i]
      Gprof.SetStats(ROOT.kFALSE)
      my.fill_profile(Gprof, ecal2[key]/y, y)
      color +=1
      if color in [10, 18, 19]:
          color+=1
      Gprof.SetLineColor(color)
      Gprof.Draw('sames')
      c1.Update()
      legend.AddEntry(Gprof, "GAN " + labels[i], "l")
      legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

# PLot ecal ratio
def plot_ecal_relative_profile(ecal1, ecal2, y, labels, out_file, p=[2, 500], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   if y.shape[0]> ecal1["n_0"].shape[0]:
      y = y[:ecal1["n_0"].shape[0]]
   Eprof = ROOT.TProfile("Eprof", "Relative error for Ecal sum vs. Ep", 50, p[0], p[1])
   Eprof.SetStats(ROOT.kFALSE)
   Eprof.SetTitle("Relative Error for sum of  Ecal energies and Ep {}-{} GeV".format(p[0], p[1]))
   my.fill_profile(Eprof, (ecal1["n_0"] - ecal1["n_0"])/ ecal1["n_0"], y)
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("(Ecal_{G4} - Ecal_{GAN})/Ecal_{G4}")
   Eprof.GetYaxis().SetRangeUser(-1, 1)
   Eprof.Draw()
   Eprof.SetLineColor(color)
   color+=1
   legend = ROOT.TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"G4","l")
   Gprofs=[]
   for i, key in enumerate(ecal2):
      Gprofs.append(ROOT.TProfile("Gprof" + str(i), "Gprof" + str(i), 50, p[0], p[1]))
      Gprof = Gprofs[i]
      Gprof.SetStats(ROOT.kFALSE)
      my.fill_profile(Gprof, (ecal1["n_0"]- ecal2[key])/ ecal1["n_0"], y)
      color +=1
      if color in [10, 18, 19]:
        color+=1
      Gprof.SetLineColor(color)
      Gprof.Draw('sames')
      c1.Update()
      legend.AddEntry(Gprof, "GAN " + labels[i], "l")
      legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_aux_relative_profile(aux1, aux2, y, out_file, labels, p=[2, 500], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.1, .1, .3, .3)
   Gprofs=[]
   Eprofs=[]
   for i, key in enumerate(aux1):
     Eprofs.append(ROOT.TProfile("Eprof" + str(i),"Eprof" + str(i), 100, p[0], p[1]))
     Gprofs.append(ROOT.TProfile("Gprof" + str(i),"Gprof" + str(i), 100, p[0], p[1]))
     Eprof= Eprofs[i]
     Gprof= Gprofs[i]
     if i== 0:
       Eprof.SetTitle("Relative Error for Primary Energy for {}-{} GeV".format(p[0], p[1]))
       Eprof.GetXaxis().SetTitle("Ep GeV")
       Eprof.GetYaxis().SetTitle("(Ep_{g4} - Ep_{predicted})/Ep")
       Eprof.GetYaxis().CenterTitle()
       Eprof.GetYaxis().SetRangeUser(-0.3, 0.3)
       my.fill_profile(Eprof, (y - 100 *aux1[key])/y, y)
       Eprof.SetLineColor(color)
       Eprof.Draw()
       c1.Update()
       legend.AddEntry(Eprof,"G4" + labels[i],"l")
       c1.Update()
       color+=2
     else:
       my.fill_profile(Eprof, (y - 100 *aux1[key])/y, y)
       Eprof.Draw('sames')
       legend.AddEntry(Eprof,"G4 " + labels[i],"l")
       color+=1
     Eprof.SetStats(0)  
     Gprof.SetStats(0)
     my.fill_profile(Gprof, (y - 100 *aux2[key])/y, y)
     Gprof.SetLineColor(color)
     color+=1
     Gprof.Draw('sames')
     c1.Update()
     legend.AddEntry(Gprof, "GAN " + labels[i], "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_hist(ecal1, ecal2, out_file, energy, labels, p=[2, 500], ifpdf=True, stest=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color=2
   hd = ROOT.TH1F("Geant4", "", 100, 0, 2.5 * p[1])
   my.fill_hist(hd, ecal1['n_0'])
   hd.Sumw2()
   hd = my.normalize(hd)              
   if energy == 0:
      hd.SetTitle("Ecal Sum Histogram for {}-{} GeV".format(p[0], p[1]))
   else:
      hd.SetTitle("Ecal Sum Histogram (Ep ={} GeV)".format(energy) )
   hd.GetXaxis().SetTitle("Ecal Sum GeV/50")
   hd.GetYaxis().SetTitle("Count")
   hd.Draw()
   hd.Draw("sames hist")
   hd.SetLineColor(color)
   color+=2
   legend = ROOT.TLegend(.7, .1, .9, .2)
   legend.AddEntry(hd,"G4" ,"l")
   hgs=[]
   pos =0
   for i, key in enumerate(ecal2):
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 100, 0, 2 * p[1]))
      hg= hgs[i]
      hg.Sumw2()
      hg.SetLineColor(color)
      color+=1
      c1.Update()
      my.fill_hist(hg, ecal2[key])
      hg =my.normalize(hg)
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      #my.stat_pos(hg, pos)
      pos+=1
      c1.Update()
      if energy == 0:
         if stest:
           ks = hd.KolmogorovTest(hg, 'UU NORM')
           ch2 = hd.Chi2Test(hg, 'UU NORM')
           glabel = "GAN {}  K={:.6f}  ch2={:.6f}".format(labels[i], ks, ch2)
         else:
           glabel = "GAN {}".format(labels[i])
         legend.AddEntry(hg, glabel , "l")
      else:
         legend.AddEntry(hg, "GAN {}".format(labels[i]) , "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_ecal_flatten_hist(event1, event2, penergy, out_file, energy, labels, p=[2, 500], ifpdf=True, log=0):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color =2
   ROOT.gPad.SetLogx()
   ROOT.gStyle.SetOptStat(111111)
   if log:
      ROOT.gPad.SetLogy()
   hd = ROOT.TH1F("Geant4", "", 100, -8, 2)
   my.BinLogX(hd)
   my.fill_hist(hd, event1.flatten())
   hd.Sumw2()
   hd =my.normalize(hd)
   if energy == 0:
      hd.SetTitle("Cell energies Histogram for {:.2f}-{:.2f} GeV".format(p[0], p[1]))
   else:
      hd.SetTitle("Cell energies Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Cell energy deposition GeV/50")
   hd.Draw()
   hd.Draw('sames hist')
   hd.SetLineColor(color)
   legend = ROOT.TLegend(.2, .7, .4, .9)
   legend.AddEntry(hd,"G4","l")
   color+=2
   hgs=[]
   pos = 0
   for i, key in enumerate(event2):
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 100, -8, 2))
      hg = hgs[i]
      hg.Sumw2()
      my.BinLogX(hg)
      my.fill_hist(hg, event2[key].flatten())
      hg =my.normalize(hg)
      hg.SetLineColor(color)
      color+=2
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
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

def plot_ecal_hits_hist(event1, event2, out_file, energy, labels, p=[2, 500], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   thresh = 0.0002 # GeV
   color = 2
   hd = ROOT.TH1F("Geant4", "", 50, 0, 4000)
   my.fill_hist(hd, my.get_hits(event1, thresh))
   if energy == 0:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV/50) for {}-{} GeV Primary Energy".format(thresh, p[0], p[1]))
   else:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV/50) for {} GeV Primary Energy".format(thresh, energy) )
   hd.GetXaxis().SetTitle("Ecal Hits")
   hd.GetYaxis().SetTitle("Count")
   hd.GetYaxis().CenterTitle()
   hd =my.normalize(hd)            
   hd.Draw()
   hd.Draw('sames hist')
   hd.SetLineColor(color)
   hd.Sumw2()
   color+=2
   hgs=[]
   pos = 0
   legend = ROOT.TLegend(.7, .1, .9, .3)
   legend.AddEntry(hd,"G4","l")
   for i, key in enumerate(event2):
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 4000))
      hg = hgs[i]
      hg.Sumw2()
      my.fill_hist(hg, my.get_hits(event2[key], thresh))
      hg.SetLineColor(color)
      hg =my.normalize(hg)
      color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
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

def plot_aux_hist(aux1, aux2, out_file, energy, labels, p=[2, 500], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hps=[]
   hgs=[]
   for i, key in enumerate(aux1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 100, 0, 600))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 100, 0, 600))
     hp= hps[i]
     hg= hgs[i]
     hp.Sumw2()
     hg.Sumw2()
     if i== 0:
       #hp.SetStats(0)
       #hp.SetTitle(" Primary Energy")
       hp.GetXaxis().SetTitle("Ep GeV")
       my.fill_hist(hp, 100 *aux1[key])
       hp.Draw()
       hp.Draw('sames hist')
       hp.SetLineColor(color)
       c1.Update()
       legend.AddEntry(hp,"G4" + labels[i],"l")
       color+=2
     else:
       my.fill_hist(hp, 100 *aux1[key])
       hp.Draw('sames')
       hp.Draw('sames hist')
       c1.Update()
       legend.AddEntry(hp,"G4" + labels[i],"l")
       color+=1
     if energy == 0:
       hp.SetTitle("Predicted Primary Energy Histogram for {}-{} GeV".format(p[0], p[1]))
     else:
       hp.SetTitle("Predicted Primary Energy Histogram for {} GeV".format(energy) )
     #hg.SetStats(0)
     #my.stat_pos(hg)
     my.fill_hist(hg, 100 *aux2[key])
     hp =my.normalize(hp)
     hg =my.normalize(hg)             
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     my.stat_pos(hg)
     c1.Update()
     legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_primary_error_hist(aux1, aux2, y, out_file, energy, labels, p=[2, 500], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.1, .1, .3, .3)
   hps=[]
   hgs=[]
   if y.shape[0]> aux1["n_0"].shape[0]:
      y = y[:aux1["n_0"].shape[0]]
   for i, key in enumerate(aux1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 20, -0.4, 0.4))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 20, -0.4, 0.4))
     hp= hps[i]
     hg= hgs[i]
     hp.Sumw2()
     hg.Sumw2()
     if i== 0:
       #hp.SetStats(0)
       if energy == 0:
          hp.SetTitle("Predicted Energy Relative Error Histogram for {}-{} GeV".format(p[0], p[1]))
       else:
          hp.SetTitle("Aux Energy Relative Error Histogram for {} GeV".format(energy) )
       hp.GetXaxis().SetTitle("Primary GeV")
       hp.GetYaxis().SetTitle("(E_p - aux)/E_p")
       hp.GetYaxis().CenterTitle()
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw()
       hp.Draw('sames hist')
       c1.Update()
       hp.SetLineColor(color)
       legend.AddEntry(hp,"G4 " + labels[i],"l")
       c1.Update()

     else:
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw('sames')
       hp.Draw('sames hist')
       legend.AddEntry(hp,"G4 " + labels[i],"l")
       color+=1
     #hg.SetStats(0)
     #my.stat_pos(hg)
     my.fill_hist(hg,  (y - aux2[key]*100)/y)
     hp =my.normalize(hp)
     hg =my.normalize(hg)
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     my.stat_pos(hg)
     c1.Update()
     legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_realfake_hist(array1, array2, out_file, energy, labels, p=[2, 500], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hps=[]
   hgs=[]
   for i, key in enumerate(array1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 20, 0, 1.3))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 20, 0, 1.3))
     hp= hps[i]
     hg= hgs[i]
     if i== 0:
       #hp.SetStats(0)
       if energy == 0:
          hp.SetTitle("Real/Fake Histogram for {}-{} GeV".format(p[0], p[1]))
       else:
          hp.SetTitle("Real/Fake Histogram for {} GeV".format(energy) )

       hp.GetXaxis().SetTitle("Real/Fake")
       my.fill_hist(hp, array1[key])
       hp.Draw()
       hp.Draw('sames hist')
       c1.Update()
       hp.Sumw2()
       hg.Sumw2()
       hp.GetYaxis().SetRangeUser(0, 1)
       hp.GetYaxis().SetTitle('count')
       hp.GetYaxis().CenterTitle()
       hp.SetLineColor(color)
       c1.Update()
       legend.AddEntry(hp,"G4" + labels[i],"l")
       c1.Update()
       color+=2
     else:
       my.fill_hist(hp, array1[key])
       hp.Draw('sames')
       hp.Draw('sames hist')
       c1.Update()
       legend.AddEntry(hp,"G4 " + labels[i],"l")
       c1.Update()
       color+=1
     ##hg.SetStats(0)
     #my.stat_pos(hg)
     my.fill_hist(hg,  array2[key])
     hp =my.normalize(hp)
     hg =my.normalize(hg)
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     my.stat_pos(hg)
     c1.Update()
     legend.AddEntry(hg, "GAN " + labels[i], "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_max(array1, array2, x, y, z, out_file1, out_file2, out_file3, energy, labels, log=0, p=[2, 500], ifpdf=True, stest=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetTitle('Weighted Histogram for point of maximum energy deposition along x, y, z axis')
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
   h1x.Sumw2()
   h1y.Sumw2()
   h1z.Sumw2()
   c1.cd(1)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist(h1x, array1[:,0])
   h1x=my.normalize(h1x)
   h1x.Draw()
   h1x.Draw('sames hist')
   h1x.GetXaxis().SetTitle("Position of Max Energy (x axis)")
   c1.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist(h1y, array1[:,1])
   h1y=my.normalize(h1y)
   h1y.Draw()
   h1y.Draw('sames hist')
   h1y.GetXaxis().SetTitle("Position of Max Energy (y axis)")
   c1.cd(3)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist(h1z, array1[:,2])
   h1z=my.normalize(h1z)
   h1z.Draw()
   h1z.Draw('sames hist')
   h1z.GetXaxis().SetTitle("Position of Max Energy (z axis)")
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
   leg = ROOT.TLegend(0.1,0.4,0.9, 0.9)
   leg.SetTextSize(0.06)
   for i, key in enumerate(array2):
      h2xs.append(ROOT.TH1F('GANx' + str(energy) + labels[i], '', x, 0, x))
      h2ys.append(ROOT.TH1F('GANy' + str(energy) + labels[i], '', y, 0, y))
      h2zs.append(ROOT.TH1F('GANz' + str(energy) + labels[i], '', z, 0, z))
      h2x=h2xs[i]
      h2y=h2ys[i]
      h2z=h2zs[i]
      h2x.Sumw2()
      h2y.Sumw2()
      h2z.Sumw2()

      h2x.SetLineColor(color)
      h2y.SetLineColor(color)
      h2z.SetLineColor(color)
      c1.cd(1)
      my.fill_hist(h2x, array2[key][:,0])
      h2x=my.normalize(h2x)
      if i==0:
         h2x.Draw()
         h2x.Draw('sames hist')
         h2x.GetXaxis().SetTitle("Position of Max Energy along x axis")
      else:
         h2x.Draw('sames')
         h2x.Draw('sames hist')
      c1.Update()
      if stest:
         ks = h1x.KolmogorovTest(h2x, "UU NORM")
         ch2 = h1x.Chi2Test(h2x, "UU NORM")
         glabel = "GAN {} X axis K = {} ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {}".format(labels[i])
      my.stat_pos(h2x)
      c1.cd(2)
      my.fill_hist(h2y, array2[key][:,1])
      h2y=my.normalize(h2y)
      if i==0:
         h2y.Draw()
         h2y.Draw('sames hist')
         h2y.GetXaxis().SetTitle("Position of Max Energy along y axis")
      else:
         h2y.Draw('sames')
         h2y.Draw('sames hist')
      c1.Update()
      my.stat_pos(h2y)
      if stest:
         ks = h1y.KolmogorovTest(h2y, "UU NORM")
         ch2 = h1y.Chi2Test(h2y, "UU NORM")
         glabel = "GAN {} Y axis K = {} ch2={}".format(labels[i], ks, ch2)
         leg.AddEntry(h2y, glabel,"l")
               
      c1.cd(3)
      my.fill_hist(h2z, array2[key][:,2])
      h2z=my.normalize(h2z)
      if i==0:
         h2z.Draw()
         h2z.Draw('sames hist')
         h2z.GetXaxis().SetTitle("Position of Max Energy (z axis)")
      else:
         h2z.Draw('sames')
         h2z.Draw('sames hist')
      c1.Update()
      if stest:
         ks = h1z.KolmogorovTest(h2z, "UU NORM")
         ch2 = h1z.Chi2Test(h2z, "UU NORM")
         glabel = "GAN {} Z axis K = {} ch2={}".format(labels[i], ks, ch2)
         leg.AddEntry(h2z, glabel,"l")
      my.stat_pos(h2z)
      color+= 2
      c1.Update()
   c1.cd(4)
   leg.Draw()
   if ifpdf:
      c1.Print(out_file2 + '.pdf')
   else:
      c1.Print(out_file2 + '.C')

   c1.cd(1)
   h1x.Draw('sames')
   h1x.Draw('sames hist')
   c1.cd(2)
   h1y.Draw('sames')
   h1y.Draw('sames hist')
   c1.cd(3)
   h1z.Draw('sames')
   h1z.Draw('sames hist')
   c1.Update()
   c1.cd(4)
   leg.AddEntry(h1x,"G4","l")
   leg.SetHeader("#splitline{Weighted Histograms for position of}{ max energy deposition along x, y, z axis}", "C")
   if not stest:
     for i, h in enumerate(h2xs):
       leg.AddEntry(h, 'GAN ' + labels[i],"l")
   leg.Draw()
   c1.Update()
   if ifpdf:
      c1.Print(out_file3 + '.pdf')
   else:
      c1.Print(out_file3 + '.C')

def plot_energy_hist_root(array1x, array1y, array1z, array2x, array2y, array2z, x, y, z, out_file1, out_file2, out_file3, energy, labels, log=0, p=[2, 500], ifpdf=True, stest=True):
   canvas = ROOT.TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   canvas.SetTitle('Weighted Histogram for energy deposition along x, y, z axis')
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
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1x, array1x)
   h1x=my.normalize(h1x)
   h1x.Draw()
   h1x.Draw('sames hist')
   h1x.GetXaxis().SetTitle("Energy deposition along x axis")
   canvas.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1y, array1y)
   h1y=my.normalize(h1y)
   h1y.Draw()
   h1y.Draw('sames hist')
   h1y.GetXaxis().SetTitle("Energy deposition along y axis")
   canvas.cd(3)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1z, array1z)
   h1z=my.normalize(h1z)
   h1z.Draw()
   h1z.Draw('sames hist')
   h1z.GetXaxis().SetTitle("Energy deposition along z axis")
   canvas.cd(4)
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file1 + '.pdf')
   else:
      canvas.Print(out_file1 + '.C')
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.06)
   h2xs=[]
   h2ys=[]
   h2zs=[]
   for i, key in enumerate(array2x):
      h2xs.append(ROOT.TH1F('GANx' + str(energy)+ labels[i], '', x, 0, x))
      h2ys.append(ROOT.TH1F('GANy' + str(energy)+ labels[i], '', y, 0, y))
      h2zs.append(ROOT.TH1F('GANz' + str(energy)+ labels[i], '', z, 0, z))
      h2x=h2xs[i]
      h2y=h2ys[i]
      h2z=h2zs[i]
      h2x.Sumw2()
      h2y.Sumw2()
      h2z.Sumw2()

      h2x.SetLineColor(color)
      h2y.SetLineColor(color)
      h2z.SetLineColor(color)
      canvas.cd(1)
      my.fill_hist_wt(h2x, array2x[key])
      h2x=my.normalize(h2x)
      if i==0:
        h2x.Draw()
        h2x.Draw('sames hist')
        h2x.GetXaxis().SetTitle("Energy deposition along x axis")
      else:
        h2x.Draw('sames')
        h2x.Draw('sames hist')
      canvas.Update()
      my.stat_pos(h2x)
      if stest:
         res=np.array
         ks= h1x.KolmogorovTest(h2x, 'WW')
         ch2 = h1x.Chi2Test(h2x, 'WW')
         glabel = "GAN {} X axis K= {}  ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {} ".format(labels[i])
      #leg.AddEntry(h2x, glabel,"l")
      canvas.Update()
      canvas.cd(2)
      my.fill_hist_wt(h2y, array2y[key])
      h2y=my.normalize(h2y)
      if i==0:
        h2y.Draw()
        h2y.Draw('sames hist')
        h2y.GetXaxis().SetTitle("Energy deposition along y axis")
      else:
        h2y.Draw('sames')
        h2y.Draw('sames hist')
      canvas.Update()
      my.stat_pos(h2y)
      if stest:
         ks= h1y.KolmogorovTest(h2y, 'WW')
         ch2 = h1y.Chi2Test(h2y, 'WW')
         glabel = "GAN {} Y axis K= {}  ch2={}".format(labels[i], ks, ch2)
         leg.AddEntry(h2y, glabel,"l")
      canvas.Update()
      canvas.cd(3)
      my.fill_hist_wt(h2z, array2z[key])
      h2z=my.normalize(h2z)
      if i==0:
        h2z.Draw()
        h2z.Draw('sames hist')
        h2z.GetXaxis().SetTitle("Energy deposition along z axis")
      else:
        h2z.Draw('sames')
        h2z.Draw('sames hist')
      canvas.Update()
      my.stat_pos(h2z)
      canvas.Update()
      if stest:
         ks= h1z.KolmogorovTest(h2z, 'WW')
         ch2 = h1z.Chi2Test(h2z, 'WW')
         glabel = "GAN {} Z axis K= {}  ch2={}".format(labels[i], ks, ch2)
         leg.AddEntry(h2z, glabel,"l")
      canvas.Update()
      color+=2
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file2 + '.pdf')
   else:
      canvas.Print(out_file2 + '.C')
   canvas.cd(1)
   h1x.Draw('sames')
   h1x.Draw('sames hist')
   canvas.cd(2)
   h1y.Draw('sames')
   h1y.Draw('sames hist')
   canvas.cd(3)
   h1z.Draw('sames')
   h1z.Draw('sames hist')
   canvas.cd(4)
   leg.AddEntry(h1x, "G4","l")
   leg.SetHeader("#splitline{Weighted Histograms for energies}{ deposited along x, y, z axis}", "C")
   if not stest:
      for i, h in enumerate(h2xs):
        leg.AddEntry(h, 'GAN ' + labels[i],"l")
      
   leg.Draw()
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file3 + '.pdf')
   else:
      canvas.Print(out_file3 + '.C')

def plot_moment(array1, array2, out_file, dim, energy, m, labels, p =[2, 500], ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   array1= array1[:, m]
   if m==0:
     if dim=='x' or dim=='y':
        bins = 51
        maxbin = 51
     else:
        bins = 25
        maxbin = 25
     minbin = 0
   else:
     bins = 50
     maxbin = np.amax(array1)+ 2
     minbin = min(0, np.amin(array1))
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hd = ROOT.TH1F("G4"+ dim + str(m), "", bins, minbin, maxbin)
   if energy == 0:
      hd.SetTitle("{} {} Moment Histogram for {}-{} GeV".format(m+1, dim, p[0], p[1]))
   else:
      hd.SetTitle("{} {} Moment Histogram for {} GeV".format(m+1, dim, energy) )
      hd.GetXaxis().SetTitle("{} Moment for {} axis".format(m+1, dim))
   my.fill_hist(hd, array1)
   hd =my.normalize(hd)
   hd.Draw()
   hd.Draw('sames hist')
   hd.SetLineColor(color)
   hd.Sumw2()
   c1.Update()
   legend.AddEntry(hd,"G4","l")
   c1.Update()
   color+=2
   hgs=[]
   for i, key in enumerate(array2):
      hgs.append(ROOT.TH1F("GAN"+ dim + str(m)+str(i), "GAN"+ dim + str(m)+str(i), bins, minbin, maxbin))
      hg= hgs[i]
      my.fill_hist(hg, array2[key][:, m])
      hg.SetLineColor(color)
      hg.Sumw2()
      hg =my.normalize(hg)
      color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      legend.AddEntry(hg,"GAN "+ str(labels[i]),"l")
      if dim == 'z':
         my.stat_pos(hg)
      else:   
         sb1=hg.GetListOfFunctions().FindObject("stats")
         sb1.SetX1NDC(.4)
         sb1.SetX2NDC(.6)
              
      c1.Update()
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

################################### Angle Plots ########################################################################
# Plot histogram of predicted angle
def plot_ang_hist(ang1, ang2, out_file, angle, angtype, labels, p, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hp=ROOT.TH1F("G4" ,"G4", 50, 0, 3)
   hp.Sumw2()
   hp.SetTitle("Angle Histogram for {:.2f} {}".format(angle, angtype) )
   hp.GetXaxis().SetTitle(angtype + ' (radians)')
   hp.GetYaxis().SetTitle('Count')
   hp.GetYaxis().CenterTitle()
   my.fill_hist(hp, ang1['n_0'])
   hp.Draw()
   hp =my.normalize(hp)
   hp.Draw('sames hist')
   c1.Update()
   hp.SetLineColor(color)
   c1.Update()
   legend.AddEntry("G4" ,"G4","l")
   color+=2
   hgs=[]
   for i, key in enumerate(ang2):
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 3))
      hg= hgs[i]
      hg.Sumw2()
  
      #hg.SetStats(0)
      #my.stat_pos(hg)
      my.fill_hist(hg, ang2[key])
      hg =my.normalize(hg)
      hg.SetLineColor(color)
      color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      my.stat_pos(hg)
      c1.Update()
      legend.AddEntry(hg, "GAN {}".format(labels[i] + ' 1'), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')
            
def plot_angle_error_hist(ang1, ang2, y, out_file, angle, angtype, labels, p, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .3)
   hps=[]
   hgs=[]
   if y.shape[0]> ang1["n_0"].shape[0]:
      y = y[:ang1["n_0"].shape[0]]
   for i, key in enumerate(ang1):
      hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 50, -1.0, 1.0))
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, -1.0, 1.0))
      hp= hps[i]
      hg= hgs[i]
      hp.Sumw2()
      hg.Sumw2()
      if i== 0:
         #hp.SetStats(0)
         hp.SetTitle("Angle Relative Error Histogram for {:.2f} {}".format(angle, angtype) )
         hp.GetXaxis().SetTitle(angtype + " radians")
         hp.GetYaxis().SetTitle("(angle_{act} - angle_{pred})/angle_{act}")
         hp.GetYaxis().CenterTitle()
         my.fill_hist(hp, (y - ang1[key])/y)
         hp.Draw()
         hp.Draw('sames hist')
         c1.Update()
         hp.SetLineColor(color)
         c1.Update()
         legend.AddEntry(hp,"G4 " + labels[i],"l")
         color+=2
      else:
         my.fill_hist(hp, (y - ang1[key])/y)
         hp.Draw('sames')
         hp.Draw('sames hist')
         c1.Update()
         legend.AddEntry(hp,"G4" + labels[i],"l")
         color+=1
      #hg.SetStats(0)
      #my.stat_pos(hg)
      my.fill_hist(hg,  (y - ang2[key])/y)
      hp =my.normalize(hp)
      hg =my.normalize(hg)
      hg.SetLineColor(color)
      color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      my.stat_pos(hg)
      c1.Update()
      legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')
                                              
def plot_angle_2Dhist(ang1, ang2, y, out_file, angtype, labels, p, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   hps=[]
   for i, key in enumerate(ang1):
      legend = ROOT.TLegend(.4, .8, .5, .9)
      hps.append(ROOT.TH2F("G4" + labels[i],"G4" + labels[i], 50, 0.5, 3, 50, 0.5, 2.5))
      hp= hps[i]
      n = y.shape[0]
      hp.SetStats(0)
      hp.SetTitle("2D Histogram for predicted angles from G4 and GAN images" )
      hp.GetXaxis().SetTitle("3d Angle from G4")
      hp.GetYaxis().SetTitle("3d Angle from GAN")
      for j in np.arange(n):
         hp.Fill(ang1[key][j], ang2[key][j])
      hp.Draw("colz")
      c1.Update()
      legend.AddEntry(hp,"2D hist for " + labels[i])
      legend.Draw()
      c1.Modified()
      c1.Update()
      if ifpdf:
        c1.Print(out_file + 'n_'+ str(i) + '.pdf')
      else:
        c1.Print(out_file + 'n_'+ str(i) + '.C')

##################################### Get plots #####################################################################

def get_plots_angle(var, labels, plots_dir, energies, angles, angtype, aindexes, m, n, ifpdf=True, stest=True, nloss=3, cell=0):
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
      maxfile = "Position_of_max_" + str(energy)
      maxlfile = "Position_of_max_" + str(energy)
      histfile = "hist_" + str(energy)
      histlfile = "hist_log" + str(energy)
      ecalfile = "ecal_" + str(energy)
      energyfile = "energy_" + str(energy)
      realfile = "realfake_" + str(energy)
      momentfile = "moment" + str(energy)
      auxfile = "Auxilliary_"+ str(energy)
      ecalerrorfile = "ecal_error" + str(energy)
      angfile = "angle_"+ str(energy)
      aerrorfile = "error_"
      allfile = 'All_energies'
      allecalfile = 'All_ecal'
      allecalrelativefile = 'All_ecal_relative'#.pdf'
      allauxrelativefile = 'All_aux_relative'#.pdf'
      allerrorfile = 'All_relative_auxerror'#.pdf'
      correlationfile = 'Corr'
      start = time.time()
      if 0 in energies:
         pmin = np.amin(var["energy" + str(energy)])
         pmax = np.amax(var["energy" + str(energy)])
         p = [int(pmin), int(pmax)]
      else:
         p = [100, 200]
                           
      if energy==0:
         plot_ecal_ratio_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                                    var["energy" + str(energy)], labels, os.path.join(comdir, allecalfile), p, ifpdf=ifpdf)
         plots+=1
         plot_ecal_relative_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                                    var["energy" + str(energy)], labels, os.path.join(comdir, allecalrelativefile), p, ifpdf=ifpdf)
         plots+=1
         plot_aux_relative_profile(var["aux_act" + str(energy)], var["aux_gan"+ str(energy)], 
                                   var["energy"+ str(energy)], os.path.join(comdir, allauxrelativefile), labels, p, ifpdf=ifpdf)
         plots+=1
         """                                                                                                                
         plot_correlation(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],    
                           var["sumsz_act"+ str(energy)], var["momentX_act" + str(energy)],                                  
                           var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)],                               
                           var["ecal_act" + str(energy)],  var["sumsx_gan"+ str(energy)],                                    
                           var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],                                     
                           var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)],                               
                           var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)],                                  
                           var["energy" + str(energy)], var["events_act" + str(energy)],                                     
                           var["events_gan" + str(energy)], os.path.join(comdir, correlationfile), labels)    
         """
      
      plot_ecal_hist(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                     os.path.join(discdir, ecalfile), energy, labels, p, stest=stest, ifpdf=ifpdf)
      plots+=1
      if cell:
         plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], var["energy" + str(energy)], 
                                os.path.join(comdir, 'flat' + 'log' + ecalfile), energy, labels, log=1, ifpdf=ifpdf)
         plots+=1     
         plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], var["energy" + str(energy)], 
                                os.path.join(comdir, 'flat' + ecalfile), energy, labels, ifpdf=ifpdf)  
      plots+=1                                                                                                             
      plot_ecal_hits_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                os.path.join(comdir, 'hits' + ecalfile), energy, labels, p, ifpdf=ifpdf)
      plots+=1
      plot_aux_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)] , 
                    os.path.join(discdir, energyfile), energy, labels, p, ifpdf=ifpdf)
      plots+=1
      plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)],
               x, y, z, os.path.join(actdir, maxfile), os.path.join(gendir, maxfile),
               os.path.join(comdir, maxfile), energy, labels, p=p, stest=stest, ifpdf=ifpdf)
      plots+=1
      plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)],
               x, y, z, os.path.join(actdir, maxlfile),
               os.path.join(gendir, maxlfile), os.path.join(comdir, 'log' + maxlfile),
               energy, labels, log=1, p=p, stest=stest, ifpdf=ifpdf)
      plots+=1
      plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],
                               var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)],
                               var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],
                               x, y, z, os.path.join(actdir, histfile), os.path.join(gendir, histfile),
                               os.path.join(comdir, histfile), energy, labels, p=p, stest=stest, ifpdf=ifpdf)
      plots+=1
      plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)],       
                            var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)],
                            var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)],
                            x, y, z, os.path.join(actdir, histlfile), os.path.join(gendir, histlfile),
                            os.path.join(comdir, histlfile), energy, labels, log=1, p=p, stest=stest, ifpdf=ifpdf)
      plots+=1
      plot_realfake_hist(var["isreal_act" + str(energy)], var["isreal_gan" + str(energy)],
                         os.path.join(discdir, realfile), energy, labels, p, ifpdf=ifpdf)
      plots+=1
      plot_primary_error_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)],
                              var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile), energy, labels, p, ifpdf=ifpdf)
      plots+=1
      plot_angle_2Dhist(var["angle_act" + str(energy)], var["angle_gan" + str(energy)],  var["angle" + str(energy)],
                        os.path.join(discdir, angfile + "ang_2D") , angtype, labels, p, ifpdf=ifpdf)
      plots+=1
      if nloss==5:
         plot_angle_2Dhist(var["angle2_act" + str(energy)], var["angle2_gan" + str(energy)],  var["angle" + str(energy)],
                           os.path.join(discdir, angfile + "ang2_2D") , angtype, labels, p, ifpdf=ifpdf)
         plots+=1
      for mmt in range(m):
         plot_moment(var["momentX_act" + str(energy)], var["momentX_gan" + str(energy)],
                     os.path.join(mdir, 'x' + str(mmt + 1) + momentfile), 'x', energy, mmt, labels, p, ifpdf=ifpdf)
         plots+=1
         plot_moment(var["momentY_act" + str(energy)], var["momentY_gan" + str(energy)],
                     os.path.join(mdir, 'y' + str(mmt + 1) + momentfile), 'y', energy, mmt, labels, p, ifpdf=ifpdf)
         plots+=1
         plot_moment(var["momentZ_act" + str(energy)], var["momentZ_gan" + str(energy)],
                     os.path.join(mdir, 'z' + str(mmt + 1) + momentfile), 'z', energy, mmt, labels, p, ifpdf=ifpdf)
         plots+=1

      ecomdir = os.path.join(comdir, 'energy_' + str(energy))
      safe_mkdir(ecomdir)
      ediscdir = os.path.join(discdir, 'energy_' + str(energy))
      safe_mkdir(ediscdir)
      eactdir = os.path.join(actdir, 'energy_' + str(energy))
      safe_mkdir(eactdir)
      egendir = os.path.join(gendir, 'energy_' + str(energy))
      safe_mkdir(egendir)
      for a, angle in zip(aindexes, angles):
         #alabels = ['ang_' + str() for _ in aindexes]
         alabels = ['angle_{:.2f}{}'.format(angle, _) for _ in labels]
         a2labels = ['angle2_{:.2f}{}'.format(angle, _) for _ in labels]
         plot_energy_hist_root(var["sumsx_act"+ str(energy) + "ang_" + str(a)], var["sumsy_act"+ str(energy)+ "ang_" + str(a)],
                                  var["sumsz_act"+ str(energy) + "ang_" + str(a)], var["sumsx_gan"+ str(energy)+ "ang_" + str(a)],
                                  var["sumsy_gan"+ str(energy)+ "ang_" + str(a)], var["sumsz_gan"+ str(energy)+ "ang_" + str(a)],
                                  x, y, z, os.path.join(eactdir, histfile + 'ang_' + str(a)), os.path.join(egendir, histfile+ 'ang_' + str(a)),
                                  os.path.join(ecomdir, histfile+ 'ang_' + str(a)), energy, alabels, p=p, stest=stest, ifpdf=ifpdf)
         plots+=1
         plot_energy_hist_root(var["sumsx_act"+ str(energy) + "ang_" + str(a)], var["sumsy_act"+ str(energy)+ "ang_" + str(a)],
                               var["sumsz_act"+ str(energy) + "ang_" + str(a)], var["sumsx_gan"+ str(energy)+ "ang_" + str(a)],
                               var["sumsy_gan"+ str(energy)+ "ang_" + str(a)], var["sumsz_gan"+ str(energy)+ "ang_" + str(a)],
                               x, y, z, os.path.join(eactdir, histfile + 'ang_' + str(a)), os.path.join(egendir, histfile+ 'ang_' + str(a)),
                               os.path.join(ecomdir, histfile+ 'logang_' + str(a)), energy, alabels, log=1, p=p, stest=stest, ifpdf=ifpdf)
         plots+=1
         plot_ang_hist(var["angle_act" + str(energy) + "ang_" + str(a)], var["angle_gan" + str(energy) + "ang_" + str(a)] ,
                       os.path.join(ediscdir, "ang_" + str(a)), angle, angtype, alabels, p=p, ifpdf=ifpdf)
         plots+=1
         plot_angle_error_hist(var["angle_act" + str(energy) + "ang_" + str(a)], var["angle_gan" + str(energy) + "ang_" + str(a)],
                               var["angle" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, aerrorfile + "ang2_" + str(a)),
                               angle, angtype, alabels, p=p, ifpdf=ifpdf)
         plots+=1

         if nloss==5:
            plot_ang_hist(var["angle2_act" + str(energy) + "ang_" + str(a)], var["angle2_gan" + str(energy) + "ang_" + str(a)] ,
                          os.path.join(ediscdir, "ang2_" + str(a)), angle, angtype, a2labels, p=p, ifpdf=ifpdf)
            plots+=1
            plot_angle_error_hist(var["angle2_act" + str(energy) + "ang_" + str(a)], var["angle2_gan" + str(energy) + "ang_" + str(a)],
                                  var["angle" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, aerrorfile + "ang2_" + str(a)),
                                  angle, angtype, a2labels, p=p, ifpdf=ifpdf)
            plots+=1
                                             
         plot_realfake_hist(var["isreal_act" + str(energy) + "ang_" + str(a)], var["isreal_gan" + str(energy)+ "ang_" + str(a)],
                            os.path.join(ediscdir, realfile  + "ang_" + str(a)), angle, alabels, p, ifpdf=ifpdf)
         plots+=1
         plot_primary_error_hist(var["aux_act" + str(energy) + "ang_" + str(a)], var["aux_gan" + str(energy) + "ang_" + str(a)],
                      var["energy" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, 'error_' + energyfile + "ang_" + str(a)),
                                 energy, alabels, p, ifpdf=ifpdf)
         plots+=1
         
   print 'Plots are saved in ', plots_dir
   plot_time= time.time()- start
   print '{} Plots are generated in {} seconds'.format(plots, plot_time)
                 
################################################# Plots for 2D coloured histograms ########################################################

def PlotEnergyHistGen(events, out_file, energy, thetas, log=0, ifC=False):
   canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,700 ,500) #make
   canvas.SetGrid()
   label = "Weighted Histograms for {} GeV".format(energy)
   canvas.Divide(2,2)
   color = 2
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.05)
   hx=[]
   hy=[]
   hz=[]
   thetas = list(reversed(thetas))
   for i, theta in enumerate(thetas):
      event = events[str(theta)]
      num = event.shape[0]
      sumx, sumy, sumz=get_sums(event)
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
      hx[i].Sumw2()
      hy[i].Sumw2()
      hz[i].Sumw2()
      canvas.cd(1)
      if log:
         gPad.SetLogy()
      my.fill_hist_wt(hx[i], sumx)
      if i ==0:
         hx[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hx[i].DrawNormalized('sames hist')
      canvas.cd(2)
      if log:
         gPad.SetLogy()
      my.fill_hist_wt(hy[i], sumy)
      if i==0:
         hy[i].DrawNormalized('sames hist')
         canvas.Update()
      else:
         hy[i].DrawNormalized('sames hist')
      canvas.cd(3)
      if log:
         gPad.SetLogy()
      my.fill_hist_wt(hz[i], sumz)
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

def MeasPython(image, mod=0):
   x_shape= image.shape[1]
   y_shape= image.shape[2]
   z_shape= image.shape[3]

   sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
   indexes = np.where(sumtot > 0)
   amask = np.ones_like(sumtot)
   amask[indexes] = 0
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
   if mod==2:
      wang = ang * z
      sumz_tot = z * zmask
      ang = np.sum(wang, axis=1)/np.sum(sumz_tot, axis=1)
   indexes = np.where(amask>0)
   ang[indexes] = 100.
   return ang

def PlotEvent(event, energy, theta, out_file, n, opt="", unit='degrees'):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   x = event.shape[0]
   y = event.shape[1]
   z = event.shape[2]
   
   ang1 = MeasPython(np.moveaxis(event, 3, 0))
   ang2 = MeasPython(np.moveaxis(event, 3, 0), mod=2)
   if unit == 'degrees':
      ang1= np.degrees(ang1)
      ang2= np.degrees(ang2)
      theta = np.degrees(theta)
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   leg.SetTextSize(0.04)
   leg.SetHeader("#splitline{Weighted Histograms for energies deposited in}{x, y and z planes}", 'C')
   hx = ROOT.TH2F('x_{:.2f}GeV_{:.2f}'.format(100 * energy, theta), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{:.2f}GeV_{:.2f}'.format(100 * energy, theta), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{:.2f}GeV_{:.2f}'.format(100 * energy, theta), '', x, 0, x, y, 0, y)
   ROOT.gPad.SetLogz()
   ROOT.gStyle.SetPalette(1)
   event = np.expand_dims(event, axis=0)
   my.FillHist2D_wt(hx, np.sum(event, axis=1))
   my.FillHist2D_wt(hy, np.sum(event, axis=2))
   my.FillHist2D_wt(hz, np.sum(event, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   my.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   my.stat_pos(hy)
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   canvas.cd(4)
   leg.AddEntry(hx, 'Energy Input = {:.2f} GeV'.format(100 * energy),"l")
   leg.AddEntry(hy, 'Theta Input  = {:.2f} {}'.format(theta, unit),"l")
   leg.AddEntry(hz, 'Computed Theta (mean)     = {:.2f} {}'.format(ang1[0], unit),"l")
   leg.AddEntry(hz, 'Computed Theta (weighted) = {:.2f} {}'.format(ang2[0], unit),"l")
   leg.Draw()
   my.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)

def PlotAngleCut(events, ang, out_file, opt=""):
   canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,700 ,500)
   canvas.Divide(2,2)
   n = events.shape[0]
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
   ROOT.gStyle.SetPalette(1)
   ROOT.gPad.SetLogz()
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.05)
   hx = ROOT.TH2F('X{} Degree'.format(str(ang)), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('Y{} Degree'.format(str(ang)), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('Z{} Degree'.format(str(ang)), '', x, 0, x, y, 0, y)
   my.FillHist2D_wt(hx, np.sum(events, axis=1))
   my.FillHist2D_wt(hy, np.sum(events, axis=2))
   my.FillHist2D_wt(hz, np.sum(events, axis=3))
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   canvas.Update()
   my.stat_pos(hx)
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hy.GetYaxis().CenterTitle()
   canvas.Update()
   my.stat_pos(hy)
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
   my.stat_pos(hz)
   canvas.Update()
   canvas.Print(out_file)
                                                                                                                                             
def PlotPosCut(events, xcut, ycut, zcut, energy, out_file, opt='colz'):
   canvas = ROOT.TCanvas("canvas" ,"Data 2D Hist" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   ROOT.gPad.SetLogz()
   ROOT.gStyle.SetPalette(1)
   n = events.shape[0]
   x = events.shape[1]
   y = events.shape[2]
   z = events.shape[3]
            
   hx = ROOT.TH2F('x_{}GeV_x={}cut'.format(str(energy), str(xcut)), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{}GeV_y={}cut'.format(str(energy), str(ycut)), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{}GeV_z={}cut'.format(str(energy), str(zcut)), '', x, 0, x, y, 0, y)
   my.FillHist2D_wt(hx, events[:, xcut, :, :])
   my.FillHist2D_wt(hy, events[:, :, ycut, :])
   my.FillHist2D_wt(hz, events[:, :, :, zcut])
   canvas.cd(1)
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   canvas.Update()
   canvas.Print(out_file)
                                                                                 
