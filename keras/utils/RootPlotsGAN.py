from os import path
import ROOT
import numpy as np

import math
import time
import numpy.core.umath_tests as umath
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import ROOTutils as my # common utility functions for root

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
def plot_ecal_ratio_profile(ecal1, ecal2, y, labels, out_file, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   if y.shape[0]> ecal1["n_0"].shape[0]:
      y = y[:ecal1["n_0"].shape[0]]
   Eprof = ROOT.TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 520)
   Eprof.SetStats(ROOT.kFALSE)
   Eprof.SetTitle("Ratio of Ecal and Ep")
   my.fill_profile(Eprof, ecal1["n_0"]/y, y)
   Eprof.GetXaxis().SetTitle("Ep GeV")
   Eprof.GetYaxis().SetTitle("Ecal/Ep")
   Eprof.GetYaxis().SetRangeUser(0, 2.)
   Eprof.Draw()
   Eprof.SetLineColor(color)
   color+=1
   legend = ROOT.TLegend(.7, .8, .9, .9)
   legend.AddEntry(Eprof,"Geant4","l")
   Gprofs=[]
   for i, key in enumerate(ecal2):
      Gprofs.append(ROOT.TProfile("Gprof" + str(i), "Gprof" + str(i), 100, 0, 520))
      Gprof = Gprofs[i]
      Gprof.SetStats(ROOT.kFALSE)
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .8, .9, .9)
   Gprofs=[]
   Eprofs=[]
   for i, key in enumerate(aux1):
     Eprofs.append(ROOT.TProfile("Eprof" + str(i),"Eprof" + str(i), 100, 0, 520))
     Gprofs.append(ROOT.TProfile("Gprof" + str(i),"Gprof" + str(i), 100, 0, 520))
     Eprof= Eprofs[i]
     Gprof= Gprofs[i]
     if i== 0:
       Eprof.SetStats(0)
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
     Gprof.SetStats(0)
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color=2
   hd = ROOT.TH1F("Geant4", "", 100, 0, 520)
   my.fill_hist(hd, ecal1['n_0'])
   if energy == 0:
      hd.SetTitle("Ecal Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Ecal Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")
   hd.Draw()
   hd.SetLineColor(color)
   color+=2
   legend = ROOT.TLegend(.4, .1, .6, .2)
   legend.AddEntry(hd,"Geant4" ,"l")
   hgs=[]
   pos =0
   for i, key in enumerate(ecal2):
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 100, 0, 520))
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
         legend = ROOT.TLegend(.4, .1, .7, .2)
         legend.AddEntry(hd,"Geant4" ,"l")
         if stest:
           ks = hd.KolmogorovTest(hg, 'UU NORM')
           ch2 = hd.Chi2Test(hg, 'UU NORM')
           glabel = "GAN {}  K={:.6f}  ch2={:.6f}".format(labels[i], ks, ch2)
         else:
           glabel = "GAN {}".format(labels[i])
         legend.AddEntry(hg, glabel , "l")
      else:
         legend = ROOT.TLegend(.7, .1, .9, .2)
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color =2
   ROOT.gPad.SetLogx()
   hd = ROOT.TH1F("Geant4", "", 100, -6, 2)
   my.BinLogX(hd)
   my.fill_hist(hd, event1.flatten())
   if energy == 0:
      hd.SetTitle("Ecal Flat Histogram for Uniform Spectrum")
   else:
      hd.SetTitle("Ecal Flat Histogram for {} GeV".format(energy) )
   hd.GetXaxis().SetTitle("Ecal GeV")
   hd.Draw()
   hd.SetLineColor(color)
   legend = ROOT.TLegend(.3, .85, .4, .9)
   legend.AddEntry(hd,"Geant4","l")
   color+=2
   hgs=[]
   pos = 0
   for i, key in enumerate(event2):
      hgs.append(ROOT.TH1F("GAN" + str(i), "GAN" + str(i), 100, -6, 2))
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   thresh = 0.0002 # GeV
   color = 2
   hd = ROOT.TH1F("Geant4", "", 50, 0, 4000)
   my.fill_hist(hd, my.get_hits(event1, thresh * 50))
   if energy == 0:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for Uniform Spectrum".format(thresh))
   else:
      hd.SetTitle("Ecal Hits Histogram (above {} GeV) for {} GeV".format(thresh, energy) )
   hd.GetXaxis().SetTitle("Ecal Hits")
   hd.Draw()
   hd.SetLineColor(color)
   color+=2
   hgs=[]
   pos = 0
   legend = ROOT.TLegend(.5, .7, .6, .9)
   legend.AddEntry(hd,"Geant4","l")
   for i, key in enumerate(event2):
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 4000))
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .8, .9, .9)
   hps=[]
   hgs=[]
   for i, key in enumerate(aux1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 100, 0, 600))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 100, 0, 600))
     hp= hps[i]
     hg= hgs[i]
     if i== 0:
       hp.SetStats(0)
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

     hg.SetStats(0)
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .8, .9, .9)
   hps=[]
   hgs=[]
   if y.shape[0]> aux1["n_0"].shape[0]:
      y = y[:aux1["n_0"].shape[0]]
   for i, key in enumerate(aux1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 20, -0.4, 0.4))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 20, -0.4, 0.4))
     hp= hps[i]
     hg= hgs[i]
     if i== 0:
       hp.SetStats(0)
       if energy == 0:
          hp.SetTitle("Aux Energy Relative Error Histogram for Uniform Spectrum")
       else:
          hp.SetTitle("Aux Energy Relative Error Histogram for {} GeV".format(energy) )
       hp.GetXaxis().SetTitle("Primary GeV")
       hp.GetYaxis().SetTitle("(E_p - aux)/E_p")
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw()
       c1.Update()
       hp.SetLineColor(color)
       legend.AddEntry("Geant4" + labels[i],"Geant4" + labels[i],"l")
       c1.Update()
       color+=2
     else:
       my.fill_hist(hp, (y - aux1[key]*100)/y)
       hp.Draw('sames')
       legend.AddEntry(hp,"Geant4" + labels[i],"l")
       color+=1
     hg.SetStats(0)
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .8, .9, .9)
   hps=[]
   hgs=[]
   for i, key in enumerate(array1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 20, 0, 1.2))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 20, 0, 1.2))
     hp= hps[i]
     hg= hgs[i]
     if i== 0:
       hp.SetStats(0)
       if energy == 0:
          hp.SetTitle("Real/Fake Histogram for Uniform Spectrum")
       else:
          hp.SetTitle("Real/Fake Histogram for {} GeV".format(energy) )

       hp.GetXaxis().SetTitle("Real/Fake")
       my.fill_hist(hp, array1[key])
       hp.Draw()
       c1.Update()
       #hp.GetYaxis().SetRangeUser(0,500)                                                                          
       hp.SetLineColor(color)
       c1.Update()
       legend.AddEntry("Geant4" + labels[i],"Geant4" + labels[i],"l")
       c1.Update()
       color+=2
     else:
       my.fill_hist(hp, array1[key])
       hp.Draw('sames')
       c1.Update()
       legend.AddEntry(hp,"Geant4" + labels[i],"l")
       c1.Update()
       color+=1
     hg.SetStats(0)
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
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                 
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
   h1x.Draw()
   c1.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist(h1y, array1[:,1])
   h1y.Draw()
   c1.cd(3)
   if log:
      ROOT.gPad.SetLogy()
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
      h2x.Sumw2()
      h2y.Sumw2()
      h2z.Sumw2()

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
   canvas = ROOT.TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make                                                         
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
   h1x.Draw()
   canvas.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1y, array1y)
   h1y.Draw()
   canvas.cd(3)
   if log:
      ROOT.gPad.SetLogy()
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
      h2xs.append(ROOT.TH1F('GANx' + str(energy)+ str(i), '', x, 0, x))
      h2ys.append(ROOT.TH1F('GANy' + str(energy)+ str(i), '', y, 0, y))
      h2zs.append(ROOT.TH1F('GANz' + str(energy)+ str(i), '', z, 0, z))
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
      h2x.Draw()
      canvas.Update()
      my.stat_pos(h2x)
      if stest:
         res=np.array
         ks= h1x.KolmogorovTest(h2x, 'WW')
         ch2 = h1x.Chi2Test(h2x, 'WW')
         glabel = "GAN {} X axis K= {}  ch2={}".format(labels[i], ks, ch2)
      else:
         glabel = "GAN {} X axis".format(labels[i])
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
         glabel = "GAN {} Y axis".format(labels[i])
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
         glabel = "GAN {} Z axis".format(labels[i])
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
   if ifpdf:
      canvas.Print(out_file3 + '.pdf')
   else:
      canvas.Print(out_file3 + '.C')

def plot_moment(array1, array2, out_file, dim, energy, m, labels, ifpdf=True):
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
     maxbin = np.amax(array1)+ 10
     minbin = min(0, np.amin(array1))
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.7, .1, .9, .2)
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
