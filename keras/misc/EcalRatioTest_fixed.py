import sys
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats, TH2F
#from ROOT import gROOT, gBenchmark
import h5py
import numpy as np
import glob
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
from array import array
import time
sys.path.insert(0,'../')
from EcalEnergyGan import generator, discriminator
try:
      import setGPU
except:
      pass

sys.path.insert(0,'../')
import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import analysis.utils.RootPlotsGAN as pl

weightdir = '../weights/'
#gStyle.SetOptStat(0)
gStyle.SetOptFit (1111) # superimpose fit results
#c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,800 ,300) #make nice
c=TCanvas("c" ,"Ecal/Ep versus Ep for Data and Generated Events" ,200 ,10 ,700 ,500)
#c.Divide(2, 1)
p =[0, 500]
bsize=10
bins = int((p[1]-p[0])/bsize)
num_events=10000
stat = 0
#filename = 'results/ecal_ratio_chpi_{}events_{}_{}GeV_bin{}_ep189_sqrt.pdf'.format(num_events, p[0], p[1], bins)
filename = 'results/ecal_ratio_multi.pdf'
#c.SetLogx ()
Eprof = TProfile("Eprof", "Sampling fraction vs. Ep;Ep [GeV];#S_{f}", bins, p[0], p[1])
if not stat: gStyle.SetOptStat(0)
#latent = 200
dscale  = 1.0
thresh = 0
#g = generator(latent)
gweight1 = '../weights/100_200GeVweights/params_generator_epoch_029.hdf5'
gweight2 = 'weights_versions/generator_epoch_029.hdf5'
gweight3 = '../weights/mae_lin_30_full/params_generator_epoch_029.hdf5'
gweight4 = 'weights_versions/generator_epoch_049_rootfit.hdf5'
gweight5 = 'weights_versions/generator_epoch_041_scaled.hdf5'

gweights = [gweight1,  gweight2, gweight3,  gweight4,  gweight5]#, gweight2]
label = ['', 'factor of 50', 'fit from ROOT', 'MAPE',  'scaled']
scales = [1, 1, 1, 1, 100]
color = [2, 6, 7, 9, 3, 4]
latent = [200, 200, 1024, 200, 200]
energies = [0]
#c.cd(1)
#Get Actual Data
datapath = '/storage/group/gpu/bigdata/LCD/NewV1/EleEscan/EleEscan_*.h5'
particle = 'Ele'
datafiles = glob.glob(datapath)
#data = gan.get_sorted(datafiles[-3:], energies, flag = True, thresh=thresh, num_events1=num_events)
X, Y = gan.GetData(datafiles[0], thresh=thresh, num_events=num_events)
#X= data["events_act0"]
#Y= data["energy0"]
#theta = data["angle0"]
Data = np.sum(X, axis=(1, 2, 3))/dscale
num_events = X.shape[0]
Eprof.Sumw2()
for j in np.arange(X.shape[0]):
      Eprof.Fill(Y[j], Data[j]/Y[j])
Eprof.SetTitle("Ratio of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep [GeV]")
Eprof.GetYaxis().SetTitle("Ecal/Ep")
Eprof.Draw()
#print('G4 data')
W1= Eprof.Integral()
#print('integral:{}'.format(Eprof.Integral()))
#print('sumw2 : {}'.format(Eprof.GetSumw2().GetSum()))
#print('bin sumw2 :', Eprof.GetBinSumw2().GetSum())
w1=[]
var1= []
#for b in np.arange(1, bins+1):
   #w1.append(Eprof.GetBinContent(b))
   #var1.append(((Eprof.GetBinError(b) * Eprof.GetBinError(b))/Eprof.GetBinEntries(b)))
   #var1.append(Eprof.GetBinError(b))
   #print('bin {} : entries {}: content {} : error {}'.format(b, Eprof.GetBinEntries(b), Eprof.GetBinContent(b), Eprof.GetBinError(b)))
#c.Update()
#Eprof.SetStats(0)
#Eprof.GetYaxis().SetRangeUser(0., 0.03)
Eprof.SetLineColor(color[0])
Eprof.SetMarkerColor(color[0])
legend = TLegend(0.7, 0.7, 0.89, 0.89)
legend.SetHeader('The range is from {} to {} with {} bins of {} GeV'.format(p[0], p[1], bins, (p[1]-p[0])/bins))
legend.AddEntry(Eprof, "G4", "l")
legend.SetBorderSize(0)

Gprof = []
for i, gweight in enumerate(gweights):
   print(i)
   Gprof.append( TProfile("Gprof" +str(i), "Gprof" + str(i), bins, p[0], p[1]))
   #Generate events
   g = generator(latent[i])
   g.load_weights(gweight)
   generated_images = gan.generate(g, num_events, [Y/100.], latent=latent[i])
   generated_images[generated_images < thresh] =0
   GData = np.sum(generated_images, axis=(1, 2, 3))/(scales[i])
   Gprof[i].Sumw2()
   
   for j in range(num_events):
      Gprof[i].Fill(Y[j], GData[j]/Y[j])
   Gprof[i].SetLineColor(color[i + 1])
   Gprof[i].SetMarkerColor(color[i + 1])
   #print('GAN data')
   W2 = Gprof[i].Integral()
   #print('integral:{}'.format(Gprof[i].Integral()))
   #print('sumw2 : {}'.format(Gprof[i].GetSumw2().GetSum()))
   #print('bin sumw2 :', Gprof[i].GetBinSumw2().GetSum())
   w2 = []
   var2 = []
   total =0
   for b in np.arange(1, bins+1):
       w2.append(Gprof[i].GetBinContent(b))
       #var2.append((Gprof[i].GetBinError(b)*Gprof[i].GetBinError(b))/Gprof[i].GetBinEntries(b))
       var2.append(Gprof[i].GetBinError(b))
       #t1 = W1*w2[b-1]
       #t2 = W2*w1[b-1]
       #statn = abs(t1 - t2)**2
       #statd = ((W1*var2[b-1])**2 + (W2*var1[b-1])**2)
       #stat = statn/statd
       #print('t1 {} : t2 {}: abs(t1 - t2)**2 {} : d {} : stat {} '.format(t1*1e+5, t2*1e+5, statn*1e+5, statd*1e+5, stat))
       #total+=stat
       #print('bin {} : entries {}: content {} : error {} : stat {} '.format(b, Gprof[i].GetBinEntries(b), Gprof[i].GetBinContent(b),
       #                  Gprof[i].GetBinError(b), stat))
   #print('chi2 stat : {}'.format(total))
   Gprof[i].Draw('sames')
   c.Modified()
   c.Update()
   if stat: r.stat_pos(Gprof[i])
   c.Update()
   Eprof.GetYaxis().SetRangeUser(0., 0.1)
   k = Eprof.KolmogorovTest(Gprof[i], 'WW P')
   chi2 = Eprof.Chi2Test(Gprof[i], 'WW P')
   print('chi2={}   k={}'.format(chi2, k))
   legend.AddEntry(Gprof[i], label[i], "l")
   
   #c.cd(2)
   legend.Draw()
   c.Update()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)

                                                                  
