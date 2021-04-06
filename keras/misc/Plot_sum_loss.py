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
ratio = 1
if not ratio:
   val = 'E_{sum}'
else:
   val = 'E_{sum}/E_{p}'
stat = 0
c=TCanvas("c" , val +" versus #E_{p} for Data and Generated Events" ,200 ,10 ,700 ,500)
p =[0, 500]
bsize=10
bins = int((p[1]-p[0])/bsize)
num_events=10000
filename = 'results/ecal_sum_ratio_multi.pdf'
Eprof = TProfile("Eprof", val + "vs. E_{p} [GeV];E_{sum}", bins, p[0], p[1])
if not stat: gStyle.SetOptStat(0)
dscale  = 1.0
thresh = 0
#g = generator(latent)
gweight1 = '../weights/100_200GeVweights/params_generator_epoch_029.hdf5' # without constraining Esum
gweight2 = 'weights_versions/generator_epoch_029.hdf5'# constraining Esum
gweight3 = '../weights/mae_lin_30_full/params_generator_epoch_029.hdf5' # MAE
gweight4 = 'weights_versions/generator_epoch_049_rootfit.hdf5'# vesion with fit from ROOT 
gweight5 = 'weights_versions/generator_epoch_041_scaled.hdf5' # version with intensities scaled

gweights = [gweight1,  gweight3,  gweight5]
label = ['GAN without constraining E_{sum}',  'GAN constraining E_{sum} (MAE)',  'GAN constraining E_{sum} (MAPE)']
scales = [1, 1, 100]
color = [2, 6, 8, 4]
#latent = [200, 200, 1024, 200, 200]
latent = [200, 1024, 200]
energies = [0]
#c.cd(1)
#Get Actual Data
datapath = '/storage/user/gkhattak/EleEscan_*.h5'
particle = 'Ele'
datafiles = glob.glob(datapath)
print('Data files: ')
X, Y = gan.GetData(datafiles[0], thresh=thresh, num_events=num_events)
Data = np.squeeze(np.sum(X, axis=(1, 2, 3))/dscale)
print(Data.shape, Y.shape)
if ratio: Data= np.divide(Data, Y)
print(Data.shape)
num_events = X.shape[0]
Eprof.Sumw2()
print(num_events)
for j in np.arange(X.shape[0]):
      Eprof.Fill(Y[j], Data[j])
Eprof.SetTitle(val + " of Ecal and Ep")
Eprof.GetXaxis().SetTitle("Ep [GeV]")
Eprof.GetYaxis().SetTitle(val)
if ratio:
   Eprof.GetYaxis().SetRangeUser(0., 0.1)
else:
   Eprof.GetYaxis().SetRangeUser(0., 12)
Eprof.Draw()
w1=[]
var1= []

Eprof.SetLineColor(color[0])
Eprof.SetMarkerColor(color[0])
if ratio:
  legend = TLegend(0.4, 0.5, 0.89, 0.89)
else:
  legend = TLegend(0.15, 0.5, 0.6, 0.89)
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
   GData = np.squeeze(np.sum(generated_images, axis=(1, 2, 3))/(scales[i]))
   if ratio: GData=GData/Y
   Gprof[i].Sumw2()
   
   for j in range(num_events):
      Gprof[i].Fill(Y[j], GData[j])
   Gprof[i].SetLineColor(color[i + 1])
   Gprof[i].SetMarkerColor(color[i + 1])
   
   w2 = []
   var2 = []
   total =0
   for b in np.arange(1, bins+1):
       w2.append(Gprof[i].GetBinContent(b))
       var2.append(Gprof[i].GetBinError(b))
   
   Gprof[i].Draw('sames')
   c.Modified()
   c.Update()
   if stat: r.stat_pos(Gprof[i])
   c.Update()
   k = Eprof.KolmogorovTest(Gprof[i], 'WW P')
   chi2 = Eprof.Chi2Test(Gprof[i], 'WW P')
   print('chi2={}   k={}'.format(chi2, k))
   legend.AddEntry(Gprof[i], label[i], "l")
   
   legend.Draw()
   c.Update()
c.Print(filename)
print ' The plot is saved in.....{}'.format(filename)

                                                                  
