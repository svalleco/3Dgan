from os import path
import ROOT
from ROOT import kFALSE, TLegend, TCanvas, gPad, TGraph, gStyle, TProfile
import os
import h5py
import numpy as np
import math
import time
import glob
import sys
import numpy.core.umath_tests as umath
#sys.path.insert(0,'../analysis')
sys.path.insert(0,'../')
#sys.path.insert(0,'/nfshome/gkhattak/keras/architectures_tested/')
import analysis.utils.GANutils as gan
import analysis.utils.ROOTutils as r
import setGPU
import keras.backend as K
def main():
   datapath = "/storage/group/gpu/bigdata/gkhattak/EleMeasured3ThetaEscan/Ele_VarAngleMeas_100_200_000.h5"
   #datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/EleEscan/EleEscan_RandomAngle_1_1.h5"
   #datapath3 = '/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_1_1.h5'
   genweight1 = "../weights/3dgan_weights_gan_training_epsilon_k2/params_generator_epoch_049.hdf5"
   genweight2 = "../../weights/3dgan_weights_bins_pow_p85/params_generator_epoch_049.hdf5"
   #genweight2 = "../keras/weights/surfsara_weights/params_generator_epoch_099.hdf5"
   #genweight3 = "../keras/weights/surfsara_128n/params_generator_epoch_193.hdf5"
   #genweight4 = "../keras/weights/surfsara_256n/params_generator_epoch_139.hdf5"
   #genweight1 = '../keras/weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5'
   #genweight2 = '../keras/weights/surfsara_2_500GeV/params_generator_epoch_068.hdf5'
   from AngleArch3dGAN import generator
   from AngleArch3dGAN_old import generator as generator2
   numdata = 1000
   scale=1
   outdir = 'results/ecal_tails_bulk_arch/'
   gan.safe_mkdir(outdir)
   outfile = os.path.join(outdir, 'Ecal')
   x, y, ang=GetAngleData(datapath, numdata)
   x = x/50.0 # convert to GeV
   print('The angle data varies from {} to {}'.format(np.amin(x[x>0]), np.amax(x)))

   power=0.85
   latent = 256 # latent space for generator
   concat=2
   K.set_image_data_format('channels_first')
   g=generator(latent) # build generator
   g.load_weights(genweight1) # load weights        
   x_gen1 = gan.generate(g, numdata, [y/100, ang], latent, concat=concat)
   x_gen1 = (1./50.) * np.power(x_gen1, 1./0.85)

   g2=generator2(latent) # build generator
   g2.load_weights(genweight2) # load weights
   x_gen2 = gan.generate(g2, numdata, [y/100, ang], latent, concat=1)
   x_gen2 = (1./50.) * np.power(x_gen2, 1./0.85)

   #g=generator(latent) # build generator
   #g.load_weights(genweight3) # load weights
   #x_gen3 = gan.generate(g, numdata, [y/100, ang], latent, concat=concat)
   #x_gen3 = (1./50.) * np.power(x_gen3, 1./0.85)

   #g=generator(latent) # build generator
   #g.load_weights(genweight4) # load weights 
   #x_gen4 = gan.generate(g, numdata, [y/100, ang], latent, concat=concat)
   #x_gen4 = (1./50.) * np.power(x_gen4, 1./0.85)


  
   labels = ['G4', 'GAN (upsampling first)', 'GAN (upsampling after each conv)']
   plot_ecal_flatten_hist([x, x_gen1, x_gen2], outfile, y, labels)
   plot_ecal_flatten_hist([x, x_gen1, x_gen2], outfile + '_log', y, labels, logy=1)
   plot_ecal_flatten_hist([x[:, 10:40, 10:40, :], x_gen1[:, 10:40, 10:40, :], x_gen2[:, 10:40, 10:40, :]], outfile + 'bulk', y, labels, logy=0)
   plot_ecal_flatten_hist([x[:, 10:40, 10:40, :], x_gen1[:, 10:40, 10:40, :], x_gen2[:, 10:40, 10:40, :]], outfile + 'bulk_log', y, labels, logy=1)
   plot_ecal_flatten_hist([x[:, :10, :10, :], x_gen1[:, :10, :10, :], x_gen2[:, :10, :10, :]], outfile + 'tail1', y, labels, set_range=1, logy=0)
   plot_ecal_flatten_hist([x[:, :10, :10, :], x_gen1[:, :10, :10, :], x_gen2[:, :10, :10, :]], outfile + 'tail1_log', y, labels, set_range=1,logy=1)
   plot_ecal_flatten_hist([x[:, 40:, 40:, :], x_gen1[:, 40:, 40:, :], x_gen2[:, 40:, 40:, :]], outfile + 'tail2', y, labels, set_range=1,logy=0)
   plot_ecal_flatten_hist([x[:, 40:, 40:, :], x_gen1[:, 40:, 40:, :], x_gen2[:, 40:, 40:, :]], outfile + 'tail2_log', y, labels, set_range=1,logy=1)
   print('Histogram is saved in ', outfile)
       
def postproc(event, f, scale):
   return f(event)/scale

def GetAngleData(datafile, numevents, ftn=0, scale=1, angtype='theta'):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents]) * scale
   if ftn!=0:
      x = ftn(x)
   ang = np.array(f.get(angtype)[:numevents])
   return x, y, ang

def GetData(datafile, numevents, scale=1, thresh=1e-6):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('energy')[:numevents])
   x=np.array(f.get('ECAL')[:numevents])
   x[x<thresh] = 0
   x = x * scale
   return x, y
                              
def GetData2(datafile, numevents, scale=1, thresh=1e-6):
   #get data for training
   print 'Loading Data from .....', datafile
   f=h5py.File(datafile,'r')
   y=np.array(f.get('target')[:numevents, 1])
   x=np.array(f.get('ECAL')[:numevents])
   x[x<thresh] = 0
   x = x * scale
   return x, y
                     

def plot_ecal_flatten_hist(events, out_file, energy, labels, logy=0, norm=1, set_range=0, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   #c1.SetGrid()
   ROOT.gPad.SetLogx()
   
   title = "Cell energy deposits for 100-200 GeV "
   legend = ROOT.TLegend(.11, .11, .3, .3)
   legend.SetBorderSize(0)
   colors = [2, 4, 6, 8]
   if logy:
      ROOT.gPad.SetLogy()
      title = title + " (log)"
   hds=[]
   for i, (event, label) in enumerate(zip(events, labels)):
      hds.append(ROOT.TH1F(label, "", 100, -12, 0))
      hd = hds[i]
      hd.SetStats(0)
      r.BinLogX(hd)
      data = event.flatten()
      r.fill_hist(hd, data)
      if norm:
        r.normalize(hd, norm-1)
      hd.SetLineColor(colors[i])
      if i ==0:                  
        hd.SetTitle(title)
        hd.GetXaxis().SetTitle("Ecal Single cell depositions [GeV]")
        hd.GetYaxis().SetTitle("Entries")
        hd.GetYaxis().CenterTitle()
        hd.Draw()
        hd.Draw('sames hist')
      else:
        hd.Draw('sames')
        hd.Draw('sames hist')
        if set_range:
         maxbin = hd.GetMaximumBin()
         val = hd.GetBinContent(maxbin)
         hds[0].SetMaximum(1.1 * val)
        c1.Update()
        #r.stat_pos(hd)
      legend.AddEntry(hd,label ,"l")
      c1.Modified()
      c1.Update()
      
   legend.Draw()
   c1.Update()
   if ifpdf:
     c1.Print(out_file + '.pdf')
   else:
     c1.Print(out_file + '.C')


if __name__ == "__main__":
   main()
