import sys
from os import path
import argparse
import h5py
import ROOT
from ROOT import TLegend, TCanvas, TGraph, gStyle, TProfile, TMultiGraph, TPaveStats
import numpy as np
from array import array
import time
import keras.backend as K
from analysis.utils.GANutils import GetData, GetAngleData, generate, safe_mkdir
from analysis.utils.RootPlotsGAN import plot_energy_hist_root as eplot
import analysis.utils.ROOTutils as my
import time

def get_parser():
   # To parse the input parameters to the script
   parser = argparse.ArgumentParser()
   parser.add_argument('--weights_train', type=str, nargs='+', default='/gkhattak/weights/best_weights/params_generator_epoch_041.hdf5', help="Complete PATH to the trained weights to test with hdf5 extension")
   parser.add_argument('--ang', default=0, help="If using angle vefsion")
   parser.add_argument('--labels', type=str, nargs='+', default='', help="labels for different weights")
   parser.add_argument('--xscales',  type=float, nargs='+', default=100., help="scaling factor for cell energies")
   parser.add_argument('--real_data', default='/eos/user/g/gkhattak/FixedAngleData/EleEscan_1_9.h5', help='Data to check the output obtained')
   parser.add_argument('--out_dir', default= 'results/short_analysis/', help='Complete PATH to save the output plot')
   parser.add_argument('--numevents', action='store', type=int, default=1000, help='Max limit for events used for validation')
   parser.add_argument('--latent', action='store', type=int, default=200, help='size of latent space to sample')
   parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
   parser.add_argument('--error', default=False, help='add relative errors to plots')
   parser.add_argument('--stest', default=False, help='add ktest to plots')
   parser.add_argument('--norm', default=True, help='normalize shower shapes')
   return parser

def main():
   parser = get_parser()
   args = parser.parse_args()
   weights_train = args.weights_train if isinstance(args.weights_train, list) else [args.weights_train]
   labels = args.labels if isinstance(args.labels, list) else [args.labels]
   ang = args.ang
   real_data = args.real_data
   out_dir = args.out_dir
   num_events= args.numevents
   latent = args.latent
   keras_dformat= args.dformat
   error = args.error
   stest = args.stest
   norm = args.norm
   xscales = args.xscales if isinstance(args.xscales, list) else [args.xscales]
   xpower = 0.85
   safe_mkdir(out_dir)
   print('{} dir created'.format(out_dir))
   if ang:
     from AngleArch3dGAN import generator, discriminator
     dscale = 50.
   else:
     from EcalEnergyGan import generator, discriminator
     dscale = 1.

   K.set_image_data_format(keras_dformat)
   
   # read data
   if ang:
     X, Y, angle = GetAngleData(real_data, angtype='theta', num_events=num_events)
   else:
     X, Y = GetData(real_data, num_events=num_events)
   X=X
   Y=Y
   X = np.squeeze(X)/dscale
   #get shape
   x = X.shape[1]
   y = X.shape[2]
   z = X.shape[3]
   xsum = np.sum(X, axis=(1, 2, 3))
   indexes = np.where(xsum > (0.2 * dscale))
   X=X[indexes]
   Y = Y[indexes]
   if ang: angle = angle[indexes]
   num_events = X.shape[0]
   images =[]
   if ang:
      gm=generator(dformat = keras_dformat)
   else:
      gm=generator(keras_dformat = keras_dformat)
   for i, gweight in enumerate(weights_train):
      gm.load_weights(gweight)
      if ang:
        angle = angle
        images.append(generate(gm, num_events, [Y/100., angle], latent=latent, concat=2))
        images[i] = np.power(images[i], 1./xpower)
      else:
        images.append(generate(gm, num_events, [Y/100.], latent=latent))
      images[i] = np.squeeze(images[i])/(xscales[i] * dscale)
   plotSF(X, images, Y, labels, out_file=out_dir +'SamplingFraction.pdf', error=error)
   plotshapes(X, images, x, y, z, Y, out_file=out_dir +'ShowerShapes',labels=labels, log=0, stest=stest, error=error, norm=norm)
   plotshapes(X, images, x, y, z, Y, out_file=out_dir +'ShowerShapes_log',labels=labels, log=1, stest=stest, error=error, norm=norm)

def plotSF(Data, gan_images, Y, labels, out_file, error=True):
   gStyle.SetOptFit (1111) # superimpose fit results
   c=TCanvas("c" ,"Sampling Fraction vs. Primary energy" ,200 ,10 ,700 ,500) #make nice
   c.SetGrid()
   color =2
   gStyle.SetOptStat(0)
   Eprof = TProfile("Eprof", "Ratio of Ecal and Ep;Ep;Ecal/Ep", 100, 0, 500)
   dsum = np.sum(Data, axis=(1,2, 3))
   dsf = dsum/Y
   for j in np.arange(Y.shape[0]):
     Eprof.Fill( Y[j], dsf[j])
   Eprof.SetTitle("Sampling Fraction (cell energy sum / primary particle energy)")
   Eprof.GetXaxis().SetTitle("Primary particle energy [GeV]")
   Eprof.GetYaxis().SetTitle("Sampling Fraction")
   Eprof.GetYaxis().SetRangeUser(0, 0.03)
   Eprof.SetLineColor(color)
   Eprof.Draw()
   legend = TLegend(0.7, 0.7, 0.9, 0.9)
   legend.AddEntry(Eprof, "Data", "l")
   Gprof = []
   for i, images in enumerate(gan_images):
      Gprof.append( TProfile("Gprof" +str(i), "Gprof" + str(i), 100, 0, 500))
      gsum = np.sum(images, axis=(1, 2, 3))
      gsf = gsum/Y
      for j in range(Y.shape[0]):
        Gprof[i].Fill(Y[j], gsf[j])
      color = color + 2
      Gprof[i].SetLineColor(color)
      Gprof[i].Draw('sames')
      c.Modified()
      sf_error = np.absolute((dsf-gsf)/dsf)
      glabel = 'GAN {}'.format(labels[i])
      if error:
         glabel = glabel + ' MRE={:.4f}'.format(np.mean(sf_error))
      legend.AddEntry(Gprof[i], glabel, "l")
      legend.Draw()
      c.Update()
   c.Print(out_file)
   print (' The plot is saved in: {}'.format(out_file))

def plotshapes(X, generated_images, x, y, z, energy, out_file, labels, log=0, p=[2, 500], norm=False, ifpdf=True, stest=True, error=True):
   canvas = ROOT.TCanvas("canvas" ,"" ,200 ,10 ,700 ,500) #make
   canvas.SetTitle('Weighted Histogram for energy deposition along x, y, z axis')
   canvas.SetGrid()
   color = 2
   canvas.Divide(2,2)
   array1x = np.sum(X, axis=(2,3))
   array1y = np.sum(X, axis=(1,3))
   array1z = np.sum(X, axis=(1,2))
   leg = ROOT.TLegend(0.1,0.4,0.9,0.9)
   leg.SetTextSize(0.06)
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
   if norm: h1x=my.normalize(h1x)
   h1x.Draw()
   h1x.Draw('sames hist')
   h1x.GetXaxis().SetTitle("Energy deposition along x axis")
   leg.AddEntry(h1x, 'G4',"l")
   canvas.cd(2)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1y, array1y)
   if norm: h1y=my.normalize(h1y)
   h1y.Draw()
   h1y.Draw('sames hist')
   h1y.GetXaxis().SetTitle("Energy deposition along y axis")
   canvas.cd(3)
   if log:
      ROOT.gPad.SetLogy()
   my.fill_hist_wt(h1z, array1z)
   if norm : h1z=my.normalize(h1z)
   h1z.Draw()
   h1z.Draw('sames hist')
   h1z.GetXaxis().SetTitle("Energy deposition along z axis")
   canvas.cd(4)
   canvas.Update()
   h2xs=[]
   h2ys=[]
   h2zs=[]
   for i, images in enumerate(generated_images):
      array2x = np.sum(images, axis=(2,3))
      array2y = np.sum(images, axis=(1,3))
      array2z = np.sum(images, axis=(1,2))
      errorx = np.divide(np.absolute(array1x-array2x), array1x, out=np.zeros_like(array1x), where=array1x!=0)
      errory = np.divide(np.absolute(array1y-array2y), array1y, out=np.zeros_like(array1y), where=array1y!=0)
      errorz = np.divide(np.absolute(array1z-array2z), array1z, out=np.zeros_like(array1z), where=array1z!=0)

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
      my.fill_hist_wt(h2x, array2x)
      if norm: h2x=my.normalize(h2x)
      h2x.Draw('sames')
      h2x.Draw('sames hist')
      canvas.Update()
      #my.stat_pos(h2x)
      if stest:
         res=np.array
         ks= h1x.KolmogorovTest(h2x, 'WW')
         #ch2 = h1x.Chi2Test(h2x, 'WW')
         glabel = "GAN {} X axis K= {}".format(labels[i], ks)
         leg.AddEntry(h2x, glabel,"l")
      canvas.Update()
      canvas.cd(2)
      my.fill_hist_wt(h2y, array2y)
      if norm: h2y=my.normalize(h2y)
      h2y.Draw('sames')
      h2y.Draw('sames hist')
      canvas.Update()
      #my.stat_pos(h2y)
      if stest:
         ks= h1y.KolmogorovTest(h2y, 'WW')
         #ch2 = h1y.Chi2Test(h2y, 'WW')
         glabel = "GAN {} Y axis K= {}".format(labels[i], ks)
         leg.AddEntry(h2y, glabel,"l")
      canvas.Update()
      canvas.cd(3)
      my.fill_hist_wt(h2z, array2z)
      if norm: h2z=my.normalize(h2z)
      h2z.Draw('sames')
      h2z.Draw('sames hist')
      canvas.Update()
      #my.stat_pos(h2z)
      canvas.Update()
      if stest:
         ks= h1z.KolmogorovTest(h2z, 'WW')
         #ch2 = h1z.Chi2Test(h2z, 'WW')
         glabel = "GAN {} Z axis K= {}".format(labels[i], ks)
         leg.AddEntry(h2z, glabel,"l")
      canvas.Update()
      color+=2
   canvas.Update()
   canvas.cd(4)
   leg.SetHeader("#splitline{Weighted Histograms for energies}{ deposited along x, y, z axis}", "C")
   if not stest:
      for i, h in enumerate(h2xs):
        glabel = 'GAN ' + labels[i]
        if error:
           tot_error = (np.mean(errorx) + np.mean(errory) + np.mean(errorz))/3.
           glabel = glabel + ' MRE {:.4f}'.format(np.mean(errorz))
        leg.AddEntry(h, glabel + labels[i],"l")     
   leg.Draw()
   canvas.Update()
   if ifpdf:
      canvas.Print(out_file + '.pdf')
   else:
      canvas.Print(out_file + '.C')

if __name__ == '__main__':
  main()
