#!/usr/bin/env python
# -*- coding: utf-8 -*-   

# This file tests the training results with the optimization function

from __future__ import print_function
import sys
import h5py
import os
import numpy as np
import glob
import numpy.core.umath_tests as umath
import time
import math
import ROOT
import argparse
from scipy import stats
from scipy.stats import wasserstein_distance as wass
if '.cern.ch' in os.environ.get('HOSTNAME'): # Here a check for host can be used
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech
except:
    pass

import utils.GANutils as gan
import utils.ROOTutils as my
sys.path.insert(0,'../')

def main():
   parser = get_parser()
   params = parser.parse_args()

   latent = params.latentsize
   datapath =params.datapath
   particle= params.particle
   angtype= params.angtype
   plotdir= params.plotdir
   sortdir= params.sortdir
   nbEvents= params.nbEvents
   binevents= params.binevents
   start = params.start
   stop = params.stop
   sl = params.sl
   ang= params.ang
   concat= params.concat
   test= params.test
   ifpdf= params.ifpdf
   grid = params.grid
   leg= params.leg
   statbox= params.statbox
   mono= params.mono
   weightdir= params.weightsdir
   xscale= params.xscale 
   yscale= params.yscale
   xpower = params.xpower
   thresh = params.thresh 
   dformat = params.dformat
   fits = params.fits if isinstance(params.fits, list) else [params.fits]
   opt = params.opt if isinstance(params.opt, list) else [params.opt]
   if ang:
     from AngleArch3dGAN import generator # architecture
     if not xscale:
       xscale=1.
     if not xpower:
       xpower = 0.85
     if not latent:
       latent = 256

   else:
     from EcalEnergyGan import generator
     if not xscale:
       xscale=100.
     if not xpower:
       xpower = 1.
     if not latent:
       latent =200
   
   if ang:
     if datapath=='reduced':
       datapath = "/storage/group/gpu/bigdata/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV
       events_per_file = 5000
       energies = [0, 110, 150, 190]
       dscale = 50.
     elif datapath=='full':
       datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
       events_per_file = 10000
       energies =[0, 100, 200, 300, 400]
       dscale = 50.
   else: 
     datapath = "/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5"
     events_per_file = 10000
     energies =[0, 100, 200, 300, 400]
     dscale = 1.
    
   if tlab:
     datapath = '/gkhattak/data/*Measured3ThetaEscan/*.h5'
   else:
     genpath =  weightdir +'/*generator*.hdf5'# path to weights
   sorted_path = 'Anglesorted'  # where sorted data is to be placed
   g= generator(latent_size=latent)
   
   gen_weights=[]
      
   gan.safe_mkdir(plotdir)

   for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
   
   gen_weights=gen_weights[start:stop]
   epoch = []
   for i in np.arange(len(gen_weights)):
      name = os.path.basename(gen_weights[i])
      num = int(filter(str.isdigit, name)[:-1])
      epoch.append(num)

   print("{} weights are found".format(len(gen_weights)))
   result = GetResults(opt, plotdir, gen_weights, g, datapath, sorted_path, particle
            , scale=xscale, power=xpower, thresh=thresh, ang=ang, concat=concat, latent=latent
            , energies=energies, nbEvents=nbEvents, eventsperfile=events_per_file, dscale=dscale
             )
   for i, op in enumerate(opt):
      r= result[i::len(opt)]
      PlotResultsRoot(r, op, plotdir, epoch, fits, sl, ang=ang)

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='full', help='HDF5 files to train from.')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle used.')
    parser.add_argument('--plotdir', action='store', type=str, default='results/best_epoch_wt_aux/', help='Directory to store the analysis plots.')
    parser.add_argument('--sortdir', action='store', type=str, default='SortedData', help='Directory to store sorted data.')
    parser.add_argument('--nbEvents', action='store', type=int, default=50000, help='Max limit for events used for Testing')
    parser.add_argument('--eventsperfile', action='store', type=int, default=5000, help='Number of events in a file')
    parser.add_argument('--binevents', action='store', type=int, default=10000, help='Number of events in each bin')
    parser.add_argument('--start', action='store', type=int, default=0, help='plot beginning from epoch')
    parser.add_argument('--stop', action='store', type=int, default=500, help='plot till epoch')
    parser.add_argument('--sl', action='store', type=int, default=10, help='select from last n epochs')
    parser.add_argument('--opt', default='mre', type=str, nargs='+', help='selction criterion: mre, chi2, ks')
    parser.add_argument('--ang', action='store', type=int, default=1, help='if inclusing angle angle')
    parser.add_argument('--concat', action='store', type=int, default=2, help='Modes related to combining conditions with latent 0)not cancatenated.. 1)concatenate angle...3) concatenate energy and angle')
    parser.add_argument('--test', action='store_false', default=True, help='Use Test data')
    parser.add_argument('--ifpdf', action='store_false', default=True, help='Whether generate pdf plots or .C plots')
    parser.add_argument('--grid', action='store_true', default=False, help='set grid')
    parser.add_argument('--leg', action='store_true', default=False, help='add legends')
    parser.add_argument('--statbox', action='store_true', default=False, help='add statboxes')
    parser.add_argument('--mono', action='store_true', default=False, help='changing line style as well as color for comparison')
    parser.add_argument('--weightsdir', action='store', type=str, default='../weights/3dgan_weights_wt_aux/', help='paths to Generator weights.')
    parser.add_argument('--xscale', action='store', type=int, help='Multiplication factors for cell energies')
    parser.add_argument('--yscale', action='store', help='Division Factor for Primary Energy.')
    parser.add_argument('--xpower', action='store', help='Power of cell energies')
    parser.add_argument('--thresh', action='store', type=int, default=0, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
    parser.add_argument('--fits', action='store', type=str, nargs='+', default=[], help='fits to use')
    return parser


def sqrt(n, scale=1):
    return np.sqrt(n * scale)

def square(n, scale=1):
    return np.square(n)/scale

def taking_power(n, xscale=1, power=1):
    return np.power(n * xscale, power)

def inv_power(n, xscale=1, power=1.):
    return np.power(n, 1./power)/xscale

#Plots results in a root file
def PlotResultsRoot(result, opt, resultdir, epochs, fits, sl=10, ang=1):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    #c1.SetGrid ()
    legend = ROOT.TLegend(.4, .7, .89, .89)
    legend.SetTextSize(0.028)
    legend.SetBorderSize(0)
    mg=ROOT.TMultiGraph()
    colors = [2, 8, 4, 6, 7]
    num = len(result)
    sf_e  = np.zeros((num))
    epoch = np.zeros((num))
      
    mins =1000
    mins_n =0
    j=0
    for i, item in enumerate(result):
      if item!=0:
        epoch[j] = epochs[i]       
        sf_e[j]=item
        j+=1
    if  num<sl: sl=num
    for i in range(num):
      if epoch[i] > num-sl:
        if sf_e[i] <= mins:
         mins = sf_e[i]
         mins_n = epoch[i]
                                         
    gs = ROOT.TGraph(j, epoch[:j], sf_e[:j])
    gs.SetLineColor(colors[0])
    gs.SetMarkerColor(colors[0])
    gs.SetMarkerStyle(5)
    mg.Add(gs)
    legend.AddEntry(gs, "min for epoch {}= {:.4f}".format(mins_n, mins), "l")
    c1.Update()
    if opt=='chi2' or opt=='ks':
       title = "-log[{}] for sampling fraction (selecting from last {} epochs);Epochs;{}"
    else:
       title = "{} for sampling fraction (selecting from last {} epochs);Epochs;{}"                    
    mg.SetTitle(title.format(opt, sl, opt))
    mg.Draw('ALP')
    mg.GetYaxis().SetRangeUser(0, 1.4 * np.amax(result))
    c1.Update()
    legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "{}_result.pdf".format(opt)))

    for i, fit in enumerate(fits):
      mg.SetTitle(title.format(opt, sl, opt))  
      
      gs.Fit(fit)
      gs.GetFunction(fit).SetLineColor(colors[0])
      gs.GetFunction(fit).SetLineStyle(2)

      if i == 0:
        legend.AddEntry(gs.GetFunction(fit), '{} ({} fit)'.format(opt, fit), "l")
      legend.Draw()
      c1.Update()
      c1.Print(os.path.join(resultdir, "{}_result_{}.pdf".format(opt, fit)))
    print ('The plot is saved to {}'.format(resultdir))

def preproc(n, scale=1):
    return n * scale

def postproc(n, scale=1):
    return n/scale
        
# results are obtained using metric and saved to a log file
def GetResults(opt, resultdir, gen_weights, g, datapath, sorted_path, particle="Ele", scale=100, power=1, thresh=1e-6, ang=1, concat=1, latent = 256, energies=[110, 150, 190], nbEvents=100000, eventsperfile=5000, dscale=1.):
    resultfile = os.path.join(resultdir,  'result_log.txt')
    file = open(resultfile,'w')
    n = len(opt)
    result = analyse(g, False,True, gen_weights, datapath, sorted_path, opt, scale, power, particle, 
                     thresh=thresh, ang=ang, concat=concat, latent=latent, energies=energies,
                     nbEvents= nbEvents, eventsperfile=eventsperfile, dscale=dscale)
    for i, r in enumerate(result):
       if i !=0 and i % n==0:
         file.write('\n')
       file.write(str(r)+'\t')
           
    #print all results together at end                                                                               
    for i in range(len(gen_weights)):                                                                                            
       print ('The results for ......',gen_weights[i])
       reslog = " The result = "
       print (reslog, result[n*i:n*i + n])
    file.close
    print ('The results are saved to {}.txt'.format(resultfile))
    return result

def GetAngleData_reduced(datafile, thresh=1e-6):
    #get data for training
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))[:, 13:38, 13:38, :]
    Y=np.array(f.get('energy'))
    eta = np.array(f.get('eta')) + 0.6
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, eta, ecal

def preproc(n, scale=1):
    return n * scale

def postproc(n, scale=1):
    return n/scale

# This function will calculate two errors derived from position of maximum along an axis and the sum of ecal along the axis
def analyse(g, read_data, save_data, gen_weights, datapath, sorted_path, optimizer, xscale=100, power=1, particle="Ele", thresh=1e-6, ang=1, latent = 256, concat=1, energies=[110, 150, 190], nbEvents=100000, eventsperfile=5000, dscale=1.):
   print ("Started")
   
   Test = False
   m = 2
   var = {}
   sorted_path= sorted_path 
   total=0   
   Trainfiles, Testfiles = gan.DivideFiles(datapath, Fractions=[.9,.1], datasetnames=["ECAL"], Particles =[particle])
   if Test:
     data_files = Testfiles
   else:
     data_files = Trainfiles
   nb_files = int(math.ceil(nbEvents/eventsperfile))
   data_files = data_files[:nb_files]
   print(data_files)
   start = time.time()
   if ang:
     var = gan.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)     
   else:
     var = gan.get_sorted(data_files, energies, flag=False, num_events1=20000, num_events2=2000, thresh=thresh)
   data_time = time.time() - start
   print ("The events were loaded in {} seconds".format(data_time))
   result = []
   for energy in energies:
     var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
     var["events_act" + str(energy)]=  np.squeeze(var["events_act" + str(energy)])/dscale
     total += var["index" + str(energy)]
     data_time = time.time() - start
   print ("{} events were put in {} bins".format(total, len(energies)))
   for gen_weight in gen_weights:
     g.load_weights(gen_weight)            
     start = time.time()
     print("Weights are loaded from {}".format(gen_weight))
     for energy in energies:
       if ang:
          angle = measPython(var["events_act"+ str(energy)])
          var["events_gan" + str(energy)] = gan.generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100, var["angle" + str(energy)] ], concat=concat, latent=latent)
          var["events_gan" + str(energy)] = np.power(var["events_gan" + str(energy)], 1./power)
       else:
          var["events_gan" + str(energy)] = gan.generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100], concat=concat, latent=latent) 
          var["events_gan" + str(energy)] = var["events_gan" + str(energy)]/xscale

       var["events_gan" + str(energy)]=  np.squeeze(var["events_gan" + str(energy)])/dscale
                     
       gen_time = time.time() - start
       ecal_act = np.sum(var["events_act"+ str(energy)], axis=(1, 2, 3))
       ecal_gan = np.sum(var["events_gan"+ str(energy)], axis=(1, 2, 3))

       var["sf_act" + str(energy)] = np.divide(ecal_act, var["energy"+ str(energy)])
       var["sf_gan" + str(energy)] = np.divide(ecal_gan, var["energy"+ str(energy)])

     print ("{} events were generated in {} seconds".format(total, gen_time))
     for opt in optimizer:
       if opt == 'mre':
         res = mre(var, energies[1:], ang=ang)
       elif opt == 'chi2':
         res = stat_test(var, [energies[0]], ang=ang, test=opt, p=[])
       elif opt == 'ks':
         res = stat_test(var, [energies[0]], ang=ang, test=opt, p=[])
       elif opt == 'wass':
         if 0 in energies:
           res = wass( var["sf_act0"],  var["sf_act0"])
       print('{} error = {}'.format(opt, res))  
       result.append(res)
   return result                                        
 
def mre(var, energies, ang=1):
   metrics = 0
   for energy in energies:
     #Relative error on mean moment value for each moment and each axis
     mean_sf_g4 = np.mean(var["sf_act"+ str(energy)])
     mean_sf_gan = np.mean(var["sf_gan"+ str(energy)])
     sf_error = np.divide(np.absolute(mean_sf_g4 - mean_sf_gan), mean_sf_g4)
     metrics +=sf_error
   metrics = metrics/len(energies)
   return metrics

def stat_test(var, energies, ang, bins=[10, 5], p=[100, 300], r = [0.018, 0.022], test='chi2', d=2):
    error=0
    for energy in energies:
      if len(p)!=2:
        pmin = np.amin(var["energy"+ str(energy)])
        pmax = np.amax(var["energy"+ str(energy)])
      else:
        pmin = p[0]
        pmax = p[1]
      if d==2:
        #r[0] = min(np.amin(var["sf_act"+ str(energy)]), np.amin(var["sf_act"+ str(energy)]))
        #r[1] = max(np.amax(var["sf_act"+ str(energy)]), np.amax(var["sf_act"+ str(energy)]))
        g4hist = ROOT.TH2F('g4hist', 'g4hist', bins[0], pmin, pmax, bins[1], r[0], r[1])
        ganhist = ROOT.TH2F('ganhist', 'ganhist', bins[0], pmin, pmax, bins[1], r[0], r[1])
        g4hist.Sumw2()
        ganhist.Sumw2()
        my.fill_hist_2D(g4hist, var["energy"+ str(energy)], var["sf_act"+ str(energy)])
        my.fill_hist_2D(ganhist, var["energy"+ str(energy)], var["sf_gan"+ str(energy)])

      else:
        g4hist = ROOT.TH1F('g4hist', 'g4hist', bins[1], r[0], r[1])
        ganhist = ROOT.TH1F('ganhist', 'ganhist', bins[1], r[0], r[1])
        g4hist.Sumw2()
        ganhist.Sumw2()
        print(var["sf_act"+ str(energy)][:10])
        print(var["sf_gan"+ str(energy)][:10])      
        my.fill_hist(g4hist, var["sf_act"+ str(energy)])
        my.fill_hist(ganhist, var["sf_gan"+ str(energy)])
      
      if test == 'ks':
        serror=g4hist.KolmogorovTest(ganhist, 'UU')
        print(test, serror)
      if test == 'chi2':
        serror = g4hist.Chi2Test(ganhist, 'UU')
        print(test, serror)
      if serror> 0:
        serror = -1 * np.log10(serror)
      
      error+=serror
    return error/len(energies)


def measPython(image): # Working version:p1 and p2 are not used. 3D angle with barycenter as reference point
    image = np.squeeze(image)
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

    #ang = np.sum(ang, axis=1)/zunmasked_events #mean
    ang = ang * z # weighted by position
    sumz_tot = z * zmask
    ang = np.sum(ang, axis=1)/np.sum(sumz_tot, axis=1)

    indexes = np.where(amask>0)
    ang[indexes] = 100.
    return ang

if __name__ == "__main__":
   main()
