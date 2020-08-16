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
import keras.backend as K
K.set_image_data_format('channels_last')
if 'tlab' in os.environ.get('HOSTNAME'): # Here a check for host can be used
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech
except:
    pass
sys.path.insert(0,'../')
import analysis.utils.GANutils as gan

def main():
    # All of the following needs to be adjusted
    from analysis.EcalEnergyGan2 import generator # architecture
    weightdir = '3dgan_weightsfixed_angle/params_generator*.hdf5'
    if tlab:
      #datapath = '/eos/project/d/dshep/LCD/V1/*scan/*.h5'
      datapath = '/eos/user/g/gkhattak/FixedAngleData/*.h5'
      genpath = '/gkhattak/weights/EnergyWeights/' + weightdir
      genpath = '/afs/cern.ch/work/g/gkhattak/3DGan_weights_surfsara/results_8276_omp27_8k_256n/*.hdf5'
    else:
      datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*VarAngleMeas_*.h5" # path to data
      genpath = "../weights/" + weightdir # path to weights
    datapath = '/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5'
    sorted_path = 'Anglesorted'  # where sorted data is to be placed
    plotsdir = 'results/optimization_fixed' # plot directory
    particle = "Ele"
    dformat= 'channels_last'
    K.set_image_data_format(dformat) # setting global format flag
    scale = 100 # scaling fatcor for energies
    threshold = 0 # threshold for cell energies
    ang = 0 # angle not included
    concat=1 # effective only for variable angle 
    power=1 # effective only for variable angle
    latent=200
    g= generator(latent_size=latent, dformat=dformat)
    start = 0
    stop = 50
    gen_weights=[]
    disc_weights=[]
    fits = ['pol1', 'pol2', 'expo']
    gan.safe_mkdir(plotsdir)
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
    epoch = []
    for i in np.arange(len(gen_weights)):
      name = os.path.basename(gen_weights[i])
      num = int(filter(str.isdigit, name)[:-1])
      epoch.append(num)
    print("{} weights are found".format(len(gen_weights)))
    result = GetResults(metric, plotsdir, gen_weights, g, datapath, sorted_path, particle
                        , scale, power=power, thresh=threshold, ang=ang, concat=concat
            , preproc = taking_power, postproc=inv_power, latent=latent, dformat=dformat
             )
    PlotResultsRoot(result, plotsdir, start, epoch, fits, ang=ang)

def sqrt(n, scale=1):
    return np.sqrt(n * scale)

def square(n, scale=1):
    return np.square(n)/scale

def taking_power(n, xscale=1, power=1):
    return np.power(n * xscale, power)

def inv_power(n, xscale=1, power=1.):
    return np.power(n, 1./power)/xscale
        

#Plots results in a root file
def PlotResultsRoot(result, resultdir, start, epochs, fits, ang=1):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    c1.SetGrid ()
    legend = ROOT.TLegend(.5, .6, .9, .9)
    legend.SetTextSize(0.028)
    mg=ROOT.TMultiGraph()
    color1 = 2
    color2 = 8
    color3 = 4
    color4 = 6
    color5 = 7
    num = len(result)
    print(num)
    total = np.zeros((num))
    energy_e = np.zeros((num))
    mmt_e = np.zeros((num))
    ang_e = np.zeros((num))
    sf_e  = np.zeros((num))
    epoch = np.zeros((num))
      
    mine = 100
    minm = 100
    mint = 100
    mina = 100
    mins = 100
    for i, item in enumerate(result):
      epoch[i] = epochs[i]  
      total[i]=item[0]
      mmt_e[i]=item[1]
      energy_e[i]=item[2]
      sf_e[i]=item[3]
      if item[0]< mint:
         mint = item[0]
         mint_n = epoch[i]
      if item[1]< minm:
         minm = item[1]
         minm_n = epoch[i]
      if item[2]< mine:
         mine = item[2]
         mine_n = epoch[i]
      if item[3]< mins:
         mins = item[3]
         mins_n = epoch[i]
      if ang:
         ang_e[i]=item[4]
         if item[4]< mina:
           mina = item[4]
           mina_n = epoch[i]
                               
    gt  = ROOT.TGraph( num- start , epoch[start:], total[start:] )
    gt.SetLineColor(color1)
    mg.Add(gt)
    legend.AddEntry(gt, "Total error min = {:.4f} (epoch {})".format(mint, mint_n), "l")
    ge = ROOT.TGraph( num- start , epoch[start:], energy_e[start:] )
    ge.SetLineColor(color2)
    legend.AddEntry(ge, "Energy error min = {:.4f} (epoch {})".format(mine, mine_n), "l")
    mg.Add(ge)
    gm = ROOT.TGraph( num- start , epoch[start:], mmt_e[start:])
    gm.SetLineColor(color3)
    mg.Add(gm)
    legend.AddEntry(gm, "Moment error  = {:.4f} (epoch {})".format(minm, minm_n), "l")
    c1.Update()
    gs = ROOT.TGraph( num- start , epoch[start:], sf_e[start:])
    gs.SetLineColor(color4)
    mg.Add(gs)
    legend.AddEntry(gs, "Sampling Fraction error  = {:.4f} (epoch {})".format(mins, mins_n), "l")
    c1.Update()
                    
    if ang:
      ga = ROOT.TGraph( num- start , epoch[start:], ang_e[start:])
      ga.SetLineColor(color5)
      mg.Add(ga)
      legend.AddEntry(ga, "Angle error  = {:4f} (epoch {})".format(mina, mina_n), "l")
      c1.Update()
                    
    mg.SetTitle("Optimization function: Mean Relative Error on shower shapes, moment and sampling fraction;Epochs;Error")
    mg.Draw('ALP')
    c1.Update()
    legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result.pdf"))

    fits = ['pol1', 'pol2', 'expo']
    for i, fit in enumerate(fits):
      mg.SetTitle("Optimization function: Mean Relative Error on shower sahpes, moments and sampling fraction({} fit);Epochs;Error".format(fit))  
      gt.Fit(fit)
      gt.GetFunction(fit).SetLineColor(color1)
      gt.GetFunction(fit).SetLineStyle(2)
    
      ge.Fit(fit)
      ge.GetFunction(fit).SetLineColor(color2)
      ge.GetFunction(fit).SetLineStyle(2)
            
      gm.Fit(fit)
      gm.GetFunction(fit).SetLineColor(color3)
      gm.GetFunction(fit).SetLineStyle(2)

      gs.Fit(fit)
      gs.GetFunction(fit).SetLineColor(color4)
      gs.GetFunction(fit).SetLineStyle(2)

      if i == 0:
        legend.AddEntry(gt.GetFunction(fit), 'Total fit', "l")
        legend.AddEntry(ge.GetFunction(fit), 'Energy fit', "l")
        legend.AddEntry(gm.GetFunction(fit), 'Moment fit', "l")  
        legend.AddEntry(gs.GetFunction(fit), 'S. Fr. fit', "l")
      legend.Draw()
      c1.Update()
      c1.Print(os.path.join(resultdir, "result_{}.pdf".format(fit)))
    print ('The plot is saved to {}'.format(resultdir))

def preproc(n, scale=1):
    return n * scale

def postproc(n, scale=1):
    return n/scale
        
# results are obtained using metric and saved to a log file
def GetResults(metric, resultdir, gen_weights, g, datapath, sorted_path, particle="Ele", scale=100, power=1, thresh=1e-6, ang=1, concat=1, preproc=preproc, postproc=postproc, latent=256, dformat='channel_last'):
    resultfile = os.path.join(resultdir,  'result_log.txt')
    file = open(resultfile,'w')
    result = []
    for i in range(len(gen_weights)):
       if i==0:
         result.append(analyse(g, False,True, gen_weights[i], datapath, sorted_path, metric, scale, power, particle, thresh=thresh, ang=ang, concat=concat, postproc=postproc, latent=latent, dformat=dformat)) # For the first time when sorted data is not saved we can make use opposite flags
       else:
         result.append(analyse(g, True, False, gen_weights[i], datapath, sorted_path, metric, scale, power, particle, thresh=thresh, ang=ang, concat=concat, postproc=postproc, latent=latent, dformat=dformat))
       #file.write(len(result[i]) * '{:.4f}\t'.format(*result[i]))
       file.write('\t'.join(str(r) for r in result[i]))
       file.write('\n')
                  
    #print all results together at end                                                                               
    for i in range(len(gen_weights)):                                                                                            
       print ('The results for ......',gen_weights[i])
       print (" The result for {} = ",)
       print ('\t'.join(str(r) for r in result[i]))
       print ('\n')
    file.close
    print ('The results are saved to {}.txt'.format(resultfile))
    return result

# If reduced data is to be used in analyse function the line:
#   var = ang.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)
# has to be replaced with:
#   var = ang.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, Data= GetAngleData_reduced, thresh=thresh)

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
def analyse(g, read_data, save_data, gen_weights, datapath, sorted_path, optimizer, xscale=100, power=1, particle="Ele", thresh=1e-6, ang=1, concat=1, preproc=preproc, postproc=postproc, latent=256, dformat='channel_last'):
   print ("Started")
   num_events=2000
   num_data = 140000
   ascale = 1
   Test = True
   m = 2
   var = {}
   energies = [50, 100, 200, 300, 400]
   sorted_path= sorted_path 
   #g =generator(latent)
   if read_data:
     start = time.time()
     var = gan.load_sorted(sorted_path + "/*.h5", energies, ang = ang)
     sort_time = time.time()- start
     print ("Events were loaded in {} seconds".format(sort_time))
   else:
     Trainfiles, Testfiles = gan.DivideFiles(datapath, Fractions=[.9,.1], datasetnames=["ECAL"], Particles =[particle])
     if Test:
       data_files = Testfiles
     else:
       data_files = Trainfiles + Testfiles
     start = time.time()
     #energies = [50, 100, 200, 250, 300, 400, 500]
     var = gan.get_sorted(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)
     data_time = time.time() - start
     print ("{} events were loaded in {} seconds".format(num_data, data_time))
     if save_data:
        gan.save_sorted(var, energies, sorted_path, ang=ang)        
   total = 0
   for energy in energies:
     var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
     total += var["index" + str(energy)]
     data_time = time.time() - start
   print ("{} events were put in {} bins".format(total, len(energies)))
   g.load_weights(gen_weights)
              
   start = time.time()
   for energy in energies:
     if ang:
        var["events_gan" + str(energy)] = gan.generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100, var["angle" + str(energy)] * ascale], concat=concat, latent=latent)
     else:
        var["events_gan" + str(energy)] = gan.generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100], concat=concat, latent=latent) 
     var["events_gan" + str(energy)] = postproc(var["events_gan" + str(energy)], xscale=xscale, power=power)
   gen_time = time.time() - start
   print ("{} events were generated in {} seconds".format(total, gen_time))
   calc={}
   print("Weights are loaded in {}".format(gen_weights))
   for energy in energies:
     var["events_act" + str(energy)]= np.squeeze(var["events_act" + str(energy)])
     var["events_gan" + str(energy)]= np.squeeze(var["events_gan" + str(energy)])
     x = var["events_act" + str(energy)].shape[1]
     y = var["events_act" + str(energy)].shape[2]
     z = var["events_act" + str(energy)].shape[3]
     var["ecal_act"+ str(energy)] = np.sum(var["events_act" + str(energy)], axis = (1, 2, 3))
     var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
     calc["sumsx_act"+ str(energy)], calc["sumsy_act"+ str(energy)], calc["sumsz_act"+ str(energy)] = gan.get_sums(var["events_act" + str(energy)])
     calc["sumsx_gan"+ str(energy)], calc["sumsy_gan"+ str(energy)], calc["sumsz_gan"+ str(energy)] = gan.get_sums(var["events_gan" + str(energy)])
     calc["momentX_act" + str(energy)], calc["momentY_act" + str(energy)], calc["momentZ_act" + str(energy)]= gan.get_moments(calc["sumsx_act"+ str(energy)], calc["sumsy_act"+ str(energy)], calc["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
     calc["momentX_gan" + str(energy)], calc["momentY_gan" + str(energy)], calc["momentZ_gan" + str(energy)] = gan.get_moments(calc["sumsx_gan"+ str(energy)], calc["sumsy_gan"+ str(energy)], calc["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
     if ang:
        calc["mtheta_act"+ str(energy)]= measPython(var["events_act" + str(energy)])
        calc["mtheta_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])

     calc["sf_act" + str(energy)] = np.divide(var["ecal_act"+ str(energy)], var["energy"+ str(energy)])
     calc["sf_gan" + str(energy)] = np.divide(var["ecal_gan"+ str(energy)], var["energy"+ str(energy)])
   return optimizer(calc, energies, m, x, y, z, ang=ang)                                        
 
def metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1):
   metricp = 0
   metrice = 0
   metrica = 0
   metrics = 0
   cut =0
   for energy in energies:
     #Relative error on mean moment value for each moment and each axis
     x_act= np.mean(var["momentX_act"+ str(energy)], axis=0)
     x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
     y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
     y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
     z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
     z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
     var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act
     var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
     var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
     #Taking absolute of errors and adding for each axis then scaling by 3
     var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
     #Summing over moments and dividing for number of moments
     var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
     metricp += var["pos_total"+ str(energy)]
     #Take profile along each axis and find mean along events
     sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis=0), np.mean(var["sumsz_act" + str(energy)], axis=0)
     sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
     var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
     var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
     var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
     #Take absolute of error and mean for all events
     var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)][cut:x-cut]))/(x-(2*cut))
     var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)][cut:y-cut]))/(y-(2*cut))
     var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/z
     
     var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
     metrice += var["eprofile_total"+ str(energy)]
     if ang:
        var["angle_error"+ str(energy)] = np.mean(np.absolute((var["mtheta_act" + str(energy)] - var[ "mtheta_gan" + str(energy)])/var["mtheta_act" + str(energy)]))
        metrica += var["angle_error"+ str(energy)]
     #sampling fraction relative error
     real_SF = np.mean(var["sf_act"+ str(energy)])
     fake_SF = np.mean(var["sf_gan"+ str(energy)])
     sf_error = np.absolute((real_SF - fake_SF)/ real_SF)
     metrics +=sf_error
   metricp = metricp/len(energies)
   metrice = metrice/len(energies)
   metrics = metrics/len(energies)
   if ang:metrica = metrica/len(energies)
   tot = metricp + metrice + metrics
   if ang:tot = tot +metrica
   result = [tot, metricp, metrice, metrics]
   if ang: result.append(metrica)
   print("Result =", result)
   return result

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
