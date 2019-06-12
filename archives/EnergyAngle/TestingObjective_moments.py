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
import utils.ROOTutils as my
if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a check for host can be used
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech
except:
    pass

import utils.GANutils as gan
sys.path.insert(0,'../')

def main():
    # All of the following needs to be adjusted
    from AngleArch3dGAN import generator # architecture
    weightdir = '3dgan_weights_gan_training/params_generator*.hdf5'
    if tlab:
      datapath = '/gkhattak/*Measured3ThetaEscan/*.h5'
      genpath = '/gkhattak/weights/' + weightdir
    else:
      datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*VarAngleMeas_*.h5" # path to data
      genpath = "../weights/" + weightdir # path to weights
    #datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
    
    sorted_path = 'Anglesorted'  # where sorted data is to be placed
    plotsdir = 'results/angle_optimization_test_gan_training_avg' # plot directory
    particle = "Ele" 
    scale = 1
    threshold = 0
    ang = 1
    concat=2
    power=0.85
    g= generator(latent_size=256)
    start = 0
    stop = 120
    gen_weights=[]
    disc_weights=[]
    fits = ['pol1', 'pol2', 'expo']
    moments = 2
    gan.safe_mkdir(plotsdir)
    for f in sorted(glob.glob(genpath)):
      gen_weights.append(f)
    gen_weights=gen_weights[:stop]
    epoch = []
    for i in np.arange(len(gen_weights)):
      name = os.path.basename(gen_weights[i])
      num = int(filter(str.isdigit, name)[:-1])
      epoch.append(num)
    print("{} weights are found".format(len(gen_weights)))
    result = GetResultsavg(metric, plotsdir, gen_weights, g, datapath, sorted_path, particle
                        , scale, power=power, thresh=threshold, ang=ang, concat=concat, m=moments,
            preproc = taking_power, postproc=inv_power
             )
    PlotResultsRoot2(result, plotsdir, start, epoch, fits, ang=ang)

def sqrt(n, scale=1):
    return np.sqrt(n * scale)

def square(n, scale=1):
    return np.square(n)/scale

def taking_power(n, xscale=1, power=1):
    return np.power(n * xscale, power)

def inv_power(n, xscale=1, power=1.):
    return np.power(n, 1./power)/xscale
        

def preproc(n, scale=1):
    return n * scale

def postproc(n, scale=1):
    return n/scale
        
#Plots results in a root file
def PlotResultsRoot(result, resultdir, start, epochs, fits, ang=1):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    c1.SetGrid ()
    legend = ROOT.TLegend(.5, .6, .9, .9)
    color = [2, 8, 4, 6, 7]
    minr = 100
    l=len(epochs)
    ep=np.zeros((l))
    res= np.zeros((l))
    minr_n = epochs[0]
    for i, epoch in enumerate(epochs):
        if result[i]< minr:
            minr = result[i]
            minr_n = epoch
        ep[i]=epoch
        res[i]=result[i]
    print(ep)
    print(res)
    gw  = ROOT.TGraph(l , ep, res )
    gw.SetLineColor(color[0])
    legend.AddEntry(gw, "ks loss = {:.4f} (epoch {})".format(minr, minr_n), "l")
    gw.SetTitle("ks loss: moments;Epochs;loss")
    gw.Draw('ALP')
    c1.Update()
    legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result.pdf"))

    fits = []#['pol1', 'pol2', 'expo']
    for i, fit in enumerate(fits):
        gw.Fit(fit)
        gw.GetFunction(fit).SetLineColor(color[i])
        gw.GetFunction(fit).SetLineStyle(2)
        legend.AddEntry(gt.GetFunction(fit), 'fit', "l")
        legend.Draw()
        c1.Update()
        c1.Print(os.path.join(resultdir, "result_{}.pdf".format(fit)))
    print ('The plot is saved to {}'.format(resultdir))

def PlotResultsRoot2(result, resultdir, start, epochs, fits, ang=1):
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
    color6 = 8
    color7 = 9
    
    num = len(result)
    print(num)
    avg = np.zeros((num))
    mx_e = np.zeros((num))
    my_e = np.zeros((num))
    mz_e = np.zeros((num))
    mx1_e  = np.zeros((num))
    my1_e  = np.zeros((num))
    mz1_e  = np.zeros((num))
    
    epoch = np.zeros((num))
    mint = 100
    minx = 100
    miny = 100
    minz = 100
    minx1 = 100
    miny1 = 100
    minz1 = 100
    
    for i, item in enumerate(result):
      epoch[i] = epochs[i]
      #avg[i]=item[0]
      mx_e[i]=item[0][0]
      my_e[i]=item[1][0]
      mz_e[i]=item[2][0]
      mx1_e[i]=item[0][1]
      my1_e[i]=item[1][1]
      mz1_e[i]=item[2][1]
      
      #if item[0]< mint:
      #   mint = item[0]
      #   mint_n = epoch[i]
      if item[0][0]< minx:
         minx = item[0][0]
         minx_n = epoch[i]
      if item[1][0]< miny:
         miny = item[1][0]
         miny_n = epoch[i]
      if item[2][0]< minz:
         minz = item[2][0]
         minz_n = epoch[i]

      if item[0][1]< minx1:
          minx1 = item[0][1]
          minx1_n = epoch[i]
      if item[1][1]< miny1:
          miny1 = item[1][1]
          miny1_n = epoch[i]
      if item[2][1]< minz1:
          minz1 = item[2][1]
          minz1_n = epoch[i]
    """
    gt  = ROOT.TGraph( num- start , epoch[start:], avg[start:] )
    gt.SetLineColor(color1)
    mg.Add(gt)
    legend.AddEntry(gt, "avg error min = {:.4f} (epoch {})".format(mint, mint_n), "l")
    """
    gex = ROOT.TGraph( num- start , epoch[start:], mx_e[start:] )
    gex.SetLineColor(color2)
    legend.AddEntry(gex, "X1 error min = {:.4f} (epoch {})".format(minx, minx_n), "l")
    mg.Add(gex)
    gey = ROOT.TGraph( num- start , epoch[start:], my_e[start:] )
    gey.SetLineColor(color3)
    legend.AddEntry(gey, "Y1 error min = {:.4f} (epoch {})".format(miny, miny_n), "l")
    mg.Add(gey)
    gez = ROOT.TGraph( num- start , epoch[start:], mz_e[start:] )
    gez.SetLineColor(color4)
    legend.AddEntry(gez, "Z1 error min = {:.4f} (epoch {})".format(minz, minz_n), "l")
    mg.Add(gez)

    gex1 = ROOT.TGraph( num- start , epoch[start:], mx1_e[start:] )
    gex1.SetLineColor(color2)
    gex1.SetLineStyle(2)
    legend.AddEntry(gex1, "X2 error min = {:.4f} (epoch {})".format(minx1, minx1_n), "l")
    mg.Add(gex1)
    gey1 = ROOT.TGraph( num- start , epoch[start:], my1_e[start:] )
    gey1.SetLineColor(color3)
    gey1.SetLineStyle(2)
    legend.AddEntry(gey1, "Y2 error min = {:.4f} (epoch {})".format(miny1, miny1_n), "l")
    mg.Add(gey1)
    gez1 = ROOT.TGraph( num- start , epoch[start:], mz1_e[start:] )
    gez1.SetLineColor(color4)
    gez1.SetLineStyle(2)
    legend.AddEntry(gez1, "Z2 error min = {:.4f} (epoch {})".format(minz1, minz1_n), "l")
    mg.Add(gez1)

    mg.SetTitle("Optimization function: KS Error on shower moment;Epochs;Error")
    mg.Draw('ALP')
    c1.Update()
    legend.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result.pdf"))
    
    gex.Draw('ALP')
    gex.SetTitle("Optimization function: KS Error on shower moment X1;Epochs;Error")
    legend1 = ROOT.TLegend(.5, .6, .9, .9)
    legend1.AddEntry(gex, "avg error min = {:.4f} (epoch {})".format(minx, minx_n), "l")
    c1.Update()
    legend1.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result_X1.pdf"))

    gey.Draw('ALP')
    gey.SetTitle("Optimization function: KS Error on shower moment Y1;Epochs;Error")
    legend2 = ROOT.TLegend(.5, .6, .9, .9)
    legend2.AddEntry(gey, "avg error min = {:.4f} (epoch {})".format(miny, miny_n), "l")
    c1.Update()
    legend2.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result_Y1.pdf"))

    gez.Draw('ALP')
    gez.SetTitle("Optimization function: KS Error on shower moment Z1;Epochs;Error")
    legend3 = ROOT.TLegend(.5, .6, .9, .9)
    legend3.AddEntry(gez, "avg error min = {:.4f} (epoch {})".format(minz, minz_n), "l")
    c1.Update()
    legend3.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result_Z1.pdf"))

    gex1.Draw('ALP')
    gex1.SetTitle("Optimization function: KS Error on shower moment X2;Epochs;Error")
    legend4 = ROOT.TLegend(.5, .6, .9, .9)
    legend4.AddEntry(gex1, "avg error min = {:.4f} (epoch {})".format(minx1, minx1_n), "l")
    c1.Update()
    legend4.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result_X2.pdf"))

    gey1.Draw('ALP')
    gey1.SetTitle("Optimization function: KS Error on shower moment Y2;Epochs;Error")
    legend5 = ROOT.TLegend(.5, .6, .9, .9)
    legend5.AddEntry(gey1, "avg error min = {:.4f} (epoch {})".format(miny1, miny1_n), "l")
    c1.Update()
    legend5.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result_Y2.pdf"))

    gez1.Draw('ALP')
    gez1.SetTitle("Optimization function: KS Error on shower moment Z2;Epochs;Error")
    legend6 = ROOT.TLegend(.5, .6, .9, .9)
    legend6.AddEntry(gez1, "avg error min = {:.4f} (epoch {})".format(minz1, minz1_n), "l")
    c1.Update()
    legend6.Draw()
    c1.Update()
    c1.Print(os.path.join(resultdir, "result_Z2.pdf"))
                                                                                                    
                                
    """
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
    """


# results are obtained using metric and saved to a log file
def GetResults(metric, resultdir, gen_weights, g, datapath, sorted_path, particle="Ele", scale=100, power=1, thresh=1e-6, ang=1, concat=1, m=2, preproc=preproc, postproc=postproc):
    resultfile = os.path.join(resultdir,  'result_log.txt')
    file = open(resultfile,'w')
    result = []
    for i in range(len(gen_weights)):
       if i==0:
         result.append(analyse(g, False,True, gen_weights[i], datapath, sorted_path, metric2, scale, power, particle, thresh=thresh, ang=ang, concat=concat, postproc=postproc, m=m)) # For the first time when sorted data is not saved we can make use opposite flags
       else:
         result.append(analyse(g, True, False, gen_weights[i], datapath, sorted_path, metric2, scale, power, particle, thresh=thresh, ang=ang, concat=concat, postproc=postproc, m=m))
       for r in result[i]:
          for item in r: file.write('{}\t'.format(item),)
       file.write('\n')
       #file.write('{}'.format(result[i]))
       
                         
    #print all results together at end                                                                               
    for i in range(len(gen_weights)):                                                                                            
       #print ('The results for ......',gen_weights[i])
       #print (" The result ={} ".format(result[i]))
       print ('\t'.join(str(r) for r in result[i]))
       print ('\n')
    file.close
    print ('The results are saved to {}.txt'.format(resultfile))
    return result

# If reduced data is to be used in analyse function the line:
#   var = ang.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)
# has to be replaced with:
#   var = ang.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, Data= GetAngleData_reduced, thresh=thresh)

# results are obtained using metric and saved to a log file
def GetResultsavg(metric, resultdir, gen_weights, g, datapath, sorted_path, particle="Ele", scale=100, power=1, thresh=1e-6, ang=1, concat=1, m=2, preproc=preproc, postproc=postproc):
    resultfile = os.path.join(resultdir,  'result_log.txt')
    file = open(resultfile,'w')
    result = []
    cumsum = []
    avg =[]
    for i in range(len(gen_weights)):
        if i==0:
            res = analyse(g, False,True, gen_weights[i], datapath, sorted_path, metric2, scale, power, particle, thresh=thresh, ang=ang, concat=concat, postproc=postproc, m=m)
            cumsum.append(res)
            result.append(res)
            avg.append(res)
            print('cumsum', cumsum)
        elif i<5:
            result.append(analyse(g, True, False, gen_weights[i], datapath, sorted_path, metric2, scale, power, particle, thresh=thresh, ang=ang, concat=concat, postproc=postproc, m=m))
            cumsum.append(np.sum(np.array(result), axis=0))
            print('cumsum', cumsum[i])
            print('dividing by...', i+1)
            print('cumsum', np.array(cumsum[i])/(i+1))
            avg.append(np.array(cumsum[i])/(i+1))
        else:
            res = analyse(g, True, False, gen_weights[i], datapath, sorted_path, metric2, scale, power, particle, thresh=thresh, ang=ang, concat=concat, postproc=postproc, m=m)
            result.append(res)
            cumsum.append(np.sum(np.array(result[i-5:]), axis=0))
            avg.append(np.array(cumsum[i])/5)
        print('Avg result', result[i])
        #for r in result[i]:
        #    for item in r: file.write('{}\t'.format(item),)
        #    file.write('\n')
    #file.close        
    #print all results together at end
    for i in range(len(gen_weights)):
        print ('The results for ......',gen_weights[i])
        print (" The result ={} ".format(result[i]))
        print (" The moving average ={} ".format(avg[i]))
    print('he results are saved to {}.txt'.format(resultfile))
    return avg
                                                                                                                 
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
def analyse(g, read_data, save_data, gen_weights, datapath, sorted_path, optimizer, xscale=100, power=1, particle="Ele", thresh=1e-6, ang=1, concat=1, m =2, preproc=preproc, postproc=postproc):
   print ("Started")
   num_events=2000
   num_data = 140000
   ascale = 1
   Test = True
   latent= 256
   var = {}
   energies = [110, 150, 190]
   #energies = [50, 100, 200, 300, 400]
   sorted_path= sorted_path 
   
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
       data_files = Trainfiles
     print(data_files)
     start = time.time()
     var = gan.get_sorted_angle(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)
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
        var["events_gan" + str(energy)] = gan.generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100], concat=concat, latent=latent, ang=ang) 
     var["events_gan" + str(energy)] = postproc(var["events_gan" + str(energy)], xscale=xscale, power=power)
   gen_time = time.time() - start
   print ("{} events were generated in {} seconds".format(total, gen_time))
   calc={}
   print("Weights are loaded in {}".format(gen_weights))
   for energy in energies:
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

def get_moment_error(array1, array2, dim, energy, m, f='ks'):
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
   hd = ROOT.TH1F("G4"+ dim + str(m)+str(energy), "", bins, minbin, maxbin)
   my.fill_hist(hd, array1)
   hd =my.normalize(hd, 1)
   hg=ROOT.TH1F("GAN"+ dim + str(m)+str(energy), "GAN"+ dim + str(m), bins, minbin, maxbin)
   my.fill_hist(hg, array2)
   hg =my.normalize(hg, 1)
   if f=='ks':
      res= hd.KolmogorovTest(hg)
   elif f=='chi2':
      res = hd.Chi2Test(hg)
   if res==0:
       res=1e-8
   return -1 * np.log(res)
   
def metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1, f='ks'):
   metricp = 0
   metrice = 0
   metrica = 0
   metrics = 0
   cut =0
   result=0
   for energy in energies:
     errorm =0
     for mom in np.arange(m):  
         errorx =  get_moment_error(var["momentX_act"+ str(energy)][:, mom], var["momentX_gan"+ str(energy)][:, mom], 'x', energy, m=mom, f=f)
         errory =  get_moment_error(var["momentY_act"+ str(energy)][:, mom], var["momentY_gan"+ str(energy)][:, mom], 'y', energy, m=mom, f=f)
         errorz =  get_moment_error(var["momentZ_act"+ str(energy)][:, mom], var["momentZ_gan"+ str(energy)][:, mom], 'z', energy, m=mom, f=f)
         errorm = errorm + (errorx + errory + errorz)/3
         print(errorx, errory, errorz)
     errorm = errorm/m
     result = result + errorm
   result= result/len(energies)
   print('Moment error={}'.format(result))
   return result

def metric2(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1, f='ks'):
   errorx = np.zeros(m)
   errory = np.zeros(m)
   errorz = np.zeros(m)
   res = []
   result=0
   for energy in energies:
       errorm =0
       for mom in np.arange(m):
           ex = get_moment_error(var["momentX_act"+ str(energy)][:, mom], var["momentX_gan"+ str(energy)][:, mom], 'x', energy, m=mom, f=f)
           ey = get_moment_error(var["momentY_act"+ str(energy)][:, mom], var["momentY_gan"+ str(energy)][:, mom], 'y', energy, m=mom, f=f)
           ez = get_moment_error(var["momentZ_act"+ str(energy)][:, mom], var["momentZ_gan"+ str(energy)][:, mom], 'z', energy, m=mom, f=f)
           errorx[mom] =  errorx[mom] + ex
           errory[mom] =  errory[mom] + ey
           errorz[mom] =  errorz[mom] + ez
           errorm = errorm + (ex + ey + ez)/3
           
       errorm = errorm/m
       result = result + errorm
   result= result/len(energies)
   for mom in np.arange(m):
       errorx[mom] = errorx[mom]/len(energies)
       errory[mom] = errory[mom]/len(energies)
       errorz[mom] = errorz[mom]/len(energies)
      
   print('Moment error={}'.format([errorx, errory, errorz]))
   return [errorx, errory, errorz]
                                                                                                                        
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
