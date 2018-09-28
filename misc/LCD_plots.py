import h5py
import numpy as np
import setGPU
import math
import sys
import ROOT
import os
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/')
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/analysis')

from utils.GANutils import perform_calculations_angle, safe_mkdir  # to calculate different Physics quantities
import utils.ROOTutils as my # common utility functions for root

def main():
    #Architecture
    from AngleArch3dGAN_sqrt import generator, discriminator

    #Weights
    disc_weight1="../weights/3Dweights_1loss_50weight_sqrt/params_discriminator_epoch_040.hdf5"
    gen_weight1= "../weights/3Dweights_1loss_50weight_sqrt/params_generator_epoch_040.hdf5"

    #Path to store results
    plotsdir = "results/LCD_paper_plots_w50_updated/"
    safe_mkdir(plotsdir)
    #Parameters
    latent = 256 # latent space
    num_data = 100000
    num_events = 5000
    events_per_file = 5000
    m = 3  # number of moments
    nloss= 4 # total number of losses...4 or 5
    concat = 1 # if concatenting angle to latent space
    cell=1 # if making plots for cell energies. Exclude for quick plots.
    corr=0 # if making correlation plots
    energies=[0, 110, 150, 190] # energy bins
    thetas = [math.radians(x) for x in [90, 62, 118]] # angle bins
    aindexes = [0, 1, 2] # numbers corressponding to different angle bins
    angtype = 'theta'# the angle data to be read from file
    particle='Ele'# partcile type
    thresh=0 # Threshold for ecal energies
    datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path

    sortdir = 'SortedAngleData'  # if saving sorted data
    gendir = 'SortedAngleGen'  # if saving generated events
    discdir = 'SortedAngleDisc' # if saving disc outputs

    Test = True # use test data
    stest = False # K and chi2 test

    #following flags are used to save sorted and GAN data and to load from sorted data. These are used while development and should be False for one time analysis
    save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
    read_data = False # True if loading previously sorted data
    save_gen =  False # True if saving generated data.
    read_gen = False # True if generated data is already saved and can be loaded
    save_disc = False # True if discriminiator data is to be saved
    read_disc =  False # True if discriminated data is to be loaded from previously saved file
    ifpdf = True # True if pdf are required. If false .C files will be generated

    flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
    dweights = [disc_weight1]#, disc_weight2]
    gweights = [gen_weight1]#, gen_weight2]
    xscales = [1]#, 1]
    ascales = [1]#, 1]
    labels = ['']#, 'epoch 40']
    d = discriminator()
    g = generator(latent)
    var= perform_calculations_angle(g, d, gweights, dweights, energies, thetas,
                                    aindexes, datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales,
                                    ascales, flags, latent, events_per_file, particle, thresh=thresh, angtype=angtype, offset=0.0,
                                    nloss=nloss, concat=concat
                                    , pre =sqrt, post =square  # Adding other preprocessing, Default is simple scaling
                                    )
    for energy in energies:
        edir = os.path.join(plotsdir, 'energy_{}'.format(energy))
        safe_mkdir(edir)
        sumx_act=[]
        sumy_act=[]
        sumz_act=[]
        sumx_gan=[]
        sumy_gan=[]
        sumz_gan=[]
        ecal_act=[]
        ecal_gan=[]
        energy_act =[]
        if energy == 0:
            p =[100, 200]
        else:
            p = [np.amin(var["energy"+ str(energy)]), np.amax(var["energy"+ str(energy)])]
        for i, theta in enumerate(thetas):
           sumx_act.append(var["sumsx_act"+ str(energy) + "ang_" + str(i)])
           sumy_act.append(var["sumsy_act"+ str(energy) + "ang_" + str(i)])
           sumz_act.append(var["sumsz_act"+ str(energy) + "ang_" + str(i)])
           sumx_gan.append(var["sumsx_gan"+ str(energy) + "ang_" + str(i)]['n_0'])
           sumy_gan.append(var["sumsy_gan"+ str(energy) + "ang_" + str(i)]['n_0'])
           sumz_gan.append(var["sumsz_gan"+ str(energy) + "ang_" + str(i)]['n_0'])
           ecal_act.append(var["ecal_act"+ str(energy) + "ang_" + str(i)]['n_0'])
           ecal_gan.append(var["ecal_gan"+ str(energy) + "ang_" + str(i)]['n_0'])
           energy_act.append(var["energy"+ str(energy) + "ang_" + str(i)])
           print(theta, var["energy"+ str(energy) + "ang_" + str(i)].shape[0])
        PlotEnergyHistY2(sumy_act, sumy_gan, os.path.join(edir, 'angle_histy3_{}'.format(energy)), energy, thetas)
        PlotEnergyHistY2(sumy_act, sumy_gan, os.path.join(edir, 'angle_histy3_log{}'.format(energy)), energy, thetas, log=1)
        plot_ecal_ratio_profile(ecal_act, ecal_gan, energy_act, thetas, os.path.join(edir, 'ecal_{}'.format(energy)), p=p, ifpdf=True)

def sqrt(n, scale=1):
    return np.sqrt(n * scale)

def square(n, scale=1):
    return np.square(n)/scale

def PlotEnergyHistY(sumy_act, sumy_gan, out_file, energy, thetas, log=0, ifC=False):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,700 ,500) #make
    canvas.SetGrid()
    if log:
       ROOT.gPad.SetLogy()
    label = "Weighted Histograms for {} GeV".format(energy)
    color = 2
    leg = ROOT.TLegend(0.7,0.6,0.9,0.9)
    #leg.SetTextSize(0.05)
    h1y=[]
    h2y=[]
    for i, theta in enumerate(thetas):
        title = label + " for{} theta".format(theta)
        theta = int(np.degrees(theta))
        num = sumy_act[i].shape[0]
        print(theta, num)
        y=sumy_act[0].shape[1]
        h1y.append(ROOT.TH1F('G4y{:d}theta_{:d}GeV'.format(theta, energy), '', y, 0, y))
        h2y.append(ROOT.TH1F('GANy{:d}theta_{:d}GeV'.format(theta, energy), '', y, 0, y))
    
        h1y[i].SetLineColor(color)
        h2y[i].SetLineColor(color)
        h1y[i].SetLineWidth(2)  
        my.fill_hist_wt(h1y[i], sumy_act[i])
        h2y[i].SetLineStyle(2)
        
        my.fill_hist_wt(h2y[i], sumy_gan[i])
        h1y[i].SetStats(0)
        h2y[i].SetStats(0)
        h1y[i]=my.normalize(h1y[i], 1)
        h2y[i]=my.normalize(h2y[i], 1)
        if i==0:
           h1y[i].GetXaxis().SetTitle("Energy deposition along Y axis")
           h1y[i].GetYaxis().SetTitle("count/integral")
           h1y[i].GetYaxis().CenterTitle()
           h1y[i].SetTitle(title)
           #h1y[i].GetYaxis().SetRangeUser(0, 1.2)
        
        if i ==0:
            h1y[i].Draw('')
            h1y[i].Draw('sames hist')
            canvas.Update()
        else:
            h1y[i].Draw('sames')
            h1y[i].Draw('sames hist')
        h2y[i].Draw('sames')
        h2y[i].Draw('sames hist')
                    
        leg.AddEntry(h1y[i], 'G4   {}\circ'.format(theta),"l")
        leg.AddEntry(h2y[i], 'GAN {}\circ'.format(theta),"l")
        
        color+=2
        canvas.Update()
    leg.Draw()
    canvas.Update()
    canvas.Print(out_file + '.pdf')
    if ifC:
       canvas.Print(out_file + '.C')
                                                                     
def PlotEnergyHistY2(sumy_act, sumy_gan, out_file, energy, thetas, log=0, ifC=False, p=[100, 200]):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,700 ,500) #make
    canvas.SetGrid()
    label = "Weighted Histograms for {} GeV".format(energy)
    leg = ROOT.TLegend(0.1,0.6,0.9,0.9)
    legs=[]
    canvas.Divide(2,2)
    #leg.SetTextSize(0.05)
    h1y=[]
    h2y=[]
    if len(thetas)>3:
        thetas=thetas[:3]
        print('Histogram will only be plotted for {:.2f} degrees'.format(thetas))
        
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        title = label + " & {} theta".format(theta)
        num = sumy_act[i].shape[0]
        canvas.cd(i+1)
        if log:
            ROOT.gPad.SetLogy()
                    
        y=sumy_act[0].shape[1]
        h1y.append(ROOT.TH1F('G4y{:d}theta_{:d}GeV'.format(theta, energy), '', y, 0, y))
        h2y.append(ROOT.TH1F('GANy{:d}theta_{:d}GeV'.format(theta, energy), '', y, 0, y))
        legs.append(ROOT.TLegend(0.1,0.6,0.2,0.9))
        h1y[i].SetLineColor(2)
        h2y[i].SetLineColor(4)
        legs[i].SetHeader('{} degrees'.format(theta))
        my.fill_hist_wt(h1y[i], sumy_act[i])
        #h2y[i].SetLineStyle(2)

        my.fill_hist_wt(h2y[i], sumy_gan[i])
        h1y[i].SetStats(0)
        h2y[i].SetStats(0)
        h1y[i]=my.normalize(h1y[i], 1)
        h2y[i]=my.normalize(h2y[i], 1)
        h1y[i].GetXaxis().SetTitle("Energy deposition along Y axis")
        h1y[i].GetYaxis().SetTitle("count/integral")
        h1y[i].GetYaxis().CenterTitle()
        h1y[i].GetYaxis().SetTitleOffset()
        h1y[i].GetYaxis().SetTitleSize(0.05)
        h1y[i].GetXaxis().SetTitleSize(0.045)
        h1y[i].GetXaxis().SetTitleOffset()
        h1y[i].SetTitle(title)
        #h1y[i].GetYaxis().SetRangeUser(0, 1.2)
        legs[i].Draw()
        canvas.Update()
        h1y[i].Draw('')
        h1y[i].Draw('sames hist')
        canvas.Update()
        if i==0:
            leg.AddEntry(h1y[i], 'G4 ',"l")
            leg.AddEntry(h2y[i], 'GAN',"l")
                
        leg.AddEntry("", 'theta({}): events={}'.format(theta, num), "")
        h2y[i].Draw('sames')
        h2y[i].Draw('sames hist')
        canvas.Update()
    canvas.cd(4)
    if energy==0:
        leg.SetHeader('Primary Energy = {}-{}GeV'.format(p[0], p[1]))
    leg.SetHeader('Primary Energy = {}GeV'.format(energy))
    leg.Draw()
    canvas.Update()
    canvas.Print(out_file + '.pdf')
    if ifC:
        canvas.Print(out_file + '.C')
                                                           
# PLot ecal ratio
def plot_ecal_ratio_profile(ecal1, ecal2, y, thetas, out_file, p=[100, 200], ifpdf=True):
    canvas = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    canvas.SetGrid()
    canvas.Divide(2,2)
    Eprofs=[]
    Gprofs=[]
    legend = ROOT.TLegend(.1, .6, .9, .9)
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        canvas.cd(i+1)
        Eprofs.append(ROOT.TProfile("Eprof" + str(i), "", 100, int(p[0]), int(p[1])))
        Gprofs.append(ROOT.TProfile("Gprof" + str(i), "", 100, int(p[0]), int(p[1])))
        Eprof=Eprofs[i]
        Eprof.SetStats(ROOT.kFALSE)
        num = ecal1[i].shape[0]
        my.fill_profile(Eprof, (ecal1[i]* 100)/y[i], y[i])
        Eprof.GetXaxis().SetTitle("Ep GeV for theta={:d} degrees".format(int(theta)))
        Eprof.GetYaxis().SetTitle("50 x Ecal/Ep")
        Eprof.GetYaxis().CenterTitle()
        Eprof.GetYaxis().SetRangeUser(0.5, 1.5)
        Eprof.Draw()
        Eprof.SetLineColor(2)
        Eprof.GetYaxis().SetTitleOffset()
        Eprof.GetYaxis().SetTitleSize(0.05)
        Eprof.GetXaxis().SetTitleSize(0.045)
        Eprof.GetXaxis().SetTitleOffset()

        Gprof = Gprofs[i]
        Gprof.SetStats(ROOT.kFALSE)
        my.fill_profile(Gprof, ecal2[i]*100/y[i], y[i])
        Gprof.SetLineColor(4)
        Gprof.Draw('sames')
                                        
        if i==0:
          legend.AddEntry(Eprof, 'G4 ',"l")
          legend.AddEntry(Gprof, 'GAN',"l")
            
        legend.AddEntry("", 'theta({}): events={}'.format(theta, num), "")                   
        canvas.Update()
    canvas.cd(4)
    legend.Draw()
    canvas.Modified()
    canvas.Update()
    if ifpdf:
        canvas.Print(out_file + '.pdf')
    else:
        canvas.Print(out_file + '.C')
           
if __name__ == "__main__":
    main()
        

                                                                      

                                                                                                                                                    
                            
