import h5py
import numpy as np
import setGPU
import math
import sys
import ROOT
import os
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/')
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/analysis')
import utils.GANutils as gan
from utils.GANutils import perform_calculations_angle, safe_mkdir  # to calculate different Physics quantities
import utils.ROOTutils as my # common utility functions for root
#sys.path.insert(1, os.path.join(sys.path[0], '..'))
try:
    import setGPU
except:
    pass

def main():
    #Architecture
    from AngleArch3dGAN_add_loss_bins import generator, discriminator

    #Weights
    disc_weight1="../weights/3dgan_weights_bins_pow/params_discriminator_epoch_026.hdf5"
    gen_weight1= "../weights/3dgan_weights_bins_pow/params_generator_epoch_026.hdf5"

    #Path to store results
    plotsdir = "results/LCD_paper_plots_grid_bins_pow/"
    safe_mkdir(plotsdir)

    #Parameters
    latent = 256 # latent space
    num_data = 100000 # data to use
    num_events = 5000 # max events in energy bin 
    events_per_file = 5000 # events in each file
    m = 2  # number of moments 
    angloss=1 # number of angle losses
    addloss= 1 # if using additional bin loss
    concat = 1 # if concatenting angle to latent space
    cell=1 # if making plots for cell energies. Exclude for quick plots.
    corr=0 # if making correlation plots
    energies=[0, 110, 150, 190] # energy bins
    thetas = [math.radians(x) for x in [62, 90, 118]] # angle bins
    aindexes = [0, 1, 2] # numbers corressponding to different angle bins
    angtype = 'theta'# the angle data to be read from file
    particle='Ele'# partcile type
    thresh=0 # Threshold for ecal energies
    power =0.75
    datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path
    states = 1
    legs=1
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
    d = discriminator(power)
    g = generator(latent)
    var= perform_calculations_angle(g, d, gweights, dweights, energies, thetas,
                                    aindexes, datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales,
                                    ascales, flags, latent, events_per_file, particle, thresh=thresh, angtype=angtype, offset=0.0,
                                    angloss=angloss, addloss=addloss, concat=concat,power=power
                                    , pre =preproc, post =postproc  # Adding other preprocessing, Default is simple scaling
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
        momentx_act=[]
        momenty_act=[]
        momentz_act=[]
        momentx_gan=[]
        momenty_gan=[]
        momentz_gan=[]
        hits_act=[]
        hits_gan=[]
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
     
           momentx_act.append(var["momentX_act"+ str(energy) + "ang_" + str(i)][:, 1])
           momenty_act.append(var["momentY_act"+ str(energy) + "ang_" + str(i)][:, 1])
           momentz_act.append(var["momentZ_act"+ str(energy) + "ang_" + str(i)][:, 1])
           momentx_gan.append(var["momentX_gan"+ str(energy) + "ang_" + str(i)]['n_0'][:, 1])
           momenty_gan.append(var["momentY_gan"+ str(energy) + "ang_" + str(i)]['n_0'][:, 1])
           momentz_gan.append(var["momentZ_gan"+ str(energy) + "ang_" + str(i)]['n_0'][:, 1])
           hits = np.sum(var["events_act"+ str(energy)+ "ang_" + str(i)] > 0.2, axis=(1, 2, 3))
           hits_act.append(hits)
           hits = np.sum(var["events_gan"+ str(energy)+ "ang_" + str(i)]['n_0'] > 0.2, axis=(1, 2, 3))
           hits_gan.append(hits)
                              
           energy_act.append(var["energy"+ str(energy) + "ang_" + str(i)])
        PlotEnergyHistGrid(sumx_act, sumx_gan, sumy_act, sumy_gan, sumz_act, sumz_gan, ecal_act, ecal_gan, energy_act, hits_act, hits_gan, os.path.join(edir, 'energy_grid{}'.format(energy)), energy, thetas, states=states,legs=legs)
        PlotMomentHistGrid(momentx_act, momenty_act, momentz_act, momentx_gan, momenty_gan, momentz_gan, os.path.join(edir, 'moment_grid{}'.format(energy)), energy, thetas, states=states, legs=legs)
        
def preproc(n, power, scale=1.):
    return np.power((n * scale), power)

def postproc(n, power, scale=1.):
    return np.power(n, 1./power)/scale

def PlotEnergyHistGrid(sumx_act, sumx_gan, sumy_act, sumy_gan, sumz_act, sumz_gan, ecal1, ecal2, penergy, hits_act, hits_gan, out_file, energy, thetas, log=0, ifC=False, p=[100, 200], states=0, legs=1):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,700 ,700) #make
    canvas.SetGrid()
   
    legs=[]
    canvas.Divide(len(thetas),5)
    h1x=[]
    h2x=[]
    h1y=[]
    h2y=[]
    h1z=[]
    h2z=[]

    h_act=[]
    h_gan=[]
    Eprofs=[]
    Gprofs=[]
    mmtx_act=[]
    mmty_act=[]
    mmtz_act=[]
    mmtx_gan=[]
    mmty_gan=[]
    mmtz_gan=[]
    
    pad=1          
    
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        legs.append(ROOT.TLegend(0.65,0.1,0.9,0.3))
        legs[pad-1].SetHeader('Sampling Fraction', "C")
        canvas.cd(pad)
     
        title = "theta={} degrees".format(theta)
        if energy!=0:
            p[0]=np.amin(penergy[i])
            p[1]=np.amax(penergy[i])
        Eprofs.append(ROOT.TProfile("Eprof" + str(i), "", 100, int(p[0]), int(p[1])))
        Gprofs.append(ROOT.TProfile("Gprof" + str(i), "", 100, int(p[0]), int(p[1])))
        Eprof=Eprofs[i]
        if states==0:
            Eprof.SetStats(0)
            Gprof.SetStats(0)
        error = np.mean(np.abs(((ecal1[i]/penergy[i]) - (ecal2[i]/penergy[i]) )/(ecal1[i]/penergy[i])))
        my.fill_profile(Eprof, 2 * ecal1[i]/penergy[i], penergy[i])
        Eprof.SetTitle(title)
        Eprof.SetTitleSize(0.8)
        Eprof.GetXaxis().SetTitle("Ep GeV")
        Eprof.GetYaxis().SetTitle("Ecal/Ep")
        Eprof.GetYaxis().CenterTitle()
        Eprof.GetYaxis().SetRangeUser(0.,0.03)
        Eprof.Draw()
        Eprof.SetLineColor(2)
        Eprof.GetYaxis().SetTitleOffset()
        Eprof.GetYaxis().SetTitleSize(0.053)
        Eprof.GetXaxis().SetTitleSize(0.051)
        Eprof.GetXaxis().SetTitleOffset()
        
        Gprof = Gprofs[i]
        legs[pad-1].AddEntry(Eprof, "MAE={:.4f}".format(error), 'l')
        my.fill_profile(Gprof, 2 * ecal2[i]/penergy[i], penergy[i])
        Gprof.SetLineColor(4)
        Gprof.Draw('sames')
        canvas.Update()
        if states:
            my.stat_pos(Gprof)
        if legs:
            legs[pad-1].Draw()
        
        canvas.Update()
        pad+=1
        
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        n_events = sumx_act[i].shape[0]
        canvas.cd(pad)
        
        if log:
          ROOT.gPad.SetLogy()
        h_act.append(ROOT.TH1F('G4_hits{:d}theta_{:d}GeV'.format(theta, energy), '', 50, 0, 500))
        h_gan.append(ROOT.TH1F('GAN_hits{:d}theta_{:d}GeV'.format(theta, energy), '', 50, 0, 500))
        legs.append(ROOT.TLegend(0.75,0.1,0.9,0.2))
        h_act[i].SetLineColor(2)
        h_gan[i].SetLineColor(4)
        legs[pad-1].SetHeader('Hits', "C")
        my.fill_hist(h_act[i], hits_act[i])
        my.fill_hist(h_gan[i], hits_gan[i])
        if states==0:
            h_act[i].SetStats(states)
            h_gan[i].SetStats(states)
        h_act[i]=my.normalize(h_act[i], 1)
        h_gan[i]=my.normalize(h_gan[i], 1)
        h_act[i].GetXaxis().SetTitle("Hits above 0.004 GeV")
        h_act[i].GetYaxis().SetTitle("normalized count")
        h_act[i].GetYaxis().CenterTitle()
        h_act[i].GetYaxis().SetTitleOffset()
        h_act[i].GetYaxis().SetTitleSize(0.053)
        h_act[i].GetXaxis().SetTitleSize(0.051)
        h_act[i].GetXaxis().SetTitleOffset()
        h_act[i].GetYaxis().SetRangeUser(0., 0.25)
        canvas.Update()
        h_act[i].Draw('')
        h_act[i].Draw('sames hist')
        canvas.Update()

        h_gan[i].Draw('sames')
        h_gan[i].Draw('sames hist')
        if legs:
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(h_gan[i])
            canvas.Update()
        pad+=1
                        
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        n_events = sumx_act[i].shape[0]
        canvas.cd(pad)
        
        if log:
          ROOT.gPad.SetLogy()
        x_shape=sumx_act[0].shape[1]
        h1x.append(ROOT.TH1F('G4x{:d}theta_{:d}GeV'.format(theta, energy), '', x_shape, 0, x_shape))
        h2x.append(ROOT.TH1F('GANx{:d}theta_{:d}GeV'.format(theta, energy), '', x_shape, 0, x_shape))
        legs.append(ROOT.TLegend(0.65,0.1,0.9,0.2))
        h1x[i].SetLineColor(2)
        h2x[i].SetLineColor(4)
        legs[pad-1].SetHeader('Shower Shapes', "C")
        my.fill_hist_wt(h1x[i], sumx_act[i])
        my.fill_hist_wt(h2x[i], sumx_gan[i])
        if states==0:
            h1x[i].SetStats(states)
            h2x[i].SetStats(states)
        h1x[i]=my.normalize(h1x[i], 1)
        h2x[i]=my.normalize(h2x[i], 1)
        h1x[i].GetXaxis().SetTitle("Position along X axis")
        h1x[i].GetYaxis().SetTitle("normalized energy")
        h1x[i].GetYaxis().CenterTitle()
        h1x[i].GetYaxis().SetTitleOffset()
        h1x[i].GetYaxis().SetTitleSize(0.053)
        h1x[i].GetXaxis().SetTitleSize(0.051)
        h1x[i].GetXaxis().SetTitleOffset()
        h1x[i].GetYaxis().SetRangeUser(0, 0.5)
        canvas.Update()
        h1x[i].Draw('')
        h1x[i].Draw('sames hist')
        canvas.Update()

        h2x[i].Draw('sames')
        h2x[i].Draw('sames hist')
        if legs:
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(h2x[i])
            canvas.Update()
                                    
        pad+=1
                        
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        n_events = sumy_act[i].shape[0]
        canvas.cd(pad)
        if log:
          ROOT.gPad.SetLogy()
        y_shape=sumy_act[0].shape[1]
        h1y.append(ROOT.TH1F('G4y{:d}theta_{:d}GeV'.format(theta, energy), '', y_shape, 0, y_shape))
        h2y.append(ROOT.TH1F('GANy{:d}theta_{:d}GeV'.format(theta, energy), '', y_shape, 0, y_shape))
        legs.append(ROOT.TLegend(0.65,0.1,0.9,0.2))
        h1y[i].SetLineColor(2)
        h2y[i].SetLineColor(4)
        legs[pad-1].SetHeader('Shower Shapes', "C")
        my.fill_hist_wt(h1y[i], sumy_act[i])
        my.fill_hist_wt(h2y[i], sumy_gan[i])
        if states==0:
            h1y[i].SetStats(states)
            h2y[i].SetStats(states)
        h1y[i]=my.normalize(h1y[i], 1)
        h2y[i]=my.normalize(h2y[i], 1)
        h1y[i].GetXaxis().SetTitle("Position along Y axis")
        h1y[i].GetYaxis().SetTitle("normalized energy")
        h1y[i].GetYaxis().CenterTitle()
        h1y[i].GetYaxis().SetTitleOffset()
        h1y[i].GetYaxis().SetTitleSize(0.053)
        h1y[i].GetXaxis().SetTitleSize(0.051)
        h1y[i].GetXaxis().SetTitleOffset()
        
        canvas.Update()
        h1y[i].Draw('')
        h1y[i].Draw('sames hist')
        canvas.Update()

        h2y[i].Draw('sames')
        h2y[i].Draw('sames hist')
        if legs:
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(h2y[i])
            canvas.Update()
        pad+=1
    
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        canvas.cd(pad)
    
        if log:
          ROOT.gPad.SetLogy()
        z_shape=sumz_act[0].shape[1]
        h1z.append(ROOT.TH1F('G4z{:d}theta_{:d}GeV'.format(theta, energy), '', z_shape, 0, z_shape))
        h2z.append(ROOT.TH1F('GANz{:d}theta_{:d}GeV'.format(theta, energy), '', z_shape, 0, z_shape))
        legs.append(ROOT.TLegend(0.65,0.2,0.9,0.1))
        h1z[i].SetLineColor(2)
        h2z[i].SetLineColor(4)
        legs[pad-1].SetHeader('Shower Shapes', "C")
        my.fill_hist_wt(h1z[i], sumz_act[i])
        my.fill_hist_wt(h2z[i], sumz_gan[i])
        if states==0:
            h1z[i].SetStats(states)
            h2z[i].SetStats(states)
        h1z[i]=my.normalize(h1z[i], 1)
        h2z[i]=my.normalize(h2z[i], 1)
        h1z[i].GetXaxis().SetTitle("Position along Z axis")
        h1z[i].GetYaxis().SetTitle("normalized energy")
        h1z[i].GetYaxis().CenterTitle()
        h1z[i].GetYaxis().SetTitleOffset()
        h1z[i].GetYaxis().SetTitleSize(0.053)
        h1z[i].GetXaxis().SetTitleSize(0.051)
        h1z[i].GetXaxis().SetTitleOffset()
   
        canvas.Update()
        h1z[i].Draw('')
        h1z[i].Draw('sames hist')
        canvas.Update()

        h2z[i].Draw('sames')
        h2z[i].Draw('sames hist')
        if legs:
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(h2z[i])
            canvas.Update()
        pad+=1
    
    canvas.Update()
    canvas.Print(out_file + '.pdf')
    if ifC:
        canvas.Print(out_file + '.C')
                        
def PlotMomentHistGrid(mx_act, my_act, mz_act, mx_gan, my_gan, mz_gan, out_file, energy, thetas, log=0, ifC=False, p=[100, 200], states=0, legs=1):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,700 ,700) #make
    canvas.SetGrid()
    label = "Weighted Histograms for {} GeV".format(energy)
    legs=[]
    canvas.Divide(len(thetas),3)
    mmtx_act=[]
    mmty_act=[]
    mmtz_act=[]
    mmtx_gan=[]
    mmty_gan=[]
    mmtz_gan=[]
    pad=1

    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        title = "theta={} degrees".format(theta)
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.7,0.1,0.9,0.2))
        maxbin = np.amax(mx_act[i])+ 2
        if maxbin > 50:
            maxbin=50
        minbin = min(0, np.amin(mx_act[i]))
        mmtx_act.append(ROOT.TH1F('G4{:d}mmtx_theta_{:d}'.format(theta, energy), '', bins, minbin, maxbin))
        mmtx_gan.append(ROOT.TH1F('GAN{:d}mmtx_theta_{:d}GeV'.format(theta, energy), '', bins, minbin, maxbin))
        mmtx_act[i].SetLineColor(2)
        mmtx_gan[i].SetLineColor(4)
        my.fill_hist(mmtx_act[i], mx_act[i])
        my.fill_hist(mmtx_gan[i], mx_gan[i])
        if states==0:
            mmtx_act[i].SetStats(states)
            mmtx_gan[i].SetStats(states)
        mmtx_act[i]=my.normalize(mmtx_act[i], 1)
        mmtx_gan[i]=my.normalize(mmtx_gan[i], 1)
        mmtx_act[i].GetXaxis().SetTitle("X2moment")
        mmtx_act[i].GetYaxis().SetTitle("normalized count")
        mmtx_act[i].GetYaxis().CenterTitle()
        mmtx_act[i].GetYaxis().SetTitleOffset()
        mmtx_act[i].GetYaxis().SetTitleSize(0.055)
        mmtx_act[i].GetXaxis().SetTitleSize(0.05)
        mmtx_act[i].GetYaxis().SetRangeUser(0., 0.9)
        mmtx_act[i].SetTitle(title)
        legs[pad-1].SetHeader('X2 moment', "C")
        canvas.Update()
        mmtx_act[i].Draw('')
        mmtx_act[i].Draw('sames hist')
        canvas.Update()

        mmtx_gan[i].Draw('sames')
        mmtx_gan[i].Draw('sames hist')
        if legs:
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(mmtx_gan[i])
            canvas.Update()
        pad+=1

    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.7,0.1,0.9,0.2))
        legs[pad-1].SetHeader('Y2 moment', "C")
        maxbin = np.amax(my_act[i])+ 2
        if maxbin > 50:
           maxbin=50
        minbin = min(0, np.amin(my_act[i]))
        mmty_act.append(ROOT.TH1F('G4{:d}mmty_theta_{:d}'.format(theta, energy), '', bins, minbin, maxbin))
        mmty_gan.append(ROOT.TH1F('GAN{:d}mmty_theta_{:d}GeV'.format(theta, energy), '', bins, minbin, maxbin))
        mmty_act[i].SetLineColor(2)
        mmty_gan[i].SetLineColor(4)
        my.fill_hist(mmty_act[i], my_act[i])
        my.fill_hist(mmty_gan[i], my_gan[i])
        if states==0:
            mmty_act[i].SetStats(states)
            mmty_gan[i].SetStats(states)
        mmty_act[i]=my.normalize(mmty_act[i], 1)
        mmty_gan[i]=my.normalize(mmty_gan[i], 1)
        mmty_act[i].GetXaxis().SetTitle("Y2moment")
        mmty_act[i].GetYaxis().SetTitle("normalized count")
        mmty_act[i].GetYaxis().CenterTitle()
        mmty_act[i].GetYaxis().SetTitleOffset()
        mmty_act[i].GetYaxis().SetTitleSize(0.055)
        mmty_act[i].GetXaxis().SetTitleSize(0.05)
        mmty_act[i].GetYaxis().SetRangeUser(0., 0.4)
     
        canvas.Update()
        mmty_act[i].Draw('')
        mmty_act[i].Draw('sames hist')
        canvas.Update()

        mmty_gan[i].Draw('sames')
        mmty_gan[i].Draw('sames hist')
        if legs:
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(mmty_gan[i])
            canvas.Update()
        pad+=1
        canvas.Update()
             
    for i, theta in enumerate(thetas):
        theta = int(np.degrees(theta))
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.7,0.1,0.9,0.2))
        legs[pad-1].SetHeader('Z2 moment', "C")
        maxbin = np.amax(mz_act[i])+ 2
        if maxbin > 50:
            maxbin=50
        minbin = min(0, np.amin(mz_act[i]))
        mmtz_act.append(ROOT.TH1F('G4y{:d}mmtz_theta_{:d}'.format(theta, energy), '', bins, minbin, maxbin))
        mmtz_gan.append(ROOT.TH1F('GANy{:d}mmtz_theta_{:d}GeV'.format(theta, energy), '', bins, minbin, maxbin))
        mmtz_act[i].SetLineColor(2)
        mmtz_gan[i].SetLineColor(4)
        my.fill_hist(mmtz_act[i], mz_act[i])
        my.fill_hist(mmtz_gan[i], mz_gan[i])
        if states==0:
            mmtz_act[i].SetStats(states)
            mmtz_gan[i].SetStats(states)
        mmtz_act[i]=my.normalize(mmtz_act[i], 1)
        mmtz_gan[i]=my.normalize(mmtz_gan[i], 1)
        mmtz_act[i].GetXaxis().SetTitle("Z2moment")
        mmtz_act[i].GetYaxis().SetTitle("normalized count")
        mmtz_act[i].GetYaxis().CenterTitle()
        mmtz_act[i].GetYaxis().SetTitleOffset()
        mmtz_act[i].GetYaxis().SetTitleSize(0.055)
        mmtz_act[i].GetXaxis().SetTitleSize(0.05)
        #mmtz_act[i].GetXaxis().SetTitleOffset()
        mmtz_act[i].GetYaxis().SetRangeUser(0., 0.2)
        canvas.Update()
        mmtz_act[i].Draw('')
        mmtz_act[i].Draw('sames hist')
        canvas.Update()

        mmtz_gan[i].Draw('sames')
        mmtz_gan[i].Draw('sames hist')
        if legs:
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(mmtz_gan[i])
            canvas.Update()
        pad+=1

    canvas.Update()
    canvas.Print(out_file + '.pdf')
    if ifC:
        canvas.Print(out_file + '.C')
                                                           
if __name__ == "__main__":
    main()
        

                                                                      

                                                                                                                                                    
                            
