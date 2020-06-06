import h5py
import numpy as np
import setGPU
import math
import sys
import ROOT
import os
sys.path.insert(0,'../keras')
sys.path.insert(0,'../keras/analysis')
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
    from AngleArch3dGAN import generator, discriminator

    #Weights
    disc_weight1="../keras/weights/surfsara_weights/params_discriminator_epoch_099.hdf5"
    gen_weight1= "../keras/weights/surfsara_weights/params_generator_epoch_099.hdf5"

    #Path to store results
    plotsdir = "results/analysis_grid_surfara_32nd"
    safe_mkdir(plotsdir)
    #Parameters
    latent = 256 # latent space
    num_data = 500000
    num_events = 10000
    num_events1 = 20000
    events_per_file = 10000
    thresh_hits = 3e-4
    m = 3  # number of moments
    angloss= 1 # total number of losses...1 or 2
    addloss= 1 # additional loss like count loss
    concat = 2 # if concatenting angle to latent space
    cell=0 # 1 if making plots for cell energies for energy bins and 2 if plotting also per angle bins. Exclude for quick plots.
    corr=1 # if making correlation plots
    # energies=[0, 50, 100, 200, 300, 400, 500] # energy bins
    energies = [0, 110, 150, 190] 
    angles = [62, 90, 118] #[math.radians(x) for x in [62, 90, 118]] # angle bins
    angtype = 'mtheta'# the angle data to be read from file
    particle='Ele'# partcile type
    thresh=0 # Threshold for ecal energies
    #datapath = "/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
    datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path
    #datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5"
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
    ifC= False # True if .C files are required. If false pdf files will be generated

    flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
    dweights = [disc_weight1]#, disc_weight2]
    gweights = [gen_weight1]#, gen_weight2]
    xscales = [1]#, 1]
    xpowers = [0.85]
    dscale = 50.0
    ascales = [1]#, 1]
    labels = ['']#, 'epoch 40']
    d = discriminator(xpowers[0])
    g = generator(latent)
    var= perform_calculations_angle(g, d, gweights, dweights, energies, angles,
                                    datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales, xpowers,
                                    ascales, dscale, flags, latent, events_per_file=events_per_file, particle=particle, thresh=thresh, angtype=angtype, offset=0.0,
                                    angloss=angloss, addloss=addloss, concat=concat, num_events1=20000
                                    , pre =taking_power, post =inv_power  # Adding other preprocessing, Default is simple scaling
    )
    states = 0
    legs=0
    norm=1
    
    for energy in energies:
        edir = os.path.join(plotsdir, 'energy_{}'.format(energy))
        safe_mkdir(edir)
        sumx_act=[var["sumsx_act"+ str(energy)]]
        sumy_act=[var["sumsy_act"+ str(energy)]]
        sumz_act=[var["sumsz_act"+ str(energy)]]
        sumx_gan=[var["sumsx_gan"+ str(energy)]['n_0']]
        sumy_gan=[var["sumsy_gan"+ str(energy)]['n_0']]
        sumz_gan=[var["sumsz_gan"+ str(energy)]['n_0']]
        ecal_act=[var["ecal_act"+ str(energy)]['n_0']]
        ecal_gan=[var["ecal_gan"+ str(energy)]['n_0']]
        energy_act =[var["energy"+ str(energy)]]
        momentx_act=[var["momentX_act"+ str(energy)][:, 1]]
        momenty_act=[var["momentY_act"+ str(energy)][:, 1]]
        momentz_act=[var["momentZ_act"+ str(energy)][:, 1]]
        momentx_gan=[var["momentX_gan"+ str(energy)]['n_0'][:, 1]]
        momenty_gan=[var["momentY_gan"+ str(energy)]['n_0'][:, 1]]
        momentz_gan=[var["momentZ_gan"+ str(energy)]['n_0'][:, 1]]
        hits_act=[np.sum(var["events_act"+ str(energy)] > thresh_hits, axis=(1, 2, 3))]
        hits_gan=[np.sum(var["events_gan"+ str(energy)]['n_0'] > thresh_hits, axis=(1, 2, 3))]
        ratio1_act=[my.ratio1_total(var["events_act"+ str(energy)])]
        ratio2_act=[my.ratio2_total(var["events_act"+ str(energy)])]
        ratio3_act=[my.ratio3_total(var["events_act"+ str(energy)])]
        ratio1_gan=[my.ratio1_total(var["events_gan"+ str(energy)]['n_0'])]
        ratio2_gan=[my.ratio2_total(var["events_gan"+ str(energy)]['n_0'])]
        ratio3_gan=[my.ratio3_total(var["events_gan"+ str(energy)]['n_0'])]
        
        p = [int(np.amin(var["energy"+ str(energy)])), int(np.amax(var["energy"+ str(energy)]))]
        
        for theta in angles:
           sumx_act.append(var["sumsx_act"+ str(energy) + "ang_" + str(theta)])
           sumy_act.append(var["sumsy_act"+ str(energy) + "ang_" + str(theta)])
           sumz_act.append(var["sumsz_act"+ str(energy) + "ang_" + str(theta)])
           sumx_gan.append(var["sumsx_gan"+ str(energy) + "ang_" + str(theta)]['n_0'])
           sumy_gan.append(var["sumsy_gan"+ str(energy) + "ang_" + str(theta)]['n_0'])
           sumz_gan.append(var["sumsz_gan"+ str(energy) + "ang_" + str(theta)]['n_0'])
           ecal_act.append(var["ecal_act"+ str(energy) + "ang_" + str(theta)]['n_0'])
           ecal_gan.append(var["ecal_gan"+ str(energy) + "ang_" + str(theta)]['n_0'])
     
           momentx_act.append(var["momentX_act"+ str(energy) + "ang_" + str(theta)][:, 1])
           momenty_act.append(var["momentY_act"+ str(energy) + "ang_" + str(theta)][:, 1])
           momentz_act.append(var["momentZ_act"+ str(energy) + "ang_" + str(theta)][:, 1])
           momentx_gan.append(var["momentX_gan"+ str(energy) + "ang_" + str(theta)]['n_0'][:, 1])
           momenty_gan.append(var["momentY_gan"+ str(energy) + "ang_" + str(theta)]['n_0'][:, 1])
           momentz_gan.append(var["momentZ_gan"+ str(energy) + "ang_" + str(theta)]['n_0'][:, 1])
           hits1 = np.sum(var["events_act"+ str(energy)+ "ang_" + str(theta)] > thresh_hits, axis=(1, 2, 3))
           hits_act.append(hits1)
           hits2 = np.sum(var["events_gan"+ str(energy)+ "ang_" + str(theta)]['n_0'] > thresh_hits, axis=(1, 2, 3))
           hits_gan.append(hits2)
           ratio1_act.append(my.ratio1_total(var["events_act"+ str(energy)+ "ang_" + str(theta)]))
           ratio2_act.append(my.ratio2_total(var["events_act"+ str(energy)+ "ang_" + str(theta)]))
           ratio3_act.append(my.ratio3_total(var["events_act"+ str(energy)+ "ang_" + str(theta)]))                 

           ratio1_gan.append(my.ratio1_total(var["events_gan"+ str(energy)+ "ang_" + str(theta)]['n_0']))
           ratio2_gan.append(my.ratio2_total(var["events_gan"+ str(energy)+ "ang_" + str(theta)]['n_0']))
           ratio3_gan.append(my.ratio3_total(var["events_gan"+ str(energy)+ "ang_" + str(theta)]['n_0']))
                              
           energy_act.append(var["energy"+ str(energy) + "ang_" + str(theta)])
        thetas = [0, 62, 90, 118]
        PlotSamplingGrid(ecal_act, ecal_gan, energy_act, hits_act, hits_gan, os.path.join(edir, 'samplig_grid{}'.format(energy)), energy, thetas, p=p, states=states,leg=legs, norm=norm, ifC=ifC)
        PlotEnergyHistGrid(sumx_act, sumx_gan, sumy_act, sumy_gan, sumz_act, sumz_gan, os.path.join(edir, 'shapes_grid{}'.format(energy)), energy, thetas, p=p, states=states,leg=legs, norm=norm, ifC=ifC)
        PlotEnergyHistGrid(sumx_act, sumx_gan, sumy_act, sumy_gan, sumz_act, sumz_gan, os.path.join(edir, 'shapes_grid_log{}'.format(energy)), energy, thetas, p=p, log=1, states=states,leg=legs, norm=norm, ifC=ifC)
        PlotMomentHistGrid(momentx_act, momenty_act, momentz_act, momentx_gan, momenty_gan, momentz_gan, os.path.join(edir, 'moment_grid{}'.format(energy)), energy, thetas, p=p, states=states, leg=legs, norm=norm, ifC=ifC)
        PlotRatioGrid(ratio1_act, ratio2_act, ratio3_act, ratio1_gan, ratio2_gan, ratio3_gan, os.path.join(edir, 'ratio_grid{}'.format(energy)), energy, thetas, p=p, states=states, leg=legs, norm=norm, ifC=ifC)        

def taking_power(n, scale=1.0, power=1.0):
    return(np.power(n * scale, power))

def inv_power(n, scale=1.0, power=1.0):
    return(np.power(n, 1.0/power))/scale
      
def PlotSamplingGrid(ecal1, ecal2, penergy, hits_act, hits_gan, out_file, energy, thetas, log=0, ifC=False, p=[100, 200], states=0, leg=1, norm=0):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,800 ,300) #make
    canvas.SetGrid()

    legs=[]
    canvas.Divide(len(thetas),2)
    
    h_act=[]
    h_gan=[]
    Eprofs=[]
    Gprofs=[]
    pad=1

    for i, theta in enumerate(thetas):
        #theta = int(np.degrees(theta))
        legs.append(ROOT.TLegend(0.65,0.1,0.9,0.3))
        #legs[pad-1].SetHeader('Sampling Fraction', "C")
        canvas.cd(pad)
        if theta==0:
            title = "theta=60-120 degrees"
            bins=50
        else:
            title = "theta={} degrees".format(theta)
            
        if energy!=0:
            p[0]=np.amin(penergy[i])
            p[1]=np.amax(penergy[i])
            bins=25
        else:
            bins=100
        Eprofs.append(ROOT.TProfile("Eprof" + str(i), "", bins, int(p[0]), int(p[1])))
        Gprofs.append(ROOT.TProfile("Gprof" + str(i), "", bins, int(p[0]), int(p[1])))
        Eprof=Eprofs[i]
        Gprof=Gprofs[i]
        Eprof.Sumw2()
        Gprof.Sumw2()
        if states==0:
            Eprof.SetStats(0)
            Gprof.SetStats(0)
        error = np.mean(np.abs(((ecal1[i]/penergy[i]) - (ecal2[i]/penergy[i]) )/(ecal1[i]/penergy[i])))
        my.fill_profile(Eprof, ecal1[i]/(penergy[i]), penergy[i])
        Eprof.SetTitle(title)
        Eprof.SetTitleSize(0.8)
        Eprof.GetXaxis().SetTitle("Ep GeV")
        Eprof.GetYaxis().SetTitle("Ecal/Ep")
        Eprof.GetYaxis().CenterTitle()
        Eprof.GetYaxis().SetRangeUser(0.,0.03)
        Eprof.Draw()
        Eprof.SetLineColor(2)
        Eprof.GetYaxis().SetTitleOffset()
        Eprof.GetYaxis().SetTitleSize(0.051)
        Eprof.GetXaxis().SetTitleSize(0.051)
        Eprof.GetXaxis().SetTitleOffset()
        legs[pad-1].AddEntry(Eprof, "G4", 'l')
        legs[pad-1].AddEntry(Gprof, "GAN", 'l')
        #legs[pad-1].AddEntry(Eprof, "MAE={:.4f}".format(error), 'l')
        my.fill_profile(Gprof, ecal2[i]/(penergy[i]), penergy[i])
        Gprof.SetLineColor(4)
        Gprof.SetLineStyle(2)
        #Gprof.SetMarkerStyle(3)
        Gprof.Draw('sames')
        canvas.Update()
        if states:
            my.stat_pos(Gprof)
        if leg:
            legs[pad-1].Draw()
        canvas.Update()
        pad+=1

    for i, theta in enumerate(thetas):
        #theta = int(np.degrees(theta))
        n_events = hits_act[i].shape[0]
        canvas.cd(pad)

        if log:
            ROOT.gPad.SetLogy()
        h_act.append(ROOT.TH1F('G4_hits{:d}theta_{:d}GeV'.format(theta, energy), '', 40, 0, 2200))
        h_gan.append(ROOT.TH1F('GAN_hits{:d}theta_{:d}GeV'.format(theta, energy), '', 40, 0, 2200))
        legs.append(ROOT.TLegend(0.65,0.1,0.9,0.3))
        h_act[i].Sumw2()
        h_gan[i].Sumw2()
        h_act[i].SetLineColor(2)
        h_gan[i].SetLineColor(4)
        h_gan[i].SetLineStyle(2)
        #h_gan[i].SetMarkerStyle(3)
        legs[pad-1].SetHeader('Hits', "C")
        my.fill_hist(h_act[i], hits_act[i])
        my.fill_hist(h_gan[i], hits_gan[i])
        if states==0:
           h_act[i].SetStats(states)
           h_gan[i].SetStats(states)
        if norm:
           h_act[i]=my.normalize(h_act[i], 1)
           h_gan[i]=my.normalize(h_gan[i], 1)
        h_act[i].GetXaxis().SetTitle("Hits above 3e-4 GeV")
        h_act[i].GetYaxis().SetTitle("normalized count")
        h_act[i].GetYaxis().CenterTitle()
        h_act[i].GetYaxis().SetTitleOffset()
        h_act[i].GetYaxis().SetTitleSize(0.053)
        h_act[i].GetXaxis().SetTitleSize(0.051)
        h_act[i].GetXaxis().SetTitleOffset()
        if i==0:
          ymax = max(h_act[i].GetMaximum(), h_gan[i].GetMaximum())
        else:
          ymax = max(ymax, h_act[i].GetMaximum(), h_gan[i].GetMaximum())
        h_act[i].GetYaxis().SetRangeUser(0., 1.2 * ymax)
        canvas.Update()
        h_act[i].Draw('')
        h_act[i].Draw('sames hist')
        canvas.Update()
        h_gan[i].Draw('sames')
        h_gan[i].Draw('sames hist')
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(h_act[i], "G4", 'l')
            legs[pad-1].AddEntry(h_gan[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(h_gan[i])
        canvas.Update()
        pad+=1
    canvas.Update()
    canvas.Print(out_file + '.pdf')
    if ifC:
        canvas.Print(out_file + '.C')
                        

def PlotEnergyHistGrid(sumx_act, sumx_gan, sumy_act, sumy_gan, sumz_act, sumz_gan, out_file, energy, thetas, log=0, ifC=False, p=[100, 200], states=0, leg=1, norm=0):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,800 ,400) #make
    canvas.SetGrid()
   
    legs=[]
    canvas.Divide(len(thetas),3)
    h1x=[]
    h2x=[]
    h1y=[]
    h2y=[]
    h1z=[]
    h2z=[]
    pad=1          
                        
    for i, theta in enumerate(thetas):
        #theta = int(np.degrees(theta))
        n_events = sumx_act[i].shape[0]
        canvas.cd(pad)
        if theta==0:
            title = "theta=60-120 degrees"
        else:
            title="theta={} degrees".format(theta)
        if log:
          ROOT.gPad.SetLogy()
        x_shape=sumx_act[0].shape[1]
        h1x.append(ROOT.TH1F('G4x{:d}theta_{:d}GeV'.format(theta, energy), '', x_shape, 0, x_shape))
        h2x.append(ROOT.TH1F('GANx{:d}theta_{:d}GeV'.format(theta, energy), '', x_shape, 0, x_shape))
        legs.append(ROOT.TLegend(0.7,0.7,0.9,0.9))
        h1x[i].SetLineColor(2)
        h2x[i].SetLineColor(4)
        h2x[i].SetLineStyle(2)
        #h2x[i].SetMarkerStyle(3)
        h1x[i].Sumw2()
        h2x[i].Sumw2()
        #legs[pad-1].SetHeader('Shower X', "C")
        my.fill_hist_wt(h1x[i], sumx_act[i])
        my.fill_hist_wt(h2x[i], sumx_gan[i])
        if states==0:
            h1x[i].SetStats(states)
            h2x[i].SetStats(states)
        if norm:
            h1x[i]=my.normalize(h1x[i])
            h2x[i]=my.normalize(h2x[i])
        h1x[i].SetTitle(title)
        h1x[i].GetXaxis().SetTitle("Position along X axis")
        if log:
            h1x[i].GetYaxis().SetTitle("normalized energy [log]")
        else:
            h1x[i].GetYaxis().SetTitle("normalized energy")
        h1x[i].GetYaxis().CenterTitle()
        h1x[i].GetYaxis().SetTitleOffset()
        h1x[i].GetYaxis().SetTitleSize(0.053)
        h1x[i].GetXaxis().SetTitleSize(0.051)
        h1x[i].GetXaxis().SetTitleOffset()
        #if not log:
          #h1x[i].GetYaxis().SetRangeUser(0, 0.5)
        canvas.Update()
        h1x[i].Draw('')
        h1x[i].Draw('sames hist')
        canvas.Update()

        h2x[i].Draw('sames')
        h2x[i].Draw('sames hist')
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(h1x[i], "G4", 'l')
            legs[pad-1].AddEntry(h2x[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(h2x[i])
        canvas.Update()
        pad+=1
                        
    for i, theta in enumerate(thetas):
        #theta = int(np.degrees(theta))
        n_events = sumy_act[i].shape[0]
        canvas.cd(pad)
        if log:
          ROOT.gPad.SetLogy()
        y_shape=sumy_act[0].shape[1]
        h1y.append(ROOT.TH1F('G4y{:d}theta_{:d}GeV'.format(theta, energy), '', y_shape, 0, y_shape))
        h2y.append(ROOT.TH1F('GANy{:d}theta_{:d}GeV'.format(theta, energy), '', y_shape, 0, y_shape))
        legs.append(ROOT.TLegend(0.7,0.7,0.9,0.9))
        h1y[i].SetLineColor(2)
        h2y[i].SetLineColor(4)
        h2y[i].SetLineStyle(2)
        #h2y[i].SetMarkerStyle(3)
        h1y[i].Sumw2()
        h2y[i].Sumw2()
        #legs[pad-1].SetHeader('Shower Y', "C")
        my.fill_hist_wt(h1y[i], sumy_act[i])
        my.fill_hist_wt(h2y[i], sumy_gan[i])
        if states==0:
            h1y[i].SetStats(states)
            h2y[i].SetStats(states)
        if norm:
            h1y[i]=my.normalize(h1y[i])
            h2y[i]=my.normalize(h2y[i])
        h1y[i].GetXaxis().SetTitle("Position along Y axis")
        if log:
            h1y[i].GetYaxis().SetTitle("normalized energy [log]")
        else:
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
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(h1y[i], "G4", 'l')
            legs[pad-1].AddEntry(h2y[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(h2y[i])
        canvas.Update()
        pad+=1
    
    for i, theta in enumerate(thetas):
        #theta = int(np.degrees(theta))
        canvas.cd(pad)
    
        if log:
          ROOT.gPad.SetLogy()
        z_shape=sumz_act[0].shape[1]
        h1z.append(ROOT.TH1F('G4z{:d}theta_{:d}GeV'.format(theta, energy), '', z_shape, 0, z_shape))
        h2z.append(ROOT.TH1F('GANz{:d}theta_{:d}GeV'.format(theta, energy), '', z_shape, 0, z_shape))
        legs.append(ROOT.TLegend(0.4,0.1,0.6,0.3))
        h1z[i].Sumw2()
        h2z[i].Sumw2()
        h1z[i].SetLineColor(2)
        h2z[i].SetLineColor(4)
        h2z[i].SetLineStyle(2)
        #h2z[i].SetMarkerStyle(3)
        #legs[pad-1].SetHeader('Shower Z', "C")
        my.fill_hist_wt(h1z[i], sumz_act[i])
        my.fill_hist_wt(h2z[i], sumz_gan[i])
        if states==0:
            h1z[i].SetStats(states)
            h2z[i].SetStats(states)
        if norm:
            h1z[i]=my.normalize(h1z[i])
            h2z[i]=my.normalize(h2z[i])
        h1z[i].GetXaxis().SetTitle("Position along Z axis")
        if log:
            h1z[i].GetYaxis().SetTitle("normalized energy [log]")
        else:
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
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(h1z[i], "G4", 'l')
            legs[pad-1].AddEntry(h2z[i], "GAN", 'l')
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
                        
def PlotRatioGrid(ratio1_act, ratio2_act, ratio3_act, ratio1_gan, ratio2_gan, ratio3_gan, out_file, energy, thetas, log=0, ifC=False, p=[100, 200], states=0, leg=1, norm=0):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,800 ,500) #make
    canvas.SetGrid()
    label = "Weighted Histograms for {} GeV".format(energy)
    legs=[]
    canvas.Divide(len(thetas),3)
    r1_act=[]
    r2_act=[]
    r3_act=[]
    r1_gan=[]
    r2_gan=[]
    r3_gan=[]
    pad=1

    for i, theta in enumerate(thetas):
        if theta==0:
            title = "theta=60-120 degrees"
        else:
            title = "theta={} degrees".format(theta)
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.5,0.7,0.9,0.9))
        r1_act.append(ROOT.TH1F('G4{:d}r1_theta_{:d}'.format(theta, energy), '', bins, 0, 1))
        r1_gan.append(ROOT.TH1F('GAN{:d}r1_theta_{:d}GeV'.format(theta, energy), '', bins, 0, 1))
        r1_act[i].SetLineColor(2)
        r1_gan[i].SetLineColor(4)
        r1_gan[i].SetLineStyle(2)
        #r1_gan[i].SetMarkerStyle(3)
        r1_act[i].Sumw2()
        r1_gan[i].Sumw2()
        
        my.fill_hist(r1_act[i], ratio1_act[i])
        my.fill_hist(r1_gan[i], ratio1_gan[i])
        if states==0:
            r1_act[i].SetStats(states)
            r1_gan[i].SetStats(states)
        if norm:
           r1_act[i]=my.normalize(r1_act[i], 1)
           r1_gan[i]=my.normalize(r1_gan[i], 1)
        r1_act[i].GetXaxis().SetTitle("Ratio first/total")
        r1_act[i].GetYaxis().SetTitle("normalized count")
        r1_act[i].GetYaxis().CenterTitle()
        r1_act[i].GetYaxis().SetTitleOffset()
        r1_act[i].GetYaxis().SetTitleSize(0.055)
        r1_act[i].GetXaxis().SetTitleSize(0.05)
        if i==0:
          ymax = max(r1_act[i].GetMaximum(), r1_gan[i].GetMaximum())
        else:
          ymax = max(ymax, r1_act[i].GetMaximum(), r1_gan[i].GetMaximum())
         
        r1_act[i].GetYaxis().SetRangeUser(0., 1.2 * ymax)
        r1_act[i].SetTitle(title)
        legs[pad-1].SetHeader('Ratio first/total', "C")
        canvas.Update()
        r1_act[i].Draw('')
        r1_act[i].Draw('sames hist')
        canvas.Update()

        r1_gan[i].Draw('sames')
        r1_gan[i].Draw('sames hist')
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(r1_act[i], "G4", 'l')
            legs[pad-1].AddEntry(r1_gan[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(r1_gan[i])
            canvas.Update()
        pad+=1

    for i, theta in enumerate(thetas):
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.1,0.7,0.5,0.9))
        legs[pad-1].SetHeader('Ratio mid/total', "C")
        r2_act.append(ROOT.TH1F('G4{:d}r2_theta_{:d}'.format(theta, energy), '', bins, 0, 1))
        r2_gan.append(ROOT.TH1F('GAN{:d}r2_theta_{:d}GeV'.format(theta, energy), '', bins, 0, 1))
        r2_act[i].SetLineColor(2)
        r2_gan[i].SetLineColor(4)
        r2_gan[i].SetLineStyle(2)
        #r2_gan[i].SetMarkerStyle(3)
        r2_act[i].Sumw2()
        r2_gan[i].Sumw2()
        my.fill_hist(r2_act[i], ratio2_act[i])
        my.fill_hist(r2_gan[i], ratio2_gan[i])
        if states==0:
            r2_act[i].SetStats(states)
            r2_gan[i].SetStats(states)
        if norm:
            r2_act[i]=my.normalize(r2_act[i], 1)
            r2_gan[i]=my.normalize(r2_gan[i], 1)
        r2_act[i].GetXaxis().SetTitle("Ratio mid/total")
        r2_act[i].GetYaxis().SetTitle("normalized count")
        r2_act[i].GetYaxis().CenterTitle()
        r2_act[i].GetYaxis().SetTitleOffset()
        r2_act[i].GetYaxis().SetTitleSize(0.055)
        r2_act[i].GetXaxis().SetTitleSize(0.05)
        if i==0:
          ymax = max(r2_act[i].GetMaximum(), r2_gan[i].GetMaximum())
        else:
          ymax = max(ymax, r2_act[i].GetMaximum(), r2_gan[i].GetMaximum())

        r2_act[i].GetYaxis().SetRangeUser(0., 1.2 * ymax)

        canvas.Update()
        r2_act[i].Draw('')
        r2_act[i].Draw('sames hist')
        canvas.Update()

        r2_gan[i].Draw('sames')
        r2_gan[i].Draw('sames hist')
        canvas.Update()
        
        if leg:
            legs[pad-1].AddEntry(r2_act[i], "G4", 'l')
            legs[pad-1].AddEntry(r2_gan[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(r2_gan[i])
            canvas.Update()
        pad+=1
        canvas.Update()

    for i, theta in enumerate(thetas):
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.5,0.7,0.9,0.9))
        legs[pad-1].SetHeader('Ratio third/total', "C")
      
        r3_act.append(ROOT.TH1F('G4y{:d}r3_theta_{:d}'.format(theta, energy), '', bins, 0, 1))
        r3_gan.append(ROOT.TH1F('GANy{:d}r3_theta_{:d}GeV'.format(theta, energy), '', bins, 0, 1))
        r3_act[i].SetLineColor(2)
        r3_gan[i].SetLineColor(4)
        r3_gan[i].SetLineStyle(2)
        #r3_gan[i].SetMarkerStyle(3)
        r3_act[i].Sumw2()
        r3_gan[i].Sumw2()
        my.fill_hist(r3_act[i], ratio3_act[i])
        my.fill_hist(r3_gan[i], ratio3_gan[i])
        if states==0:
            r3_act[i].SetStats(states)
            r3_gan[i].SetStats(states)
        if norm:
            r3_act[i]=my.normalize(r3_act[i], 1)
            r3_gan[i]=my.normalize(r3_gan[i], 1)

        r3_act[i].GetXaxis().SetTitle("Ratio third/total")
        r3_act[i].GetYaxis().SetTitle("normalized count")
        r3_act[i].GetYaxis().CenterTitle()
        r3_act[i].GetYaxis().SetTitleOffset()
        r3_act[i].GetYaxis().SetTitleSize(0.055)
        r3_act[i].GetXaxis().SetTitleSize(0.05)
        if i==0:
          ymax = max(r3_act[i].GetMaximum(), r3_gan[i].GetMaximum())
        else:
          ymax = max(ymax, r3_act[i].GetMaximum(), r3_gan[i].GetMaximum())

        r3_act[i].GetYaxis().SetRangeUser(0., 1.2 * ymax)

        canvas.Update()
        r3_act[i].Draw('')
        r3_act[i].Draw('sames hist')
        canvas.Update()

        r3_gan[i].Draw('sames')
        r3_gan[i].Draw('sames hist')
        canvas.Update()
        
        if leg:
            legs[pad-1].AddEntry(r3_act[i], "G4", 'l')
            legs[pad-1].AddEntry(r3_gan[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(r3_gan[i])
            canvas.Update()
        pad+=1

    canvas.Update()
    canvas.Print(out_file + '.pdf')
    if ifC:
        canvas.Print(out_file + '.C')

def PlotMomentHistGrid(mx_act, my_act, mz_act, mx_gan, my_gan, mz_gan, out_file, energy, thetas, log=0, ifC=False, p=[100, 200], states=0, leg=1, norm=0):
    canvas = ROOT.TCanvas("canvas" ,"abc" ,200 ,10 ,800 ,500) #make
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
        if theta==0:
            title = "theta=60-120 degrees"
        else:
            title = "theta={} degrees".format(theta)
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.6,0.7,0.9,0.9))
        maxbin = np.amax(mx_act[i])+ 2
        if maxbin > 50:
            maxbin=50
        minbin = min(0, np.amin(mx_act[i]))
        mmtx_act.append(ROOT.TH1F('G4{:d}mmtx_theta_{:d}'.format(theta, energy), '', bins, minbin, maxbin))
        mmtx_gan.append(ROOT.TH1F('GAN{:d}mmtx_theta_{:d}GeV'.format(theta, energy), '', bins, minbin, maxbin))
        mmtx_act[i].SetLineColor(2)
        mmtx_gan[i].SetLineColor(4)
        mmtx_gan[i].SetLineStyle(2)
        #mmtx_gan[i].SetMarkerStyle(3)
        mmtx_act[i].Sumw2()
        mmtx_gan[i].Sumw2()
        my.fill_hist(mmtx_act[i], mx_act[i])
        my.fill_hist(mmtx_gan[i], mx_gan[i])
        if states==0:
            mmtx_act[i].SetStats(states)
            mmtx_gan[i].SetStats(states)
        if norm:
            mmtx_act[i]=my.normalize(mmtx_act[i], 1)
            mmtx_gan[i]=my.normalize(mmtx_gan[i], 1)

        mmtx_act[i].GetXaxis().SetTitle("2nd X moment")
        mmtx_act[i].GetYaxis().SetTitle("normalized count")
        mmtx_act[i].GetYaxis().CenterTitle()
        mmtx_act[i].GetYaxis().SetTitleOffset()
        mmtx_act[i].GetYaxis().SetTitleSize(0.055)
        mmtx_act[i].GetXaxis().SetTitleSize(0.05)
        mmtx_act[i].SetTitle(title)
        if i==0:
          ymax = max(mmtx_act[i].GetMaximum(), mmtx_gan[i].GetMaximum())
        else:
          ymax = max(ymax, mmtx_act[i].GetMaximum(), mmtx_gan[i].GetMaximum())

        mmtx_act[i].GetYaxis().SetRangeUser(0., 1.2 * ymax)

        legs[pad-1].SetHeader('X2 moment', "C")
        canvas.Update()
        mmtx_act[i].Draw('')
        mmtx_act[i].Draw('sames hist')
        canvas.Update()

        mmtx_gan[i].Draw('sames')
        mmtx_gan[i].Draw('sames hist')
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(mmtx_act[i], "G4", 'l')
            legs[pad-1].AddEntry(mmtx_gan[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(mmtx_gan[i])
            canvas.Update()
        pad+=1

    for i, theta in enumerate(thetas):
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.6,0.7,0.9,0.9))
        legs[pad-1].SetHeader('Y2 moment', "C")
        maxbin = np.amax(my_act[i])+ 2
        if maxbin > 50:
           maxbin=50
        minbin = min(0, np.amin(my_act[i]))
        mmty_act.append(ROOT.TH1F('G4{:d}mmty_theta_{:d}'.format(theta, energy), '', bins, minbin, maxbin))
        mmty_gan.append(ROOT.TH1F('GAN{:d}mmty_theta_{:d}GeV'.format(theta, energy), '', bins, minbin, maxbin))
        mmty_act[i].SetLineColor(2)
        mmty_gan[i].SetLineColor(4)
        mmty_gan[i].SetLineStyle(2)
        #mmty_gan[i].SetMarkerStyle(3)
        mmty_act[i].Sumw2()
        mmty_gan[i].Sumw2()
        my.fill_hist(mmty_act[i], my_act[i])
        my.fill_hist(mmty_gan[i], my_gan[i])
        if states==0:
            mmty_act[i].SetStats(states)
            mmty_gan[i].SetStats(states)
        if norm:
            mmty_act[i]=my.normalize(mmty_act[i], 1)
            mmty_gan[i]=my.normalize(mmty_gan[i], 1)

        mmty_act[i].GetXaxis().SetTitle("2nd Y moment")
        mmty_act[i].GetYaxis().SetTitle("normalized count")
        mmty_act[i].GetYaxis().CenterTitle()
        mmty_act[i].GetYaxis().SetTitleOffset()
        mmty_act[i].GetYaxis().SetTitleSize(0.055)
        mmty_act[i].GetXaxis().SetTitleSize(0.05)
        if i==0:
          ymax = max(mmty_act[i].GetMaximum(), mmty_gan[i].GetMaximum())
        else:
          ymax = max(ymax, mmty_act[i].GetMaximum(), mmty_gan[i].GetMaximum())

        mmty_act[i].GetYaxis().SetRangeUser(0., 1.2 * ymax)
     
        canvas.Update()
        mmty_act[i].Draw('')
        mmty_act[i].Draw('sames hist')
        canvas.Update()

        mmty_gan[i].Draw('sames')
        mmty_gan[i].Draw('sames hist')
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(mmty_act[i], "G4", 'l')
            legs[pad-1].AddEntry(mmty_gan[i], "GAN", 'l')
            legs[pad-1].Draw()
        canvas.Update()
        if states:
            my.stat_pos(mmty_gan[i])
            canvas.Update()
        pad+=1
        canvas.Update()
             
    for i, theta in enumerate(thetas):
        bins = 50
        canvas.cd(pad)
        legs.append(ROOT.TLegend(0.6,0.7,0.9,0.9))
        legs[pad-1].SetHeader('Z2 moment', "C")
        maxbin = np.amax(mz_act[i])+ 2
        if maxbin > 50:
            maxbin=50
        minbin = min(0, np.amin(mz_act[i]))
        mmtz_act.append(ROOT.TH1F('G4y{:d}mmtz_theta_{:d}'.format(theta, energy), '', bins, minbin, maxbin))
        mmtz_gan.append(ROOT.TH1F('GANy{:d}mmtz_theta_{:d}GeV'.format(theta, energy), '', bins, minbin, maxbin))
        mmtz_act[i].SetLineColor(2)
        mmtz_gan[i].SetLineColor(4)
        mmtz_gan[i].SetLineStyle(2)
        #mmtz_gan[i].SetMarkerStyle(3)
        mmtz_act[i].Sumw2()
        mmtz_gan[i].Sumw2()
        my.fill_hist(mmtz_act[i], mz_act[i])
        my.fill_hist(mmtz_gan[i], mz_gan[i])
        if states==0:
            mmtz_act[i].SetStats(states)
            mmtz_gan[i].SetStats(states)
        if norm:
            mmtz_act[i]=my.normalize(mmtz_act[i], 1)
            mmtz_gan[i]=my.normalize(mmtz_gan[i], 1)
            #mmtz_act[i].GetYaxis().SetRangeUser(0., 0.2)
        mmtz_act[i].GetXaxis().SetTitle("2nd Z moment")
        mmtz_act[i].GetYaxis().SetTitle("normalized count")
        mmtz_act[i].GetYaxis().CenterTitle()
        mmtz_act[i].GetYaxis().SetTitleOffset()
        mmtz_act[i].GetYaxis().SetTitleSize(0.055)
        mmtz_act[i].GetXaxis().SetTitleSize(0.05)
        if i==0:
          ymax = max(mmtz_act[i].GetMaximum(), mmtz_gan[i].GetMaximum())
        else:
          ymax = max(ymax, mmtz_act[i].GetMaximum(), mmtz_gan[i].GetMaximum())

        mmtz_act[i].GetYaxis().SetRangeUser(0., 1.2 * ymax)

        canvas.Update()
        mmtz_act[i].Draw('')
        mmtz_act[i].Draw('sames hist')
        canvas.Update()

        mmtz_gan[i].Draw('sames')
        mmtz_gan[i].Draw('sames hist')
        canvas.Update()
        if leg:
            legs[pad-1].AddEntry(mmtz_act[i], "G4", 'l')
            legs[pad-1].AddEntry(mmtz_gan[i], "GAN", 'l')
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
        

                                                                      

                                                                                                                                                    
                            
