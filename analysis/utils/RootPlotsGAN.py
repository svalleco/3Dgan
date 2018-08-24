########### Plots for Angle version ###################
import os
import ROOT
import numpy as np
import math
import time
import numpy.core.umath_tests as umath
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import RootPlotsGAN as rp # common root plots
import ROOTutils as my # utility functions for root
from GANutils import safe_mkdir # utility functions for variable angle
 
# Plot histogram of predicted angle 
def plot_ang1_hist(ang1, ang2, out_file, angle, angtype, labels, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make 
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.8, .1, .9, .2)
   hps=[]
   hgs=[]
   
   for i, key in enumerate(ang1):
     hps.append(ROOT.TH1F("G4" + labels[i] ,"G4" + labels[i], 50, 0, 3))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 3))
     hp= hps[i]
     hg= hgs[i]
     hp.Sumw2()
     hg.Sumw2()
     if i== 0:
       #hp.SetStats(0)
       hp.SetTitle("Predicted Angle for Global" + angtype)
       hp.GetXaxis().SetTitle(angtype + ' (radians)')
       hp.GetYaxis().SetTitle('Count')
       hp.GetYaxis().CenterTitle()
       my.fill_hist(hp, ang1[key])
       hp.Draw()
       hp.Draw('sames hist')
       c1.Update()
       hp.SetLineColor(color)
       c1.Update()
       legend.AddEntry("G4" + labels[i],"G4" + labels[i] + '1',"l")
       color+=2
     else:
       my.fill_hist(hp, ang1[key])
       hp.Draw('sames')
       hp.Draw('sames hist')
       c1.Update()
       legend.AddEntry(hp,"G4" + labels[i],"l")
       color+=1

     hp.SetTitle("Angle Histogram for {:.2f} {}".format(angle, angtype) )
     #hg.SetStats(0)
     #my.stat_pos(hg)
     my.fill_hist(hg, ang2[key])
     hp =my.normalize(hp)
     hg =my.normalize(hg)
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     my.stat_pos(hg)
     c1.Update()
     legend.AddEntry(hg, "GAN {}".format(labels[i] + ' 1'), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

# Plot histogram of predicted angle
def plot_ang2_hist(ang1, ang2, out_file, angle, angtype, labels, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   #angle = angle - (math.pi/2)
   legend = ROOT.TLegend(.4, .8, .5, .9)
   hps=[]
   hgs=[]
   for i, key in enumerate(ang1):
      hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 50, 0, 3.))
      hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, 0, 3.))
      hp= hps[i]
      hg= hgs[i]
      hp.Sumw2()
      hg.Sumw2()
      if i== 0:
         #hp.SetStats(0)
         hp.SetTitle("Predicted Angle")
         hp.GetXaxis().SetTitle(angtype + ' radians')
         my.fill_hist(hp, ang1[key])
         hp.Draw()
         hp.Draw('sames hist')
         c1.Update()
         hp.SetLineColor(color)
         c1.Update()
         legend.AddEntry("G4" + labels[i],"G4" + labels[i] + '2',"l")
         color+=2
      else:
         my.fill_hist(hp, ang1[key])
         hp.Draw('sames')
         hp.Draw('sames hist')
         c1.Update()
         legend.AddEntry(hp,"G4" + labels[i] + ' 2',"l")
         color+=1
      hp.SetTitle("Angle Histogram for {:.2f} {}".format(angle, angtype) )
      #hg.SetStats(0)
      #my.stat_pos(hg)
      my.fill_hist(hg, ang2[key])
      hp =my.normalize(hp)
      hg =my.normalize(hg)
      hg.SetLineColor(color)
      color+=1
      hg.Draw('sames')
      hg.Draw('sames hist')
      c1.Update()
      my.stat_pos(hg)
      c1.Update()
      legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
      legend.Draw()
      c1.Modified()
      c1.Update()
      if ifpdf:
         c1.Print(out_file + '.pdf')
      else:
         c1.Print(out_file + '.C')
         # Calculate error for predicted angle


def plot_angle_error_hist(ang1, ang2, y, out_file, angle, angtype, labels, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.8, .1, .9, .2)
   hps=[]
   hgs=[]
   if y.shape[0]> ang1["n_0"].shape[0]:
      y = y[:ang1["n_0"].shape[0]]
   for i, key in enumerate(ang1):
     hps.append(ROOT.TH1F("G4" + labels[i],"G4" + labels[i], 50, -1.0, 1.0))
     hgs.append(ROOT.TH1F("GAN" + labels[i], "GAN" + labels[i], 50, -1.0, 1.0))
     hp= hps[i]
     hg= hgs[i]
     hp.Sumw2()
     hg.Sumw2()
     if i== 0:
       #hp.SetStats(0)
       hp.SetTitle("Angle Relative Error Histogram for {:.2f} {}".format(angle, angtype) )
       hp.GetXaxis().SetTitle(angtype + " radians")
       hp.GetYaxis().SetTitle("(angle_{act} - angle_{pred})/angle_{act}")
       hp.GetYaxis().CenterTitle()
       my.fill_hist(hp, (y - ang1[key])/y)
       hp.Draw()
       hp.Draw('sames hist')
       c1.Update()
       hp.SetLineColor(color)
       c1.Update()
       legend.AddEntry(hp,"G4 " + labels[i],"l")
       color+=2
     else:
       my.fill_hist(hp, (y - ang1[key])/y)
       hp.Draw('sames')
       hp.Draw('sames hist')
       c1.Update()
       legend.AddEntry(hp,"G4" + labels[i],"l")
       color+=1
     #hg.SetStats(0)
     #my.stat_pos(hg)
     my.fill_hist(hg,  (y - ang2[key])/y)
     hp =my.normalize(hp)
     hg =my.normalize(hg)
     hg.SetLineColor(color)
     color+=1
     hg.Draw('sames')
     hg.Draw('sames hist')
     c1.Update()
     my.stat_pos(hg)
     c1.Update()
     legend.AddEntry(hg, "GAN {}".format(labels[i]), "l")
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C')

def plot_angle_2Dhist(ang1, ang2, y, out_file, angtype, labels, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   color = 2
   legend = ROOT.TLegend(.4, .8, .5, .9)
   hps=[]
   for i, key in enumerate(ang1):
      hps.append(ROOT.TH2F("G4" + labels[i],"G4" + labels[i], 50, 0.5, 3, 50, 0.5, 2.5))
      hp= hps[i]
      n = y.shape[0]
      if i== 0:
        hp.SetStats(0)
        hp.SetTitle("2D Histogram for predicted angles from G4 and GAN images" )
        hp.GetXaxis().SetTitle("3d Angle from G4")
        hp.GetYaxis().SetTitle("3d Angle from GAN")
        for j in np.arange(n):
          hp.Fill(ang1[key][j], ang2[key][j])
    
        hp.Draw("colz")
        c1.Update()
        hp.SetLineColor(color)
        c1.Update()
        legend.AddEntry(hp,"G4" + labels[i],"l")
        color+=2
      else:
        for j in np.arange(n): 
          hp.Fill(ang1[key][j], ang2[key][j])
        hp.Draw('sames')
        c1.Update()
        legend.AddEntry(hp,"G4" + labels[i],"l")
        color+=1
      c1.Update()
      c1.Modified()
      c1.Update()
      if ifpdf:
        c1.Print(out_file + '.pdf')
      else:
        c1.Print(out_file + '.C')
                                                                                                                                                                 
def get_plots_angle(var, labels, plots_dir, energies, angles, angtype, aindexes, m, n, ifpdf=True, stest=True, nloss=3):
    actdir = plots_dir + 'Actual'
    safe_mkdir(actdir)
    discdir = plots_dir + 'disc_outputs'
    safe_mkdir(discdir)
    gendir = plots_dir + 'Generated'
    safe_mkdir(gendir)
    comdir = plots_dir + 'Combined'
    safe_mkdir(comdir)
    mdir = plots_dir + 'Moments'
    safe_mkdir(mdir)
    start = time.time()
    plots = 0
    for energy in energies:

       x=var["events_act" + str(energy)].shape[1]
       y=var["events_act" + str(energy)].shape[2]
       z=var["events_act" + str(energy)].shape[3]
       maxfile = "Position_of_max_" + str(energy)                                                        
       maxlfile = "Position_of_max_" + str(energy)                                                   
       histfile = "hist_" + str(energy)
       histlfile = "hist_log" + str(energy)
       ecalfile = "ecal_" + str(energy)
       energyfile = "energy_" + str(energy)
       realfile = "realfake_" + str(energy)
       momentfile = "moment" + str(energy)
       auxfile = "Auxilliary_"+ str(energy)
       ecalerrorfile = "ecal_error" + str(energy)
       angfile = "angle_"+ str(energy)
       aerrorfile = "error_"
       allfile = 'All_energies'
       allecalfile = 'All_ecal'
       allecalrelativefile = 'All_ecal_relative'#.pdf'                                                             
       allauxrelativefile = 'All_aux_relative'#.pdf'                                                               
       allerrorfile = 'All_relative_auxerror'#.pdf'                                                                
       correlationfile = 'Corr'
       start = time.time()
       if energy==0:
          rp.plot_ecal_ratio_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                                     var["energy" + str(energy)], labels, os.path.join(comdir, allecalfile))
          plots+=1

          rp.plot_ecal_relative_profile(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)],
                                     var["energy" + str(energy)], labels, os.path.join(comdir, allecalrelativefile))
          plots+=1
                    
          rp.plot_aux_relative_profile(var["aux_act" + str(energy)], var["aux_gan"+ str(energy)], 
                                     var["energy"+ str(energy)], os.path.join(comdir, allauxrelativefile), labels)
          plots+=1
          """
          rp.plot_correlation(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], 
                           var["sumsz_act"+ str(energy)], var["momentX_act" + str(energy)], 
                           var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)], 
                           var["ecal_act" + str(energy)],  var["sumsx_gan"+ str(energy)], 
                           var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], 
                           var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], 
                           var["momentZ_gan" + str(energy)], var["ecal_gan" + str(energy)], 
                           var["energy" + str(energy)], var["events_act" + str(energy)], 
                           var["events_gan" + str(energy)], os.path.join(comdir, correlationfile), labels)
          """
       rp.plot_ecal_hist(var["ecal_act" + str(energy)], var["ecal_gan" + str(energy)], 
                         os.path.join(discdir, ecalfile), energy, labels, stest=stest)
       plots+=1

       """
       rp.plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], 
                                 os.path.join(comdir, 'flat' + 'log' + ecalfile), energy, labels, log=1)                                                                            
       plots+=1

       rp.plot_ecal_flatten_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)],
                                                                  os.path.join(comdir, 'flat' + ecalfile), energy, labels)  
       plots+=1
       """      
       
       rp.plot_ecal_hits_hist(var["events_act" + str(energy)], var["events_gan" + str(energy)], 
                         os.path.join(comdir, 'hits' + ecalfile), energy, labels)
       plots+=1
       rp.plot_aux_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)] , 
                         os.path.join(discdir, energyfile), energy, labels)
       plots+=1
       rp.plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)], 
                  x, y, z, os.path.join(actdir, maxfile), os.path.join(gendir, maxfile), 
                  os.path.join(comdir, maxfile), energy, labels, stest=stest)
       plots+=1
       rp.plot_max(var["max_pos_act" + str(energy)], var["max_pos_gan" + str(energy)], 
                  x, y, z, os.path.join(actdir, maxlfile), 
                 os.path.join(gendir, maxlfile), os.path.join(comdir, 'log' + maxlfile), 
                   energy, labels, log=1, stest=stest)
       plots+=1
       rp.plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], 
                             var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)], 
                             var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], 
                             x, y, z, os.path.join(actdir, histfile), os.path.join(gendir, histfile), 
                             os.path.join(comdir, histfile), energy, labels, stest=stest)
       plots+=1
       rp.plot_energy_hist_root(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], 
                             var["sumsz_act"+ str(energy)], var["sumsx_gan"+ str(energy)], 
                             var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], 
                             x, y, z, os.path.join(actdir, histlfile), os.path.join(gendir, histlfile), 
                             os.path.join(comdir, histlfile), energy, labels, log=1, stest=stest)
       plots+=1
       rp.plot_realfake_hist(var["isreal_act" + str(energy)], var["isreal_gan" + str(energy)], 
                             os.path.join(discdir, realfile), energy, labels)
       plots+=1
       rp.plot_primary_error_hist(var["aux_act" + str(energy)], var["aux_gan" + str(energy)], 
                             var["energy" + str(energy)], os.path.join(discdir, 'error_' + energyfile), energy, labels)
       plots+=1

       plot_angle_2Dhist(var["angle_act" + str(energy)], var["angle_gan" + str(energy)],  var["angle" + str(energy)],
                             os.path.join(discdir, angfile + "ang_2D") , angtype, labels)
       plots+=1
       if nloss==4:
          plot_angle_2Dhist(var["angle2_act" + str(energy)], var["angle2_gan" + str(energy)],  var["angle" + str(energy)],
                             os.path.join(discdir, angfile + "ang2_2D") , angtype, labels)
          plots+=1
                 
       for mmt in range(m):     
          rp.plot_moment(var["momentX_act" + str(energy)], var["momentX_gan" + str(energy)], 
                             os.path.join(mdir, 'x' + str(mmt + 1) + momentfile), 'x', energy, mmt, labels)
          plots+=1
          rp.plot_moment(var["momentY_act" + str(energy)], var["momentY_gan" + str(energy)], 
                             os.path.join(mdir, 'y' + str(mmt + 1) + momentfile), 'y', energy, mmt, labels)
          plots+=1
          rp.plot_moment(var["momentZ_act" + str(energy)], var["momentZ_gan" + str(energy)], 
                             os.path.join(mdir, 'z' + str(mmt + 1) + momentfile), 'z', energy, mmt, labels)
          plots+=1

       ecomdir = os.path.join(comdir, 'energy_' + str(energy))
       safe_mkdir(ecomdir)
       ediscdir = os.path.join(discdir, 'energy_' + str(energy))
       safe_mkdir(ediscdir)
       eactdir = os.path.join(actdir, 'energy_' + str(energy))
       safe_mkdir(eactdir)
       egendir = os.path.join(gendir, 'energy_' + str(energy))
       safe_mkdir(egendir)
       for a, angle in zip(aindexes, angles):
          #alabels = ['ang_' + str() for _ in aindexes]
          alabels = ['angle_' for _ in aindexes]
          a2labels = ['angle2_' for _ in aindexes]
          rp.plot_energy_hist_root(var["sumsx_act"+ str(energy) + "ang_" + str(a)], var["sumsy_act"+ str(energy)+ "ang_" + str(a)], 
                                var["sumsz_act"+ str(energy) + "ang_" + str(a)], var["sumsx_gan"+ str(energy)+ "ang_" + str(a)], 
                                var["sumsy_gan"+ str(energy)+ "ang_" + str(a)], var["sumsz_gan"+ str(energy)+ "ang_" + str(a)], 
                                 x, y, z, os.path.join(eactdir, histfile + 'ang_' + str(a)), os.path.join(egendir, histfile+ 'ang_' + str(a)), 
                                 os.path.join(ecomdir, histfile+ 'ang_' + str(a)), energy, alabels, stest=stest)
          plots+=1
          rp.plot_energy_hist_root(var["sumsx_act"+ str(energy) + "ang_" + str(a)], var["sumsy_act"+ str(energy)+ "ang_" + str(a)],
                                var["sumsz_act"+ str(energy) + "ang_" + str(a)], var["sumsx_gan"+ str(energy)+ "ang_" + str(a)],
                                var["sumsy_gan"+ str(energy)+ "ang_" + str(a)], var["sumsz_gan"+ str(energy)+ "ang_" + str(a)],
                                 x, y, z, os.path.join(eactdir, histfile + 'ang_' + str(a)), os.path.join(egendir, histfile+ 'ang_' + str(a)),
                                 os.path.join(ecomdir, histfile+ 'logang_' + str(a)), energy, alabels, log=1, stest=stest)

          plots+=1
          plot_ang1_hist(var["angle_act" + str(energy) + "ang_" + str(a)], var["angle_gan" + str(energy) + "ang_" + str(a)] , 
                                 os.path.join(ediscdir, "ang_" + str(a)), angle, angtype, alabels)
          plots+=1
          if nloss==4:
              plot_ang2_hist(var["angle2_act" + str(energy) + "ang_" + str(a)], var["angle2_gan" + str(energy) + "ang_" + str(a)] ,
                                 os.path.join(ediscdir, "ang2_" + str(a)), angle, angtype, a2labels)
          plots+=1
          rp.plot_realfake_hist(var["isreal_act" + str(energy) + "ang_" + str(a)], var["isreal_gan" + str(energy)+ "ang_" + str(a)], 
                                 os.path.join(ediscdir, realfile  + "ang_" + str(a)), angle, alabels)
          plots+=1
          rp.plot_primary_error_hist(var["aux_act" + str(energy) + "ang_" + str(a)], var["aux_gan" + str(energy) + "ang_" + str(a)], 
                                  var["energy" + str(energy) + "ang_" + str(a)], 
                                 os.path.join(ediscdir, 'error_' + energyfile + "ang_" + str(a)), energy, alabels)
          plots+=1
         
          plot_angle_error_hist(var["angle_act" + str(energy) + "ang_" + str(a)], var["angle_gan" + str(energy) + "ang_" + str(a)], 
                                var["angle" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, aerrorfile + "ang1_" + str(a)), 
                                 angle, angtype, alabels)

          plots+=1
          if nloss==4:
              plot_angle_error_hist(var["angle2_act" + str(energy) + "ang_" + str(a)], var["angle2_gan" + str(energy) + "ang_" + str(a)],
                                var["angle" + str(energy) + "ang_" + str(a)], os.path.join(ediscdir, aerrorfile + "ang2_" + str(a)),
                                  angle, angtype, a2labels)
              plots+=1          

    print 'Plots are saved in ', plots_dir
    plot_time= time.time()- start
    print '{} Plots are generated in {} seconds'.format(plots, plot_time)
