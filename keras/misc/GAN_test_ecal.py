from __future__ import absolute_import, division, print_function, unicode_literals
import h5py 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams.update({'lines.markeredgewidth': 1})
import ROOT
import sys
sys.path.insert(0,'../keras/analysis')
import utils.ROOTutils as r
import utils.GANutils as gan

def main():
    validation_files =["validation_GANtrain_G4test.h5", "validation_GANtrain_GANtest.h5", "validation_G4train_G4test.h5", "validation_G4train_GANtest_ECAL.h5"]
    data = []
    for f in validation_files:
      data.append(h5py.File(f, 'r'))    
    save_path = 'results/triforce_validation_net_out_wthcal/'
    gan.safe_mkdir(save_path)
    ###################################################
    Ele_ID = 11
    ChPi_ID = 211
    GAN_ID = 0
    GEANT_ID = 1
    num_events  = 5000
    leg = True
    particles = ['electrons', 'charged pions']
    
    ###################################################
    results =[]
    for i, d in enumerate(data):
      params ={}
      Ele_indices = np.absolute(np.array(d['pdgID'])) == Ele_ID
      ChPi_indices = np.absolute(np.array(d['pdgID'])) == ChPi_ID
      params['Ele_energy'] = d['energy'][Ele_indices][:num_events]
      params['ChPi_energy'] = d['energy'][ChPi_indices][:num_events]
      params['ChPi_ecal'] = d['ECAL_E'][ChPi_indices][:num_events]
      params['ChPi_hcal'] = d['HCAL_E'][ChPi_indices][:num_events]
      params['Ele_ecal'] = d['ECAL_E'][Ele_indices][:num_events]
      params['Ele_hcal'] = d['HCAL_E'][Ele_indices][:num_events]
      params['Ele_ratio'] = params['Ele_hcal']/params['Ele_ecal']
      params['Ele_sum'] = params['Ele_hcal']+ params['Ele_ecal']
      params['Ele_reg_energy_prediction'] = d['reg_energy_prediction'][Ele_indices][:num_events]
      params['ChPi_reg_energy_prediction'] = d['reg_energy_prediction'][ChPi_indices][:num_events]
      params['Ele_raw'] = inv_tf(params['Ele_reg_energy_prediction'], params['Ele_ecal'] + params['Ele_hcal'], f=10)
      params['ChPi_raw'] = inv_tf(params['ChPi_reg_energy_prediction'], params['ChPi_ecal'] + params['ChPi_hcal'], f=10)
      params['Ele_out'] = d['reg_energy_output'][Ele_indices][:num_events]
      params['ChiP_out'] = d['reg_energy_output'][ChPi_indices][:num_events]
      params['Ele_class_prediction'] = d['class_prediction'][Ele_indices][:num_events]
      params['ChPi_class_prediction'] = d['class_prediction'][ChPi_indices][:num_events]
      params['Ele_class_prediction_accuracy'] = float(sum(params['Ele_class_prediction'])) / float(len(params['Ele_class_prediction']))
      params['ChPi_class_prediction_accuracy'] = 1.0 - (float(sum(params['ChPi_class_prediction'])) / float(len(params['ChPi_class_prediction'])))
      params['Ele_error'] = (np.sqrt(params['Ele_class_prediction'].shape[0]) * params['Ele_class_prediction_accuracy'])/params['Ele_class_prediction'].shape[0]
      params['ChPi_error'] =(np.sqrt(params['ChPi_class_prediction'].shape[0]) * params['ChPi_class_prediction_accuracy'])/params['ChPi_class_prediction'].shape[0]
      print('The validation results for {}'.format(validation_files[i]))
      print('The number of electrons events was {}'.format(params['Ele_class_prediction'].shape[0]))
      print('The number of charged pions events was {}'.format(params['ChPi_class_prediction'].shape[0]))
      print('minimum Ele energy is {}'.format(np.amin(params['Ele_energy'])))
      print('minimum ChPi energy is {}'.format(np.amin(params['ChPi_energy'])))
      print('The number of electrons events was {}'.format(params['Ele_class_prediction'].shape[0]))
      print('The number of charged pions events was {}'.format(params['ChPi_class_prediction'].shape[0]))
      print('maximum Ele energy is {}'.format(np.amax(params['Ele_energy'])))
      print('maximum ChPi energy is {}'.format(np.amax(params['ChPi_energy'])))
      print('maximum Ele predicted energy is {}'.format(np.amax(params['Ele_reg_energy_prediction'])))
      print('maximum ChPi predicted energy is {}'.format(np.amax(params['ChPi_reg_energy_prediction'])))
      print('##################################################################################')

      results.append(params)
    p = particles[0]
    title = 'trained on GAN $e^{-}$ and G4 $\pi$'
    rtitle = 'trained on GAN e and G4 pions' 
    PLotClassBarPython([results[0]['Ele_class_prediction_accuracy'], results[1]['Ele_class_prediction_accuracy']], 
                        [results[0]['Ele_error'], results[1]['Ele_error']], 
                        ['G4', 'GAN'], p, title, save_path + p + '_GAN_train_class_accuracy')
    PlotRegressionProf([results[0]['Ele_reg_energy_prediction'], results[1]['Ele_reg_energy_prediction']],
                       [results[0]['Ele_energy'], results[1]['Ele_energy']], 
                       p, ['G4', 'GAN'], rtitle, save_path + p + "_GAN_train_regression_prof.pdf", leg=leg)
    PlotRegressionScat([results[0]['Ele_reg_energy_prediction'], results[1]['Ele_reg_energy_prediction']],
                       [results[0]['Ele_energy'], results[1]['Ele_energy']], p, ['G4', 'GAN'], rtitle, save_path + p + "_GAN_train_regression_scat.pdf", leg=leg )
   
     
    title = 'trained on G4 $e^{-}$ and G4 $\pi$'
    rtitle = 'trained on G4 e and G4 pions'
    PLotClassBarPython([results[2]['Ele_class_prediction_accuracy'], results[3]['Ele_class_prediction_accuracy']],
                        [results[2]['Ele_error'], results[3]['Ele_error']],
                        ['G4', 'GAN'], p, title, save_path + p + '_G4_train_class_accuracy')
    PlotRegressionProf([results[2]['Ele_reg_energy_prediction'], results[3]['Ele_reg_energy_prediction']],
                       [results[2]['Ele_energy'], results[3]['Ele_energy']],
                       p, ['G4', 'GAN'], rtitle, save_path + p + "_G4_train_regression_prof.pdf", leg=leg)

    PlotRegressionScat([results[2]['Ele_reg_energy_prediction'], results[3]['Ele_reg_energy_prediction']],
                       [results[2]['Ele_energy'], results[3]['Ele_energy']], p, ['G4', 'GAN'], rtitle, save_path + p + "_G4_train_regression_scat.pdf", leg=leg )
     
    PlotRegressionScat([results[2]['Ele_reg_energy_prediction']-results[2]['Ele_hcal'], results[3]['Ele_reg_energy_prediction'] - results[3]['Ele_hcal']],
                       [results[2]['Ele_energy'], results[3]['Ele_energy']], p, ['G4', 'GAN'], rtitle, save_path + p + "_G4_train_regression_scat_wthcal.pdf", leg=leg )


    PlotRegressionScat([results[2]['Ele_raw'], results[3]['Ele_raw']],
                       [results[2]['Ele_energy'], results[3]['Ele_energy']], p, ['G4 reco raw', 'GAN reco raw'], rtitle, save_path + p + "_G4_train_regression_inv_scat.pdf", error =False, leg=leg )

    PlotRegressionScat([results[2]['Ele_out'], results[3]['Ele_out']],
                       [results[2]['Ele_energy'], results[3]['Ele_energy']], p, ['G4 net out', 'GAN net out'], rtitle, save_path + p + "_G4_train_regression_raw_scat.pdf", error =False, leg=leg )


    PlotRegressionScat([results[2]['Ele_sum'], results[3]['Ele_sum']],
                       [results[2]['Ele_energy'], results[3]['Ele_energy']], p, ['G4 sum', 'GAN sum'], rtitle, save_path + p + "_G4_train_regression_sum_scat.pdf", leg=leg )

    PlotRegressionScat([results[2]['Ele_ecal'], results[3]['Ele_ecal']],
                       [results[2]['Ele_energy'], results[3]['Ele_energy']], p, ['G4 ecal sum', 'GAN ecal sum'], rtitle, save_path + p + "_G4_train_regression_ecal_scat.pdf", leg=leg )

    PlotHist([results[2]['Ele_ratio'], results[3]['Ele_ratio']],
                       ['Ele', 'Ele'], ['G4 ratio', 'GAN ratio'], 'HCAL?ECAL', rtitle, save_path + p + "_G4_train_regression_ratio_scat.pdf", leg=leg )
    
    p = particles[1]
    title = 'trained on GAN $e^{-}$ and G4 $\pi$'
    rtitle = 'trained on GAN e and G4 pi'
    PLotClassBarPython([results[0]['ChPi_class_prediction_accuracy'], results[2]['ChPi_class_prediction_accuracy']],
                        [results[0]['ChPi_error'], results[2]['ChPi_error']],
                        ['net 1', 'net 2'], p, "", save_path + p + '_class_accuracy')
    PlotRegressionProf([results[0]['ChPi_reg_energy_prediction'], results[2]['ChPi_reg_energy_prediction']], 
                       [results[0]['ChPi_energy'], results[2]['ChPi_energy']],
                       p, ['net 1', 'net 2'], "", save_path + p + "_regression_prof.pdf", leg=leg)
    PlotRegressionScat([results[0]['ChPi_reg_energy_prediction'], results[2]['ChPi_reg_energy_prediction']], 
                       [results[0]['ChPi_energy'], results[2]['ChPi_energy']],
                       p, ['net1', 'net2'], rtitle, save_path + p + "_regression_scat.pdf", leg=leg )

def inv_tf(energy, ecal_sum, f=10.0):
    return (energy - ecal_sum)/f

def tf(raw, ecal_sum, f=10.0):
    return((raw * f) + ecal_sum) 

def PLotClassBarPython(class_list, error_list, labels, particle, title, out_file):
    plt.figure()
    width = 0.4
    x_val = np.arange(len(class_list) + 2)
    color = ['b', 'r', 'g']
    for i, class_pred in enumerate(class_list):
       plt.bar(i+1, [class_pred], width, yerr = error_list[i], align='center', alpha=0.5, color=color[i], capsize=4)
    
    plt.xticks(range(len(class_list) + 2), [""] + labels + [""])
    for i, v in enumerate(class_list):
       plt.text(x_val[i+1]-0.03 , v+0.03 , '{:.4f}'.format(v))
    plt.ylim([0.,  1.1])
    plt.title("Classification accuracy {} ({})".format(particle, title))
    plt.savefig(out_file + ".pdf")


def PlotClassBar(g4_class, g4_class_error, gan_class, gan_class_error, particle, out_file, leg=True):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    labels=['GEANT4', 'GAN']
    title = "Classification Accuracy for {}".format(particle)
    legend = ROOT.TLegend(.1, .7, .3, .9)
    color =2
    mg = ROOT.TMultiGraph()
        
    graph1 = ROOT.TGraphErrors()
    graph2 = ROOT.TGraphErrors()
    graph1.SetPoint(1, 1, g4_class)
    graph2.SetPoint(1, 2, gan_class)
    graph1.SetPointError(1, 0.5, g4_class_error)
    graph2.SetPointError(1, 0.5, gan_class_error)
    graph1.SetFillColorAlpha(2, 0.5)
    graph2.SetFillColorAlpha(4, 0.5)
    
    mg.Add(graph1)
    mg.Add(graph2)
    mg.GetXaxis().SetLimits(0, 3)

    mg.SetTitle(title)
    mg.GetYaxis().SetTitle("Accuracy")
    mg.GetYaxis().CenterTitle()
    ROOT.gStyle.SetBarWidth(0.5)
    mg.GetYaxis().SetRangeUser(0.,1.1)    
    mg.Draw('ABFE')
    
    legend.AddEntry(graph1, 'G4' ,"f")
    legend.AddEntry(graph2, 'GAN' ,"f")
    c1.Modified()
    c1.Update()

    if leg:legend.Draw()
    c1.Print(out_file)

def PlotClassBar2(g4_class, g4_class_error, gan_class, gan_class_error, particle, out_file, leg=True):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                           
    labels=['GEANT4', 'GAN']
    title = "Classification Accuracy for {}".format(particle)
    legend = ROOT.TLegend(.1, .7, .3, .9)
    color =2
    graph1 = ROOT.TH1F("g4", "", 2, 0, 2)
    graph2 = ROOT.TH1F("gan", "", 2, 0, 2)
    
    graph1.Fill(0, g4_class)
    
    graph2.Fill(1, gan_class)
    
    graph1.GetXaxis().SetBinLabel(1, 'G4')
    graph1.GetXaxis().SetBinLabel(2, 'GAN')
    graph1.SetStats(0)
    graph2.SetStats(0)
    graph1.SetBinError(1, g4_class_error)
    graph2.SetBinError(2, gan_class_error)
    graph1.SetFillColorAlpha(2, 0.5)
    graph2.SetFillColorAlpha(4, 0.5)
    graph1.SetBarWidth(0.4)
    graph1.SetBarOffset(0.1)
    graph1.GetXaxis().SetRangeUser(0, 3)
    graph1.SetTitle(title)
    graph1.GetYaxis().SetTitle("Accuracy")
    graph1.GetYaxis().CenterTitle()
    graph2.SetBarWidth(0.4)
    graph2.SetBarOffset(0.1)
    graph1.GetYaxis().SetRangeUser(0.,1.1)
    graph1.Draw('bf text0')
    graph2.Draw('bf text0 same')

    legend.AddEntry(graph1, 'G4' ,"f")
    legend.AddEntry(graph2, 'GAN' ,"f")
    c1.Modified()
    c1.Update()

    if leg:legend.Draw()
    c1.Print(out_file)

def PlotRegressionProf(reg_list, e_list, particle, labels, title, out_file, leg=True, reverse=0):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    p =[int(np.amin(e_list[0])), int(np.amax(e_list[0]))]
    title = "Predicted primary energy for {} ({})".format(particle, title)
    legend = ROOT.TLegend(.2, .6, .5, .8)
    legend.SetBorderSize(0)
    color =2
    profs =[]
    if reverse:
      reg_list, e_list, labels = reg_list[::-1], e_list[::-1], labels[::-1]
    for i, reg in enumerate(reg_list):
      profs.append(ROOT.TProfile(labels[i], labels[i], 100, p[0], p[1]*1.1))
      profs[i].SetStats(0)
      r.fill_profile(profs[i], e_list[i], reg_list[i])
      profs[i].SetLineColor(color)
      if i== 0:
        profs[i].SetTitle(title)
        profs[i].GetXaxis().SetTitle("Ep [GeV]")
        profs[i].GetYaxis().SetTitle("Predicted Ep [GeV]")
        profs[i].GetYaxis().CenterTitle()
        profs[i].Draw()
        profs[i].Draw('sames hist')
      else:
        profs[i].Draw('sames')
        profs[i].Draw('sames hist')
      legend.AddEntry(profs[i], labels[i] ,"l")
      c1.Modified()
      c1.Update()
      color+=2
    if leg:legend.Draw()
    c1.Print(out_file)

def PlotRegressionScat(reg_list, e_list, particle, labels, title, out_file, error=True, leg=True, reverse=0):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    p =[int(np.amin(e_list[0])), int(np.amax(e_list[0]))]
    title = "Predicted primary energy for {} ({})".format(particle, title)
    legend = ROOT.TLegend(.15, .7, .5, .85)
    legend.SetBorderSize(0)
    color = 2
    mg = ROOT.TMultiGraph()
    graphs=[]
    mins=[]  
    maxs=[]
    if reverse:
      reg_list, e_list, labels = reg_list[::-1], e_list[::-1], labels[::-1]

    for i, reg in enumerate(reg_list):
      graphs.append(ROOT.TGraph())
      r.fill_graph(graphs[i], e_list[i], reg)
      graphs[i].SetMarkerColor(color)
      graphs[i].SetLineColor(color)
      mg.Add(graphs[i])
      mins.append(np.amin(reg))
      maxs.append(np.amax(reg))
      if i== 0:
        mg.SetTitle(title)
        mg.GetXaxis().SetTitle("Ep [GeV]")
        mg.GetYaxis().SetTitle("Predicted Ep [GeV]")
        mg.GetYaxis().CenterTitle()
      if error:
        perror = np.absolute((e_list[i]-reg)/e_list[i])
        legend.AddEntry(graphs[i], labels[i] + ' MAE {:.4f}({:.4f})'.format(np.mean(perror), np.std(perror)),"l")
      else:
        legend.AddEntry(graphs[i], labels[i] + ' {:.4f}({:.4f})'.format(np.mean(reg), np.std(reg)),"l")
      color+=2
    ymin = 0 if min(mins) > 0 else -6
    ymax = 1.1 * max(maxs) if maxs>100 else 16 
                 
    mg.GetYaxis().SetRangeUser(ymin, ymax)
    mg.Draw('AP')
    c1.Modified()
    c1.Update()

    if leg:legend.Draw()
    c1.Print(out_file)

   
def PlotScat(ylist, xlist, particles, labels, labely, title, out_file, leg=True, p=[2, 500]):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    legend = ROOT.TLegend(.1, .7, .3, .9)
    legend.SetBorderSize(0)
    color =2
    mg = ROOT.TMultiGraph()
    graphs =[]
    
    for i, (x, y) in enumerate(zip(xlist, ylist)):
      graphs.append(ROOT.TGraph())
      r.fill_graph(graphs[i], x, y)
      graphs[i].SetMarkerColor(color)
      graphs[i].SetLineColor(color)
      mg.Add(graphs[i])
      legend.AddEntry(graphs[i], labels[i] ,"l")
      color+=1
    mg.SetTitle(title)
    mg.GetXaxis().SetTitle("Ep [GeV]")
    mg.GetYaxis().SetTitle(labely)
    mg.GetYaxis().CenterTitle()
    mg.Draw('AP')
    c1.Modified()
    c1.Update()

    if leg:legend.Draw()
    c1.Print(out_file)

def PlotHist(dlist, particles, labels, labelx, title, out_file, leg=True, p=[2, 500]):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500)
    legend = ROOT.TLegend(.1, .7, .3, .9)
    legend.SetBorderSize(0)
    color =2
    hists =[]
    
    for i, d in enumerate(dlist):
      hists.append(ROOT.TH1F("", "", 100, 0, 12))
      r.fill_hist(hists[i], d)
      hists[i].SetLineColor(color)
      legend.AddEntry(hists[i], labels[i] ,"l")
      color+=1
      if i==0:
        hists[i].SetTitle(title)
        hists[i].GetXaxis().SetTitle(labelx)
        hists[i].GetYaxis().SetTitle('count')
        hists[i].GetYaxis().CenterTitle()
        hists[i].Draw()
      else:
        hists[i].Draw('sames')
    c1.Modified()
    c1.Update()

    if leg:legend.Draw()
    c1.Print(out_file)



if __name__ == "__main__":
    main()
