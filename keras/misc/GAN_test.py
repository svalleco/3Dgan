from __future__ import absolute_import, division, print_function, unicode_literals
import h5py as h5
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

data = h5.File("validation_results.h5")
GAN_ID = 0
GEANT_ID = 1
particle = 'Ele'
save_path = 'results/triforce_plots_Ele2/'
gan.safe_mkdir(save_path)
leg = False
def main():
    GAN_indices = np.array(data['pdgID']) == GAN_ID
    GEANT_indices = np.array(data['pdgID']) == GEANT_ID
    
    GAN_energy = data['energy'][GAN_indices]
    GEANT_energy = data['energy'][GEANT_indices]
    GAN_reg_energy_prediction = data['reg_energy_prediction'][GAN_indices]
    GEANT_reg_energy_prediction = data['reg_energy_prediction'][GEANT_indices]
    for key in data:
       print( key)
    rcParams['axes.titlepad'] = 20
    plt.scatter(GAN_energy, GAN_reg_energy_prediction, s=1, label='GAN')
    plt.scatter(GEANT_energy, GEANT_reg_energy_prediction, s=1, label='GEANT')
    plt.title("Energy Predictions from Regression Nets for GAN and GEANT4 on {} Samples".format(particle))
    plt.xlabel("True Energy")
    plt.ylabel("Predicted Energy")
    plt.legend()
    plt.savefig(save_path + "Pi0_GAN_GEANT_energy_regression_comparison.pdf")

    GAN_class_prediction = data['class_prediction'][GAN_indices]
    GEANT_class_prediction = data['class_prediction'][GEANT_indices]

    GAN_class_prediction_accuracy = np.sum(GAN_class_prediction) / len(GAN_class_prediction)
    GEANT_class_prediction_accuracy = np.sum(GEANT_class_prediction) / len(GEANT_class_prediction)
    GAN_error = (np.sqrt(GAN_class_prediction.shape[0]) * GAN_class_prediction_accuracy)/GAN_class_prediction.shape[0]
    GEANT_error =(np.sqrt(GEANT_class_prediction.shape[0]) * GEANT_class_prediction_accuracy)/GEANT_class_prediction.shape[0]
    print(GAN_error, GEANT_error)
    plt.clf()
    width = 0.4
    x_val = np.arange(4)
    plt.bar(1, [GAN_class_prediction_accuracy], width, yerr = GAN_error, align='center', alpha=0.5, color='b', capsize=4)
    plt.bar(2, [GEANT_class_prediction_accuracy], width, yerr = GEANT_error, align='center', alpha=0.5, color='r', capsize=4)
    plt.xticks(range(4), ["", "GAN", "GEANT", ""])
    for i, v in enumerate([GAN_class_prediction_accuracy, GEANT_class_prediction_accuracy]):
       plt.text(x_val[i+1]-0.03 , v+0.03 , '{:.2f}'.format(v))
    plt.ylim([0., max(GAN_class_prediction_accuracy, GEANT_class_prediction_accuracy) * 1.1])
    plt.title("Accuracy of Classification Nets on GAN and GEANT4 {} Samples".format(particle))
    plt.savefig(save_path + particle + "_GAN_GEANT_class_accuracy_comparison.pdf")
    PlotRegressionProf(GEANT_reg_energy_prediction, GEANT_energy, GAN_reg_energy_prediction, GAN_energy, particle, save_path + particle + "_regression_prof.pdf", leg=leg)
    PlotRegressionScat(GEANT_reg_energy_prediction, GEANT_energy, GAN_reg_energy_prediction, GAN_energy, particle, save_path + particle + "_regression_scat.pdf", leg=leg)
    PlotClassBar(GEANT_class_prediction_accuracy, GEANT_error, GAN_class_prediction_accuracy, GAN_error, particle, save_path + particle + "_classification_bar.pdf", leg=True)
   
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

def PlotRegressionProf(g4_reg, g4_e, gan_reg, gan_e, particle, out_file, leg=True):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
    p =[int(np.amin(g4_e)), int(np.amax(g4_e))]
    title = "Predicted primary energy for {}".format(particle)
    legend = ROOT.TLegend(.1, .7, .3, .9)
    color =2
   
    pg4 = ROOT.TProfile("G4", "G4", 100, p[0], p[1]*1.1)
    pgan = ROOT.TProfile("GAN", "GAN", 100, p[0], p[1]*1.1)
    pg4.SetStats(0)
    pgan.SetStats(0)
    r.fill_profile(pg4, g4_e, g4_reg)
    r.fill_profile(pgan, gan_e, gan_reg)
    pg4.SetLineColor(2)
    pgan.SetLineColor(4)
    pg4.SetTitle(title)
    pg4.GetXaxis().SetTitle("Ep [GeV]")
    pg4.GetYaxis().SetTitle("Predicted Ep [GeV]")
    pg4.GetYaxis().CenterTitle()
    pg4.Draw()
    pg4.Draw('sames hist')
    pgan.Draw('sames')
    pgan.Draw('sames hist')
    legend.AddEntry(pg4, 'G4' ,"l")
    legend.AddEntry(pgan, 'GAN' ,"l")
    c1.Modified()
    c1.Update()

    if leg:legend.Draw()
    c1.Print(out_file)

def PlotRegressionScat(g4_reg, g4_e, gan_reg, gan_e, particle, out_file, leg=True):
    c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make                                                                                                                                                                                                    
    p =[int(np.amin(g4_e)), int(np.amax(g4_e))]
    title = "Predicted primary energy for {}".format(particle)
    legend = ROOT.TLegend(.1, .7, .3, .9)
    legend.SetBorderSize(0)
    color =2
    mg = ROOT.TMultiGraph()
    pg4 = ROOT.TGraph()
    pgan = ROOT.TGraph()

    r.fill_graph(pg4, g4_e, g4_reg)
    r.fill_graph(pgan, gan_e, gan_reg)
    pg4.SetMarkerColor(2)
    pgan.SetMarkerColor(4)
    mg.Add(pgan)
    mg.Add(pg4)
    mg.SetTitle(title)
    mg.GetXaxis().SetTitle("Ep [GeV]")
    mg.GetYaxis().SetTitle("Predicted Ep [GeV]")
    mg.GetYaxis().CenterTitle()
    mg.Draw('AP')
    
    legend.AddEntry(pg4, 'G4' ,"p")
    legend.AddEntry(pgan, 'GAN' ,"p")
    c1.Modified()
    c1.Update()

    if leg:legend.Draw()
    c1.Print(out_file)

   
if __name__ == "__main__":
    main()
