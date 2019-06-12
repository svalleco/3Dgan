import numpy as np
import ROOT
import os
import sys
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/analysis')
import utils.GANutils as gan

epochs = np.arange(60)

result = np.array([0.03713994,  0.03678386,  0.03201763,  0.02883984,  0.02663808,  0.03088229, 0.02635102,  0.01341125,  0.01793237,  0.0290715,   0.01881253,  0.00886392, 0.01717307,  0.02223882,  0.01832704,  0.01291435,  0.00622016,  0.02725404, 0.02293936,  0.00884851,  0.01031577,  0.01686671,  0.02305532,  0.01544341, 0.02109731,  0.01237523,  0.01025029,  0.00702774,  0.01468448,  0.01101928, 0.01257938,  0.00921587,  0.01517361,  0.00805735,  0.01238083,  0.01961428, 0.01453228,  0.01937219,  0.01145564,  0.01673119,  0.01087116,  0.01211864, 0.01114888,  0.01392543,  0.01103229,  0.01020601,  0.00897073,  0.00816365, 0.01320744,  0.01989233,  0.00783365,  0.00944511,  0.01527615, 0., 0.01200496, 0., 0., 0., 0., 0.01381917])
start = 0
fits = 'pol2'
resultdir = 'gromov/'
gan.safe_mkdir(resultdir)
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
legend.AddEntry(gw, "Gromov Wass = {:.4f} (epoch {})".format(minr, minr_n), "l")
gw.SetTitle("Optimization function: Gromov Wasserstein;Epochs;loss")
gw.Draw('ALP')
c1.Update()
legend.Draw()
c1.Update()
c1.Print(os.path.join(resultdir, "result.pdf"))

for i, fit in enumerate(fits):
    gw.Fit(fit)
    gw.GetFunction(fit).SetLineColor(color[i])
    gw.GetFunction(fit).SetLineStyle(2)
    if i == 0:
        legend.AddEntry(gt.GetFunction(fit), 'fit', "l")
legend.Draw()
c1.Update()
c1.Print(os.path.join(resultdir, "result_{}.pdf".format(fit)))
print ('The plot is saved to {}'.format(resultdir))
                                                                                                                                                                                                            

 
