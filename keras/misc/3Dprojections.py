# 2D projections for 3 D events from GAN and GEANT4
from os import path
import os
import sys
import numpy as np
import argparse
sys.path.insert(0,'../')
import utils.GANutils as gan
import utils.RootPlotsGAN as pl
import utils.ROOTutils as r
import ROOT
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
try:
    import setGPU #if Caltech
except:
    pass

def main():
   ROOT.gStyle.SetCanvasPreferGL(1)
   parser = get_parser()
   params = parser.parse_args()

   datapath =params.datapath
   events_per_file = params.eventsperfile
   energies = params.energies if isinstance(params.energies, list) else [params.energies]
   latent = params.latentsize
   particle= params.particle
   angtype= params.angtype
   plotsdir= params.outdir+'/'
   concat= params.concat
   gweight= params.gweight 
   xscale= params.xscale
   ascale= params.ascale
   yscale= params.yscale
   xpower= params.xpower 
   thresh = params.thresh
   dformat = params.dformat
   ang = params.ang
   logz = params.logz
   ntup = params.ntup
   ifC = params.ifC
   num = params.num
   gan.safe_mkdir(plotsdir) # make plot directory
   tolerance2=0.05
   opt="box2z"
   
   
   if ang:
     from AngleArch3dGAN import generator
     dscale=50.
     if not xscale:
       xscale=1.
     if not xpower:
       xpower = 0.85
     if not latent:
       latent = 256
     if not ascale:
       ascale = 1

     if datapath=='reduced':
       datapath = "/storage/group/gpu/bigdata/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV
       events_per_file = 5000
       energies = [0, 110, 150, 190]
     elif datapath=='full':
       datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
       events_per_file = 10000
       energies = [50, 100, 200, 300, 400, 500]
     else: 
       datapath = datapath + "/*scan/*scan_RandomAngle_*.h5"
     thetas = [62, 90, 118]      
   else:
     from EcalEnergyGan import generator
     dscale=1
     if not xscale:
       xscale=100.
     if not xpower:
       xpower = 1
     if not latent:
       latent = 200
     if not ascale:
       ascale = 1

     if datapath=='full':
       datapath ='/storage/group/gpu/bigdata/LCD/NewV1/*scan/*scan_*.h5'
       energies = [50, 100, 200, 300, 400, 500]
     else:
       datapath =  datapath+ "/*scan/*scan_*.h5"
     events_per_file = 10000    
     
   datafiles = gan.GetDataFiles(datapath, Particles=[particle]) # get list of files
   if ang:
     var = gan.get_sorted_angle(datafiles[-2:], energies, True, num_events1=1000, num_events2=1000, angtype=angtype, thresh=0.0)#get data from last two files
     g = generator(latent, dformat=dformat) 
     g.load_weights(gweight)
     for energy in energies: # for each energy bin
        edir = os.path.join(plotsdir, 'energy{}'.format(energy))
        gan.safe_mkdir(edir)
        rad = np.radians(thetas)
        for index, a in enumerate(rad): # for each angle bin
          adir = os.path.join(edir, 'angle{}'.format(thetas[index]))
          gan.safe_mkdir(adir)
          if a==0:
            var["events_act" + str(energy) + "ang_" + str(index)] = var["events_act" + str(energy)]/dscale # data in units of GeV * dscale
            var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)] # energy labels
            var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)]  # angle labels
            var["index" + str(energy)+ "ang_" + str(index)] = var["events_act" + str(energy) + "ang_" + str(index)].shape[0] # number of events
          else:
            indexes = np.where(((var["angle" + str(energy)]) > a - tolerance2) & ((var["angle" + str(energy)]) < a + tolerance2)) # all events with angle within a bin                                     
            var["events_act" + str(energy) + "ang_" + str(index)] = var["events_act" + str(energy)][indexes]/dscale
            var["energy" + str(energy) + "ang_" + str(index)] = var["energy" + str(energy)][indexes]
            var["angle" + str(energy) + "ang_" + str(index)] = var["angle" + str(energy)][indexes]
            var["index" + str(energy)+ "ang_" + str(index)] = var["events_act" + str(energy) + "ang_" + str(index)].shape[0]

          var["events_act" + str(energy) + "ang_" + str(index)] = applythresh(var["events_act" + str(energy) + "ang_" + str(index)], thresh) # remove energies below threshold
          var["events_gan" + str(energy) + "ang_" + str(index)]= gan.generate(g, var["index" + str(energy)+ "ang_" + str(index)],  # generate events
                                                                           [var["energy" + str(energy)+ "ang_" + str(index)]/yscale,
                                                                            (var["angle"+ str(energy)+ "ang_" + str(index)]) * ascale], latent, concat=2)
          var["events_gan" + str(energy) + "ang_" + str(index)]= inv_power(var["events_gan" + str(energy) + "ang_" + str(index)], xpower=xpower)/dscale # post processing
          var["events_gan" + str(energy) + "ang_" + str(index)]= applythresh(var["events_gan" + str(energy) + "ang_" + str(index)], thresh) # remove energies below threshold
          for n in np.arange(min(num, var["index" + str(energy)+ "ang_" + str(index)])): # plot events
            PlotEvent3d_python(var["events_act" + str(energy) + "ang_" + str(index)][n], var["events_gan" + str(energy) + "ang_" + str(index)][n],
                         var["energy" + str(energy) + "ang_" + str(index)][n],
                         var["angle" + str(energy) + "ang_" + str(index)][n],
                          os.path.join(adir, 'Event{}'.format(n)), n, opt=opt, logz=logz, ifC=ifC, ntup=ntup)

   else:
     g = generator(latent, dformat=dformat)
     g.load_weights(gweight)
     var = gan.get_sorted(datafiles[-2:], energies, True, num_events1=50, num_events2=50, thresh=0.0)#get data from last two files
     for energy in energies: # for each energy bin
        edir = os.path.join(plotsdir, 'energy{}'.format(energy))
        gan.safe_mkdir(edir)
        var["events_act" + str(energy)] = var["events_act" + str(energy)]/dscale # data in units of GeV * dscale
        var["energy" + str(energy)] = var["energy" + str(energy)] # energy labels
        var["index" + str(energy)] = var["events_act" + str(energy)].shape[0] # number of events
        var["events_act" + str(energy)] = applythresh(var["events_act" + str(energy)], thresh)
        var["events_gan" + str(energy)]= gan.generate(g, var["index" + str(energy)],
                                                      [var["energy" + str(energy)]/yscale], latent=latent)
        var["events_gan" + str(energy)]= var["events_gan" + str(energy)]/(xscale* dscale) # post processing
        var["events_gan" + str(energy)]= applythresh(var["events_gan" + str(energy)], thresh)# remove energies below threshold
        for n in np.arange(min(num, var["index" + str(energy)])): # plot events
            PlotEvent3d(var["events_act" + str(energy)][n], var["events_gan" + str(energy)][n],
                         var["energy" + str(energy)][n],
                         None,
                         os.path.join(edir, 'Event{}'.format(n)), n, opt=opt, logz=logz)

   print('Plots are saved in {}'.format(plotsdir))

def get_parser():
    # defaults apply at caltech
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--latentsize', action='store', type=int, help='size of random N(0, 1) latent space to sample')    #parser.add_argument('--model', action='store', default=AngleArch3dgan, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='full', help='HDF5 files to train from.')
    parser.add_argument('--eventsperfile', action='store', type=int, default=1000, help='Number of events in a file')
    parser.add_argument('--energies', action='store', type=int, nargs='+', default=[0], help='Energy bins')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle.')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle used.')
    parser.add_argument('--outdir', action='store', type=str, default='results/2d_projections', help='Directory to store the analysis plots.')
    parser.add_argument('--nbEvents', action='store', type=int, default=100000, help='Max limit for events used for Testing')
    parser.add_argument('--concat', action='store', type=int, default=2, help='Modes related to combining conditions with latent 0)not cancatenated.. 1)concatenate angle...3) concatenate energy and angle')
    parser.add_argument('--gweight', action='store', type=str, default='../weights/3dgan_weights_gan_training_epsilon_2_500GeV/params_generator_epoch_021.hdf5', help='Generator weights')
    parser.add_argument('--xscale', action='store', type=int, help='Multiplication factors for cell energies')
    parser.add_argument('--ascale', action='store', type=int, help='Multiplication factors for angles')
    parser.add_argument('--yscale', action='store', default=100., help='Division Factor for Primary Energy')
    parser.add_argument('--xpower', action='store', help='Power of cell energies')
    parser.add_argument('--thresh', action='store', default=1e-4, help='Threshold for cell energies')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last', help='keras image format')
    parser.add_argument('--ang', action='store', default=1, type=int, help='if variable angle')
    parser.add_argument('--ifC', action='store', default=0, type=int, help='Generate .C files')
    parser.add_argument('--logz', action='store', default=0, type=int, help='log of energies')
    parser.add_argument('--ntup', action='store', default=0, type=int, help='draw as ntuple')
    parser.add_argument('--num', action='store', default=10, type=int, help='number of events to plot')
    return parser

def power(n, xscale=1, xpower=1):
   return np.power(n/xscale, xpower)

def inv_power(n, xscale=1, xpower=1):
   return np.power(n, 1./xpower) / xscale

def applythresh(n, thresh):
   n[n<thresh]=0
   return n

def PlotEvent3d_python(aevent, gevent, energy, theta, out_file, n, opt="", unit='degrees', label="", logz=0, ifC=0, ntup=0):
   x = (aevent.shape[0])
   y = (aevent.shape[1])
   z = (aevent.shape[2])
   cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue",  "cyan", "yellow", "red"])
   #cmap.clim(1e-6, 1e-1)
   aevent, gevent = np.squeeze(aevent), np.squeeze(gevent)
   XX1 = []
   YY1 = []
   ZZ1 = []
   XX2 = []
   YY2 = []
   ZZ2 = []

   g4data = []
   gandata = []
   for i in np.arange(x):
     for j in np.arange(y):
       for k in np.arange(z):
         if aevent[i, j, k] > 0:
           XX1.append(k)
           YY1.append(i)
           ZZ1.append(j)
           g4data.append(aevent[i, j, k])
         if gevent[i, j, k] > 0:
           XX2.append(k)
           YY2.append(i)
           ZZ2.append(j)
           gandata.append(gevent[i, j, k])
   g4data = np.array(g4data)
   gandata = np.array(gandata)
   fig=plt.figure(figsize=(20,10))
   ax1 = fig.add_subplot(121, projection='3d')
   pnt3d1=ax1.scatter(XX1, YY1, ZZ1, c=g4data, s=-100/np.log(g4data), alpha = 0.8, norm=colors.LogNorm(), cmap=cmap)
   fig.colorbar(pnt3d1, pad =0.1, fraction = 0.025)
   fig.suptitle('{:.2f} GeV and $\theta$ {:.2f}$\circ$'.format(energy, np.degrees(theta)))
   ax1.set_title('G4')
   fig.set_label("Energy")
   ax1.set_xlabel('z [layers]')
   ax1.set_ylabel('x [cells]')
   ax1.set_zlabel('y [cells]')
   ax1.set_ylim(0, 51)   
   ax1.set_xlim(0, 25)   
   ax1.set_zlim(0, 51)
   pnt3d1.set_clim(1e-6, 1e-1)
   plt.tight_layout() #Without this function are the y-labels cut off
   #cbar=plt.colorbar(pnt3d1)

   ax2 = fig.add_subplot(122, projection='3d')
   pnt3d2=ax2.scatter(XX2, YY2, ZZ2, c=gandata, s=-100/np.log(gandata), alpha = 0.8, norm=colors.LogNorm(), cmap=cmap)
   fig.colorbar(pnt3d2, pad =0.1, fraction = 0.025)
   ax2.set_title('GEANT')
   fig.set_label("Energy")
   ax2.set_xlabel('z [layers]')
   ax2.set_ylabel('x [cells]')
   ax2.set_zlabel('y [cells]')
   ax2.set_ylim(0, 51)
   ax2.set_xlim(0, 25)
   ax2.set_zlim(0, 51)
   pnt3d2.set_clim(1e-6, 1e-1)
   plt.tight_layout()
   #cbar=plt.colorbar(pnt3d2)
   print(out_file + '.pdf is printed')
   plt.savefig(out_file + '.pdf')

def PlotEvent3d(aevent, gevent, energy, theta, out_file, n, opt="", unit='degrees', label="", logz=0, ifC=0, ntup=1):
   x = aevent.shape[0]
   y = aevent.shape[1]
   z = aevent.shape[2]
   if logz:
      aevent=np.where(aevent>0, np.log(aevent), 0)
      gevent=np.where(gevent>0, np.log(gevent), 0)
   if x==25:
     canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,800 ,400)
   else:
     canvas = ROOT.TCanvas("canvas" ,"GAN Hist" ,200 ,10 ,900 ,400)
   lsize = 0.03 # axis label size
   tsize = 0.08 # axis title size
   tmargin = 0.02
   bmargin = 0.15
   lmargin = 0.15
   rmargin = 0.15
   #if theta:
     #ang1 = MeasPython(np.moveaxis(aevent, 3, 0))
     #ang1 = MeasPython(np.moveaxis(aevent, 3, 0), mod=2)
     #ang2 = MeasPython(np.moveaxis(gevent, 3, 0), mod=2)
     #if unit == 'degrees':
        #ang1= np.degrees(ang1)
        #ang2= np.degrees(ang2)
        #theta = np.degrees(theta)
     
     #title = ROOT.TPaveLabel(0.1,0.95,0.9,0.99,"Ep = {:.2f} GeV #theta={:.2f} #circ  meas#theta G4={:.2f} #circ meas#theta GAN={:.2f} #circ ".format(energy, theta, ang1[0], ang2[0]))
   #else:
   title = ROOT.TPaveLabel(0.1,0.95,0.9,0.99,"Ep = {:.2f} GeV #theta = {:.2f}#circ".format(energy, np.degrees(theta)))
   title.SetFillStyle(0)
   title.SetLineColor(0)
   title.SetBorderSize(1)
   title.Draw()
   graphPad = ROOT.TPad("Graphs","Graphs",0.01,0.01,0.95,0.95)
   graphPad.Draw()

   graphPad.Divide(2,1)
   
   #h3d_g4 = ROOT.TH3F('G4_{:.2f}GeV'.format(energy), '', x, 0, x, y, 0, y, z, 0, z)
   #h3d_gan = ROOT.TH3F('GAN_{:.2f}GeV'.format(energy), '', x, 0, x, y, 0, y, z, 0, z)
   if ntup:
     h3d_g4 = ROOT.TNtuple('','', 'X:Y:Z:energy')
     h3d_gan = ROOT.TNtuple('','', 'X:Y:Z:energy')
     htemp = ROOT.TH2F()
     ROOT.gPad.GetPrimitive("htemp")
   else:
     h3d_g4 = ROOT.TH3F('G4_{:.2f}GeV'.format(energy), '', 4*x, 0, x, 4*y, 0, y, 4*z, 0, z)
     h3d_gan = ROOT.TH3F('GAN_{:.2f}GeV'.format(energy), '', 4*x, 0, x, 4*y, 0, y, 4*z, 0, z) 
     h3d_g4.SetStats(0)
     h3d_gan.SetStats(0)
   ROOT.gStyle.SetPalette(1)
   aevent = np.squeeze(aevent)
   gevent = np.squeeze(gevent)
   if ntup: 
      r.Filltuple3D_wt(h3d_g4, aevent)
      r.Filltuple3D_wt(h3d_gan, gevent)
   else:
      aevent = np.expand_dims(aevent, axis=0)
      gevent = np.expand_dims(gevent, axis=0)
      r.FillHist3D_wt(h3d_g4, aevent)
      r.FillHist3D_wt(h3d_gan, gevent)
   
   if logz:
     Min = -4
     Max = -2
   else:
     Min = 1e-6
     Max = 1e-1
   canvas.Update()
   graphPad.cd(1)
   #ROOT.gPad.SetLogz(1)
   if ntup: 
      h3d_g4.Draw('X:Y:Z:energy', '', 'logcolz')
   else: 
      h3d_g4.Draw(opt)
      h3d_g4.GetXaxis().SetTitle('X')
      h3d_g4.GetYaxis().SetTitle('Y')
      h3d_g4.GetZaxis().SetTitle('Z')
      h3d_g4.GetYaxis().CenterTitle()
      h3d_g4.GetXaxis().CenterTitle()
      h3d_g4.GetZaxis().CenterTitle()
      h3d_g4.GetXaxis().SetLabelSize(lsize)
      h3d_g4.GetZaxis().SetLabelSize(lsize)
      h3d_g4.GetYaxis().SetLabelSize(lsize)
      h3d_g4.GetXaxis().SetTitleSize(tsize)
      h3d_g4.GetYaxis().SetTitleSize(tsize)
      h3d_g4.GetZaxis().SetTitleSize(tsize)
      h3d_g4.SetMinimum(Min)
      h3d_g4.SetMaximum(Max)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(rmargin)
   """   
   if (not ntup) and ('colz' in opt):
      canvas.Update()
      palette = h3d_g4.GetListOfFunctions().FindObject("palette")
      palette.SetX1NDC(0.9)
      palette.SetX2NDC(0.95)
      palette.SetY1NDC(0.2)
      palette.SetY2NDC(0.8)
      canvas.Modified()
   """
   canvas.Update()
   
   graphPad.cd(2)
   #ROOT.gPad.SetLogz(1)
   if ntup:
      h3d_gan.Draw('X:Y:Z:energy','', 'logcolz')
   else:
      h3d_gan.Draw(opt)
      h3d_gan.GetXaxis().SetTitle('X')
      h3d_gan.GetYaxis().SetTitle('Y')
      h3d_gan.GetZaxis().SetTitle('Z')

      h3d_gan.GetYaxis().CenterTitle()
      h3d_gan.GetXaxis().CenterTitle()
      h3d_gan.GetZaxis().CenterTitle()
      h3d_gan.GetXaxis().SetLabelSize(lsize)
      h3d_gan.GetYaxis().SetLabelSize(lsize)
      h3d_gan.GetZaxis().SetLabelSize(lsize)
      h3d_gan.GetXaxis().SetTitleSize(tsize)
      h3d_gan.GetYaxis().SetTitleSize(tsize)
      h3d_gan.GetZaxis().SetTitleSize(tsize)
      h3d_gan.SetMinimum(Min)
      h3d_gan.SetMaximum(Max)
   ROOT.gPad.SetTopMargin(tmargin)
   ROOT.gPad.SetBottomMargin(bmargin)
   ROOT.gPad.SetLeftMargin(lmargin)
   ROOT.gPad.SetRightMargin(rmargin)

   canvas.Update()
   
   canvas.Print(out_file + '.pdf')
   if ifC: canvas.Print(out_file + '.C')



if __name__ == "__main__":
    main()


   
