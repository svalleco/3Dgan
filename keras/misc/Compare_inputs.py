from __future__ import print_function
from os import path
import ROOT
import h5py
import numpy as np
import keras.backend as K
import tensorflow as tf
#import tensorflow.python.ops.image_ops_impl as image 
import time
import sys
sys.path.insert(0,'../keras')
sys.path.insert(0,'../keras/analysis')
import utils.GANutils as gan
import utils.ROOTutils as roo
from skimage import measure
import math
from AngleArch3dGAN import generator, discriminator
try:
  import setGPU
except:
  pass

def main():
  latent = 256  #latent space
  power=0.85    #power for cell energies used in training
  thresh =0.0   #threshold used
  particle = 'Ele'
  num_files=40
  events_per_file = 10000
  feat = ['HCAL', 'HCAL_E', 'ECAL', 'energy', 'theta', 'phi', 'pdgID', 'eta', "recoEta", "recoPhi", 'recoTheta', 'mtheta'] 
  outdir = 'results/angle_data_correlation/'
  gan.safe_mkdir(outdir) 
  datapath = "/storage/group/gpu/bigdata/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech
  data_files = gan.GetDataFiles(datapath, [particle]) # get list of files
  data = GetAngleData(data_files[0], features=feat)
  plot_angle_2Dhist(data['theta'], data['recoTheta'], outdir + 'theta_recoTheta', 'theta', 'reco theta')
  plot_angle_2Dhist(data['theta'], data['mtheta'], outdir + 'theta_mTheta', 'theta', 'measured angle')
  print('The results are saved in {}'.format(outdir))

def GetAngleData(datafile, features=['HCAL_E', 'ECAL_E','ECAL', 'energy', 'theta', 'phi','eta', 'pdgID']):
    #get data for training                                                                                                       
    print ('Loading Data from .....', datafile)
    data={}
    f=h5py.File(datafile,'r')
    ecal =np.array(f.get('ECAL_E'))
    #ecal_sum = np.sum(ecal, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    data['ECAL_E'] = ecal[indexes]
    print(data['ECAL_E'].shape[0])
    for feat in features:
      print(feat)
      if feat=='mtheta':
        data[feat] = gan.measPython(data['ECAL'])
      else:
        data[feat] = np.array(f.get(feat))[indexes]
        data[feat] = data[feat].astype(np.float32)
      print(data[feat].shape) 
    print('eta', data['eta'][:5])
    print('reco_eta', data['recoEta'][:5])
    print('theta', data['theta'][:5])
    print('reco_theta', data['recoTheta'][:5])
    print('measured theta', data['mtheta'][:5])
    print('phi', data['phi'][:5])
    print('reco_phi', data['recoPhi'][:5])
    return data

def plot_angle_2Dhist(ang1, ang2, out_file, angtype1, angtype2, ifpdf=True):
   c1 = ROOT.TCanvas("c1" ,"" ,200 ,10 ,700 ,500) #make
   c1.SetGrid()
   legend = ROOT.TLegend(.4, .8, .5, .9)
   hp = ROOT.TH2F("G4","G4" , 50, 0.5, 3, 50, 0.5, 2.5)
   n = ang1.shape[0]
   hp.SetStats(0)
   hp.SetTitle("2D Histogram for angle data" )
   hp.GetXaxis().SetTitle(angtype1)
   hp.GetYaxis().SetTitle(angtype2)
   for j in np.arange(n):
     hp.Fill(ang1[j], ang2[j])
   hp.Draw("colz")
   c1.Update()
   legend.AddEntry(hp,"2D hist " )
   legend.Draw()
   c1.Modified()
   c1.Update()
   if ifpdf:
      c1.Print(out_file + '.pdf')
   else:
      c1.Print(out_file + '.C') 
               
if __name__ == "__main__":
  main()
