import numpy as np
import h5py
import sys
import ROOT 
from ROOT import TTree, TFile, AddressOf, gROOT, std, vector
gROOT.ProcessLine("#include <vector>");
events = []

if len(sys.argv) != 4:
  print('Usage: python h5toroot.py [h5 input file name] [root output file name] [max number of events]')
  sys.exit()

ifile = h5py.File(sys.argv[1],'r')
ofile = ROOT.TFile(sys.argv[2],'RECREATE')
maxn = int(sys.argv[3])
ecalTree = TTree('ecalTree', "ecal Tree")

#hcal = np.array(ifile['HCAL'])
ecal = np.array(ifile['ECAL'])
#target = np.array(ifile['target'])

#print('TARGET:', target.shape)
#print('ECAL:', ecal.shape)
#print('HCAL:', hcal.shape)
en = 0
ei = ecal.shape[1]
ej = ecal.shape[2]
ek = ecal.shape[3]

#hi = hcal.shape[1]
#hj = hcal.shape[2]
#hk = hcal.shape[3]

#ti = target.shape[1]
#tj = target.shape[2]

#assert(ecal.shape[0] == hcal.shape[0] == target.shape[0])

nevents = ecal.shape[0]
if ecal.shape[0] > maxn:
  nevents = maxn
vec_x = ROOT.std.vector(int)()
vec_y = ROOT.std.vector(int)()
vec_z = ROOT.std.vector(int)()
vec_E = ROOT.std.vector(float)()

nevents = maxn


ecalTree.Branch('x',vec_x)
ecalTree.Branch('y',vec_y)
ecalTree.Branch('z',vec_z)
ecalTree.Branch('E',vec_E)

for e in range(nevents):

  ec = ecal[e]
  #hc = hcal[e]
  #tg = target[e]
  
  vec_x.clear()
  vec_y.clear()
  vec_z.clear()
  vec_E.clear()
 
  for i in range(ei):
    for j in range(ej):
      for k in range(ek):

        energy = ec[i][j][k]
        if energy > 0:
           vec_E.push_back(energy)
           vec_x.push_back(i)
           vec_y.push_back(j)
           vec_z.push_back(k)
  ecalTree.Fill()
  en += 1

ofile.Write()
ofile.Close()
