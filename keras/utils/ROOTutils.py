# To make logrimthic bins for a root histogram                                                                    
import numpy as np
import ROOT
                                                                                                  
def BinLogX(h):
   axis = h.GetXaxis()
   bins = axis.GetNbins()
   From = axis.GetXmin()
   to = axis.GetXmax()
   width = (to - From) / bins
   new_bins = np.zeros(bins + 1)

   for i in np.arange(bins + 1):
     new_bins[i] = ROOT.TMath.Power(10, From + i * width)
   axis.Set(bins, new_bins)
   new_bins=None

#Fill a Tgraph with arrays
def fill_graph(arrayx, arrayy):
   n = arrayx.shape[0]
   arrayx = np.squeeze(arrayx)
   print n
   x = np.zeros(n)
   y = np.zeros(n)
   print x.shape
   print arrayx.shape
   for i in np.arange(n):
      x[i] = arrayx[i]
      y[i] = arrayy[i]
   graph = ROOT.TGraph(n, y, x)
   return graph
      

#Fill a histogram from 1D numpy array                                                                                                                                                                            
def fill_hist(hist, array):
   [hist.Fill(_) for _ in array]

#Fill a weighted histogram for event from numpy array
def fill_hist_wt(hist, weight):
   l=weight.shape[1]
   array= np.arange(0, l, 1)
   for i in array:
     for j in np.arange(weight.shape[0]):
        hist.Fill(i, weight[j, i])

#Fill a 2D histogram for event from numpy array                                                                                                                                                             
def fill_hist_2D(hist, array):
#   l=array.shape[0]
   x = array.shape[0]
   y = array.shape[1]
#   for i in np.arange(l):
   for j in np.arange(x):
     for k in np.arange(y):
       hist.Fill(j, k, array[j, k])


#Hits above a threshold                                                                                                                                                                                           
def get_hits(events, thresh=0.0002):
   hit_array = events>thresh
   hits = np.sum(hit_array, axis=(1, 2, 3))
   return hits

def ratio1_total(events):
   mid= int(events.shape[3]/2)
   print 'mid={}'.format(mid)
   array1=np.sum(events[:, :, :, :mid], axis=(1, 2, 3))
   total=np.sum(events[:, :, :, :], axis=(1, 2, 3))
   return array1/total

# Position of state box
def stat_pos(a, pos=0):
  # Upper left                                                                                                                                                                                                     
  if pos==0:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.0)
   sb1.SetX2NDC(.2)
  # Upper right                                                                                                                                                                                                    
  if pos==1:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.7)
   sb1.SetX2NDC(.9)
   sb1.SetY1NDC(.7)
   sb1.SetY2NDC(.9)
  # Lower right                                                                                                                                                                                                    
  if pos==2:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.1)
   sb1.SetX2NDC(.3)
   sb1.SetY1NDC(.1)
   sb1.SetY2NDC(.3)
  # Lower left                                                                                                                                                                                                     
  if pos==3:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.7)
   sb1.SetX2NDC(.9)
   sb1.SetY1NDC(.1)
   sb1.SetY2NDC(.3)
  # Upper Center                                                                                                                                                                                                   
  if pos==4:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.4)
   sb1.SetX2NDC(.6)
   sb1.SetY1NDC(.7)
   sb1.SetY2NDC(.9)
  #lower Center                                                                                                                                                                                                    
  if pos==5:
   sb1=a.GetListOfFunctions().FindObject("stats")
   sb1.SetX1NDC(.4)
   sb1.SetX2NDC(.6)
   sb1.SetY1NDC(.1)
   sb1.SetY2NDC(.3)
  return sb1

# Fill root profile                                                                                                                                                                                                
def fill_profile(prof, x, y):
  for i in range(len(y)):
      prof.Fill(y[i], x[i])
