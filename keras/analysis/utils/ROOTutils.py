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

#Fill a histogram from 1D numpy array                                                                                                                                                                              
def fill_hist(hist, array):
   [hist.Fill(_) for _ in array]

def normalize(hist, mod=0):
   if mod==0:
      norm = hist.GetEntries()
      hist.Scale(1/norm)
   elif mod==1:   
      if hist.Integral()!=0:
         hist.Scale(1/hist.Integral())
   return hist

#Fill a weighted histogram for event from numpy array
def fill_hist_wt(hist, weight):
   l=weight.shape[1]
   array= np.arange(0, l, 1)
   for i in array:
     for j in np.arange(weight.shape[0]):
        hist.Fill(i, weight[j, i])

def FillHist2D_wt(hist, array):
   array= np.squeeze(array, axis=3)
   dim1 = array.shape[0]
   dim2 = array.shape[1]
   dim3 = array.shape[2]
   bin1 = np.arange(dim1)
   bin2 = np.arange(dim2)
   bin3 = np.arange(dim3)
   count = 0
   for j in bin2:
      for k in bin3:
         for i in bin1:
            hist.Fill(j, k, array[i, j, k])
            count+=1
                                                                            
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
