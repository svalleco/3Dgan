
## This script loads weights into architectures for generator and discriminator. Different Physics quantities are then calculated and plotted for a 100-200 GeV events from LCD variable angle dataset##
from utils.GANutils import perform_calculations_angle  # to calculate different Physics quantities
from utils.RootPlotsGAN import get_plots_angle         # to make plots with ROOT
import os
import h5py
import numpy as np
import math
import sys
if os.environ.get('HOSTNAME') == 'tlab-gpu-gtx1080ti-06.cern.ch': # Here a check for host can be used        
    tlab = True
else:
    tlab= False

try:
    import setGPU #if Caltech                                                                                
except:
    pass


sys.path.insert(0,'../')

def main():
   #Architecture 
   from AngleArch3dGAN_newbins import generator, discriminator
   
   #Weights
   disc_weight1="/gkhattak/weights/3Dweights_newbins2/params_discriminator_epoch_059.hdf5"
   gen_weight1= "/gkhattak/weights/3Dweights_newbins2/params_generator_epoch_059.hdf5"
      
   #Path to store results
   plots_dir = "results/analysis_newbins2_ep59/"

   #Parameters
   latent = 256 # latent space
   num_data = 100000 
   num_events = 2000
   events_per_file = 5000
   m = 3  # number of moments 
   angloss= 1 # total number of losses...1 or 2
   addloss= 1 # additional loss like count loss
   concat = 1 # if concatenting angle to latent space
   cell=0 # 1 if making plots for cell energies for energy bins and 2 if plotting also per angle bins. Exclude for quick plots.
   corr=0 # if making correlation plots
   energies=[0, 110, 150, 190] # energy bins
   angles = [62, 90, 118] #[math.radians(x) for x in [62, 90, 118]] # angle bins
   angtype = 'theta'# the angle data to be read from file
   particle='Ele'# partcile type
   thresh=0 # Threshold for ecal energies
   #datapath = "/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
   if tlab:
      datapath = '/gkhattak/*Measured3ThetaEscan/*.h5'
   else:
      datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path
   
   sortdir = 'SortedAngleData'  # if saving sorted data
   gendir = 'SortedAngleGen'  # if saving generated events
   discdir = 'SortedAngleDisc' # if saving disc outputs
      
   Test = True # use test data
   stest = True # K and chi2 test
   
   #following flags are used to save sorted and GAN data and to load from sorted data. These are used while development and should be False for one time analysis
   save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
   read_data = False # True if loading previously sorted data  
   save_gen =  False # True if saving generated data. 
   read_gen = False # True if generated data is already saved and can be loaded
   save_disc = False # True if discriminiator data is to be saved
   read_disc =  False # True if discriminated data is to be loaded from previously saved file
   ifpdf = True # True if pdf are required. If false .C files will be generated
 
   flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   dweights = [disc_weight1]#, disc_weight2]
   gweights = [gen_weight1]#, gen_weight2]
   xscales = [1]#, 1]
   xpowers = [0.85]
   ascales = [1]#, 1]
   labels = ['']#, 'epoch 40']
   d = discriminator(xpowers[0])
   g = generator(latent)
   var= perform_calculations_angle(g, d, gweights, dweights, energies, angles, 
                datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales, xpowers,
                ascales, flags, latent, events_per_file, particle, thresh=thresh, angtype=angtype, offset=0.0,
                angloss=angloss, addloss=addloss, concat=concat
                , pre =taking_power, post =inv_power  # Adding other preprocessing, Default is simple scaling                 
   )
   
   get_plots_angle(var, labels, plots_dir, energies, angles, angtype, m, len(gweights), ifpdf, stest, angloss=angloss, addloss=addloss, cell=cell, corr=corr)

def sqrt(n, scale=1):
   return np.sqrt(n * scale)

def square(n, scale=1):
   return np.square(n)/scale
        
def taking_power(n, scale=1.0, power=1.0):
   return(np.power(n * scale, power))

def inv_power(n, scale=1.0, power=1.0):
   return(np.power(n, 1.0/power))/scale
   
# If using reduced Ecal 25x25x25 then use the following function as argument to perform_calculations_angle, Data=GetAngleDataEta_reduced
def GetAngleDataEta_reduced(datafile, thresh=1e-6):
    #get data for training                                                                                        
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))[:, 13:38, 13:38, :]
    Y=np.array(f.get('energy'))
    eta = np.array(f.get('eta')) + 0.6
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    return X, Y, eta, ecal

if __name__ == "__main__":
    main()
