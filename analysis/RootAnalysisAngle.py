## This script loads weights into architectures for generator and discriminator. Different Physics quantities are then calculated and plotted for a 100-200 GeV events from LCD variable angle dataset##
from utils.GANutils import perform_calculations_angle  # to calculate different Physics quantities
from utils.RootPlotsGAN import get_plots_angle         # to make plots with ROOT
import h5py
import numpy as np
import setGPU
import math
import sys
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')

def main():
   #Architecture 
   from AngleArch3dGAN_sqrt import generator, discriminator

   #Weights
   disc_weight1="../weights/3Dweights_1loss_50weight_sqrt/params_discriminator_epoch_059.hdf5"
   gen_weight1= "../weights/3Dweights_1loss_50weight_sqrt/params_generator_epoch_059.hdf5"

   disc_weight2="../weights/3Dweights_1loss_50weight_sqrt/params_discriminator_epoch_040.hdf5"
   gen_weight2= "../weights/3Dweights_1loss_50weight_sqrt/params_generator_epoch_040.hdf5"
      
   #Path to store results
   plots_dir = "results/sqrt_ep59_ep40/"

   #Parameters
   latent = 256 # latent space
   num_data = 100000 
   num_events = 2000
   events_per_file = 5000
   m = 2  # number of moments 
   nloss= 4 # total number of losses...4 or 5
   concat = 1 # if concatenting angle to latent space
   cell=0 # if making plots for cell energies. Exclude for quick plots.
   energies=[0, 110, 150, 190] # energy bins
   angles = [math.radians(x) for x in [62, 85, 90, 105, 118]] # angle bins
   aindexes = [0, 1, 2, 3, 4] # numbers corressponding to different angle bins
   angtype = 'theta'# the angle data to be read from file
   particle='Ele'# partcile type
   thresh=0 # Threshold for ecal energies
   #datapath = "/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
   #datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # imperium
   datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path
   sortdir = 'SortedAngleData'  # if saving sorted data
   gendir = 'SortedAngleGen'  # if saving generated events
   discdir = 'SortedAngleDisc' # if saving disc outputs
      
   Test = True # use test data
   stest = False # K and chi2 test
   
   #following flags are used to save sorted and GAN data and to load from sorted data. These are used while development and should be False for one time analysis
   save_data = True # True if the sorted data is to be saved. It only saves when read_data is false
   read_data = True # True if loading previously sorted data  
   save_gen =  True # True if saving generated data. 
   read_gen = True # True if generated data is already saved and can be loaded
   save_disc = True # True if discriminiator data is to be saved
   read_disc =  True # True if discriminated data is to be loaded from previously saved file
   ifpdf = True # True if pdf are required. If false .C files will be generated
 
   flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   dweights = [disc_weight1, disc_weight2]
   gweights = [gen_weight1, gen_weight2]
   xscales = [1, 1]
   ascales = [1, 1]
   labels = ['epoch 59', 'epoch 40']
   d = discriminator()
   g = generator(latent)
   var= perform_calculations_angle(g, d, gweights, dweights, energies, angles, 
                aindexes, datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales, 
                ascales, flags, latent, events_per_file, particle, thresh=thresh, angtype=angtype, offset=0.0,
                nloss=nloss, concat=concat
                , pre =sqrt, post =square  # Adding other preprocessing, Default is simple scaling                 
   )
   
   get_plots_angle(var, labels, plots_dir, energies, angles, angtype, aindexes,  m, len(gweights), ifpdf, stest, nloss=nloss, cell=cell)

def sqrt(n, scale=1):
   return np.sqrt(n * scale)

def square(n, scale=1):
   return np.square(n)/scale
        
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
