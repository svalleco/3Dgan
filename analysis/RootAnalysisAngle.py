## Plots for variable angle ##
from utils.GANutils import perform_calculations_angle
from utils.RootPlotsGAN import get_plots_angle 
import h5py
import numpy as np
import setGPU
import math
import sys
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')

def main():
   #Architectures 
   from AngleArch3dGAN_sqrt import generator, discriminator
   
   disc_weights="/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_50weight/params_discriminator_epoch_057.hdf5"
   gen_weights= "/nfshome/gkhattak/3Dgan/weights/3Dweights_1loss_50weight/params_generator_epoch_057.hdf5"

   plots_dir = "results/sqrt_plots_testing_ep57/"
   latent = 256
   num_data = 100000
   num_events = 2000
   m = 2
   nloss= 3
   concat = 1
   cell=0
   energies=[0, 110, 150, 190]
   #angles = [-0.5, -0.25, 0, 0.25, 0.5]
   angles = [math.radians(x) for x in [62, 85, 90, 105, 118]]
   aindexes = [0, 1, 2, 3, 4]
   angtype = 'theta'
   particle='Ele'
   thresh=0
   #datapath = "/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
   #datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # imperium
   datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"
   sortdir = 'SortedEAngleData'
   angledir = 'SortedAngleData'
   gendir = 'AngleGen'  
   discdir = 'AngleDisc'
   genangdir = 'AngleGenAngle'
    
   Test = True # use test data
   stest = False # K and chi2 test
   
   #following flags are used to save sorted and GAN data and to load from sorted data
   save_data = False # True if the sorted data is to be saved. It only saves when read_data is false
   read_data = False # True if loading previously sorted data  
   save_gen =  False # True if saving generated data. 
   read_gen = False # True if generated data is already saved and can be loaded
   save_disc = False # True if discriminiator data is to be saved
   read_disc =  False # True if discriminated data is to be loaded from previously saved file
   ifpdf = False # True if pdf are required. If false .C files will be generated
 
   flags =[Test, save_data, read_data, save_gen, read_gen, save_disc, read_disc]
   dweights = [disc_weights]
   gweights = [gen_weights]
   xscales = [1]
   ascales = [1]
   labels = ['']
   d = discriminator()
   g = generator(latent)
   var= perform_calculations_angle(g, d, gweights, dweights, energies, angles, 
                aindexes, datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales, 
                ascales, flags, latent, particle, thresh=thresh, angtype=angtype, offset=0.0,
                nloss=nloss, concat=concat
                , pre =sqrt, post =square                   
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
