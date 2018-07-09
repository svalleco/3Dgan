## Plots for variable angle ##
from GANutilsANG4_concat import perform_calculations_angle
from RootPlotsAngle4 import get_plots_angle 
import h5py
import numpy as np
import setGPU
import math

def main():
   #Architectures 
   from EcalCondGanAngle_3d import generator, discriminator
   
   disc_weights="3d_angleweights/params_discriminator_epoch_000.hdf5"
   gen_weights= "3d_anglewights/params_generator_epoch_000.hdf5"

   plots_dir = "3d_angleplots/"
   latent = 256
   num_data = 100000
   num_events = 2000
   m = 2
   energies=[0, 110, 150, 190]
   #angles = [-0.5, -0.25, 0, 0.25, 0.5]
   angles = [math.radians(x) for x in [62, 85, 90, 105, 118]]
   aindexes = [0, 1, 2, 3, 4]
   particle='Ele'
   angtype='theta'
   #datapath = "/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate
   datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # imperium
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
   xscales = [2]
   ascales = [1]
   labels = ['']
   d = discriminator()
   g = generator(latent)
   var= perform_calculations_angle(g, d, gweights, dweights, energies, angles, 
                aindexes, datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales, 
                                   ascales, flags, latent, particle,
                                    thresh=1e-5, angtype=angtype, offset=0.0)
   get_plots_angle(var, labels, plots_dir, energies, angles, angtype, aindexes, m, len(gweights), ifpdf, stest)

# If using reduced Ecal 25x25x25 then use the following function as argument to perform_calculations_angle, Data=GetAngleDataEta_reduced
def GetAngleData_reduced(datafile, thresh=1e-6, angtype='theta', offset=0.0):
    #get data for training                                                                                        
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))[:, 13:38, 13:38, :]
    Y=np.array(f.get('energy'))
    ang = np.array(f.get(angtype))
    ang = ang + offset
    X[X < thresh] = 0
    X = np.expand_dims(X, axis=-1)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    print(X.shape)
    return X, Y, ang, ecal

if __name__ == "__main__":
    main()
