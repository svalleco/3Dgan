## Plots for variable angle ##
from GANutilsANG import perform_calculations_angle
from RootPlotsAngle import get_plots_angle 

def main():
   #Architectures 
   from EcalCondGan6 import generator, discriminator
   
   disc_weights="params_discriminator_epoch_048.hdf5"
   gen_weights= "params_generator_epoch_048.hdf5"

   plots_dir = "angle_plots_testing/"
   latent = 256
   num_data = 100000
   num_events = 2000
   m = 2
   energies=[0, 50, 100, 200, 250, 300, 400, 500]
   angles = [-0.5, -0.25, 0, 0.25, 0.5]
   aindexes = [0, 1, 2, 3, 4]
   particle='Ele'
   datapath = "/eos/project/d/dshep/LCD/DDHEP/*scan_RandomAngle_*_MERGED/*Escan_RandomAngle_*.h5" #cern
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
   ascales = [10]
   labels = ['']
   d = discriminator()
   g = generator(latent)
   var= perform_calculations_angle(g, d, gweights, dweights, energies, angles, 
                aindexes, datapath, sortdir, gendir, discdir, num_data, num_events, m, xscales, 
                ascales, flags, latent, particle)
   get_plots_angle(var, labels, plots_dir, energies, angles, aindexes, m, len(gweights), ifpdf, stest)
    
if __name__ == "__main__":
    main()
