####   This script takes a fixed number of events from fixed energy point data files. It also generates events at fixed energies by GAN. It is possible to use the code without incorporating data files by setting get_actual=0, It is also possible to use generated events from a file by setting get_gen= 1 otherwise events will be generated. ##    
import os, h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
plt.switch_backend('Agg')
import time

from vegan import discriminator as build_discriminator
from vegan import generator as build_generator

#Get VEGAN params
gen_weights='veganweights/params_generator_epoch_029.hdf5'
disc_weights='veganweights/params_discriminator_epoch_029.hdf5'
batch_size=100
latent_space =200                                                            

num_events_act=5000
num_events_gan=5000

plots_dir = 'fixed_plots/'
filename= 'Gen_ecal_3000'
# Other params
save = 0
get_actual = 1    # Get actual data from file
get_gen = 0       # Get generated data from file
energies=[10, 50, 100, 150, 200, 300, 400, 500]  #Energy for generating events. Should be superset of data energies
denergies=[10, 50, 100, 200]                     #Energy for data events                            
var = {}
num_files = np.ceil(num_events_act/1000) # As each data file has 1000 events
num = 0                                     
      
# Histogram Functions                                                                                 
def plot_max(array, index, out_file, num_fig, plot_label, pos=0):
   ## Plot the Histogram of Maximum energy deposition location on all axis                                                                                                                            
   bins = np.arange(0, 25, 1)
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index-1, 0]))+ '({:.2f})'.format(np.std(array[0:index-1, 0]))
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= label, normed =1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index-1, 1]))+ '({:.2f}'.format(np.std(array[0:index-1, 1]))
   plt.hist(array[0:index-1, 1], bins=bins, histtype='step', label=label, normed =1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.xlabel('Position')

   plt.subplot(223)
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index-1, 2]))+ '({:.2f})'.format(np.std(array[0:index-1, 2]))
   plt.hist(array[0:index-1, 2], bins=bins, histtype='step', label=label, normed =1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.xlabel('Position')
   plt.ylabel('Events')
   plt.savefig(out_file)
                                   
def plot_flat_energy(array, index, out_file, num_fig, plot_label):
   ### Plot Histogram of energy flat distribution along all three axis                               
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(array[:index, 0].flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend(fontsize='xx-small')
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[:index, 1].flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend(fontsize='xx-small')
   plt.xlabel('Energy')

   plt.subplot(223)
   plt.hist(array[:index, 2].flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend(fontsize='xx-small')
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_energy(array, index, out_file, num_fig, plot_label, color='blue', style='-', pos=0):
   ### Plot Histogram of energy                                                                      
   plt.figure(num_fig)
   ebins=np.arange(0, 600, 5)
   label= plot_label + '\n{:.2f}'.format(np.mean(array))+ '({:.2f})'.format(np.std(array))
   plt.hist(array, bins=ebins, histtype='step', label=label, color=color, ls=style, normed =1)
   plt.xticks([0, 10, 50, 100, 150, 200, 300, 400, 500, 600])
   plt.xlabel('Energy GeV')
   plt.ylabel('Events')
   pos = 0 if energy <= 300 else 2
   plt.legend(title='   Mean(std)', loc=pos)
   plt.savefig(out_file)

def plot_energy_axis(array, index, out_file, num_fig, plot_label, pos=0):
   ### Plot total energy deposition cell by cell along x, y, z axis                     
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   sumx_array = array[0:index, 0].sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumx_array))+ '({:.2f})'.format(np.std(sumx_array))
   plt.plot(sumx_array/index, label=plot_label)
   plt.ylabel('ECAL Energy')
   plt.legend(loc=2, fontsize='x-small')
   plt.subplot(222)
   plt.title('Y-axis')
   sumy_array =array[0:index, 1].sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumy_array))+ '({:.2f})'.format(np.std(sumy_array))
   plt.plot(sumy_array/index, label=plot_label)
   plt.legend(loc=2, fontsize='x-small')
   plt.xlabel('Position')

   plt.subplot(223)
   sumz_array =array[0:index, 2].sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumz_array))+ '({:.2f})'.format(np.std(sumz_array))
   plt.plot( sumz_array/index, label=plot_label)
   plt.legend(loc=8, fontsize='x-small')
   plt.xlabel('Z axis Position')
   plt.ylabel('ECAL Energy')

   plt.savefig(out_file)
                          
def plot_energy_mean(array, index, out_file, num_fig, plot_label):
   ### Plot total energy deposition cell by cell along x, y, z axis                                                                                                               
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index, 0].mean(axis = 0), label=plot_label)
   plt.legend(fontsize='xx-small')
   plt.ylabel('Mean Energy')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index, 1].mean(axis = 0), label=plot_label)
   plt.legend(fontsize='xx-small')
   plt.xlabel('Position')

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index, 2].mean(axis = 0), label=plot_label)
   plt.xlabel('Position')
   plt.legend(fontsize='xx-small')
   plt.ylabel('Mean Energy')
   plt.savefig(out_file)

def plot_real(array, index, out_file, num_fig, plot_label):
   ## Plot the disc real/fake flag                                                                                                                                
   plt.figure(num_fig)
   bins = np.arange(0, 1, 0.01)
   plt.figure(num_fig)
   plt.title('Real/ Fake')
   label= plot_label + '{:.2f}'.format(np.mean(array[0:index-1, 0]))+ '({:.2f})'.format(np.std(array[0:index-1, 0]))
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= label, normed =1)
   plt.legend()
   plt.ylabel('Events')
   plt.xlabel('Real/fake')
   plt.savefig(out_file)

def plot_error(energy, array2, index, out_file, num_fig, plot_label, pos=2):
   # plot error                                                                                                                                                                   
   plt.figure(num_fig)
   array1 = np.multiply(energy, np.ones((index, 1)))
   bins = np.linspace(-150, 150, 30)
   label= plot_label + '\n{:.2f}'.format(np.mean(np.absolute(array1-array2))) + '({:.2f})'.format(np.std(np.absolute(array1-array2)))
   plt.hist(np.multiply(100, array1-array2), bins=bins, histtype='step', label=label, normed =1)
   plt.xlabel('error GeV')
   plt.ylabel('Number of events')
   plt.legend(title='                       Mean     ( std )', loc=pos)
   plt.savefig(out_file)

def plot_ecal(array, index, out_file, num_fig, plot_label):
   # plot ecal sum                                                                                                                                             
   bins = np.linspace(0, 11, 50)
   plt.figure(num_fig)
   ecal_array=np.sum(array[0:index-1], axis=(1, 2, 3))
   label= plot_label + '\n{:.2f}'.format(np.mean(ecal_array))+ '({:.2f})'.format(np.std(ecal_array))
   plt.title('ECAL SUM')
   plt.xlabel('ECAL SUM')
   plt.ylabel('Events')
   plt.hist(ecal_array, bins=bins, histtype='step', label=label, normed =1)
   pos = 0 if energy <= 300 else 2
   plt.legend(loc=pos)
   plt.savefig(out_file)

## Get Full data                                                                                                                                                                                                  
if (get_actual):  # if actual data has to be used
   for energy in denergies:
      num = 0
      while num < num_files:
         infile = "/eos/project/d/dshep/LCD/FixedEnergy/Ele_" + str(energy) + "GeV/Ele_"+ str(energy) + "GeV_" + str(num) + ".h5"
         d=h5py.File(infile,'r')
         c=np.array(d.get('ECAL'))
         e=d.get('target')
         var["X" + str(num)]=np.array(c[:num_events_act])
         y=np.array(e[:num_events_act,1])
         var["Y"+ str(num)]=np.expand_dims(y, axis=-1)
         if num == 0:
            var["X" + str(energy)] = var["X" + str(num)]
            var["Y" + str(energy)] = var["Y" + str(num)]
         else:
            var["X" + str(energy)] = np.concatenate((var["X" + str(energy)], var["X" + str(num)]))
            var["Y" + str(energy)] = np.concatenate((var["Y" + str(energy)], var["Y" + str(num)]))
         num+=1
      var["X" + str(energy)][ var["X" + str(energy)] < 1e-6] = 0
      var["events_act" + str(energy)] = np.zeros((num_events_act, 25, 25, 25))
      var["max_pos_act_" + str(energy)] = np.zeros((num_events_act, 3))
      var["sum_act" + str(energy)] = np.zeros((num_events_act, 3, 25))
      var["energy_act" + str(energy)] = np.zeros((num_events_act, 1))
      var["isreal_act" + str(energy)] =np.zeros((num_events_act, 1))
             
### Get Generated Data                                                                   
if get_gen:    # if generated data is to be loaded from file
   for energy in energies:
      var["filename" + str(energy)]= filename + str(energy) + '.h5'
      f=h5py.File(var["filename" + str(energy)],'r')
      var["generated_images" + str(energy)] = np.array(f.get('ECAL'))
      var["isreal" + str(energy)]= f.get('ISREAL')
      var["aux_out" + str(energy)] = f.get('AUX')
      if num_events_gan > len(var["generated_images" + str(energy)]):
         print " Number of events in Gen file is less than ", num_events_gan
         num_events_gan = len(var["generated_images" + str(energy)])
         print " Thus number of events for generated is now ", num_events_gan
      var["energy_gan" +str(energy)] = var["aux_out" + str(energy)]
else:        # if images are to be generated
   g = build_generator(latent_space, return_intermediate=False)
   g.load_weights(gen_weights)
   d = build_discriminator()
   d.load_weights(disc_weights)
   for energy in energies:
      noise = np.random.normal(0, 1, (num_events_gan, latent_space))
      sampled_labels = np.multiply(energy, np.ones((num_events_gan, 1)))
      sampled_labels = sampled_labels/100
      generator_in = np.multiply(sampled_labels, noise)
      var["generated_images" + str(energy)] = g.predict(generator_in, verbose=False, batch_size=batch_size)
      var["isreal" + str(energy)], var["aux_out" + str(energy)] = np.array(d.predict(var["generated_images" + str(energy)], verbose=False, batch_size=batch_size))
      var["generated_images" + str(energy)] = np.squeeze(var["generated_images" + str(energy)])
      var["energy_gan" +str(energy)] = var["aux_out" + str(energy)]

# Initialization of generated parameters                                                                                                                                                                          
for energy in energies:
   var["events_gan" + str(energy)] = np.zeros((num_events_gan, 25, 25, 25))
   var["max_pos_gan_" + str(energy)] = np.zeros((num_events_gan, 3))
   var["sum_gan" + str(energy)] = np.zeros((num_events_gan, 3, 25))
   var["isreal_gan" + str(energy)] = var["isreal" + str(energy)]

## Use Discriminator for actual images and perform calculations                                                        
if get_actual:
     for energy in denergies:
        if (get_gen == 0):
           var["image" + str(energy)] = np.expand_dims(var["X" + str(energy)], axis=-1)
           var["isreal2_" +str(energy)], var["aux_out2_" + str(energy)] = np.array(d.predict(var["image" + str(energy)], verbose=False, batch_size=batch_size))
           var["energy_act" +str(energy)] = var["aux_out2_" + str(energy)]
        #calculations for actual
        for j in range(num_events_act):
           var["events_act" + str(energy)][j]= var["X" + str(energy)][j]
           var["max_pos_act_" + str(energy)][j] = np.unravel_index(var["events_act" + str(energy)][j].argmax(), (25, 25, 25))
           var["sum_act" + str(energy)][j, 0] = np.sum(var["events_act" + str(energy)][j], axis=(1,2))
           var["sum_act" + str(energy)][j, 1] = np.sum(var["events_act" + str(energy)][j], axis=(0,2))
           var["sum_act" + str(energy)][j, 2] = np.sum(var["events_act" + str(energy)][j], axis=(0,1))
   
### Calculations for generated
for energy in energies:                                                             
   for j in range(num_events_gan):
      var["events_gan" + str(energy)][j]= var["generated_images" + str(energy)][j]
      var["max_pos_gan_" + str(energy)][j] = np.unravel_index(var["events_gan" + str(energy)][j].argmax(), (25, 25, 25))
      var["sum_gan" + str(energy)][j, 0] = np.sum(var["events_gan" + str(energy)][j], axis=(1,2))
      var["sum_gan" + str(energy)][j, 1] = np.sum(var["events_gan" + str(energy)][j], axis=(0,2))
      var["sum_gan" + str(energy)][j, 2] = np.sum(var["events_gan" + str(energy)][j], axis=(0,1))

## Generate Data table to screen
if (get_actual==1):
   print "Actual Data"                                                 
   print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t"
   for energy in denergies:
      print "%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(energy, num_events_act, np.amax(var["events_act" + str(energy)]), str(np.unravel_index(var["events_act" + str(energy)].argmax(), (num_events_act, 25, 25, 25))), np.mean(var["events_act" + str(energy)]), np.amin(var["events_act" + str(energy)]))
 
#### Generate GAN table to screen                                                                                
print "Generated Data"
print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t"
for energy in energies:
   print "%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(energy, num_events_gan, np.amax(var["events_gan" + str(energy)]), str(np.unravel_index(var["events_gan" + str(energy)].argmax(), (num_events_gan, 25, 25, 25))), np.mean(var["events_gan" + str(energy)]), np.amin(var["events_gan" + str(energy)]))


def safe_mkdir(path):
    '''
    Safe mkdir (i.e., don't create if already exists,
    and no violation of race conditions)
    '''
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception


## Make folders for plots
gendir = plots_dir + 'Generated'
safe_mkdir(gendir)
discdir = plots_dir + 'disc_outputs'
safe_mkdir(discdir)
if get_actual:
   actdir = plots_dir + 'Actual'
   safe_mkdir(actdir)
   comdir = plots_dir + 'Combined'
   safe_mkdir(comdir)
fig = 1
## Make plots for generated data

for energy in energies:
   maxfile = "Position_of_max_" + str(energy) + ".pdf"
   histfile = "hist_" + str(energy) + ".pdf"
   ecalfile = "ecal_" + str(energy) + ".pdf"
   energyfile = "energy_" + str(energy) + ".pdf"
   realfile = "real_gan" + str(energy) + ".pdf"
   errorfile = "error_gan" + str(energy) + ".pdf"
   label = "GAN " + str(energy)
   sampled_energy = energy/100
   plot_max(var["max_pos_gan_" + str(energy)], num_events_gan, os.path.join(gendir, maxfile), fig, label)
   fig+=1
   plot_energy_axis(var["sum_gan" + str(energy)], num_events_gan, os.path.join(gendir, histfile), fig, label)
   fig+=1
   plot_energy(np.multiply(100, var["energy_gan" + str(energy)]), num_events_gan, os.path.join(gendir, energyfile), fig, label, 'green')
   fig+=1
   plot_ecal(var["events_gan" + str(energy)], num_events_gan, os.path.join(gendir, ecalfile), fig, label)
   fig+=1
   plot_real(var["isreal_gan" + str(energy)], num_events_gan, os.path.join(discdir, realfile), fig, label)
   fig+=1
   plot_error(sampled_energy,  var["energy_gan" + str(energy)], num_events_gan, os.path.join(discdir, errorfile), fig, label, pos=0)
   fig+=1
if get_actual == 1:
   for energy in denergies:
      maxfile = "Position_of_max_" + str(energy) + ".pdf"
      histfile = "hist_" + str(energy) + ".pdf"
      energyfile = "energy_" + str(energy) + ".pdf"
      realfile = "real_act" + str(energy) + ".pdf"
      errorfile = "error_act" + str(energy) + ".pdf"
      ecalfile = "ecal_" + str(energy) + ".pdf"
      label = "DATA " + str(energy)
      plot_max(var["max_pos_act_" + str(energy)], num_events_act, os.path.join(actdir, maxfile), fig, label)
      fig+=1
      plot_energy_axis(var["sum_act" + str(energy)], num_events_act, os.path.join(actdir, histfile), fig, label)
      fig+=1
      plot_ecal(var["events_act" + str(energy)], num_events_act, os.path.join(actdir, ecalfile), fig, label)
      fig+=1
      if get_gen == 0:
         plot_energy(np.multiply(100, var["energy_act" + str(energy)]), num_events_act, os.path.join(actdir, energyfile), fig, label, 'green')
         fig+=1
         plot_real(var["isreal_act" + str(energy)], num_events_act, os.path.join(discdir, realfile), fig, label)
         fig+=1
         plot_error(energy/100,  var["energy_act" + str(energy)], num_events_act, os.path.join(discdir, errorfile), fig, label, pos=0)
         fig+=1
      ## Super imposed plots
      maxfile = "Position_of_max_" + str(energy) + ".pdf"
      histfile = "hist_" + str(energy) + ".pdf"
      ecalfile = "ecal_" + str(energy) + ".pdf"
      energyfile = "energy_" + str(energy) + ".pdf"
      errorfile = "error_" + str(energy) + ".pdf"
      realfile = "real_" + str(energy) + ".pdf"
      dlabel = "Data " + str(energy)
      glabel = "GAN " + str(energy)
      plot_max(var["max_pos_act_" + str(energy)], num_events_act, os.path.join(comdir, maxfile), fig, dlabel)
      plot_max(var["max_pos_gan_" + str(energy)], num_events_gan, os.path.join(comdir, maxfile), fig, glabel)
      fig+=1
      plot_energy_axis(var["sum_act" + str(energy)], num_events_act, os.path.join(comdir, histfile), fig, dlabel)
      plot_energy_axis(var["sum_gan" + str(energy)], num_events_gan, os.path.join(comdir, histfile), fig, glabel)
      fig+=1
      plot_ecal(var["events_act" + str(energy)], num_events_act, os.path.join(comdir, ecalfile), fig, dlabel)
      plot_ecal(var["events_gan" + str(energy)], num_events_gan, os.path.join(comdir, ecalfile), fig, glabel)
      fig+=1
      if (get_gen==0):
         plot_energy(np.multiply(100, var["energy_act" + str(energy)]), num_events_act, os.path.join(comdir, energyfile), fig, dlabel, 'blue')
         plot_energy(np.multiply(100, var["energy_gan" + str(energy)]), num_events_gan, os.path.join(comdir, energyfile), fig, glabel, 'green')
         fig+=1
         plot_real(var["isreal_act" + str(energy)], num_events_act, os.path.join(discdir, realfile), fig, dlabel)
         plot_real(var["isreal_gan" + str(energy)], num_events_gan, os.path.join(discdir, realfile), fig, glabel)
         fig+=1
         plot_error(energy/100,  var["energy_act" + str(energy)], num_events_act, os.path.join(discdir, errorfile), fig, dlabel, pos=0)
         plot_error(energy/100,  var["energy_gan" + str(energy)], num_events_gan, os.path.join(discdir, errorfile), fig, glabel, pos=0)
         fig+=1
   for energy in denergies:
      ## plots for all energies together
      flatfile = "Flat_energy_all.pdf"
      histallfile = "hist_all.pdf"
      meanallfile = "hist_mean_all.pdf"
      dlabel = "Data " + str(energy)
      glabel = "GAN " + str(energy)
      plot_flat_energy(var["sum_act" + str(energy)], num_events_act, os.path.join(actdir, flatfile), fig, dlabel)
      plot_energy_axis(var["sum_act" + str(energy)], num_events_act, os.path.join(actdir, histallfile), fig+1, dlabel)
      plot_energy_mean(var["sum_act" + str(energy)], num_events_act, os.path.join(actdir, meanallfile), fig+3, dlabel)
   fig+=4

## Make combined plots                                                                                                                                                                                            
for energy in energies:
   flatfile = "Flat_energy_all.pdf"
   histallfile = "hist_all.pdf"
   meanallfile = "hist_mean_all.pdf"
   dlabel = "Data " + str(energy)
   glabel = "GAN " + str(energy)
   plot_flat_energy(var["sum_gan" + str(energy)], num_events_gan, os.path.join(gendir, flatfile), fig, glabel)
   plot_energy_axis(var["sum_gan" + str(energy)], num_events_gan, os.path.join(gendir, histallfile), fig+1, glabel)
   plot_energy_mean(var["sum_gan" + str(energy)], num_events_gan, os.path.join(gendir, meanallfile), fig+3, glabel)
fig+=4

### Save generated image data to file
if (save):
   for energy in energies:
      generated_images = (var["generated_images" + str(energy)]) 
      generated_images = np.squeeze(generated_images)
      with h5py.File(var["filename" + str(energy)],'w') as outfile:
         outfile.create_dataset('ECAL',data=generated_images)
         outfile.create_dataset('LABELS',data= np.multiply(energy, np.ones((num_events_gan, 1))))
         outfile.create_dataset('AUX',data=var["aux_out" + str(energy)])
         outfile.create_dataset('ISREAL',data=var["isreal" + str(energy)])
         print "Generated ECAL saved to ", var["filename" + str(energy)]   
      
print "Images are saved in", plots_dir



