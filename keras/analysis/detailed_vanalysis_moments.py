####   This script takes a certain number of events from data file. Then generates events at same energies by GAN. It puts the events in user specified bins and compare the different quantities through plots ##
import os, h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
plt.switch_backend('Agg')
import time
import numpy.core.umath_tests as umath

from arch10 import discriminator as build_discriminator
from arch10 import generator as build_generator

#Get VEGAN params
n_jets = 150000         #events to take from data file
latent_space =200
num_events=3000         #events in each bin
save=0                #whether save generated output in a file
get_gen=0             # whether load from file. In that case discrimination will not be performed but the file must be generated using same energies as the data file.
m=2                   #moments

#Get weights
gen_weights='params_generator_epoch_019.hdf5'
disc_weights='params_discriminator_epoch_019.hdf5'
filename = 'Gen_cont_' + str(n_jets) + 'events.h5'
datafile = "/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5"
plots_dir = 'cont_moments_arch10epoch20/'
## Get Full data                                                               
d=h5py.File(datafile,'r')
c=np.array(d.get('ECAL'))
e=d.get('target')
X=np.array(c[:n_jets])
y=np.array(e[:n_jets,1])
Y=np.expand_dims(y, axis=-1)
tolerance = 5
print " Data is loaded"
print X.shape
print Y.shape
X[X < 1e-6] = 0
energies=[50, 100, 150, 200, 300, 400, 500]  #Energy bins

# Histogram Functions
def plot_max(array, index, out_file, num_fig, plot_label, pos=0):
   ## Plot the Histogram of Maximum energy deposition location on all axis                                            
   bins = np.arange(0, 25, 1)
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index-1, 0]))+ '({:.2f})'.format(np.std(array[0:index-1, 0])) 
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= label, normed=1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index-1, 1]))+ '({:.2f}'.format(np.std(array[0:index-1, 1])) 
   plt.hist(array[0:index-1, 1], bins=bins, histtype='step', label=label, normed=1)
   plt.legend(loc=pos, fontsize='xx-small')
   plt.xlabel('Position')

   plt.subplot(223)
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index-1, 2]))+ '({:.2f})'.format(np.std(array[0:index-1, 2])) 
   plt.hist(array[0:index-1, 2], bins=bins, histtype='step', label=label, normed=1)
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
   plt.legend()
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[:index, 1].flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend()
   plt.xlabel('Energy')

   plt.subplot(223)
   plt.hist(array[:index, 2].flatten(), bins='auto', histtype='step', label=plot_label)
   plt.legend()
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_energy(array, index, out_file, num_fig, plot_label, color='blue', style='-', pos=0):
   ### Plot Histogram of energy                                                                                       
   plt.figure(num_fig)
   ebins=np.arange(0, 600, 5)
   array = array[:index-1]
   label= plot_label + '\n{:.2f}'.format(np.mean(array))+ '({:.2f})'.format(np.std(array)) 
   plt.hist(array[:index-1], bins=ebins, histtype='step', label=label, color=color, ls=style, normed=1)
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
   sumx_array = array[0:index-1, 0].sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumx_array))+ '({:.2f})'.format(np.std(sumx_array))
   plt.plot(sumx_array/index, label=plot_label)
   plt.ylabel('ECAL Energy')
   plt.legend(loc=2, fontsize='xx-small')

   plt.subplot(222)
   plt.title('Y-axis')
   sumy_array =array[0:index-1, 1].sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumy_array))+ '({:.2f})'.format(np.std(sumy_array))
   plt.plot(sumy_array/index, label=plot_label)
   plt.legend(loc=2, fontsize='small')
   plt.xlabel('Position')

   plt.subplot(223)
   sumz_array =array[0:index-1, 2].sum(axis = 0)
   label= plot_label + '\n{:.2f}'.format(np.mean(sumz_array))+ '({:.2f})'.format(np.std(sumz_array))
   plt.plot( sumz_array/index, label=plot_label)
   plt.legend(loc=8, fontsize='small')
   plt.xlabel('Z axis Position')
   plt.ylabel('ECAL Energy')

   plt.savefig(out_file)

def plot_energy_mean(array, index, out_file, num_fig, plot_label):
   ### Plot total energy deposition cell by cell along x, y, z axis                                                   

   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index-1, 0].mean(axis = 0), label=plot_label)
   plt.legend()
   plt.ylabel('Mean Energy')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index-1, 1].mean(axis = 0), label=plot_label)
   plt.legend()
   plt.xlabel('Position')

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index-1, 2].mean(axis = 0), label=plot_label)
   plt.xlabel('Position')
   plt.legend()
   plt.ylabel('Mean Energy')
   plt.savefig(out_file)

def plot_real(array, index, out_file, num_fig, plot_label):
   ## Plot the disc real/fake flag                                                                                    
   plt.figure(num_fig)
   bins = np.arange(0, 1, 0.01)
   plt.figure(num_fig)
   plt.title('Real/ Fake')
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index-1, 0]))+ '({:.2f})'.format(np.std(array[0:index-1, 0]))
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= label, normed=1)
   plt.legend()
   plt.ylabel('Events')
   plt.xlabel('Real/fake')
   plt.savefig(out_file)

def plot_error(array1, array2, index, out_file, num_fig, plot_label, pos=2):
   # plot error                                                                                                       
   plt.figure(num_fig)
   bins = np.linspace(-150, 150, 30)
   label= plot_label + '\n{:.2f}'.format(np.mean(np.absolute(array1-array2))) + '({:.2f})'.format(np.std(np.absolute(array1-array2)))
   plt.hist(np.multiply(100, array1[:index-1]- array2[:index-1]), bins=bins, histtype='step', label=label)
   plt.xlabel('error GeV')
   plt.ylabel('Number of events')
   plt.legend(title='                       Mean     ( std )', loc=pos)
   plt.savefig(out_file)

def plot_ecal(array, index, out_file, num_fig, plot_label):
   # plot ecal sum                                                                                                    
   bins = np.linspace(0, 11, 50)
   plt.figure(num_fig)
   ecal_array=np.sum(array, axis=(1, 2, 3))
   ecal_array= ecal_array[:index-1]
   label= plot_label + '\n{:.2f}'.format(np.mean(ecal_array))+ '({:.2f})'.format(np.std(ecal_array))
   plt.title('ECAL SUM')
   plt.xlabel('ECAL SUM')
   plt.ylabel('Events')
   plt.hist(ecal_array, bins=bins, histtype='step', label=label, normed=1)
   pos = 0 if energy <= 300 else 2                                      
   plt.legend(loc=pos)
   plt.savefig(out_file)

def plot_moment(array, index, out_file, num_fig, plot_label):
   # plot error                         
   plt.figure(num_fig)
   bins = np.linspace(0, 4, 30)
   label= plot_label + '\n{:.2f}'.format(np.mean(array[0:index]))+ '({:.2f})'.format(np.std(array[0:index]))
   plt.hist(array[0:index], bins='auto', histtype='step', label=label)
   plt.xlabel('X2 moment')
   plt.ylabel('Number of events')
   plt.legend(title='                       Mean     ( std )')
   plt.savefig(out_file)

# Initialization of parameters

var = {}
for energy in energies:
   var["index" + str(energy)] = 0
   var["events_act" + str(energy)] = np.zeros((num_events, 25, 25, 25))
   var["max_pos_act_" + str(energy)] = np.zeros((num_events, 3))
   var["sumact" + str(energy)] = np.zeros((num_events, 3, 25))
   var["energy_sampled" + str(energy)] = np.zeros((num_events, 1))
   var["energy_act" + str(energy)] = np.zeros((num_events, 1))
   var["isreal_act" + str(energy)] =np.zeros((num_events, 1))
   var["events_gan" + str(energy)] = np.zeros((num_events, 25, 25, 25))
   var["max_pos_gan_" + str(energy)] = np.zeros((num_events, 3))
   var["sumgan" + str(energy)] = np.zeros((num_events, 3, 25))
   var["energy_sampled" + str(energy)] = np.zeros((num_events, 1))
   var["energy_gan" + str(energy)] = np.zeros((num_events, 1))
   var["isreal_gan" + str(energy)] =np.zeros((num_events, 1))
   var["x_act" + str(energy)] = np.zeros((num_events, m))
   var["y_act" + str(energy)] =np.zeros((num_events, m))
   var["z_act" + str(energy)] =np.zeros((num_events, m))
   var["x_gan" + str(energy)] =np.zeros((num_events, m))
   var["y_gan" + str(energy)] =np.zeros((num_events, m))
   var["z_gan" + str(energy)] =np.zeros((num_events, m))
if get_gen:
    f=h5py.File(filename,'r')
    generated_images = np.array(f.get('ECAL'))
    isreal= f.get('ISREAL')
    aux_out = f.get('AUX')
    if n_jets > len(generated_images):
         print " Number of events in Gen file is less than ", n_jets
    print "Generated file ", filename, " is loaded"
else:
   ### Get Generated Data                                                                     
   g = build_generator(latent_space, return_intermediate=False)
   g.load_weights(gen_weights)
   noise = np.random.normal(0, 1, (n_jets, latent_space))
   #sampled_labels = np.ones((n_jets, 1), dtype=np.float)                                
   #sampled_labels = np.random.uniform(0, 5, (n_jets, 1))                     
   sampled_labels = Y/100  #take sampled labels from actual data                              
   print sampled_labels[:10]
   generator_in = np.multiply(sampled_labels, noise)
   start = time.time()
   generated_images = g.predict(generator_in, verbose=False, batch_size=100)
   end = time.time()
   gen_time = end - start
   print generated_images.shape
   d = build_discriminator()
   d.load_weights(disc_weights)
   start =time.time()
   isreal, aux_out, ecal_out = np.array(d.predict(generated_images, verbose=False, batch_size=100))
   end = time.time()
   disc_time = end - start
   generated_images = np.squeeze(generated_images)
   print generated_images.shape
   ## Use Discriminator for actual images                                                         
   d = build_discriminator()
   d.load_weights(disc_weights)
   image = np.expand_dims(X, axis=-1)
   start =time.time()
   isreal2, aux_out2, ecal_out2 = np.array(d.predict(image, verbose=False, batch_size=100))
   end = time.time()
   disc2_time = end - start

## Sorting data in bins
size_data = int(X.shape[0])
for i in range(size_data):
   for energy in energies:
      if Y[i] > energy-tolerance and Y[i] < energy+tolerance and var["index" + str(energy)] < num_events:
         var["events_act" + str(energy)][var["index" + str(energy)]]= X[i]
         var["events_gan" + str(energy)][var["index" + str(energy)]]= generated_images[i]
         var["energy_sampled" + str(energy)][var["index" + str(energy)]] = Y[i]/100
         if get_gen==0:
            var["energy_act" + str(energy)][var["index" + str(energy)]] = aux_out2[i]
            var["energy_gan" + str(energy)][var["index" + str(energy)]] = aux_out[i]
            var["isreal_act" + str(energy)][var["index" + str(energy)]] = isreal2[i]
            var["isreal_gan" + str(energy)][var["index" + str(energy)]] = isreal[i]
         var["index" + str(energy)]= var["index" + str(energy)] + 1

# calculations 
for energy in energies:
   for j in range(num_events):
      var["max_pos_act_" + str(energy)][j] = np.unravel_index(var["events_act" + str(energy)][j].argmax(), (25, 25, 25))
      var["sumact" + str(energy)][j, 0] = np.sum(var["events_act" + str(energy)][j], axis=(1,2))
      var["sumact" + str(energy)][j, 1] = np.sum(var["events_act" + str(energy)][j], axis=(0,2))
      var["sumact" + str(energy)][j, 2] = np.sum(var["events_act" + str(energy)][j], axis=(0,1))
      var["max_pos_gan_" + str(energy)][j] = np.unravel_index(var["events_gan" + str(energy)][j].argmax(), (25, 25, 25))
      var["sumgan" + str(energy)][j, 0] = np.sum(var["events_gan" + str(energy)][j], axis=(1,2))
      var["sumgan" + str(energy)][j, 1] = np.sum(var["events_gan" + str(energy)][j], axis=(0,2))
      var["sumgan" + str(energy)][j, 2] = np.sum(var["events_gan" + str(energy)][j], axis=(0,1))
   # Moments Computations
   totalE = np.sum(var["sumact" + str(energy)][:var["index" + str(energy)], 0], axis=1)
   ecal_size = 25
   ECAL_midX = np.zeros(var["index" + str(energy)])
   ECAL_midY = np.zeros(var["index" + str(energy)])
   ECAL_midZ = np.zeros(var["index" + str(energy)])
   for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      sumx = var["sumact" + str(energy)][0:(var["index" + str(energy)]), 0]
      print (moments.shape)
      print (sumx.shape)
      print (totalE.shape)
      ECAL_momentX = umath.inner1d(sumx, moments)/totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      var["x_act"+ str(energy)][:var["index" + str(energy)],i]= ECAL_momentX
   for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = umath.inner1d(var["sumact" + str(energy)][:var["index" + str(energy)], 1], moments)/totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      var["y_act"+ str(energy)][:var["index" + str(energy)],i]= ECAL_momentY
   for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = umath.inner1d(var["sumact" + str(energy)][:var["index" + str(energy)], 2], moments)/totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      var["z_act"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentZ
   totalE = np.sum(var["sumgan" + str(energy)][:var["index" + str(energy)], 0], axis=1)
   ECAL_midX = np.zeros(var["index" + str(energy)])
   ECAL_midY = np.zeros(var["index" + str(energy)])
   ECAL_midZ = np.zeros(var["index" + str(energy)])
   for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = umath.inner1d(var["sumgan" + str(energy)][:var["index" + str(energy)], 0], moments)/totalE
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      var["x_gan"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentX
   for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = umath.inner1d(var["sumgan" + str(energy)][:var["index" + str(energy)], 1], moments)/totalE
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      var["y_gan"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentY
   for i in range(m):
      relativeIndices = np.tile(np.arange(ecal_size), (var["index" + str(energy)],1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = umath.inner1d(var["sumgan" + str(energy)][:var["index" + str(energy)], 2], moments)/totalE
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      var["z_gan"+ str(energy)][:var["index" + str(energy)], i]= ECAL_momentZ

#### Generate Data table to screen
print "Actual Data"                                                 
print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Momentx2\t\t Momenty2\t\t Momentz2\t\t"
for energy in energies:
   print "%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f \t\t%f \t\t%f" %(energy, var["index" +str(energy)], np.amax(var["events_act" + str(energy)]), str(np.unravel_index(var["events_act" + str(energy)].argmax(), (var["index" + str(energy)], 25, 25, 25))), np.mean(var["events_act" + str(energy)]), np.mean(var["x_act"+ str(energy)][:, 1]), np.mean(var["y_act"+ str(energy)][:, 1]), np.mean(var["z_act"+ str(energy)][:, 1]))

#### Generate GAN table to screen                                                                                
print "Generated Data"
print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Momentx2\t\t Momenty2\t\t Momentz2\t\t"
for energy in energies:
   print "%d \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f \t\t%f \t\t%f" %(energy, var["index" +str(energy)], np.amax(var["events_gan" + str(energy)]), str(np.unravel_index(var["events_gan" + str(energy)].argmax(), (var["index" + str(energy)], 25, 25, 25))), np.mean(var["events_gan" + str(energy)]), np.mean(var["x_gan"+ str(energy)][:, 1]), np.mean(var["y_gan"+ str(energy)][:, 1]), np.mean(var["z_gan"+ str(energy)][:, 1]))

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
discdir = plots_dir + 'disc_outputs'
safe_mkdir(discdir)
actdir = plots_dir + 'Actual'
safe_mkdir(actdir)
gendir = plots_dir + 'Generated'
safe_mkdir(gendir)
comdir = plots_dir + 'Combined'
safe_mkdir(comdir)
mdir = plots_dir + 'Moments'
safe_mkdir(mdir)
print (m)
fig = 1

## Make seperate plots
for energy in energies:
   maxfile = "Position_of_max_" + str(energy) + ".pdf"
   histfile = "hist_" + str(energy) + ".pdf"
   ecalfile = "ecal_" + str(energy) + ".pdf"
   energyfile = "energy_" + str(energy) + ".pdf"
   drealfile = "realfake_act" + str(energy) + ".pdf"
   grealfile = "realfake_gan" + str(energy) + ".pdf"
   derrorfile = "error_act" + str(energy) + ".pdf"
   gerrorfile = "error_gan" + str(energy) + ".pdf"
   dlabel = "Data " + str(energy)
   glabel = "Data " + str(energy)

   plot_max(var["max_pos_act_" + str(energy)], var["index" + str(energy)], os.path.join(actdir, maxfile), fig, dlabel)
   fig+=1
   plot_energy_axis(var["sumact" + str(energy)], var["index" + str(energy)], os.path.join(actdir, histfile), fig, dlabel)
   fig+=1
   plot_ecal(var["events_act" + str(energy)], var["index" + str(energy)], os.path.join(actdir, ecalfile), fig, dlabel)
   fig+=1
   plot_max(var["max_pos_gan_" + str(energy)], var["index" + str(energy)], os.path.join(gendir, maxfile), fig, glabel)
   fig+=1
   plot_energy_axis(var["sumgan" + str(energy)], var["index" + str(energy)], os.path.join(gendir, histfile), fig, glabel)
   fig+=1
   plot_ecal(var["events_gan" + str(energy)], var["index" + str(energy)], os.path.join(gendir, ecalfile), fig, glabel)
   fig+=1
   if get_gen==0:
      plot_energy(np.multiply(100, var["energy_act" + str(energy)]), var["index" + str(energy)], os.path.join(actdir, energyfile), fig, dlabel, 'green')
      fig+=1
      plot_real(var["isreal_act" + str(energy)], var["index" + str(energy)], os.path.join(discdir, drealfile), fig, glabel)
      fig+=1
      plot_error(var["energy_sampled" + str(energy)],  var["energy_act" + str(energy)], var["index" + str(energy)], os.path.join(discdir, derrorfile), fig, dlabel, pos=0)
      fig+=1
      plot_energy(np.multiply(100, var["energy_gan" + str(energy)]), var["index" + str(energy)], os.path.join(gendir, energyfile), fig, glabel, 'green')
      fig+=1
      plot_real(var["isreal_gan" + str(energy)], var["index" + str(energy)], os.path.join(discdir, grealfile), fig, glabel)
      fig+=1
      plot_error(var["energy_sampled" + str(energy)],  var["energy_gan" + str(energy)], var["index" + str(energy)], os.path.join(discdir, gerrorfile), fig, glabel, pos=0)
      fig+=1
   plt.close("all")
## Make combined plots
for energy in energies:
   flatfile = "Flat_energy_all.pdf"
   histallfile = "hist_all.pdf"
   meanallfile = "hist_mean_all.pdf"
   dlabel = "Data " + str(energy)
   glabel = "GAN " + str(energy)
   plot_flat_energy(var["sumact" + str(energy)], var["index" + str(energy)], os.path.join(actdir, flatfile), fig, dlabel)
   plot_energy_axis(var["sumact" + str(energy)], var["index" + str(energy)], os.path.join(actdir, histallfile), fig+1, dlabel)
   plot_energy_mean(var["sumact" + str(energy)], var["index" + str(energy)], os.path.join(actdir, meanallfile), fig+3, dlabel)
   plot_flat_energy(var["sumgan" + str(energy)], var["index" + str(energy)], os.path.join(gendir, flatfile), fig+4, glabel)
   plot_energy_axis(var["sumgan" + str(energy)], var["index" + str(energy)], os.path.join(gendir, histallfile), fig+5, glabel)
   plot_energy_mean(var["sumgan" + str(energy)], var["index" + str(energy)], os.path.join(gendir, meanallfile), fig+6, glabel)
plt.close("all")
fig+=7
## Make superimposed plots                                                                                                                                                                                       
for energy in energies:
   maxfile = "Position_of_max_" + str(energy) + ".pdf"
   histfile = "hist_" + str(energy) + ".pdf"
   ecalfile = "ecal_" + str(energy) + ".pdf"
   energyfile = "energy_" + str(energy) + ".pdf"
   errorfile = "error_" + str(energy) + ".pdf"
   realfile = "realfake_" + str(energy) + ".pdf"
   dlabel = "Data " + str(energy)
   glabel = "GAN " + str(energy)
   xfile = "xmoment" + str(energy) + "_"
   yfile = "ymoment" + str(energy) + "_"
   zfile = "zmoment" + str(energy) + "_"
   plot_max(var["max_pos_act_" + str(energy)], var["index" + str(energy)], os.path.join(comdir, maxfile), fig, dlabel)
   plot_max(var["max_pos_gan_" + str(energy)], var["index" + str(energy)], os.path.join(comdir, maxfile), fig, glabel)
   fig+=1
   plot_energy_axis(var["sumact" + str(energy)], var["index" + str(energy)], os.path.join(comdir, histfile), fig, dlabel)
   plot_energy_axis(var["sumgan" + str(energy)], var["index" + str(energy)], os.path.join(comdir, histfile), fig, glabel)
   fig+=1
   plot_ecal(var["events_act" + str(energy)], var["index" + str(energy)], os.path.join(comdir, ecalfile), fig, dlabel)
   plot_ecal(var["events_gan" + str(energy)], var["index" + str(energy)], os.path.join(comdir, ecalfile), fig, glabel)
   fig+=1
   if get_gen==0:
      plot_energy(np.multiply(100, var["energy_act" + str(energy)]), var["index" + str(energy)], os.path.join(comdir, energyfile), fig, dlabel, 'blue')
      plot_energy(np.multiply(100, var["energy_gan" + str(energy)]), var["index" + str(energy)], os.path.join(comdir, energyfile), fig, glabel, 'green')
      fig+=1
      plot_real(var["isreal_act" + str(energy)], var["index" + str(energy)], os.path.join(discdir, realfile), fig, dlabel)
      plot_real(var["isreal_gan" + str(energy)], var["index" + str(energy)], os.path.join(discdir, realfile), fig, glabel)
      fig+=1
      plot_error(var["energy_sampled" + str(energy)],  var["energy_act" + str(energy)], var["index" + str(energy)], os.path.join(discdir, errorfile), fig, dlabel, pos=0)
      plot_error(var["energy_sampled" + str(energy)],  var["energy_gan" + str(energy)], var["index" + str(energy)], os.path.join(discdir, errorfile), fig, glabel, pos=0)
      fig+=1
      for mmt in range(m):
         mfile=xfile + str(mmt)
         plot_moment(var["x_act" + str(energy)][:, mmt], var["index" + str(energy)], os.path.join(mdir, mfile), fig, dlabel)
         plot_moment(var["x_gan" + str(energy)][:, mmt], var["index" + str(energy)], os.path.join(mdir, mfile), fig, glabel)
         fig+=1
         mfile=yfile + str(mmt)
         plot_moment(var["y_act" + str(energy)][:, mmt], var["index" + str(energy)], os.path.join(mdir, mfile), fig, dlabel)
         plot_moment(var["y_gan" + str(energy)][:, mmt], var["index" + str(energy)], os.path.join(mdir, mfile), fig, glabel)
         fig+=1
         mfile=zfile + str(mmt)
         plot_moment(var["z_act" + str(energy)][:, mmt], var["index" + str(energy)], os.path.join(mdir, mfile), fig, dlabel)
         plot_moment(var["z_gan" + str(energy)][:, mmt], var["index" + str(energy)], os.path.join(mdir, mfile), fig, glabel)
         fig+=1

plt.close("all")
# Plot for all
plot_ecal(X, 4 * num_events, os.path.join(comdir, 'ECAL_sum.pdf'), fig, 'Data')
plot_ecal(generated_images, 4 * num_events, os.path.join(comdir, 'ECAL_sum.pdf'), fig, 'GAN')
fig+=1
plot_energy(Y, len(energies) * num_events, os.path.join(comdir, 'energy.pdf'), fig, 'Primary Energy')
plt.close('all')

### Save generated image data to file
if (save):
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=sampled_labels)
      outfile.create_dataset('AUX',data=aux_out)
      outfile.create_dataset('ISREAL',data=isreal)
   print "Generated ECAL saved to ", filename   
if get_gen == 0:
   print 'Generation time for %d events was %.3f'%(n_jets, gen_time)
   print 'Discremenation time for %d generated events was %.3f'%(n_jets, disc_time)
   print 'Discremenation time for %d actual events was %.3f' %(n_jets, disc2_time)
print 'Plots are saved in ', plots_dir



