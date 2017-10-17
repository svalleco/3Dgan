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
gen_weights='maelin50_full/params_generator_epoch_049.hdf5'
disc_weights='maelin50_full/params_discriminator_epoch_049.hdf5'
#gen_weights='full_maelin30/params_generator_epoch_029.hdf5'
#disc_weights='full_maelin30/params_discriminator_epoch_029.hdf5'
latent_space =200
num_events=1000

# Other params
save = 1
get_actual = 1
filename10 = 'Gen_full_10.h5'
filename50 = 'Gen_full_50.h5'
filename100 = 'Gen_full_100.h5'
filename150 = 'Gen_full_150.h5'
filename200 = 'Gen_full_200.h5'
filename300 = 'Gen_full_300.h5'
filename400 = 'Gen_full_400.h5'
filename500 = 'Gen_full_500.h5'

## Get Full data
if (get_actual):      
   d=h5py.File("/eos/project/d/dshep/LCD/FixedEnergy/Ele_10GeV/Ele_10GeV_0.h5",'r')
   c=np.array(d.get('ECAL'))
   e=d.get('target')
   X10=np.array(c[:num_events])
   y=np.array(e[:num_events,1])
   Y10=np.expand_dims(y, axis=-1)
   print X10.shape
   print Y10.shape
   X10[X10 < 1e-6] = 0

   d=h5py.File("/eos/project/d/dshep/LCD/FixedEnergy/Ele_50GeV/Ele_50GeV_0.h5",'r')
   c=np.array(d.get('ECAL'))
   e=d.get('target')
   X50=np.array(c[:num_events])
   y=np.array(e[:num_events,1])
   Y50=np.expand_dims(y, axis=-1)
   X50[X50 < 1e-6] = 0
   
   d=h5py.File("/eos/project/d/dshep/LCD/FixedEnergy/Ele_100GeV/Ele_100GeV_0.h5",'r')
   c=np.array(d.get('ECAL'))
   e=d.get('target')
   X100=np.array(c[:num_events])
   y=np.array(e[:num_events,1])
   Y100=np.expand_dims(y, axis=-1)
   X100[X100 < 1e-6] = 0
   
   d=h5py.File("/eos/project/d/dshep/LCD/FixedEnergy/Ele_200GeV/Ele_200GeV_0.h5",'r')
   c=np.array(d.get('ECAL'))
   e=d.get('target')
   X200=np.array(c[:num_events])
   y=np.array(e[:num_events,1])
   Y200=np.expand_dims(y, axis=-1)
   X200[X200 < 1e-6] = 0

# Histogram Functions

def plot_max(array, index, out_file, num_fig, energy):
   ## Plot the Histogram of Maximum energy deposition location on all axis
   bins = np.arange(0, 25, 1)
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= str(energy))
   plt.legend()
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[0:index-1, 1], bins=bins, histtype='step', label=str(energy))
   plt.legend()
   plt.xlabel('Position')

   plt.subplot(223)                                                                           
   plt.hist(array[0:index-1, 2], bins=bins, histtype='step', label=str(energy))
   plt.legend(loc=1)
   plt.xlabel('Position')
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_energy(array, index, out_file, num_fig, energy):
   ### Plot Histogram of energy flat distribution along all three axis                                        
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(array[:index, 0].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[:index, 1].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()
   plt.xlabel('ECAL Cell Energy')
   
   plt.subplot(223)
   plt.hist(array[:index, 2].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_energy2(array, index, out_file, num_fig, energy, color='blue', style='-'):
   ### Plot Histogram of energy 
   plt.figure(num_fig)
   ebins=np.arange(0, 500, 5)
   label= energy + '  {:.2f}'.format(np.mean(array))+ ' ( {:.2f}'.format(np.std(array)) + ' )'
   plt.hist(array, bins=ebins, histtype='step', label=label, color=color, ls=style)
   plt.xticks([0, 10, 50, 100, 150, 200, 300, 400, 500])
   plt.xlabel('Energy GeV')
   plt.ylabel('Events')
   plt.legend(title='                      Mean     (std)', loc=0)
   plt.savefig(out_file)
                                                                                                                                                 
def plot_energy_hist(array, index, out_file, num_fig, energy):
   ### Plot total energy deposition cell by cell along x, y, z axis                                             
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index, 0].sum(axis = 0)/index, label=str(energy))
   plt.ylabel('ECAL Energy/Events')
   plt.legend()

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index, 1].sum(axis = 0)/index, label=str(energy))
   plt.legend()
   plt.xlabel('Position')

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index, 2].sum(axis = 0)/index, label=str(energy))
   plt.legend()                                                          
   plt.xlabel('Position')
   plt.ylabel('ECAL Energy/Events')

   plt.savefig(out_file)

def plot_energy_mean(array, index, out_file, num_fig, energy):
   ### Plot total energy deposition cell by cell along x, y, z axis      
   
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index, 0].mean(axis = 0), label=str(energy))
   plt.legend()
   plt.ylabel('Mean Energy')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index, 1].mean(axis = 0), label=str(energy))
   plt.legend()
   plt.xlabel('Position')

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index, 2].mean(axis = 0), label=str(energy))
   plt.xlabel('Position')
   plt.legend()
   plt.ylabel('Mean Energy')
   plt.savefig(out_file)

def plot_real(array, index, out_file, num_fig, energy):
   ## Plot the disc real/fake flag
   plt.figure(num_fig)
   bins = np.arange(0, 1, 0.01)
   plt.figure(num_fig)
   plt.title('Real/ Fake')
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= str(energy))
   plt.legend()
   plt.ylabel('Events')
   plt.xlabel('Real/fake')
   plt.savefig(out_file)

def plot_error(array1, array2, index, out_file, num_fig, energy, pos=2):
   # plot error
   plt.figure(num_fig)
   bins = np.linspace(-100, 100, 30)
   label= energy + '  {:.2f} '.format(np.multiply(100, np.mean(np.absolute(array1-array2)))) + ' ( {:.2f}'.format(np.multiply(100, np.std(array1-array2)))+ ' )'
   plt.hist(np.multiply(100, array1-array2), bins=bins, histtype='step', label=label)
   plt.xlabel('error GeV')
   plt.ylabel('Number of events')
   plt.legend(title='                       Mean     ( std )', loc=pos)
   plt.savefig(out_file)

def plot_ecal(array, index, out_file, num_fig, energy):
   # plot ecal sum
   bins = np.linspace(0, 11, 50)
   plt.figure(num_fig)
   plt.title('ECAL SUM')
   plt.xlabel('ECAL SUM')
   plt.ylabel('Events')
   plt.hist(np.sum(array, axis=(1, 2, 3)), bins=bins, histtype='step', label=energy)
   plt.legend(loc=0)
   plt.savefig(out_file)

# Initialization of parameters                                                 
index10 = num_events
index50 = num_events
index100 = num_events
index150 = num_events
index200 = num_events
index300 = num_events
index400 = num_events
index500 = num_events

#Initialization of arrays for actual events 
events_act10 = np.zeros((num_events, 25, 25, 25))
max_pos_act_10 = np.zeros((num_events, 3))                                              
events_act50 = np.zeros((num_events, 25, 25, 25))
max_pos_act_50 = np.zeros((num_events, 3))
events_act100 = np.zeros((num_events, 25, 25, 25))
max_pos_act_100 = np.zeros((num_events, 3))
events_act200 = np.zeros((num_events, 25, 25, 25))
max_pos_act_200 = np.zeros((num_events, 3))
sum_act10 = np.zeros((num_events, 3, 25))
sum_act50 = np.zeros((num_events, 3, 25))
sum_act100 = np.zeros((num_events, 3, 25))
sum_act200 = np.zeros((num_events, 3, 25))
energy_sampled10 = np.multiply(0.1, np.ones((num_events, 1)))
energy_sampled50 = np.multiply(0.5, np.ones((num_events, 1)))
energy_sampled100 = np.ones((num_events, 1))
energy_sampled150 = np.multiply(1.5, np.ones((num_events, 1)))
energy_sampled200 = np.multiply(2, np.ones((num_events, 1)))
energy_sampled300 = np.multiply(3, np.ones((num_events, 1)))
energy_sampled400 = np.multiply(4, np.ones((num_events, 1)))
energy_sampled500 = np.multiply(5, np.ones((num_events, 1)))
energy_act10 = np.zeros((num_events, 1))
energy_act50 = np.zeros((num_events, 1))
energy_act100 = np.zeros((num_events, 1))
energy_act200 = np.zeros((num_events, 1))
energy_act300 = np.zeros((num_events, 1))

#Initialization of arrays for generated images                                             
events_gan10 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_10 = np.zeros((num_events, 3))
events_gan50 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_50 = np.zeros((num_events, 3))
events_gan100 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_100 = np.zeros((num_events, 3))
events_gan150 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_150 = np.zeros((num_events, 3))
events_gan200 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_200 = np.zeros((num_events, 3))
events_gan300 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_300 = np.zeros((num_events, 3))
events_gan400 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_400 = np.zeros((num_events, 3))
events_gan500 = np.zeros((num_events, 25, 25, 25))
max_pos_gan_500 = np.zeros((num_events, 3))
sum_gan10 = np.zeros((num_events, 3, 25))
sum_gan50 = np.zeros((num_events, 3, 25))
sum_gan100 = np.zeros((num_events, 3, 25))
sum_gan150 = np.zeros((num_events, 3, 25))
sum_gan200 = np.zeros((num_events, 3, 25))
sum_gan300 = np.zeros((num_events, 3, 25))
sum_gan400 = np.zeros((num_events, 3, 25))
sum_gan500 = np.zeros((num_events, 3, 25))
energy_gan10 = np.zeros((num_events, 1))
energy_gan50 = np.zeros((num_events, 1))
energy_gan100 = np.zeros((num_events, 1))
energy_gan150 = np.zeros((num_events, 1))
energy_gan200 = np.zeros((num_events, 1))
energy_gan300 = np.zeros((num_events, 1))
energy_gan400 = np.zeros((num_events, 1))
energy_gan500 = np.zeros((num_events, 1))

### Get Generated Data
## events for 10 GeV                                                                     
g = build_generator(latent_space, return_intermediate=False)
g.load_weights(gen_weights)
noise = np.random.normal(0, 1, (num_events, latent_space))                    
sampled_labels = energy_sampled10                             
generator_in = np.multiply(sampled_labels, noise)
start = time.time()
generated_images10 = g.predict(generator_in, verbose=False, batch_size=100)
end = time.time()
gen_time = end - start
print generated_images10.shape
print gen_time
d = build_discriminator()
d.load_weights(disc_weights)
start =time.time()
isreal10, aux_out10 = np.array(d.predict(generated_images10, verbose=False, batch_size=100))
end = time.time()
disc_time = end - start
generated_images10 = np.squeeze(generated_images10)
print generated_images10.shape
print disc_time

## events for 50 GeV                                                                          
noise = np.random.normal(0, 1, (num_events, latent_space))
sampled_labels = energy_sampled50
generator_in = np.multiply(sampled_labels, noise)
generated_images50 = g.predict(generator_in, verbose=False, batch_size=100)
isreal50, aux_out50 = np.array(d.predict(generated_images50, verbose=False, batch_size=100))
generated_images50 = np.squeeze(generated_images50)

## events for 100 GeV                                                                         
noise = np.random.normal(0, 1, (num_events, latent_space))
sampled_labels = energy_sampled100
generator_in = np.multiply(sampled_labels, noise)
generated_images100 = g.predict(generator_in, verbose=False, batch_size=100)
isreal100, aux_out100 = np.array(d.predict(generated_images100, verbose=False, batch_size=100))
generated_images100 = np.squeeze(generated_images100)

## events for 150 GeV                                                                         
noise = np.random.normal(0, 1, (num_events, latent_space))
sampled_labels = energy_sampled150
generator_in = np.multiply(sampled_labels, noise)
generated_images150 = g.predict(generator_in, verbose=False, batch_size=100)
isreal150, aux_out150 = np.array(d.predict(generated_images150, verbose=False, batch_size=100))
generated_images150 = np.squeeze(generated_images150)

## events for 200 GeV                                                                         
noise = np.random.normal(0, 1, (num_events, latent_space))
sampled_labels = energy_sampled200
generator_in = np.multiply(sampled_labels, noise)
generated_images200 = g.predict(generator_in, verbose=False, batch_size=100)
isreal200, aux_out200 = np.array(d.predict(generated_images200, verbose=False, batch_size=100))
generated_images200 = np.squeeze(generated_images200)

## events for 300 GeV                                                                         
noise = np.random.normal(0, 1, (num_events, latent_space))
sampled_labels = energy_sampled300
generator_in = np.multiply(sampled_labels, noise)
generated_images300 = g.predict(generator_in, verbose=False, batch_size=100)
isreal300, aux_out300 = np.array(d.predict(generated_images300, verbose=False, batch_size=100))
generated_images300 = np.squeeze(generated_images300)

## events for 400 GeV                                                                                      
noise = np.random.normal(0, 1, (num_events, latent_space))
sampled_labels = energy_sampled400
generator_in = np.multiply(sampled_labels, noise)
generated_images400 = g.predict(generator_in, verbose=False, batch_size=100)
isreal400, aux_out400 = np.array(d.predict(generated_images400, verbose=False, batch_size=100))
generated_images400 = np.squeeze(generated_images400)

## events for 500 GeV                                                                                      
noise = np.random.normal(0, 1, (num_events, latent_space))
sampled_labels = energy_sampled500
generator_in = np.multiply(sampled_labels, noise)
generated_images500 = g.predict(generator_in, verbose=False, batch_size=100)
isreal500, aux_out500 = np.array(d.predict(generated_images500, verbose=False, batch_size=100))
generated_images500 = np.squeeze(generated_images500)

## Use Discriminator for actual images                                                        

if (get_actual):
     ## events for 10 GeV
     image10 = np.expand_dims(X10, axis=-1)
     isreal2_10, aux_out2_10 = np.array(d.predict(image10, verbose=False, batch_size=100))
     ## events for 50 GeV                                                                     
     image50 = np.expand_dims(X50, axis=-1)
     isreal2_50, aux_out2_50 = np.array(d.predict(image50, verbose=False, batch_size=100))
     ## events for 100 GeV                                                                    
     image100 = np.expand_dims(X100, axis=-1)
     isreal2_100, aux_out2_100 = np.array(d.predict(image50, verbose=False, batch_size=100))
     ## events for 200 GeV                                                                    
     image200 = np.expand_dims(X200, axis=-1)
     isreal2_200, aux_out2_200 = np.array(d.predict(image200, verbose=False, batch_size=100))
                                              
     #calculations for actual
     for j in range(num_events):
        events_act10[j]= X10[j]
        events_act50[j]= X50[j]
        events_act100[j]= X100[j]
        events_act200[j]= X200[j]
        max_pos_act_10[j] = np.unravel_index(events_act10[j].argmax(), (25, 25, 25))
        max_pos_act_50[j] = np.unravel_index(events_act50[j].argmax(), (25, 25, 25))
        max_pos_act_100[j] = np.unravel_index(events_act100[j].argmax(), (25, 25, 25))
        max_pos_act_200[j] = np.unravel_index(events_act200[j].argmax(), (25, 25, 25))
        sum_act10[j, 0] = np.sum(events_act10[j], axis=(1,2))
        sum_act10[j, 1] = np.sum(events_act10[j], axis=(0,2))
        sum_act10[j, 2] = np.sum(events_act10[j], axis=(0,1))
        sum_act50[j, 0] = np.sum(events_act50[j], axis=(1,2))
        sum_act50[j, 1] = np.sum(events_act50[j], axis=(0,2))
        sum_act50[j, 2] = np.sum(events_act50[j], axis=(0,1))
        sum_act100[j, 0] = np.sum(events_act100[j], axis=(1,2))
        sum_act100[j, 1] = np.sum(events_act100[j], axis=(0,2))
        sum_act100[j, 2] = np.sum(events_act100[j], axis=(0,1))
        sum_act200[j, 0] = np.sum(events_act200[j], axis=(1,2))
        sum_act200[j, 1] = np.sum(events_act200[j], axis=(0,2))
        sum_act200[j, 2] = np.sum(events_act200[j], axis=(0,1))
   
### Calculations for generated                                                             
for j in range(num_events):
   events_gan10[j]= generated_images10[j]
   events_gan50[j]= generated_images50[j]
   events_gan100[j]= generated_images100[j]
   events_gan150[j]= generated_images150[j]
   events_gan200[j]= generated_images200[j]
   events_gan300[j]= generated_images300[j]
   events_gan400[j]= generated_images400[j]
   events_gan500[j]= generated_images500[j]
   max_pos_gan_10[j] = np.unravel_index(events_gan10[j].argmax(), (25, 25, 25))
   max_pos_gan_50[j] = np.unravel_index(events_gan50[j].argmax(), (25, 25, 25))
   max_pos_gan_100[j] = np.unravel_index(events_gan100[j].argmax(), (25, 25, 25))
   max_pos_gan_150[j] = np.unravel_index(events_gan150[j].argmax(), (25, 25, 25))
   max_pos_gan_200[j] = np.unravel_index(events_gan200[j].argmax(), (25, 25, 25))
   max_pos_gan_300[j] = np.unravel_index(events_gan300[j].argmax(), (25, 25, 25))
   max_pos_gan_400[j] = np.unravel_index(events_gan400[j].argmax(), (25, 25, 25))
   max_pos_gan_500[j] = np.unravel_index(events_gan500[j].argmax(), (25, 25, 25))
   sum_gan10[j, 0] = np.sum(events_gan50[j], axis=(1,2))
   sum_gan10[j, 1] = np.sum(events_gan50[j], axis=(0,2))
   sum_gan10[j, 2] = np.sum(events_gan50[j], axis=(0,1))
   sum_gan50[j, 0] = np.sum(events_gan50[j], axis=(1,2))
   sum_gan50[j, 1] = np.sum(events_gan50[j], axis=(0,2))
   sum_gan50[j, 2] = np.sum(events_gan50[j], axis=(0,1))
   sum_gan100[j, 0] = np.sum(events_gan100[j], axis=(1,2))
   sum_gan100[j, 1] = np.sum(events_gan100[j], axis=(0,2))
   sum_gan100[j, 2] = np.sum(events_gan100[j], axis=(0,1))
   sum_gan150[j, 0] = np.sum(events_gan150[j], axis=(1,2))
   sum_gan150[j, 1] = np.sum(events_gan150[j], axis=(0,2))
   sum_gan150[j, 2] = np.sum(events_gan150[j], axis=(0,1))
   sum_gan200[j, 0] = np.sum(events_gan200[j], axis=(1,2))
   sum_gan200[j, 1] = np.sum(events_gan200[j], axis=(0,2))
   sum_gan200[j, 2] = np.sum(events_gan200[j], axis=(0,1))
   sum_gan300[j, 0] = np.sum(events_gan300[j], axis=(1,2))
   sum_gan300[j, 1] = np.sum(events_gan300[j], axis=(0,2))
   sum_gan300[j, 2] = np.sum(events_gan300[j], axis=(0,1))
   sum_gan400[j, 0] = np.sum(events_gan400[j], axis=(1,2))
   sum_gan400[j, 1] = np.sum(events_gan400[j], axis=(0,2))
   sum_gan400[j, 2] = np.sum(events_gan400[j], axis=(0,1))
   sum_gan500[j, 0] = np.sum(events_gan500[j], axis=(1,2))
   sum_gan500[j, 1] = np.sum(events_gan500[j], axis=(0,2))
   sum_gan500[j, 2] = np.sum(events_gan500[j], axis=(0,1))

## Generate Data table to screen
if (get_actual):
   print "Actual Data"                                                 
   print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t"
   print "50 \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(index10, np.amax(events_act10), str(np.unravel_index(events_act10.argmax(), (index10, 25, 25, 25))), np.mean(events_act10), np.amin(events_act10))
   print "50 \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(index50, np.amax(events_act50), str(np.unravel_index(events_act50.argmax(), (index50, 25, 25, 25))), np.mean(events_act50), np.amin(events_act50))
   print "100 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index100, np.amax(events_act100), str(np.unravel_index(events_act100.argmax(), (index100, 25, 25, 25))), np.mean(events_act100), np.amin(events_act100))
   print "200 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index200, np.amax(events_act200), str(np.unravel_index(events_act200.argmax(), (index200, 25, 25, 25))), np.mean(events_act200), np.amin(events_act200))
 
#### Generate GAN table to screen                                                                                
print "Generated Data"
print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t"
print "10 \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(index10, np.amax(events_gan10), str(np.unravel_index(events_gan10.argmax(), (index10, 25, 25, 25))), np.mean(events_gan10), np.amin(events_gan10))
print "50 \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(index50, np.amax(events_gan50), str(np.unravel_index(events_gan50.argmax(), (index50, 25, 25, 25))), np.mean(events_gan50), np.amin(events_gan50))
print "100 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index100, np.amax(events_gan100), str(np.unravel_index(events_gan100.argmax(), (index100, 25, 25, 25))), np.mean(events_gan100), np.amin(events_gan100))
print "150 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index150, np.amax(events_gan150), str(np.unravel_index(events_gan150.argmax(), (index150, 25, 25, 25))), np.mean(events_gan150), np.amin(events_gan150))
print "200 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index200, np.amax(events_gan200), str(np.unravel_index(events_gan200.argmax(), (index200, 25, 25, 25))), np.mean(events_gan200), np.amin(events_gan200))
print "300 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index300, np.amax(events_gan300), str(np.unravel_index(events_gan300.argmax(), (index300, 25, 25, 25))), np.mean(events_gan300), np.amin(events_gan300))
print "400 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index400, np.amax(events_gan400), str(np.unravel_index(events_gan400.argmax(), (index400, 25, 25, 25))), np.mean(events_gan400), np.amin(events_gan400))
print "500 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index500, np.amax(events_gan500), str(np.unravel_index(events_gan500.argmax(), (index500, 25, 25, 25))), np.mean(events_gan500), np.amin(events_gan500))

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
discdir = 'fixed_plots/disc_outputs'
safe_mkdir(discdir)
actdir = 'fixed_plots/Actual'
safe_mkdir(actdir)
gendir = 'fixed_plots/Generated'
safe_mkdir(gendir)
comdir = 'fixed_plots/Combined'
safe_mkdir(comdir)

## Make plots for generated data  
plot_real(isreal10, index10, os.path.join(discdir, 'real_10.pdf'), 1, 'GAN 10')
plot_real(isreal50, index50, os.path.join(discdir, 'real_50.pdf'), 2, 'GAN 50')
plot_real(isreal100, index100, os.path.join(discdir, 'real_100.pdf'), 3, 'GAN 100')
plot_real(isreal150, index150, os.path.join(discdir, 'real_150.pdf'), 4, 'GAN 150')
plot_real(isreal200, index200, os.path.join(discdir, 'real_200.pdf'), 5, 'GAN 200')
plot_real(isreal300, index300, os.path.join(discdir, 'real_300.pdf'), 6, 'GAN 300')
plot_real(isreal400, index400, os.path.join(discdir, 'real_400.pdf'), 47, 'GAN 400')
plot_real(isreal500, index500, os.path.join(discdir, 'real_500.pdf'), 48, 'GAN 500')

plot_error(energy_sampled10, aux_out10, index10, os.path.join(discdir, 'error_10.pdf'), 7, 'GAN 10')
plot_error(energy_sampled50, aux_out50, index50, os.path.join(discdir, 'error_50.pdf'), 8, 'GAN 50')
plot_error(energy_sampled100, aux_out100, index100, os.path.join(discdir, 'error_100.pdf'), 9, 'GAN 100')
plot_error(energy_sampled200, aux_out200, index200, os.path.join(discdir, 'error_200.pdf'), 10, 'GAN 200')
plot_error(energy_sampled300, aux_out300, index300, os.path.join(discdir, 'error_300.pdf'), 11, 'GAN 300')
plot_error(energy_sampled400, aux_out400, index400, os.path.join(discdir, 'error_400.pdf'), 49, 'GAN 400')
plot_error(energy_sampled500, aux_out500, index500, os.path.join(discdir, 'error_500.pdf'), 50, 'GAN 500')

plot_max(max_pos_gan_10, index10, os.path.join(comdir, 'Position_of_max_10.pdf'), 12, 'GAN 10')
plot_max(max_pos_gan_50, index50, os.path.join(comdir, 'Position_of_max_50.pdf'), 13, 'GAN 50')
plot_max(max_pos_gan_100, index100, os.path.join(comdir, 'Position_of_max_100.pdf'), 14, 'GAN 100')
plot_max(max_pos_gan_150, index150, os.path.join(comdir, 'Position_of_max_150.pdf'), 15, 'GAN 150')
plot_max(max_pos_gan_200, index200, os.path.join(comdir, 'Position_of_max_200.pdf'), 16, 'GAN 200')
plot_max(max_pos_gan_300, index300, os.path.join(comdir, 'Position_of_max_300.pdf'), 17, 'GAN 300')      
plot_max(max_pos_gan_400, index400, os.path.join(comdir, 'Position_of_max_400.pdf'), 51, 'GAN 400')   
plot_max(max_pos_gan_500, index500, os.path.join(comdir, 'Position_of_max_500.pdf'), 52, 'GAN 500')
                  
plot_energy_hist(sum_gan10, index10, os.path.join(comdir, 'hist_10.pdf'), 18, 'GAN 10')
plot_energy_hist(sum_gan50, index50, os.path.join(comdir, 'hist_50.pdf'), 19, 'GAN 50')
plot_energy_hist(sum_gan100, index100, os.path.join(comdir, 'hist_100.pdf'), 20, 'GAN 100')
plot_energy_hist(sum_gan150, index150, os.path.join(comdir, 'hist_150.pdf'), 21, 'GAN 150')
plot_energy_hist(sum_gan200, index200, os.path.join(comdir, 'hist_200.pdf'), 22, 'GAN 200')
plot_energy_hist(sum_gan300, index300, os.path.join(comdir, 'hist_300.pdf'), 23, 'GAN 300')
plot_energy_hist(sum_gan400, index400, os.path.join(comdir, 'hist_400.pdf'), 53, 'GAN 400')
plot_energy_hist(sum_gan500, index500, os.path.join(comdir, 'hist_500.pdf'), 54, 'GAN 500')

## Make plots for real data   
if (get_actual):
   plot_real(isreal2_10, index10, os.path.join(discdir, 'real_10_act.pdf'), 61, 'Data 10')
   plot_real(isreal2_50, index50, os.path.join(discdir, 'real_50_act.pdf'), 62, 'Data 50')
   plot_real(isreal2_100, index100, os.path.join(discdir, 'real_100_act.pdf'), 63, 'Data 100')
   plot_real(isreal2_200, index200, os.path.join(discdir, 'real_200_act.pdf'), 65, 'Data 200')
   plot_error(energy_sampled10, aux_out2_10, index10, os.path.join(discdir, 'error_10_act.pdf'), 67, 'Data 10', 0)
   plot_error(energy_sampled50, aux_out2_50, index50, os.path.join(discdir, 'error_50_act.pdf'), 68, 'Data 50', 0)
   plot_error(energy_sampled100, aux_out2_100, index100, os.path.join(discdir, 'error_100_act.pdf'), 69, 'Data 100')
   plot_error(energy_sampled200, aux_out2_200, index200, os.path.join(discdir, 'error_200_act.pdf'), 70, 'Data 200', 0)
   plot_max(max_pos_act_10, index10, os.path.join(comdir, 'Position_of_max_10_act.pdf'), 72, 'Data 50')
   plot_max(max_pos_act_50, index50, os.path.join(comdir, 'Position_of_max_50_act.pdf'), 73, 'Data 50')
   plot_max(max_pos_act_100, index100, os.path.join(comdir, 'Position_of_max_100_act.pdf'), 74, 'Data 100')
   plot_max(max_pos_act_200, index200, os.path.join(comdir, 'Position_of_max_200_act.pdf'), 76, 'Data 200')
   plot_energy_hist(sum_act10, index10, os.path.join(comdir, 'hist_10_act.pdf'), 78, 'Data 10')
   plot_energy_hist(sum_act50, index50, os.path.join(comdir, 'hist_50_act.pdf'), 79, 'Data 50')
   plot_energy_hist(sum_act100, index100, os.path.join(comdir, 'hist_100_act.pdf'), 80, 'Data 100')
   plot_energy_hist(sum_act200, index200, os.path.join(comdir, 'hist_200_act.pdf'), 82, 'Data 200')
   plot_energy(sum_act10, index10, os.path.join(actdir, 'Flat_energy.pdf'), 25, 10)
   plot_energy(sum_act50, index50, os.path.join(actdir, 'Flat_energy.pdf'), 25, 50)
   plot_energy(sum_act100, index100, os.path.join(actdir, 'Flat_energy.pdf'),25, 100)
   plot_energy(sum_act200, index200, os.path.join(actdir, 'Flat_energy.pdf'),25, 200)
   plot_energy_hist(sum_act10, index10, os.path.join(actdir, 'hist_all.pdf'), 26, 'Data 10')
   plot_energy_hist(sum_act50, index50, os.path.join(actdir, 'hist_all.pdf'), 26, 'Data 50')
   plot_energy_hist(sum_act100, index100, os.path.join(actdir, 'hist_all.pdf'), 26, 'Data 100')
   plot_energy_hist(sum_act200, index200, os.path.join(actdir, 'hist_all.pdf'), 26, 'Data 200')
   plot_energy_mean(sum_act10, index10, os.path.join(actdir, 'hist_mean_all.pdf'), 27, 'Data 10')
   plot_energy_mean(sum_act50, index50, os.path.join(actdir, 'hist_mean_all.pdf'), 27, 'Data 50')
   plot_energy_mean(sum_act100, index100, os.path.join(actdir, 'hist_mean_all.pdf'), 27, 'Data 100')
   plot_energy_mean(sum_act200, index200, os.path.join(actdir, 'hist_mean_all.pdf'), 27, 'Data 200')
   X = np.concatenate((X10, X50, X100, X200))
   plot_ecal(X, 4 * num_events, os.path.join(comdir, 'ECAL_sum.pdf'), 28, 'All Data')
   plot_energy2(np.multiply(100, aux_out2_10), index10, os.path.join(comdir, 'energy10_act.pdf'), 84, 'Data 10', 'green')
   plot_energy2(np.multiply(100, aux_out2_50), index50, os.path.join(comdir, 'energy50_act.pdf'), 85, 'Data 50', 'green')
   plot_energy2(np.multiply(100, aux_out2_100), index100, os.path.join(comdir, 'energy100_act.pdf'), 86, 'Data 100', 'green')
   plot_energy2(np.multiply(100, aux_out2_200), index200, os.path.join(comdir, 'energy200_act.pdf'), 88, 'Data 200', 'green')
   plot_ecal(events_act10, num_events, os.path.join(comdir, 'ECAL_sum10_act.pdf'), 91, 'Data 10')
   plot_ecal(events_act50, num_events, os.path.join(comdir, 'ECAL_sum50_act.pdf'), 92, 'Data 50')
   plot_ecal(events_act100, num_events, os.path.join(comdir, 'ECAL_sum100_act.pdf'), 93, 'Data 100')
   plot_ecal(events_act200, num_events, os.path.join(comdir, 'ECAL_sum200_act.pdf'), 95, 'Data 200')
Y = np.concatenate((energy_sampled10, energy_sampled50, energy_sampled100, energy_sampled150, energy_sampled200, energy_sampled300, energy_sampled400, energy_sampled500))
plot_energy2(np.multiply(100, Y), 6 * num_events, os.path.join(comdir, 'energy.pdf'), 29, 'Primary Energy')
generated_images = np.concatenate((generated_images10, generated_images50, generated_images100, generated_images150, generated_images200, generated_images300, generated_images400, generated_images500))
plot_ecal(generated_images, 6 * num_events, os.path.join(comdir, 'ECAL_sum.pdf'), 28, 'GAN')

## Plots for Generated
plot_max(max_pos_gan_10, index10, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 10')
plot_max(max_pos_gan_50, index50, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 50')
plot_max(max_pos_gan_100, index100, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 100')
plot_max(max_pos_gan_150, index150, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 150')
plot_max(max_pos_gan_200, index200, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 200')
plot_max(max_pos_gan_300, index300, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 300')
plot_max(max_pos_gan_400, index400, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 400')
plot_max(max_pos_gan_500, index500, os.path.join(gendir, 'Position_of_max.pdf'), 30, 'GAN 500')

plot_energy(sum_gan10, index10, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 10')
plot_energy(sum_gan50, index50, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 50')
plot_energy(sum_gan100, index100, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 100')
plot_energy(sum_gan150, index150, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 150')
plot_energy(sum_gan200, index200, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 200')
plot_energy(sum_gan300, index300, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 300')
plot_energy(sum_gan400, index400, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 400')
plot_energy(sum_gan500, index500, os.path.join(gendir, 'Flat_energy.pdf'), 31, 'GAN 500')

plot_energy_hist(sum_gan10, index10, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 10')
plot_energy_hist(sum_gan50, index50, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 50')
plot_energy_hist(sum_gan100, index100, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 100')
plot_energy_hist(sum_gan150, index150, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 150')
plot_energy_hist(sum_gan200, index200, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 200')
plot_energy_hist(sum_gan300, index300, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 300')
plot_energy_hist(sum_gan400, index400, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 400')
plot_energy_hist(sum_gan500, index500, os.path.join(gendir, 'hist_all.pdf'), 32, 'GAN 500')

plot_energy_mean(sum_gan10, index10, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 10)
plot_energy_mean(sum_gan50, index50, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 50)
plot_energy_mean(sum_gan100, index100, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 100)
plot_energy_mean(sum_gan150, index150, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 150)
plot_energy_mean(sum_gan200, index200, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 200)
plot_energy_mean(sum_gan300, index300, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 300)
plot_energy_mean(sum_gan400, index400, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 400)
plot_energy_mean(sum_gan500, index500, os.path.join(gendir, 'hist_mean_all.pdf'), 33, 500)

#plot_energy2(np.multiply(100, energy_sampled10), index10, os.path.join(comdir, 'energy10.pdf'), 34, 'Primary 10', 'red', '--')
#plot_energy2(np.multiply(100, energy_sampled50), index50, os.path.join(comdir, 'energy50.pdf'), 35, 'Primary 50', 'blue', '--')
#plot_energy2(np.multiply(100, energy_sampled100), index100, os.path.join(comdir, 'energy100.pdf'), 36, 'Primary 100', 'green', '--')
#plot_energy2(np.multiply(100, energy_sampled150), index150, os.path.join(comdir, 'energy150.pdf'), 37, 'Primary 150', 'yellow', '--')
#plot_energy2(np.multiply(100, energy_sampled200), index200, os.path.join(comdir, 'energy200.pdf'), 38, 'Primary 200', 'cyan', '--')
#plot_energy2(np.multiply(100, energy_sampled300), index300, os.path.join(comdir, 'energy300.pdf'), 39, 'Primary 300', 'magenta', '--')
#plot_energy2(np.multiply(100, energy_sampled400), index400, os.path.join(comdir, 'energy400.pdf'), 39, 'Primary 400', 'magenta', '--')                                                                           #plot_energy2(np.multiply(100, energy_sampled500), index500, os.path.join(comdir, 'energy500.pdf'), 39, 'Primary 500', 'magenta', '--')                                                                            
plot_energy2(np.multiply(100, aux_out10), index10, os.path.join(comdir, 'energy10.pdf'), 34, 'GAN 10')
plot_energy2(np.multiply(100, aux_out50), index50, os.path.join(comdir, 'energy50.pdf'), 35, 'GAN 50')
plot_energy2(np.multiply(100, aux_out100), index100, os.path.join(comdir, 'energy100.pdf'), 36, 'GAN 100')
plot_energy2(np.multiply(100, aux_out150), index150, os.path.join(comdir, 'energy150.pdf'), 37, 'GAN 150')
plot_energy2(np.multiply(100, aux_out200), index200, os.path.join(comdir, 'energy200.pdf'), 38, 'GAN 200')
plot_energy2(np.multiply(100, aux_out300), index300, os.path.join(comdir, 'energy300.pdf'), 39, 'GAN 300')
plot_energy2(np.multiply(100, aux_out400), index400, os.path.join(comdir, 'energy400.pdf'), 55, 'GAN 400')
plot_energy2(np.multiply(100, aux_out500), index500, os.path.join(comdir, 'energy500.pdf'), 56, 'GAN 500')

#plot_energy2(np.multiply(100, energy_sampled10), index10, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 10', 'red', '--')
#plot_energy2(np.multiply(100, energy_sampled50), index50, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 50', 'blue', '--')
#plot_energy2(np.multiply(100, energy_sampled100), index100, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 100', 'green', '--')
#plot_energy2(np.multiply(100, energy_sampled150), index150, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 150', 'yellow', '--')
#plot_energy2(np.multiply(100, energy_sampled200), index200, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 200', 'cyan', '--')
#plot_energy2(np.multiply(100, energy_sampled300), index300, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 300', 'magenta', '--')
#plot_energy2(np.multiply(100, energy_sampled400), index400, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 400', 'magenta', '--')                                                                          #plot_energy2(np.multiply(100, energy_sampled500), index500, os.path.join(comdir, 'energy_all.pdf'), 40, 'Primary 500', 'magenta', '--')                                                                           
plot_energy2(np.multiply(100, aux_out10), index10, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 10', 'red', '-')
plot_energy2(np.multiply(100, aux_out50), index50, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 50', 'blue', '-')
plot_energy2(np.multiply(100, aux_out100), index100, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 100', 'green', '-')
plot_energy2(np.multiply(100, aux_out150), index150, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 150', 'yellow', '-')
plot_energy2(np.multiply(100, aux_out200), index200, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 200', 'cyan', '-')
plot_energy2(np.multiply(100, aux_out300), index300, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 300', 'magenta', '-')
plot_energy2(np.multiply(100, aux_out400), index400, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 400', 'red', '-')
plot_energy2(np.multiply(100, aux_out500), index500, os.path.join(comdir, 'energy_all.pdf'), 40, 'GAN 500', 'blue', '-')

plot_ecal(events_gan10, num_events, os.path.join(comdir, 'ECAL_sum10.pdf'), 41, 'GAN 10')
plot_ecal(events_gan50, num_events, os.path.join(comdir, 'ECAL_sum50.pdf'), 42, 'GAN 50')
plot_ecal(events_gan100, num_events, os.path.join(comdir, 'ECAL_sum100.pdf'), 43, 'GAN 100')
plot_ecal(events_gan150, num_events, os.path.join(comdir, 'ECAL_sum150.pdf'), 44, 'GAN 150')
plot_ecal(events_gan200, num_events, os.path.join(comdir, 'ECAL_sum200.pdf'), 45, 'GAN 200')
plot_ecal(events_gan300, num_events, os.path.join(comdir, 'ECAL_sum300.pdf'), 46, 'GAN 300')
plot_ecal(events_gan400, num_events, os.path.join(comdir, 'ECAL_sum400.pdf'), 57, 'GAN 400')
plot_ecal(events_gan500, num_events, os.path.join(comdir, 'ECAL_sum500.pdf'), 58, 'GAN 500')

### Save generated image data to file
if (save):
   generated_images = (generated_images10) 
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename10,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled10)
      outfile.create_dataset('AUX',data=aux_out10)
      outfile.create_dataset('ISREAL',data=isreal10)
   print "Generated ECAL saved to ", filename10   
   generated_images = (generated_images50)
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename50,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled50)
      outfile.create_dataset('AUX',data=aux_out50)
      outfile.create_dataset('ISREAL',data=isreal50)
   print "Generated ECAL saved to ", filename50
   generated_images = (generated_images100)
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename100,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled100)
      outfile.create_dataset('AUX',data=aux_out100)
      outfile.create_dataset('ISREAL',data=isreal100)
   print "Generated ECAL saved to ", filename100
   generated_images = (generated_images150)
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename150,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled150)
      outfile.create_dataset('AUX',data=aux_out150)
      outfile.create_dataset('ISREAL',data=isreal150)
   print "Generated ECAL saved to ", filename150
   generated_images = (generated_images200)
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename200,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled200)
      outfile.create_dataset('AUX',data=aux_out200)
      outfile.create_dataset('ISREAL',data=isreal200)
   print "Generated ECAL saved to ", filename200
   generated_images = (generated_images300)
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename300,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled300)
      outfile.create_dataset('AUX',data=aux_out300)
      outfile.create_dataset('ISREAL',data=isreal300)
   print "Generated ECAL saved to ", filename300
   generated_images = (generated_images400)
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename400,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled400)
      outfile.create_dataset('AUX',data=aux_out400)
      outfile.create_dataset('ISREAL',data=isreal400)
   print "Generated ECAL saved to ", filename400
   generated_images = (generated_images500)
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename300,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=energy_sampled500)
      outfile.create_dataset('AUX',data=aux_out500)
      outfile.create_dataset('ISREAL',data=isreal500)
   print "Generated ECAL saved to ", filename500
print 'Plots are saved in', ' fixed_plots/disc_outputs, ', 'fixed_plots/Actual, ', 'fixed_plots/Generated and ', 'fixed_plots/Combined'



