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
n_jets = 100000
gen_weights='params_generator_epoch_049.hdf5'
disc_weights='params_discriminator_epoch_049.hdf5'
latent_space =200
num_events=600

# Other params
save = 0
filename = 'Gen_full_100000.h5'
## Get Full data                                                               
d=h5py.File("/afs/cern.ch/work/g/gkhattak/public/Ele_v1_1_2.h5",'r')
c=np.array(d.get('ECAL'))
e=d.get('target')
X=np.array(c[:n_jets])
y=np.array(e[:n_jets,1])
Y=np.expand_dims(y, axis=-1)
print X.shape
print Y.shape
X[X < 1e-6] = 0
# Histogram Functions

def plot_max(array, index, out_file, num_fig, energy):
   ## Plot the Histogram of Maximum energy deposition location on all axis
   bins = np.arange(0, 25, 1)
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(array[0:index-1, 0], bins=bins, histtype='step', label= str(energy), normed=1)
   plt.legend()
   plt.ylabel('Events')

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[0:index-1, 1], bins=bins, histtype='step', label=str(energy), normed=1)
   plt.legend()
   plt.xlabel('Position')
   #plt.ylabel('Events')              

   plt.subplot(223)
   #plt.title('Z-axis')                                                                             
   plt.hist(array[0:index-1, 2], bins=bins, histtype='step', label=str(energy), normed=1)
   plt.legend(loc=1)
   plt.xlabel('Position')
   plt.ylabel('Events')
   plt.savefig(out_file)

def plot_energy(array, index, out_file, num_fig, energy):
   ### Plot Histogram of energy deposition along all three axis                                        
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.hist(array[:index, 0].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()

   plt.subplot(222)
   plt.title('Y-axis')
   plt.hist(array[:index, 1].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()

   plt.subplot(223)
   plt.hist(array[:index, 2].flatten(), bins='auto', histtype='step', label=str(energy))
   plt.legend()
   plt.savefig(out_file)

def plot_energy_hist(array, index, out_file, num_fig, energy):
   ### Plot total energy deposition cell by cell along x, y, z axis                                             
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index, 0].sum(axis = 0)/index, label=str(energy))
   plt.legend()

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index, 1].sum(axis = 0)/index, label=str(energy))
   plt.legend()

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index, 2].sum(axis = 0)/index, label=str(energy))
   plt.legend()                                                          

   plt.savefig(out_file)

def plot_energy_mean(array, index, out_file, num_fig, energy):
   ### Plot total energy deposition cell by cell along x, y, z axis      
   
   plt.figure(num_fig)
   plt.subplot(221)
   plt.title('X-axis')
   plt.plot(array[0:index, 0].mean(axis = 0), label=str(energy))
   plt.legend()

   plt.subplot(222)
   plt.title('Y-axis')
   plt.plot(array[0:index, 1].mean(axis = 0), label=str(energy))
   plt.legend()

   plt.subplot(223)
   plt.title('Z-axis')
   plt.plot(array[0:index, 2].mean(axis = 0), label=str(energy))
   plt.legend()
   plt.savefig(out_file)

# Initialization of parameters                                                 
index50 = 0
index100 = 0
index150 = 0
index200 = 0
index300 = 0
index400 = 0
index500 = 0

#Initialization of arrays for actual events                                                
events_act50 = np.zeros((num_events, 25, 25, 25))
max_pos_act_50 = np.zeros((num_events, 3))
events_act100 = np.zeros((num_events, 25, 25, 25))
max_pos_act_100 = np.zeros((num_events, 3))
events_act150 = np.zeros((num_events, 25, 25, 25))
max_pos_act_150 = np.zeros((num_events, 3))
events_act200 = np.zeros((num_events, 25, 25, 25))
max_pos_act_200 = np.zeros((num_events, 3))
events_act300 = np.zeros((num_events, 25, 25, 25))
max_pos_act_300 = np.zeros((num_events, 3))
events_act400 = np.zeros((num_events, 25, 25, 25))
max_pos_act_400 = np.zeros((num_events, 3))
events_act500 = np.zeros((num_events, 25, 25, 25))
max_pos_act_500 = np.zeros((num_events, 3))
sum_act50 = np.zeros((num_events, 3, 25))
sum_act100 = np.zeros((num_events, 3, 25))
sum_act150 = np.zeros((num_events, 3, 25))
sum_act200 = np.zeros((num_events, 3, 25))
sum_act300 = np.zeros((num_events, 3, 25))
sum_act400 = np.zeros((num_events, 3, 25))
sum_act500 = np.zeros((num_events, 3, 25))
energy_sampled50 = np.zeros((num_events, 1))
energy_sampled100 = np.zeros((num_events, 1))
energy_sampled150 = np.zeros((num_events, 1))
energy_sampled200 = np.zeros((num_events, 1))
energy_sampled300 = np.zeros((num_events, 1))
energy_sampled400 = np.zeros((num_events, 1))
energy_sampled500 = np.zeros((num_events, 1))
energy_act50 = np.zeros((num_events, 1))
energy_act100 = np.zeros((num_events, 1))
energy_act150 = np.zeros((num_events, 1))
energy_act200 = np.zeros((num_events, 1))
energy_act300 = np.zeros((num_events, 1))
energy_act400 = np.zeros((num_events, 1))
energy_act500 = np.zeros((num_events, 1))

#Initialization of arrays for generated images                                             
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
sum_gan50 = np.zeros((num_events, 3, 25))
sum_gan100 = np.zeros((num_events, 3, 25))
sum_gan150 = np.zeros((num_events, 3, 25))
sum_gan200 = np.zeros((num_events, 3, 25))
sum_gan300 = np.zeros((num_events, 3, 25))
sum_gan400 = np.zeros((num_events, 3, 25))
sum_gan500 = np.zeros((num_events, 3, 25))
energy_gan50 = np.zeros((num_events, 1))
energy_gan100 = np.zeros((num_events, 1))
energy_gan150 = np.zeros((num_events, 1))
energy_gan200 = np.zeros((num_events, 1))
energy_gan300 = np.zeros((num_events, 1))
energy_gan400 = np.zeros((num_events, 1))
energy_gan500 = np.zeros((num_events, 1))

### Get Generated Data                                                                     
g = build_generator(latent_space, return_intermediate=False)
g.load_weights(gen_weights)
noise = np.random.normal(0, 1, (n_jets, latent_space))
#sampled_labels = np.ones((n_jets, 1), dtype=np.float)                                     
#sampled_labels = np.random.uniform(0, 5, (n_jets, 1))                     
sampled_labels = Y/100  #take sampled labels from actual data                              
print Y.shape
generator_in = np.multiply(sampled_labels, noise)
start = time.time()
generated_images = g.predict(generator_in, verbose=False, batch_size=100)
end = time.time()
gen_time = end - start
print generated_images.shape
d = build_discriminator()
d.load_weights(disc_weights)
start =time.time()
isreal, aux_out = np.array(d.predict(generated_images, verbose=False, batch_size=100))
end = time.time()
disc_time = end - start
generated_images = np.squeeze(generated_images)
print generated_images.shape

## Use Discriminator for actual images                                                         
d = build_discriminator()
d.load_weights(disc_weights)
image = np.expand_dims(X, axis=-1)
start =time.time()
isreal2, aux_out2 = np.array(d.predict(image, verbose=False, batch_size=100))
end = time.time()
disc2_time = end - start

## Sorting data in bins
size_data = int(X.shape[0])
for i in range(size_data):
   if Y[i] > 45 and Y[i] < 55 and index50 < num_events:
     events_act50[index50] = X[i]
     events_gan50[index50] = generated_images[i]
     energy_sampled50[index50] = Y[i]
     energy_act50[index50] = aux_out2[i]
     energy_gan50[index50] = aux_out[i]
     index50 = index50 + 1
   elif Y[i] > 95 and Y[i] < 105 and index100 < num_events:
     events_act100[index100] = X[i]
     events_gan100[index100] = generated_images[i]
     energy_sampled100[index100] = Y[i]
     energy_act100[index100] = aux_out2[i]
     energy_gan100[index100] = aux_out[i]
     index100 = index100 + 1
   elif Y[i] > 145 and Y[i] < 155 and index150 < num_events:
     events_act150[index150] = X[i]
     events_gan150[index150] = generated_images[i]
     energy_sampled150[index150] = Y[i]
     energy_act150[index150] = aux_out2[i]
     energy_gan150[index150] = aux_out[i]
     index150 = index150 + 1
   elif Y[i] > 195 and Y[i] < 205 and index200 < num_events:
     events_act200[index200] = X[i]
     events_gan200[index200] = generated_images[i]
     energy_sampled200[index200] = Y[i]
     energy_act200[index200] = aux_out2[i]
     energy_gan200[index200] = aux_out[i]
     index200 = index200 + 1
   elif Y[i] > 295 and Y[i] < 305 and index300 < num_events:
     events_act300[index300] = X[i]
     events_gan300[index300] = generated_images[i]
     energy_sampled300[index300] = Y[i]     
     energy_act300[index200] = aux_out2[i]
     energy_gan300[index300] = aux_out[i]
     index300 = index300 + 1
   elif Y[i] > 395 and Y[i] < 405 and index400 < num_events:
     events_act400[index400] = X[i]
     events_gan400[index400] = generated_images[i]
     energy_sampled400[index400] = Y[i]
     energy_act400[index400] = aux_out2[i]
     energy_gan400[index400] = aux_out[i]
     index400 = index400 + 1
   elif Y[i] > 495 and Y[i] < 505 and index500 < num_events:
     events_act500[index500] = X[i]
     events_gan500[index500] = generated_images[i]
     energy_sampled500[index500] = Y[i]
     energy_act500[index500] = aux_out2[i]
     energy_gan500[index500] = aux_out[i]
     index500 = index500 + 1

### Calculations for actual
for j in range(num_events):
   max_pos_act_50[j] = np.unravel_index(events_act50[j].argmax(), (25, 25, 25))
   max_pos_act_100[j] = np.unravel_index(events_act100[j].argmax(), (25, 25, 25))
   max_pos_act_150[j] = np.unravel_index(events_act150[j].argmax(), (25, 25, 25))
   max_pos_act_200[j] = np.unravel_index(events_act200[j].argmax(), (25, 25, 25))
   max_pos_act_300[j] = np.unravel_index(events_act300[j].argmax(), (25, 25, 25))
   max_pos_act_400[j] = np.unravel_index(events_act400[j].argmax(), (25, 25, 25))
   max_pos_act_500[j] = np.unravel_index(events_act500[j].argmax(), (25, 25, 25))
   sum_act50[j, 0] = np.sum(events_act50[j], axis=(1,2))
   sum_act50[j, 1] = np.sum(events_act50[j], axis=(0,2))
   sum_act50[j, 2] = np.sum(events_act50[j], axis=(0,1))
   sum_act100[j, 0] = np.sum(events_act100[j], axis=(1,2))
   sum_act100[j, 1] = np.sum(events_act100[j], axis=(0,2))
   sum_act100[j, 2] = np.sum(events_act100[j], axis=(0,1))
   sum_act150[j, 0] = np.sum(events_act150[j], axis=(1,2))
   sum_act150[j, 1] = np.sum(events_act150[j], axis=(0,2))
   sum_act150[j, 2] = np.sum(events_act150[j], axis=(0,1))
   sum_act200[j, 0] = np.sum(events_act200[j], axis=(1,2))
   sum_act200[j, 1] = np.sum(events_act200[j], axis=(0,2))
   sum_act200[j, 2] = np.sum(events_act200[j], axis=(0,1))
   sum_act300[j, 0] = np.sum(events_act300[j], axis=(1,2))
   sum_act300[j, 1] = np.sum(events_act300[j], axis=(0,2))
   sum_act300[j, 2] = np.sum(events_act300[j], axis=(0,1))
   sum_act400[j, 0] = np.sum(events_act400[j], axis=(1,2))
   sum_act400[j, 1] = np.sum(events_act400[j], axis=(0,2))
   sum_act400[j, 2] = np.sum(events_act400[j], axis=(0,1))
   sum_act500[j, 0] = np.sum(events_act500[j], axis=(1,2))
   sum_act500[j, 1] = np.sum(events_act500[j], axis=(0,2))
   sum_act500[j, 2] = np.sum(events_act500[j], axis=(0,1))

print max_pos_act_50[0:10]
### Calculations for generated                                                             
for j in range(num_events):
   max_pos_gan_50[j] = np.unravel_index(events_gan50[j].argmax(), (25, 25, 25))
   max_pos_gan_100[j] = np.unravel_index(events_gan100[j].argmax(), (25, 25, 25))
   max_pos_gan_150[j] = np.unravel_index(events_gan150[j].argmax(), (25, 25, 25))
   max_pos_gan_200[j] = np.unravel_index(events_gan200[j].argmax(), (25, 25, 25))
   max_pos_gan_300[j] = np.unravel_index(events_gan300[j].argmax(), (25, 25, 25))
   max_pos_gan_400[j] = np.unravel_index(events_gan400[j].argmax(), (25, 25, 25))
   max_pos_gan_500[j] = np.unravel_index(events_gan500[j].argmax(), (25, 25, 25))
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

#### Generate Data table to screen
print "Actual Data"                                                 
print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t"
print "50 \t\t%d \t\t%f \t\t%s \t\t%f \t\t%f" %(index50, np.amax(events_act50), str(np.unravel_index(events_act50.argmax(), (index50, 25, 25, 25))), np.mean(events_act50), np.amin(events_act50))
print "100 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index100, np.amax(events_act100), str(np.unravel_index(events_act100.argmax(), (index100, 25, 25, 25))), np.mean(events_act100), np.amin(events_act100))
print "150 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index150, np.amax(events_act150), str(np.unravel_index(events_act150.argmax(), (index150, 25, 25, 25))), np.mean(events_act150), np.amin(events_act150))
print "200 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index200, np.amax(events_act200), str(np.unravel_index(events_act200.argmax(), (index200, 25, 25, 25))), np.mean(events_act200), np.amin(events_act200))
print "300 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index300, np.amax(events_act300), str(np.unravel_index(events_act300.argmax(), (index300, 25, 25, 25))), np.mean(events_act300), np.amin(events_act300))
print "400 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index400, np.amax(events_act400), str(np.unravel_index(events_act400.argmax(), (index400, 25, 25, 25))), np.mean(events_act400), np.amin(events_act400))
print "500 \t\t%d \t\t%f \t\t%s  \t\t%f \t\t%f" %(index500, np.amax(events_act500), str(np.unravel_index(events_act500.argmax(), (index500, 25, 25, 25))), np.mean(events_act500), np.amin(events_act500))


#### Generate GAN table to screen                                                                                
print "Generated Data"
print "Energy\t\t Events\t\tMaximum Value\t\t Maximum loc\t\t\t Mean\t\t\t Minimum\t\t"
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
discdir = 'analysis_plots/disc_outputs'
safe_mkdir(discdir)
actdir = 'analysis_plots/Actual'
safe_mkdir(actdir)
gendir = 'analysis_plots/Generated'
safe_mkdir(gendir)
comdir = 'analysis_plots/Combined'
safe_mkdir(comdir)

## Make plots for generated data  
plt.figure(1)
bins = np.linspace(0, 1, 30)
plt.hist(isreal, bins='auto', histtype='step', label='GAN', color='green')
plt.legend(ncol=2, mode='expand')
plt.xlabel('real/fake')
plt.ylabel('Number of events')
#plt.ylim(ymax= n_jets)
plt.tight_layout()
plt.savefig(os.path.join(discdir, 'real.pdf'))

plt.figure(2)
plt.hist(sampled_labels-aux_out, bins='auto', histtype='step', label='GAN', color='green')
plt.legend(ncol=2, mode='expand')
plt.xlabel('error')
plt.ylabel('Number of events')
plt.tight_layout()
plt.savefig(os.path.join(discdir, 'error.pdf'))
                                                          
## Make plots for real data                                                
plt.figure(1)
bins = np.linspace(0, 1, 30)
plt.hist(isreal2, bins='auto', histtype='step', label='Actual', color='blue')
plt.legend(ncol=2, mode='expand')
#plt.ylim(ymax= n_jets)
plt.tight_layout()
plt.savefig(os.path.join(discdir, 'real.pdf'))

plt.figure(2)
plt.hist(sampled_labels-aux_out2, bins='auto', histtype='step', label='Actual', color='blue')
plt.legend(ncol=2, mode='expand')
plt.tight_layout()
plt.savefig(os.path.join(discdir, 'error.pdf'))

print max_pos_act_50.shape
print max_pos_act_50

plot_max(max_pos_act_50, index50, os.path.join(comdir, 'Position_of_max_50.pdf'), 3, 'Data 50')
plot_max(max_pos_act_100, index100, os.path.join(comdir, 'Position_of_max_100.pdf'), 4, 'Data 100')
plot_max(max_pos_act_150, index150, os.path.join(comdir, 'Position_of_max_150.pdf'), 5, 'Data 150')
plot_max(max_pos_act_200, index200, os.path.join(comdir, 'Position_of_max_200.pdf'), 6, 'Data 200')
plot_max(max_pos_act_300, index300, os.path.join(comdir, 'Position_of_max_300.pdf'), 7, 'Data 300')
plot_max(max_pos_act_400, index400, os.path.join(comdir, 'Position_of_max_400.pdf'), 8, 'Data 400')
#plot_max(max_pos_act_500, index500, os.path.join(actdir, 'Position_of_max.pdf'), 3, 500)

plot_max(max_pos_gan_50, index50, os.path.join(comdir, 'Position_of_max_50.pdf'), 3, 'GAN 50')
plot_max(max_pos_gan_100, index100, os.path.join(comdir, 'Position_of_max_100.pdf'), 4, 'GAN 100')
plot_max(max_pos_gan_150, index150, os.path.join(comdir, 'Position_of_max_150.pdf'), 5, 'GAN 150')
plot_max(max_pos_gan_200, index200, os.path.join(comdir, 'Position_of_max_200.pdf'), 6, 'GAN 200')
plot_max(max_pos_gan_300, index300, os.path.join(comdir, 'Position_of_max_300.pdf'), 7, 'GAN 300')
plot_max(max_pos_gan_400, index400, os.path.join(comdir, 'Position_of_max_400.pdf'), 8, 'GAN 400')
#plot_max(max_pos_act_500, index500, os.path.join(actdir, 'Position_of_max.pdf'), 3, 500)     

plot_energy_hist(sum_act50, index50, os.path.join(comdir, 'hist_50.pdf'), 9, 'Data 50')
plot_energy_hist(sum_act100, index100, os.path.join(comdir, 'hist_100.pdf'), 10, 'Data 100')
plot_energy_hist(sum_act150, index150, os.path.join(comdir, 'hist_150.pdf'), 11, 'Data 150')
plot_energy_hist(sum_act200, index200, os.path.join(comdir, 'hist_200.pdf'), 12, 'Data 200')
plot_energy_hist(sum_act300, index300, os.path.join(comdir, 'hist_300.pdf'), 13, 'Data 300')
plot_energy_hist(sum_act400, index400, os.path.join(comdir, 'hist_400.pdf'), 14, 'Data 400')
#plot_energy_hist(sum_act500, index500, os.path.join(actdir, 'hist_500.pdf'), 17, 500)        

plot_energy_hist(sum_gan50, index50, os.path.join(comdir, 'hist_50.pdf'), 9, 'GAN 50')
plot_energy_hist(sum_gan100, index100, os.path.join(comdir, 'hist_100.pdf'), 10, 'GAN 100')
plot_energy_hist(sum_gan150, index150, os.path.join(comdir, 'hist_150.pdf'), 11, 'GAN 150')
plot_energy_hist(sum_gan200, index200, os.path.join(comdir, 'hist_200.pdf'), 12, 'GAN 200')
plot_energy_hist(sum_gan300, index300, os.path.join(comdir, 'hist_300.pdf'), 13, 'GAN 300')
plot_energy_hist(sum_gan400, index400, os.path.join(comdir, 'hist_400.pdf'), 14, 'GAN 400')
#plot_energy_hist(sum_act500, index500, os.path.join(actdir, 'hist_500.pdf'), 17, 500)          

plot_energy(sum_act50, index50, os.path.join(actdir, 'Flat_energy.pdf'), 15, 50)
plot_energy(sum_act100, index100, os.path.join(actdir, 'Flat_energy.pdf'),15, 100)
plot_energy(sum_act150, index150, os.path.join(actdir, 'Flat_energy.pdf'),15, 150)
plot_energy(sum_act200, index200, os.path.join(actdir, 'Flat_energy.pdf'),15, 200)
plot_energy(sum_act300, index300, os.path.join(actdir, 'Flat_energy.pdf'),15, 300)
plot_energy(sum_act400, index400, os.path.join(actdir, 'Flat_energy.pdf'),15, 400)
plot_energy(sum_act500, index500,  os.path.join(actdir, 'Flat_energy.pdf'),15, 500)

plot_energy_hist(sum_act50, index50, os.path.join(actdir, 'hist_all.pdf'), 16, 'Data 50')
plot_energy_hist(sum_act100, index100, os.path.join(actdir, 'hist_all.pdf'), 16, 'Data 100')
plot_energy_hist(sum_act150, index150, os.path.join(actdir, 'hist_all.pdf'), 16, 'Data 150')
plot_energy_hist(sum_act200, index200, os.path.join(actdir, 'hist_all.pdf'), 16, 'Data 200')
plot_energy_hist(sum_act300, index300, os.path.join(actdir, 'hist_all.pdf'), 16, 'Data 300')
plot_energy_hist(sum_act400, index400, os.path.join(actdir, 'hist_all.pdf'), 16, 'Data 400')
#plot_energy_hist(sum_act500, index500, os.path.join(actdir, 'hist_all.pdf'), 18, 500)

plot_energy_mean(sum_act50, index50, os.path.join(actdir, 'hist_mean_all.pdf'), 17, 'Data 50')
plot_energy_mean(sum_act100, index100, os.path.join(actdir, 'hist_mean_all.pdf'), 17, 'Data 100')
plot_energy_mean(sum_act150, index150, os.path.join(actdir, 'hist_mean_all.pdf'), 17, 'Data 150')
plot_energy_mean(sum_act200, index200, os.path.join(actdir, 'hist_mean_all.pdf'), 17, 'Data 200')
plot_energy_mean(sum_act300, index300, os.path.join(actdir, 'hist_mean_all.pdf'), 17, 'Data 300')
plot_energy_mean(sum_act400, index400, os.path.join(actdir, 'hist_mean_all.pdf'), 17, 'Data 400')
#plot_energy_mean(sum_act500, index500, os.path.join(actdir, 'hist_mean_all.pdf'), 19, 500)

plt.figure(20)
plt.title('Energy')
ebins=np.arange(0, 500, 10)
plt.hist(Y, bins=ebins, histtype='step', label='Primary Energy')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy.pdf'))

plt.figure(21)
plt.title('ECAL SUM')
plt.hist(np.sum(X, axis=(1, 2, 3)), bins='auto', histtype='step', label='Data')
plt.legend()
plt.savefig(os.path.join(comdir, 'ECAL_sum.pdf'))

plt.figure(21)
plt.hist(np.sum(generated_images, axis=(1, 2, 3)), bins='auto', histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'ECAL_sum.pdf'))

print Y.shape
print np.sum(X, axis=(1, 2, 3)).shape
print aux_out.shape
print np.sum(generated_images, axis=(1, 2, 3)).shape

#plt.figure(22)
#plt.title('Ratio of Primary and ECAL')
#ratio_act=np.divide(Y, np.sum(X, axis=(1, 2, 3)))
#plt.hist(ratio_act, bins='auto', histtype='step', label='Data')
#plt.legend(loc=8)
#plt.savefig(os.path.join(comdir, 'ECAL_ratio.pdf'))

#ratio_gan=np.divide(Y, np.sum(generated_images, axis=(1, 2, 3)))
#plt.figure(22)
#plt.hist(ratio_gan, bins='auto', histtype='step', label='GAN')
#plt.legend(loc=8)
#plt.savefig(os.path.join(comdir, 'ECAL_ratio.pdf'))

## Plots for Generated

plot_max(max_pos_gan_50, index50, os.path.join(gendir, 'Position_of_max.pdf'), 18, 'GAN 50')
plot_max(max_pos_gan_100, index100, os.path.join(gendir, 'Position_of_max.pdf'), 18, 'GAN 100')
plot_max(max_pos_gan_150, index150, os.path.join(gendir, 'Position_of_max.pdf'), 18, 'GAN 150')
plot_max(max_pos_gan_200, index200, os.path.join(gendir, 'Position_of_max.pdf'), 18, 'GAN 200')
plot_max(max_pos_gan_300, index300, os.path.join(gendir, 'Position_of_max.pdf'), 18, 'GAN 300')
plot_max(max_pos_gan_400, index400, os.path.join(gendir, 'Position_of_max.pdf'), 18, 'GAN 400')
#plot_max(max_pos_gan_500, index500, os.path.join(actdir, 'Position_of_max.pdf'), 23, 500)

plot_energy(sum_gan50, index50, os.path.join(gendir, 'Flat_energy.pdf'), 19, 'GAN 50')
plot_energy(sum_gan100, index100, os.path.join(gendir, 'Flat_energy.pdf'), 19, 'GAN 100')
plot_energy(sum_gan150, index150, os.path.join(gendir, 'Flat_energy.pdf'), 19, 'GAN 150')
plot_energy(sum_gan200, index200, os.path.join(gendir, 'Flat_energy.pdf'), 19, 'GAN 200')
plot_energy(sum_gan300, index300, os.path.join(gendir, 'Flat_energy_300.pdf'), 19, 'GAN 300')
plot_energy(sum_gan400, index400, os.path.join(gendir, 'Flat_energy_400.pdf'), 19, 'GAN 400')
#plot_energy(sum_gan500, os.path.join(gendir, 'Flat_energy_500.pdf'), 30, 500)

plot_energy_hist(sum_gan50, index50, os.path.join(gendir, 'hist_all.pdf'), 23, 'GAN 50')
plot_energy_hist(sum_gan100, index100, os.path.join(gendir, 'hist_all.pdf'), 23, 'GAN 100')
plot_energy_hist(sum_gan150, index150, os.path.join(gendir, 'hist_all.pdf'), 23, 'GAN 150')
plot_energy_hist(sum_gan200, index200, os.path.join(gendir, 'hist_all.pdf'), 23, 'GAN 200')
plot_energy_hist(sum_gan300, index300, os.path.join(gendir, 'hist_all.pdf'), 23, 'GAN 300')
plot_energy_hist(sum_gan400, index400, os.path.join(gendir, 'hist_all.pdf'), 23, 'GAN 400')
#plot_energy_hist(sum_gan500, index500, os.path.join(actdir, 'hist_500.pdf'), 37, 500)

plot_energy_mean(sum_gan50, index50, os.path.join(gendir, 'hist_mean_all.pdf'), 24, 50)
plot_energy_mean(sum_gan100, index100, os.path.join(gendir, 'hist_mean_all.pdf'), 24, 100)
plot_energy_mean(sum_gan150, index150, os.path.join(gendir, 'hist_mean_all.pdf'), 24, 150)
plot_energy_mean(sum_gan200, index200, os.path.join(gendir, 'hist_mean_all.pdf'), 24, 200)
plot_energy_mean(sum_gan300, index300, os.path.join(gendir, 'hist_mean_all.pdf'), 24, 300)
plot_energy_mean(sum_gan400, index400, os.path.join(gendir, 'hist_mean_all.pdf'), 24, 400)
plot_energy_mean(sum_gan500, index500, os.path.join(gendir, 'hist_mean_all.pdf'), 24, 500)

plt.figure(20)
ebins=np.arange(0, 500, 10)
plt.hist(np.multiply(100, aux_out), bins=ebins, histtype='step', label='GAN')
plt.hist(np.multiply(100, aux_out2), bins=ebins, histtype='step', label='Data')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy.pdf'))

plt.figure(25)                                                   
plt.title('GAN versus Primary for 50 GeV')
plt.hist(energy_sampled50, bins=ebins, histtype='step', label='Primary')
plt.hist(np.multiply(100, energy_act50), bins=ebins, histtype='step', label='Actual')
plt.hist(np.multiply(100, energy_gan50), bins=ebins, histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy50.pdf'))

plt.figure(26)
plt.title('GAN versus Primary for 100 GeV')
plt.hist(energy_sampled100, bins=ebins, histtype='step', label='Primary')
plt.hist(np.multiply(100, energy_act100), bins=ebins, histtype='step', label='Actual')
plt.hist(np.multiply(100, energy_gan100), bins=ebins, histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy100.pdf'))

plt.figure(27)
plt.title('GAN versus Primary for 150 GeV')
plt.hist(energy_sampled150, bins=ebins, histtype='step', label='Primary')
plt.hist(np.multiply(100, energy_act150), bins=ebins, histtype='step', label='Actual')
plt.hist(np.multiply(100, energy_gan150), bins=ebins, histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy150.pdf'))

plt.figure(28)
plt.title('GAN versus Primary for 200 GeV')
plt.hist(energy_sampled200, bins=ebins, histtype='step', label='Primary')
plt.hist(np.multiply(100, energy_act200), bins=ebins, histtype='step', label='Actual')
plt.hist(np.multiply(100, energy_gan200), bins=ebins, histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy200.pdf'))

plt.figure(29)
plt.title('GAN versus Primary for 300 GeV')
plt.hist(energy_sampled300, bins=ebins, histtype='step', label='Primary')
plt.hist(np.multiply(100, energy_act300), bins=ebins, histtype='step', label='Actual')
plt.hist(np.multiply(100, energy_gan300), bins=ebins, histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy300.pdf'))

plt.figure(30)
plt.title('GAN versus Primary for 400 GeV')
plt.hist(energy_sampled400, bins=ebins, histtype='step', label='Primary')
plt.hist(np.multiply(100, energy_act400), bins=ebins, histtype='step', label='Actual')
plt.hist(np.multiply(100, energy_gan400), bins=ebins, histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy400.pdf'))

plt.figure(31)
plt.title('GAN versus Primary for 500 GeV')
plt.hist(energy_sampled500, bins=ebins, histtype='step', label='Primary')
plt.hist(np.multiply(100, energy_act500), bins=ebins, histtype='step', label='Actual')
plt.hist(np.multiply(100, energy_gan500), bins=ebins, histtype='step', label='GAN')
plt.legend()
plt.savefig(os.path.join(comdir, 'energy500.pdf'))

### Save generated image data to file
if (save):
   generated_images = (generated_images) 
   generated_images = np.squeeze(generated_images)
   with h5py.File(filename,'w') as outfile:
      outfile.create_dataset('ECAL',data=generated_images)
      outfile.create_dataset('LABELS',data=sampled_labels)
      outfile.create_dataset('AUX',data=aux_out)
      outfile.create_dataset('ISREAL',data=isreal)
   print "Generated ECAL saved to ", filename   
print gen_time
print disc_time
print disc2_time




