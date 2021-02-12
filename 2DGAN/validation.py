from __future__ import print_function

import tensorflow as tf
print(tf.__version__)

import os
from PTF_m4_1_ReLU import *
from Functions_v1_5_8 import *

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse
import os
from six.moves import range
import sys
import h5py 
import numpy as np
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import math as math

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params')
    parser.add_argument('--weightdir', action='store', default='')
    parser.add_argument('--outpath', action='store', default='')
    return parser

data = '/eos/user/r/redacost/EleScans/'

parser = get_parser()
params = parser.parse_args()
outpath = params.outpath
weightdir = params.weightdir

lrate_g = 0.0005
lrate_d = 0.00010   #lr_d should be roughly the same as the ratio from g to d parameters
nb_epochs = 30
percent = 100                      #take just 10 percent of all data for training and testing
nb_train_files = 20           #20
ReLU_epoch = 3
save_only_best_weights = True
latent_size = 200
batch_size =  128                   #128     #batch_size must be less or equal test size, otherwise error
keras_dformat = 'channels_first'    #last for CPU, first for GPU
wtf = 6.0                           #weight true fake loss
wa = 0.2                            #weight auxiliary loss
we = 0.1                            #weight ecal loss
batch_size = 128

epoch = 30

gweights = params.weightdir
generator = generator_ReLU(keras_dformat = keras_dformat, latent_size=200)



#validation script
validation_metric = validate(generator, percent=percent, keras_dformat=keras_dformat, data_path=data)
Gromov_Wasserstein_distance = analyse(generator, read_data=False, save_data=False, gen_weights="", data_path=data, sorted_path="", optimizer = Gromov_metric)
##############################
#loss dict
# discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)   #mean disc loss for all epochs
# generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
# discriminator_test_loss = np.mean(np.array(disc_test_loss_list), axis=0)
# generator_test_loss = np.mean(np.array(gen_test_loss_list), axis=0)

# train_history['generator'].append(generator_train_loss)
# train_history['discriminator'].append(discriminator_train_loss)
# train_history['validation'].append(validation_metric)
# train_history['Gromov_Wasserstein_validation'].append(Gromov_Wasserstein_distance)
# test_history['generator'].append(generator_test_loss)
# test_history['discriminator'].append(discriminator_test_loss)
# end_d = time.time()

#calculate time for epoch
# end_batch = time.time()
# e = int(end_batch-start_epoch)
# print('Time for Epoch: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

# #save history
# pickle.dump([train_history, test_history], open(save_folder+'/Pickle/3dgan-history.pkl', 'wb'))
# pickle.dump({'train': train_history, 'test': test_history}, open(save_folder+'/Pickle/3dgan-history_dict.pkl', 'wb'))

#print loss table and plot generated image; also save them
# loss_table(train_history, test_history, save_folder, epoch, validation_metric, save=True, timeforepoch = e)
plot_gen_image_tf(latent_size, epoch)
# plot_loss(train_history, test_history, save_folder, save=True)
# plot_validation(train_history, save_folder)
# plot_gromov_w_distance(train_history, save_folder)
