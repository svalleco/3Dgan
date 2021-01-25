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



#set parameters
save_folder = "TF_K_1"         #Name of the Folder in which the results schould be safed

train_file_by_file = True              #True if you want to import the trainingsdata file by file
train_on_folder = False                #True if you want to train on all data files within this folder, False if you want to train just with one file
#train_data_path = "/eos/home-f/frehm/TF2/Data/EleEscan_1_1/EleEscan_1_1.h5" #file
#train_folder_path = "/eos/home-f/frehm/TF2/Data//"     #there has to be a folder inside
#train_file_by_file_path = "/eos/home-f/frehm/TF2/File_by_file/"         #data on eos
#train_file_by_file_path = "/data/frehm/File_by_file/"                  #data on GPU

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


#model summary
generator_ReLU(keras_dformat=keras_dformat).summary()
discriminator(keras_dformat=keras_dformat).summary()

#create output folders for saving files and weights
create_folder(save_folder+'/')
create_folder(save_folder+'/Weights/')
create_folder(save_folder+'/Weights/disc')
create_folder(save_folder+'/Weights/gen')
create_folder(save_folder+'/Pickle/')
create_folder(save_folder+'/Save_Images/')
print('*************************************************************************************')

#import trainingsdata
if train_file_by_file == False:
    X,y = import_data(train_on_folder, train_data_path, train_folder_path)
    print('*************************************************************************************') 

    #do data preperation    
    X_train, X_test, y_train, y_test, ecal_train, ecal_test, nb_train, nb_test = data_preperation(X, y, keras_dformat, batch_size, percent)
    y_train = np.expand_dims(y_train, axis=-1)  #macht jeden Eintrag in der Liste zu einer Unterliste [1,2,3]->[[1],[2],[3]]
    y_test  = np.expand_dims(y_test,  axis=-1)
    X_train = tf.constant(X_train)
    X_test  = tf.constant(X_test)
    y_train = np.expand_dims(y_train, axis=-1)  #macht jeden Eintrag in der Liste zu einer Unterliste [1,2,3]->[[1],[2],[3]]
    y_test  = np.expand_dims(y_test,  axis=-1)
    y_train = tf.constant(y_train)
    y_test  = tf.constant(y_test)
    ecal_train = tf.constant(ecal_train)
    ecal_test  = tf.constant(ecal_test)
    print("X_train_shape: ", X_train.shape)
    print("X_test_shape: ", X_test.shape)
    print("ecal_train_shape: ", ecal_train.shape)
    print("ecal_test_shape: ", ecal_test.shape)
    print('*************************************************************************************')
else:
    #files = create_files_list(train_file_by_file_path)
    X_test = []
    y_test = []
    ecal_test = []

Trainfiles = ['gs://renato_bucket/EleEscanstf/EleEscan_1_1.tfrecords',\
            'gs://renato_bucket/EleEscanstf/EleEscan_1_2.tfrecords']
Testfiles = ['gs://renato_bucket/EleEscanstf/EleEscan_1_3.tfrecords']

def epoch_cycle(batch_size=128):
    from PTF_m4_1_ReLU import discriminator, generator_LeakyReLU
    #from Functions_v1_5_8 import *
    import tfrecordconverter as tfconvert


    #-------------------------------------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------------------------------------

    #create models
    #discriminator.build()

    tpu_address = os.environ["TPU_NAME"]
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

    #setting up parallel strategy
    #strategy = tf.distribute.MirroredStrategy() #initialize parallel strategy
    strategy = tf.distribute.TPUStrategy(cluster_resolver)

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # global_batch_size = batch_size * strategy.num_replicas_in_sync

    BATCH_SIZE_PER_REPLICA = batch_size
    batch_size = batch_size * strategy.num_replicas_in_sync

    with strategy.scope():
        discriminator = discriminator(keras_dformat = keras_dformat)
        generator = generator_LeakyReLU(keras_dformat = keras_dformat, latent_size = 200)

    #generator = tf.saved_model.load(save_folder + "/Generator")
    #print(discriminator.trainable_variables)
    #print(generator.trainable_variables)



    #define plots during training after each epoch

    #%matplotlib notebook            #this is for 3D projection
    epoch=0

    #function for plotting a training images
    if train_file_by_file == False:
        plot_images(X, epoch, save_folder, save=False, number=1)

    def plot_gen_image_tf(latent_size, epoch=0):   #generate a image 
        tf.random.set_seed(1)     
        noise, gen_aux, generator_input, gen_ecal = func_for_gen(1,latent_size=latent_size)
        tf.random.set_seed(None)
        generated_images = generator(generator_input)
        try:
            generated_images = np.squeeze(generated_images, axis=(1,))
        except:
            generated_images = np.squeeze(generated_images, axis=(-1,))
        Xgen = generated_images
        plot_images(Xgen, epoch, save_folder, save=True, number=1)
        return

    #plot_gen_image_tf(latent_size)

    #learningrate decay with epochs
    def l_dec(initial_lrate, epoch, start_decay=80, decay_rate=0.012):
        epoch = epoch - 1 #because training starts at epoch 1
        if epoch < start_decay:
            k = 0.0
        else:
            k = decay_rate #0.07
            epoch = epoch - start_decay
        lrate = initial_lrate * math.exp(-k*epoch)
        return lrate

    ep_list=[]
    lr_list=[]
    for i in range(1,nb_epochs*20):
        ep_list.append(i)
        lr_list.append(l_dec(lrate_g, i))
    ep_list = [i / 20 for i in ep_list]

    plt.plot(ep_list,lr_list)
    plt.title("Learningrate Decay")
    plt.xlabel("Epochs")
    plt.ylabel("Learnrate")
    plt.show()
    print("lr last value: ", round(lr_list[-1],5))
    #print(lr_list)

    #-------------------------------------------------------------------------------------------------------
    # training
    #-------------------------------------------------------------------------------------------------------

    verbose = 'true'

    train_history = defaultdict(list)   #create a dict with an empty list 
    test_history = defaultdict(list)





    print ('Loading Data from .....')
    
    time_start_file = time.time()
    # Get the dataset from the trainfile
    dataset = tfconvert.RetrieveTFRecordpreprocessing(Trainfiles, batch_size)
    datasettest = tfconvert.RetrieveTFRecordpreprocessing(Testfiles, batch_size)

    time_elapsed = time.time() - time_start_file
    print("Get Dataset: " + str(time_elapsed))
    time_start_file = time.time()
    
    #distribute the dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    dist_dataset_test = strategy.experimental_distribute_dataset(datasettest)


    time_elapsed = time.time() - time_start_file
    print("Distribute dataset: " + str(time_elapsed))
    time_start_file = time.time()

    dist_dataset_iter = iter(dist_dataset)

    dist_dataset_iter_test = iter(dist_dataset_test)

    steps_per_epoch = int( 180000 // (batch_size))
    steps_per_epoch_test =int( 20000 // (batch_size))

    ########################################################################################
    # Training Functions
    ########################################################################################

    def training_step(dataset):
        #create batches
        image_batch  = dataset.get('X')
        energy_batch = dataset.get('Y')
        ecal_batch   = dataset.get('ecal')

        b_size = energy_batch.get_shape().as_list()[0]

        #discriminator true training
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(discriminator.trainable_variables)
            d_loss_true = disc_loss(generator, discriminator, image_batch, energy_batch, ecal_batch, batch_size, BATCH_SIZE_PER_REPLICA, label ="ones", wtf=wtf, wa=wa, we=we, epoch=epoch)
            d_grads = tape.gradient( d_loss_true[0] , discriminator.trainable_variables )
        optimizer_d.apply_gradients( zip( d_grads , discriminator.trainable_variables) )

        #discriminator fake training
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(discriminator.trainable_variables)
            noise, gen_aux, generator_input, gen_ecal = func_for_gen(nb_test=batch_size, epoch=epoch) 
            generated_images = generator(generator_input)
            d_loss_fake = disc_loss(generator, discriminator, generated_images, gen_aux, gen_ecal, batch_size, BATCH_SIZE_PER_REPLICA, label = "zeros", wtf=wtf, wa=wa, we=we, epoch=epoch)
            #d_loss_fake[0] = tf.nn.compute_average_loss(binary_example_loss, global_batch_size=global_batch_size)
            d_grads = tape.gradient( d_loss_fake[0] , discriminator.trainable_variables )
        optimizer_d.apply_gradients( zip( d_grads , discriminator.trainable_variables) )

        gen_losses = []
        for i in range(2):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(generator.trainable_variables)
                g_loss = gen_loss(generator, discriminator, batch_size=batch_size, batch_size_per_replica=BATCH_SIZE_PER_REPLICA, epoch=epoch, wtf=wtf, wa=wa, we=we)
                g_grads = tape.gradient( g_loss[0] , generator.trainable_variables )
            optimizer_g.apply_gradients( zip( g_grads , generator.trainable_variables ) )
            gen_losses.append([g_loss[0], g_loss[1], g_loss[2], g_loss[3]])

        #total_loss, loss_true_fake, loss_aux, loss_ecal
        return d_loss_true[0], d_loss_true[1], d_loss_true[2], d_loss_true[3], \
                d_loss_fake[0], d_loss_fake[1], d_loss_fake[2], d_loss_fake[3], \
                gen_losses[0], gen_losses[1], gen_losses[2], gen_losses[3], \
                gen_losses[4], gen_losses[5], gen_losses[6], gen_losses[7]

    @tf.function
    def distributed_training_step(dataset):
        #X_train, epoch_disc_loss, epoch_gen_loss = 

        #print('check100')
        gen_losses = []

        #fake, energy, ang, ecal
        d_loss_true_1, d_loss_true_2, d_loss_true_3, d_loss_true_4, \
        d_loss_fake_1, d_loss_fake_2, d_loss_fake_3, d_loss_fake_4, \
        g_loss_1, g_loss_2, g_loss_3, g_loss_4, \
        g_loss_5, g_loss_6, g_loss_7, g_loss_8  = strategy.run(training_step, args=(next(dataset),))
        
        d_loss_true_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_true_1, axis=None)
        d_loss_true_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_true_2, axis=None)
        d_loss_true_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_true_3, axis=None)
        d_loss_true_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_true_4, axis=None)
        d_loss_true = [d_loss_true_1, d_loss_true_2, d_loss_true_3, d_loss_true_4]

        d_loss_fake_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_fake_1, axis=None)
        d_loss_fake_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_fake_2, axis=None)
        d_loss_fake_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_fake_3, axis=None)
        d_loss_fake_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss_fake_4, axis=None)
        d_loss_fake = [d_loss_fake_1, d_loss_fake_2, d_loss_fake_3, d_loss_fake_4]

    
        g_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_1, axis=None)
        g_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_2, axis=None)
        g_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_3, axis=None)
        g_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_4, axis=None)
        g_loss = [g_loss_1, g_loss_2, g_loss_3, g_loss_4]

        gen_losses.append(g_loss)

        g_loss_5 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_5, axis=None)
        g_loss_6 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_6, axis=None)
        g_loss_7 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_7, axis=None)
        g_loss_8 = strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss_8, axis=None)
        g_loss = [g_loss_5, g_loss_6, g_loss_7, g_loss_8]

        gen_losses.append(g_loss)

        return d_loss_true, d_loss_fake, gen_losses

    ########################################################################################
    # Testing Functions
    ########################################################################################

    def testing_step(dataset):
        image_batch  = dataset.get('X')
        energy_batch = dataset.get('Y')
        ecal_batch   = dataset.get('ecal')

        d_test_loss_true = disc_loss(generator, discriminator, image_batch, energy_batch, ecal_batch, batch_size, label ="ones", wtf=wtf, wa=wa, we=we)
        noise, gen_aux, generator_input, gen_ecal = func_for_gen(nb_test=batch_size, epoch=epoch) 
        generated_images = generator(generator_input)
        d_test_loss_fake = disc_loss(generator, discriminator, generated_images, gen_aux, gen_ecal, batch_size, label = "zeros", wtf=wtf, wa=wa, we=we)

        gen_test_loss = gen_loss(generator, discriminator, batch_size)

        return d_test_loss_true[0], d_test_loss_true[1], d_test_loss_true[2], d_test_loss_true[3], \
                d_test_loss_fake[0], d_test_loss_fake[1], d_test_loss_fake[2], d_test_loss_fake[3], \
                gen_test_loss[0], gen_test_loss[1], gen_test_loss[2], gen_test_loss[3]

    @tf.function
    def distributed_testing_step(dataset):
        d_test_loss_true_1, d_test_loss_true_2, d_test_loss_true_3, d_test_loss_true_4, \
        d_test_loss_fake_1, d_test_loss_fake_2, d_test_loss_fake_3, d_test_loss_fake_4, \
        gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4 = strategy.run(testing_step, args=(next(dataset),))

        d_test_loss_true_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_true_1, axis=None)
        d_test_loss_true_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_true_2, axis=None)
        d_test_loss_true_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_true_3, axis=None)
        d_test_loss_true_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_true_4, axis=None)
        d_test_loss_true = [d_test_loss_true_1, d_test_loss_true_2, d_test_loss_true_3, d_test_loss_true_4]

        d_test_loss_fake_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_fake_1, axis=None)
        d_test_loss_fake_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_fake_2, axis=None)
        d_test_loss_fake_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_fake_3, axis=None)
        d_test_loss_fake_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, d_test_loss_fake_4, axis=None)
        d_test_loss_fake = [d_test_loss_fake_1, d_test_loss_fake_2, d_test_loss_fake_3, d_test_loss_fake_4]

        gen_test_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_1, axis=None)
        gen_test_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_2, axis=None)
        gen_test_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_3, axis=None)
        gen_test_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_4, axis=None)
        gen_test_loss = [gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4]

        
        return d_test_loss_true, d_test_loss_fake, gen_test_loss

    ########################################################################################
    # Main Cycle
    ########################################################################################

    start_epoch = 1             #i want to start traing at epoch 1
    for epoch in range(start_epoch, nb_epochs + 1):
        print('Epoch {} of {}'.format(epoch, nb_epochs))
        start_epoch = time.time()
        #lr_d = l_dec(lrate_d, epoch)
        #lr_g = l_dec(lrate_g, epoch)

        with strategy.scope():
            optimizer_d = tf.optimizers.Adam(lrate_d)
            optimizer_g = tf.optimizers.Adam(lrate_g)

        #nb_batches = int(X_train.shape[0] / batch_size)
        epoch_gen_loss = []
        epoch_disc_loss = []
        
        
        if epoch == ReLU_epoch:

            with strategy.scope():
                generator = generator_ReLU(keras_dformat = keras_dformat, latent_size = 200)
                load_epoch = str(ReLU_epoch-1)
                gweights = save_folder + "/Weights/gen/params_generator_epoch_" + load_epoch + ".hdf5"
                print(gweights)
                generator.load_weights(gweights)

        #training

        for epochstep in range(steps_per_epoch):
            if train_file_by_file == True:
                lr_d = l_dec(lrate_d, epoch*20+epochstep-20)
                lr_g = l_dec(lrate_g, epoch*20+epochstep-20)
                with strategy.scope():
                    optimizer_d = tf.optimizers.Adam(lr_d)
                    optimizer_g = tf.optimizers.Adam(lr_g)

                print("Learnrate Generator:     ", lr_g)
                print("Learnrate Discriminator: ", lr_d)

            d_loss_true, d_loss_fake, gen_losses = distributed_training_step(dist_dataset_iter)

            d_loss_true = [el.numpy() for el in d_loss_true]
            d_loss_fake = [el.numpy() for el in d_loss_fake]
            gen_losses = [el.numpy() for el in gen_losses]

            d_loss = []
            d_loss.append( (d_loss_true[0] + d_loss_fake[0])/2)
            d_loss.append( (d_loss_true[1] + d_loss_fake[1])/2)
            d_loss.append( (d_loss_true[2] + d_loss_fake[2])/2)
            d_loss.append( (d_loss_true[3] + d_loss_fake[3])/2)
            epoch_disc_loss.append([d_loss[0].numpy(), d_loss[1].numpy(), d_loss[2].numpy(), d_loss[3].numpy()])  

            epoch_gen_loss.append(np.mean(gen_losses, axis = 0))

        
        #testing
        disc_test_loss_list = []
        gen_test_loss_list = []

        for epochstep in range(steps_per_epoch_test):
            #create batches

            d_test_loss_true, d_test_loss_fake, gen_test_loss = distributed_testing_step(dist_dataset_iter_test)

            d_test_loss_true = [el.numpy() for el in d_test_loss_true]
            d_test_loss_fake = [el.numpy() for el in d_test_loss_fake]
            gen_test_loss = [el.numpy() for el in gen_test_loss]

            d_test_loss = []
            d_test_loss.append( (d_test_loss_true[0] + d_test_loss_fake[0])/2)
            d_test_loss.append( (d_test_loss_true[1] + d_test_loss_fake[1])/2)
            d_test_loss.append( (d_test_loss_true[2] + d_test_loss_fake[2])/2)
            d_test_loss.append( (d_test_loss_true[3] + d_test_loss_fake[3])/2)
            disc_test_loss_list.append([d_test_loss[0].numpy(), d_test_loss[1].numpy(), d_test_loss[2].numpy(), d_test_loss[3].numpy()])

            gen_test_loss_list.append([gen_test_loss[0].numpy(), gen_test_loss[1].numpy(), gen_test_loss[2].numpy(), gen_test_loss[3].numpy()])


        ########################################################################################
        # Validation
        ########################################################################################

        ###############################
        #validation script
        validation_metric = validate(generator, percent=percent, keras_dformat=keras_dformat, data_path=train_file_by_file_path)
        Gromov_Wasserstein_distance = analyse(generator, read_data=False, save_data=False, gen_weights="", data_path=train_file_by_file_path,
        sorted_path="", optimizer = Gromov_metric)
        ##############################
        #loss dict
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)   #mean disc loss for all epochs
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        discriminator_test_loss = np.mean(np.array(disc_test_loss_list), axis=0)
        generator_test_loss = np.mean(np.array(gen_test_loss_list), axis=0)

        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        train_history['validation'].append(validation_metric)
        train_history['Gromov_Wasserstein_validation'].append(Gromov_Wasserstein_distance)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        end_d = time.time()

        #calculate time for epoch
        end_batch = time.time()
        e = int(end_batch-start_epoch)
        print('Time for Epoch: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

        #save history
        pickle.dump([train_history, test_history], open(save_folder+'/Pickle/3dgan-history.pkl', 'wb'))
        pickle.dump({'train': train_history, 'test': test_history}, open(save_folder+'/Pickle/3dgan-history_dict.pkl', 'wb'))

        #print loss table and plot generated image; also save them
        loss_table(train_history, test_history, save_folder, epoch, validation_metric, save=True, timeforepoch = e)
        plot_gen_image_tf(latent_size, epoch)
        plot_loss(train_history, test_history, save_folder, save=True)
        plot_validation(train_history, save_folder)
        plot_gromov_w_distance(train_history, save_folder)

        #dump data
        if epoch >=ReLU_epoch-1 or epoch >= 5:
            generator.save_weights(save_folder+"/Weights/gen/params_generator_epoch_"+str(epoch)+".hdf5",overwrite=True)
            discriminator.save_weights(save_folder+"/Weights/disc/params_discriminator_epoch_"+str(epoch)+".hdf5",overwrite=True)
    

        
    
    if save_only_best_weights:
        #delete some weight files
        #print(train_history['validation'])
        #print(train_history['validation'].index(np.min(train_history['validation']))+1)
        best_weights_index = train_history['validation'].index(np.min(train_history['validation']))+1
        print("Best Weights Epoch: ", best_weights_index)
        for i in range(len(train_history['validation'])+1):
            if i != best_weights_index:
                try:
                    print("remove weights epoch: " + str(i))
                    os.remove(save_folder+"/Weights/gen/params_generator_epoch_"+str(i)+".hdf5")
                    os.remove(save_folder+"/Weights/disc/params_discriminator_epoch_"+str(i)+".hdf5")
                except:
                    print(str(i) + " is not saved")
    
    return

epoch_cycle()

