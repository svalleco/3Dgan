#Training for GAN

from __future__ import print_function
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

## setting seed ###
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(1)
#import random
#random.seed(1)
#os.environ['PYTHONHASHSEED'] = '0' 
##################

GLOBAL_BATCH_SIZE = 64

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
#import keras
import argparse
import sys
import h5py 
import numpy as np
import time
import math
import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import analysis.utils.GANutils as gan
import TfRecordConverter as tfconvert


# if '.cern.ch' in os.environ.get('HOSTNAME'): # Here a check for host can be used to set defaults accordingly
#     tlab = True
# else:
#     tlab= False
    
# try:
#     import setGPU #if Caltech
# except:
#     pass

#from memory_profiler import profile # used for memory profiling
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from tensorflow.keras.utils import Progbar

#minimize function
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
from tensorflow.python.distribute import parameter_server_strategy

#Config
config = tf.compat.v1.ConfigProto(log_device_placement=True)
# config.gpu_options.allow_growth = True
# main_session = tf.compat.v1.InteractiveSession(config=config)

#tf.config.experimental_run_functions_eagerly(True)


def main():
    #Architectures to import
    from ParallelAngleArch3dGANv2 import generator, discriminator #if there is any parallel changes to the architecture this needs to change
    
    #Values to be set by user
    parser = get_parser()
    params = parser.parse_args()
    nb_epochs = params.nbepochs #Total Epochs
    batch_size = params.batchsize #batch size
    latent_size = params.latentsize #latent vector size
    verbose = params.verbose
    datapath = params.datapath# Data path
    outpath = params.outpath # training output
    nEvents = params.nEvents# maximum number of events used in training
    ascale = params.ascale # angle scale
    yscale = params.yscale # scaling energy
    weightdir = 'weights/3dgan_weights_' + params.name
    pklfile = 'results/3dgan_history_' + params.name + '.pkl'# loss history
    resultfile = 'results/3dgan_analysis' + params.name + '.pkl'# optimization metric history
    timefile = 'results/3dgan_times_' + params.name + '.pkl'
    prev_gweights = 'weights/' + params.prev_gweights
    prev_dweights = 'weights/' + params.prev_dweights
    xscale = params.xscale
    xpower = params.xpower
    analyse=params.analyse # if analysing
    loss_weights=[params.gen_weight, params.aux_weight, params.ang_weight, params.ecal_weight]
    dformat=params.dformat
    thresh = params.thresh # threshold for data
    angtype = params.angtype
    particle = params.particle
    warm = params.warm
    lr = params.lr
    events_per_file = 5000
    energies = [0, 110, 150, 190]

    # if tlab:
    #    if not warm:
    #      datapath = 'path4'
    #    outpath = '/gkhattak/'
         
    # if datapath=='path1':
    #    datapath = "/data/shared/gkhattak/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV                                                         
    # elif datapath=='path2':
    #    datapath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # culture plate                              
    #    events_per_file = 10000
    #    energies = [0, 50, 100, 200, 250, 300, 400, 500]
    # elif datapath=='path3':
    #    datapath = "/data/shared/LCDLargeWindow/varangle/*scan/*scan_RandomAngle_*.h5" # caltech                                                      
    #    events_per_file = 10000
    #    energies = [0, 50, 100, 200, 250, 300, 400, 500]
    # elif datapath=='path4':
    #    datapath = "/eos/user/g/gkhattak/VarAngleData/*Measured3ThetaEscan/*.h5"  # Data path 100-200 GeV                                             
    # elif datapath=='path5':
    #    datapath = "/gkhattak/data/*RandomAngle100GeV/*.h5"
    #    energies = [0, 10, 50, 90]

    weightdir = outpath + 'weights/3dgan_weights_' + params.name
    pklfile = outpath + 'results/3dgan_history_' + params.name + '.pkl'# loss history
    resultfile = outpath + 'results/3dgan_analysis' + params.name + '.pkl'# optimization metric history   
    timefile = outpath + 'results/3dgan_times_' + params.name + '.pkl'
    prev_gweights = params.prev_gweights #outpath + 'weights/' + params.prev_gweights
    prev_dweights = params.prev_dweights #outpath + 'weights/' + params.prev_dweights

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

    # Building discriminator and generator
    gan.safe_mkdir(weightdir)
    with strategy.scope():
        d=discriminator(xpower, dformat=dformat)
        g=generator(latent_size, dformat=dformat)


    # GAN training 
    Gan3DTrainAngle(strategy,d, g, datapath, nEvents, weightdir, pklfile, timefile ,nb_epochs=nb_epochs, batch_size=batch_size, batch_size_per_replica=BATCH_SIZE_PER_REPLICA,
                    latent_size=latent_size, loss_weights=loss_weights, lr=lr, xscale = xscale, xpower=xpower, angscale=ascale,
                    yscale=yscale, thresh=thresh, angtype=angtype, analyse=analyse, resultfile=resultfile,
                    energies=energies, dformat=dformat, particle=particle, verbose=verbose, warm=warm,
                    prev_gweights= prev_gweights, prev_dweights=prev_dweights   )

def get_parser():
    parser = argparse.ArgumentParser(description='3D GAN Params' )
    parser.add_argument('--nbepochs', action='store', type=int, default=60, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=64, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=256, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='path2', help='HDF5 files to train from.')
    parser.add_argument('--outpath', action='store', type=str, default='', help='Dir to save output from a training.')
    parser.add_argument('--dformat', action='store', type=str, default='channels_last')
    parser.add_argument('--nEvents', action='store', type=int, default=400000, help='Maximum Number of events used for Training')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
    parser.add_argument('--xpower', action='store', type=float, default=0.85, help='pre processing of cell energies by raising to a power')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--ascale', action='store', type=int, default=1, help='Multiplication factor for angle input')
    parser.add_argument('--analyse', action='store', default=False, help='Whether or not to perform analysis')
    parser.add_argument('--gen_weight', action='store', type=float, default=3, help='loss weight for generation real/fake loss')
    parser.add_argument('--aux_weight', action='store', type=float, default=0.1, help='loss weight for auxilliary energy regression loss')
    parser.add_argument('--ang_weight', action='store', type=float, default=25, help='loss weight for angle loss')
    parser.add_argument('--ecal_weight', action='store', type=float, default=0.5, help='loss weight for ecal sum loss')
    parser.add_argument('--hist_weight', action='store', type=float, default=0.1, help='loss weight for additional bin count loss')
    parser.add_argument('--thresh', action='store', type=int, default=0., help='Threshold for cell energies')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle to use for Training. It can be theta, mtheta or eta')
    parser.add_argument('--particle', action='store', type=str, default='Ele', help='Type of particle')
    parser.add_argument('--lr', action='store', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--warm', action='store', default=False, help='Start from pretrained weights or random initialization')
    parser.add_argument('--prev_gweights', type=str, default='3dgan_weights_gan_training_epsilon_k2/params_generator_epoch_131.hdf5', help='Initial generator weights for warm start')
    parser.add_argument('--prev_dweights', type=str, default='3dgan_weights_gan_training_epsilon_k2/params_discriminator_epoch_131.hdf5', help='Initial discriminator weights for warm start')
    parser.add_argument('--name', action='store', type=str, default='gan_training', help='Unique identifier can be set for each training')
    return parser

# A histogram function that counts cells in different bins
def hist_count(x, p=1.0, daxis=(1, 2, 3)):
    limits=np.array([0.05, 0.03, 0.02, 0.0125, 0.008, 0.003]) # bin boundaries used
    limits= np.power(limits, p)
    bin1 = np.sum(np.where(x>(limits[0]) , 1, 0), axis=daxis)
    bin2 = np.sum(np.where((x<(limits[0])) & (x>(limits[1])), 1, 0), axis=daxis)
    bin3 = np.sum(np.where((x<(limits[1])) & (x>(limits[2])), 1, 0), axis=daxis)
    bin4 = np.sum(np.where((x<(limits[2])) & (x>(limits[3])), 1, 0), axis=daxis)
    bin5 = np.sum(np.where((x<(limits[3])) & (x>(limits[4])), 1, 0), axis=daxis)
    bin6 = np.sum(np.where((x<(limits[4])) & (x>(limits[5])), 1, 0), axis=daxis)
    bin7 = np.sum(np.where((x<(limits[5])) & (x>0.), 1, 0), axis=daxis)
    bin8 = np.sum(np.where(x==0, 1, 0), axis=daxis)
    bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
    bins[np.where(bins==0)]=1 # so that an empty bin will be assigned a count of 1 to avoid unstability
    return bins

#get data for training
def GetDataAngle(datafile, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1):
    print ('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    X=np.array(f.get('ECAL'))* xscale
    Y=np.array(f.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X=X[indexes]
    Y=Y[indexes]
    if angtype in f:
      ang = np.array(f.get(angtype))[indexes]
    else:
      ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=ecal[indexes]
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal


def GetDataAngleParallel(dataset, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1):
    #print ('Loading Data from .....', dataset)
    #replica_context = tf.distribute.get_replica_context()
    #tf.print("Replica id: ", replica_context.replica_id_in_sync_group, " of ", replica_context.num_replicas_in_sync)


    
    # tf.print(dataset.get('energy'))
    # Y=tf.math.divide(dataset.get('energy'),yscale)
    # main_session.run(tf.compat.v1.global_variables_initializer())
    # print(Y.eval(session=main_session))
    
    X=np.array(dataset.get('ECAL'))* xscale
    Y=np.array(dataset.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X=X[indexes]
    Y=Y[indexes]
    if angtype in dataset:
      ang = np.array(dataset.get(angtype))[indexes]
    else:
      ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=ecal[indexes]
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)

    Y = [[el] for el in Y]
    ang = [[el] for el in ang]
    ecal = [[el] for el in ecal]

    final_dataset = {'X': X,'Y': Y, 'ang': ang, 'ecal': ecal}

    return final_dataset

#Create global loss objects
# binary_crossentropy_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
# mean_absolute_percentage_error_object = tf.keras.losses.MeanAbsolutePercentageError(reduction='none')
# mae_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE) 

def get_numpy_values(value_tensor):
    value_tensor = value_tensor.numpy()
    #tf.make_ndarray(tensor)
    return value_tensor

def compute_global_loss(labels, predictions, global_batch_size, loss_weights=[3, 0.1, 25, 0.1]):
    #print(predictions)

    #can be initialized outside 
    binary_crossentropy_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    mean_absolute_percentage_error_object = tf.keras.losses.MeanAbsolutePercentageError(reduction=tf.keras.losses.Reduction.NONE)
    mae_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE) 
    
    #tf.print(predictions[0])
    #predictions[0].numpy()

    #pred = tf.py_function(get_numpy_values, [predictions[0]], Tout=tf.float32)
    #print(pred)


    #print(predictions[0])

    # y_true = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    # labels[0] = [[el] for el in y_true]
    # y_pred = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    # predictions[0] = [[el] for el in y_pred]

    binary_example_loss = binary_crossentropy_object(labels[0], predictions[0], sample_weight=loss_weights[0])
    mean_example_loss_1 = mean_absolute_percentage_error_object(labels[1], predictions[1], sample_weight=loss_weights[1])
    #tf.print(mean_example_loss_1)
    # for el in binary_example_loss:
    #     tf.print(el)
    mae_example_loss = mae_object(labels[2], predictions[2], sample_weight=loss_weights[2])
    mean_example_loss_2 = mean_absolute_percentage_error_object(labels[3], predictions[3], sample_weight=loss_weights[3])
    
    binary_loss = tf.nn.compute_average_loss(binary_example_loss, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[0])
    mean_loss_1 = tf.nn.compute_average_loss(mean_example_loss_1, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[1])
    mae_loss = tf.nn.compute_average_loss(mae_example_loss, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[2])
    mean_loss_2 = tf.nn.compute_average_loss(mean_example_loss_2, global_batch_size=global_batch_size)#, sample_weight=1/loss_weights[3])
    
    #print(binary_loss)
    
    return [binary_loss, mean_loss_1, mae_loss, mean_loss_2]

def Gan3DTrainAngle(strategy, discriminator, generator, datapath, nEvents, WeightsDir, pklfile, timefile, nb_epochs=30, batch_size=128, batch_size_per_replica=64 ,latent_size=200, loss_weights=[3, 0.1, 25, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
    
    start_init = time.time()
    f = [0.9, 0.1] # train, test fractions 

    #loss_ftn = hist_count # function used for additional loss
    
    # apply settings according to data format
    if dformat=='channels_last':
       daxis=4 # channel axis
       daxis2=(1, 2, 3) # axis for sum
    else:
       daxis=1 # channel axis
       daxis2=(2, 3, 4) # axis for sum

    #Create loss objects and optimizers

    with strategy.scope():
        optimizer_discriminator = RMSprop(lr)
        optimizer_generator = RMSprop(lr)

    # with strategy.scope():
    #     # build the discriminator
    #     print('[INFO] Building discriminator')
    #     discriminator.compile(
    #         optimizer=RMSprop(lr),
    #         loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error'],
    #         loss_weights=loss_weights
    #     )

    #     # build the generator
    #     print('[INFO] Building generator')
    #     generator.compile(
    #         optimizer=RMSprop(lr),
    #         loss='binary_crossentropy'
    #     )
 
    # build combined Model
    
    with strategy.scope():
        latent = Input(shape=(latent_size, ), name='combined_z')   
        fake_image = generator( latent)
        discriminator.trainable = False
        fake, aux, ang, ecal = discriminator(fake_image) #remove add_loss

        combined = Model(
            inputs=[latent],
            outputs=[fake, aux, ang, ecal], # remove add_loss
            name='combined_model'
        )
        
        combined.compile()
        # combined.compile(
        #     optimizer=RMSprop(lr),
        #     loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error'],
        #     loss_weights=loss_weights
        # )


    #initialize with previous weights
    if warm:
        generator.load_weights(prev_gweights)
        print('Generator initialized from {}'.format(prev_gweights))
        discriminator.load_weights(prev_dweights)
        print('Discriminator initialized from {}'.format(prev_dweights))

    # Getting All available Data sorted in test train fraction
    #Trainfiles, Testfiles = gan.DivideFiles(datapath, f, datasetnames=["ECAL"], Particles =[particle])
    discriminator.trainable = True # to allow updates to moving averages for BatchNormalization     
    
    Trainfiles = ['gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_000.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_001.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_002.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_003.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_004.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_005.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_006.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_007.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_008.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_009.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_010.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_011.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_012.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_013.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_014.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_015.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_016.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_017.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_018.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_019.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_020.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_021.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_022.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_023.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_024.tfrecords']
    Testfiles = ['gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_025.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_026.tfrecords',\
                'gs://renato-tpu-bucket/tfrecordsprepoc/Ele_VarAngleMeas_100_200_027.tfrecords']
    
    print(Trainfiles)
    print(Testfiles)




    # @tf.function
    # def rundist(dataset):
    #     # print(tf.executing_eagerly())
    #     print(tf.executing_eagerly())
    #     #tf.config.run_functions_eagerly(True)
    #     strategy.run(GetDataAngleParallel, args=(dataset,))
    #     # tf.config.run_functions_eagerly(False)


    # for element in dist_dataset:
    #     print(element.get('energy'))
    #     #element.get[0].numpy()
    #     rundist(element)
    #     return

    # for debuging 
    #print(dist_dataset)
    # for element in Y_train_dist_dataset.as_numpy_iterator():
    #     print('---------------------------------------------------------------')
    #     print(element)
    #     print('---------------------------------------------------------------')



    nb_Test = int(nEvents * f[1]) # The number of test events calculated from fraction of nEvents
    nb_Train = int(nEvents * f[0]) # The number of train events calculated from fraction of nEvents

    #------------------------------Probably not needed

    #The number of actual batches used will be min(available batches & nb_Train)
    nb_train_batches = int(nb_Train/batch_size)
    nb_test_batches = int(nb_Test/batch_size)
    print('The max train batches can be {} batches while max test batches can be {}'.format(nb_train_batches, nb_test_batches))  
    
    #------------------------------


    #create history and finish initiation
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    init_time = time.time()- start_init
    analysis_history = defaultdict(list)
    time_history = defaultdict(list)
    print('Initialization time is {} seconds'.format(init_time))

    def minimize(tape, optimizer, loss, trainable_variables):
        with tape:
            if isinstance(optimizer, lso.LossScaleOptimizer):
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(loss, trainable_variables)

        aggregate_grads_outside_optimizer = (optimizer._HAS_AGGREGATE_GRAD and not isinstance(strategy.extended, parameter_server_strategy.ParameterServerStrategyExtended))

        if aggregate_grads_outside_optimizer:
            gradients = optimizer._aggregate_gradients(zip(gradients,trainable_variables))

        if isinstance(optimizer, lso.LossScaleOptimizer):
            gradients = optimizer.get_unscaled_gradients(gradients)
        
        gradients = optimizer._clip_gradients(gradients)  # pylint: disable=protected-access
        
        if trainable_variables:
            if aggregate_grads_outside_optimizer:
                optimizer.apply_gradients(zip(gradients, trainable_variables), experimental_aggregate_gradients=False)
            else:
                optimizer.apply_gradients(zip(gradients, trainable_variables))




    def Discriminator_Train_steps(dataset):
        print('Discriminator')
        start = time.time()
        # Get a single batch    
        image_batch = dataset.get('X')#.numpy()
        energy_batch = dataset.get('Y')#.numpy()
        ecal_batch = dataset.get('ecal')#.numpy()
        ang_batch = dataset.get('ang')#.numpy()
        #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

        

        # filefortests = '/data/redacost/filefortests.pkl'
        # with open(filefortests, 'rb') as f:
        #     x = pickle.load(f) 
        # noise = np.asarray(x['noise'])
        #tf.print(noise)

        b_size = energy_batch.get_shape().as_list()[0]#.numpy()[0]
        print(b_size)

        
        # Generate Fake events with same energy and angle as data batch
        noise = tf.random.normal((batch_size_per_replica, latent_size-2), 0, 1)
        #noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
        generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
        generated_images = generator(generator_ip, training=False)
        #tf.print(generated_images) #same image

        # Train discriminator first on real batch 
        fake_batch = gan.BitFlip(np.ones(batch_size_per_replica).astype(np.float32))
        #fake_batch = x['ganflip1']
        fake_batch = [[el] for el in fake_batch]
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        print(time.time()-start)
        time1 = time.time()


        with tf.GradientTape() as tape:
            predictions = discriminator(image_batch, training=True)
            real_batch_loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)
        
        #print('Discriminator real')
        #tf.print(predictions)
        
        minimize(tape, optimizer_discriminator, real_batch_loss, discriminator.trainable_variables)

    
        #gradients = tape.gradient(real_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
        #optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights
    
        print(time.time()-time1)
        time2=time.time()

        #tf.print(real_batch_loss)
        #return real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3], real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3]

        #Train discriminato on the fake batch
        fake_batch = gan.BitFlip(np.zeros(batch_size_per_replica).astype(np.float32))
        #fake_batch = x['ganflip2']
        fake_batch = [[el] for el in fake_batch]
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        with tf.GradientTape() as tape:
            predictions = discriminator(generated_images, training=True)
            fake_batch_loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)
        
        minimize(tape, optimizer_discriminator, fake_batch_loss, discriminator.trainable_variables)
        #gradients = tape.gradient(fake_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
        #optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

        print(time.time()-time2)
        print('begin 3')

        # tf.print('Discriminator false')
        # tf.print(predictions)

        #tf.print(real_batch_loss)
        # tf.print(real_batch_loss)
        # tf.print(fake_batch_loss)

        #return losses separatly for reduce op

        print('Generator') 
        start = time.time()
        

        # filefortests = '/data/redacost/filefortests.pkl'
        # with open(filefortests, 'rb') as f:
        #     x = pickle.load(f) 
        # noise = np.asarray(x['noise'])

        # generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
        # generated_images = generator(generator_ip, training=False)
        # tf.print(generated_images)

        trick = np.ones(batch_size_per_replica).astype(np.float32)
        #trick = x['trick']

        fake_batch = [[el] for el in trick]
        labels = [fake_batch, tf.reshape(energy_batch, (-1,1)), ang_batch, ecal_batch]
        #labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

        time1 = time.time() - start
        print(time1)
        start=time.time()

        gen_losses = []
        # Train generator twice using combined model
        for _ in range(2):
            start=time.time()
            #noise = x['noise']
            #noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
            noise = tf.random.normal((batch_size_per_replica, latent_size-2), 0, 1)
            generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1) # sampled angle same as g4 theta
            time1 = time.time() - start
            print(time1)
            start=time.time()        

            with tf.GradientTape() as tape:
                #predictions = combined(generator_ip, training= True)
                generated_images = generator(generator_ip ,training= True)
                #tf.print(generated_images)
                predictions = discriminator(generated_images , training=True)
                loss = compute_global_loss(labels, predictions, batch_size, loss_weights=loss_weights)
            
            # tf.print('--Generator------------')
            # tf.print(predictions)
            # tf.print('---------------------------')
            minimize(tape, optimizer_generator, loss, generator.trainable_variables)
            #gradients = tape.gradient(loss, generator.trainable_variables) # model.trainable_variables or  model.trainable_weights
            #optimizer_generator.apply_gradients(zip(gradients, generator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

            time1 = time.time() - start
            print(time1)

            for el in loss:
                gen_losses.append(el)

        return real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3], fake_batch_loss[0], fake_batch_loss[1], fake_batch_loss[2], fake_batch_loss[3], \
                gen_losses[0], gen_losses[1], gen_losses[2], gen_losses[3], gen_losses[4], gen_losses[5], gen_losses[6], gen_losses[7]   

    def Test_steps(dataset):    
        # Get a single batch    
        image_batch = dataset.get('X')#.numpy()
        energy_batch = dataset.get('Y')#.numpy()
        ecal_batch = dataset.get('ecal')#.numpy()
        ang_batch = dataset.get('ang')#.numpy()
        #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

        # Generate Fake events with same energy and angle as data batch
        noise = np.random.normal(0, 1, (batch_size_per_replica, latent_size-2)).astype(np.float32)
        generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
        generated_images = generator(generator_ip, training=False)

        # concatenate to fake and real batches
        X = tf.concat((image_batch, generated_images), axis=0)
        y = np.array([1] * batch_size_per_replica + [0] * batch_size_per_replica).astype(np.float32)
        ang = tf.concat((ang_batch, ang_batch), axis=0)
        ecal = tf.concat((ecal_batch, ecal_batch), axis=0)
        aux_y = tf.concat((energy_batch, energy_batch), axis=0)
        #add_loss= tf.concat((add_loss_batch, add_loss_batch), axis=0)

        y = [[el] for el in y]

        labels = [y, aux_y, ang, ecal]
        disc_eval = discriminator(X, training=False)
        disc_eval_loss = compute_global_loss(labels, disc_eval, batch_size, loss_weights=loss_weights)
        
        trick = np.ones(batch_size_per_replica).astype(np.float32) #original doest have astype
        fake_batch = [[el] for el in trick]
        labels = [fake_batch, energy_batch, ang_batch, ecal_batch]
        gen_eval = combined(generator_ip, training=False)
        
        gen_eval_loss = compute_global_loss(labels, gen_eval, batch_size, loss_weights=loss_weights)


        #disc_eval_loss = discriminator.evaluate( X, [y, aux_y, ang, ecal, add_loss], verbose=False, batch_size=batch_size)
        #gen_eval_loss = combined.evaluate(generator_ip, [np.ones(batch_size), energy_batch, ang_batch, ecal_batch, add_loss_batch], verbose=False, batch_size=batch_size)

        return disc_eval_loss[0], disc_eval_loss[1], disc_eval_loss[2], disc_eval_loss[3], gen_eval_loss[0], gen_eval_loss[1], gen_eval_loss[2], gen_eval_loss[3]
        #return disc_eval_loss, gen_eval_loss


    @tf.function
    def distributed_discriminator_train_step(dataset):
        #X_train, epoch_disc_loss, epoch_gen_loss = 

        #print('check100')
        gen_losses = []

        #fake, energy, ang, ecal
        real_batch_loss_1, real_batch_loss_2, real_batch_loss_3, real_batch_loss_4, \
        fake_batch_loss_1, fake_batch_loss_2, fake_batch_loss_3, fake_batch_loss_4, \
        gen_batch_loss_1, gen_batch_loss_2, gen_batch_loss_3, gen_batch_loss_4, \
        gen_batch_loss_5, gen_batch_loss_6, gen_batch_loss_7, gen_batch_loss_8  = strategy.run(Discriminator_Train_steps, args=(next(dataset),))
        
        real_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_1, axis=None)
        real_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_2, axis=None)
        real_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_3, axis=None)
        real_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_4, axis=None)
        real_batch_loss = [real_batch_loss_1, real_batch_loss_2, real_batch_loss_3, real_batch_loss_4]

        fake_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_1, axis=None)
        fake_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_2, axis=None)
        fake_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_3, axis=None)
        fake_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_4, axis=None)
        fake_batch_loss = [fake_batch_loss_1, fake_batch_loss_2, fake_batch_loss_3, fake_batch_loss_4]

    
        gen_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_1, axis=None)
        gen_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_2, axis=None)
        gen_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_3, axis=None)
        gen_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_4, axis=None)
        gen_batch_loss = [gen_batch_loss_1, gen_batch_loss_2, gen_batch_loss_3, gen_batch_loss_4]

        gen_losses.append(gen_batch_loss)

        gen_batch_loss_5 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_5, axis=None)
        gen_batch_loss_6 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_6, axis=None)
        gen_batch_loss_7 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_7, axis=None)
        gen_batch_loss_8 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_8, axis=None)
        gen_batch_loss = [gen_batch_loss_5, gen_batch_loss_6, gen_batch_loss_7, gen_batch_loss_8]

        gen_losses.append(gen_batch_loss)

        return real_batch_loss, fake_batch_loss, gen_losses


    @tf.function
    def distributed_test_step(dataset):
        disc_test_loss_1, disc_test_loss_2, disc_test_loss_3, disc_test_loss_4, \
        gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4 = strategy.run(Test_steps, args=(next(dataset),))

        disc_test_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_1, axis=None)
        disc_test_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_2, axis=None)
        disc_test_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_3, axis=None)
        disc_test_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_4, axis=None)
        disc_test_loss = [disc_test_loss_1, disc_test_loss_2, disc_test_loss_3, disc_test_loss_4]

        gen_test_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_1, axis=None)
        gen_test_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_2, axis=None)
        gen_test_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_3, axis=None)
        gen_test_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_4, axis=None)
        gen_test_loss = [gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4]

        
        return disc_test_loss, gen_test_loss


    # Dataset preparation

    #if index % 100 == 0:
    print ('Loading Data from .....')
    
    time_start_file = time.time()
    # Get the dataset from the trainfile
    dataset, datasetsize = tfconvert.RetrieveTFRecordpreprocessing(Trainfiles, batch_size)
    datasettest, datasetsizetest = tfconvert.RetrieveTFRecordpreprocessing(Testfiles, batch_size)

    time_elapsed = time.time() - time_start_file
    print("Get Dataset: " + str(time_elapsed))
    time_start_file = time.time()
    
    #distribute the dataset
    #dist_dataset = strategy.experimental_distribute_datasets_from_function(lambda _: tfconvert.RetrieveTFRecordpreprocessing(Trainfiles, 128))
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    dist_dataset_test = strategy.experimental_distribute_dataset(datasettest)


    time_elapsed = time.time() - time_start_file
    print("Distribute dataset: " + str(time_elapsed))
    time_start_file = time.time()

    dist_dataset_iter = iter(dist_dataset)
    steps_per_epoch =int( datasetsize // (batch_size))

    dist_dataset_iter_test = iter(dist_dataset_test)
    steps_per_epoch_test =int( datasetsizetest // (batch_size))

    

    #print(next(dist_dataset_iter))

    #return

    #nb = 0
    #for b in dist_dataset:
    #    nb += 1
    #    print(nb)
    #return

    #print(dist_dataset_iter)
    #return

    # Start training
    for epoch in range(nb_epochs):
        #dist_dataset_iter.initializer
        epoch_start = time.time()
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))


        #--------------------------------------------------------------------------------------------
        #------------------------------ Main Training Cycle -----------------------------------------
        #--------------------------------------------------------------------------------------------

        #Get the data for each training file

        nb_file=0
        epoch_gen_loss = []
        epoch_disc_loss = []
        index = 0
        file_index=0
        nbatch = 0


        #Training
        #add Trainfiles, nb_train_batches, progress_bar, daxis, daxis2, loss_ftn, combined
        #for batch in dist_dataset:
        #print(nbatch)
        #nbatch += 1
        #file_time = time.time()
        #print(batch.get('Y'))
        #print(nbatch)

        #this_batch_size =128 #not necessary can be removed
        print('Number of Batches: ', steps_per_epoch)
        
        for _ in range(steps_per_epoch):
            #Discriminator Training
            file_time = time.time()
            real_batch_loss, fake_batch_loss, gen_losses = distributed_discriminator_train_step(dist_dataset_iter)

            #Configure the loss so it is equal to the original values
            real_batch_loss = [el.numpy() for el in real_batch_loss]
            real_batch_loss_total_loss = np.sum(real_batch_loss)
            new_real_batch_loss = [real_batch_loss_total_loss]
            for i_weights in range(len(real_batch_loss)):
                new_real_batch_loss.append(real_batch_loss[i_weights] / loss_weights[i_weights])
            real_batch_loss = new_real_batch_loss

            fake_batch_loss = [el.numpy() for el in fake_batch_loss]
            fake_batch_loss_total_loss = np.sum(fake_batch_loss)
            new_fake_batch_loss = [fake_batch_loss_total_loss]
            for i_weights in range(len(fake_batch_loss)):
                new_fake_batch_loss.append(fake_batch_loss[i_weights] / loss_weights[i_weights])
            fake_batch_loss = new_fake_batch_loss

            #real_batch_loss = [el * w for el in real_batch_loss]
            # print(real_batch_loss)
            # print(fake_batch_loss)

            # if index == 9:
            #     return

            # index +=1
            # continue

            #if ecal sum has 100% loss(generating empty events) then end the training 
            if fake_batch_loss[3] == 100.0 and index >10:
                print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
                generator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
                discriminator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
                print ('real_batch_loss', real_batch_loss)
                print ('fake_batch_loss', fake_batch_loss)
                sys.exit()

            # append mean of discriminator loss for real and fake events 
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])

            #return

            gen_losses[0] = [el.numpy() for el in gen_losses[0]]
            gen_losses_total_loss = np.sum(gen_losses[0])
            new_gen_losses = [gen_losses_total_loss]
            for i_weights in range(len(gen_losses[0])):
                new_gen_losses.append(gen_losses[0][i_weights] / loss_weights[i_weights])
            gen_losses[0] = new_gen_losses

            gen_losses[1] = [el.numpy() for el in gen_losses[1]]
            gen_losses_total_loss = np.sum(gen_losses[1])
            new_gen_losses = [gen_losses_total_loss]
            for i_weights in range(len(gen_losses[1])):
                new_gen_losses.append(gen_losses[1][i_weights] / loss_weights[i_weights])
            gen_losses[1] = new_gen_losses

            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]

            epoch_gen_loss.append(generator_loss)
            #index +=1

            print('Time taken by batch', str(nbatch) ,' was', str(time.time()-file_time) , 'seconds.')
            nbatch += 1

        #print(generator_loss)
        #return

        #break

        #break
    #print ('Total batches were {}'.format(index))

        #return
        

        #X_train, Y_train, ang_train, ecal_train = GetDataAngle(Trainfiles[0], xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
        print('Time taken by epoch{} was {} seconds.'.format(epoch, time.time()-epoch_start))
        train_time = time.time() - epoch_start

        #continue

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # print(discriminator_train_loss)
        # print(generator_train_loss)

        # return
        #--------------------------------------------------------------------------------------------
        #------------------------------ Main Testing Cycle ------------------------------------------
        #--------------------------------------------------------------------------------------------

        #read first test file
        disc_test_loss=[]
        gen_test_loss =[]
        nb_file=0
        index=0
        file_index=0

        # Test process will also be accomplished in batches to reduce memory consumption
        print('\nTesting for epoch {}:'.format(epoch))
        test_start = time.time()

        #continue


        # repeat till data is available
        for _ in range(steps_per_epoch_test):

            disc_eval_loss, gen_eval_loss = distributed_test_step(dist_dataset_iter_test)

            #Configure the loss so it is equal to the original values
            disc_eval_loss = [el.numpy() for el in disc_eval_loss]
            disc_eval_loss_total_loss = np.sum(disc_eval_loss)
            new_disc_eval_loss = [disc_eval_loss_total_loss]
            for i_weights in range(len(disc_eval_loss)):
                new_disc_eval_loss.append(disc_eval_loss[i_weights] / loss_weights[i_weights])
            disc_eval_loss = new_disc_eval_loss

            gen_eval_loss = [el.numpy() for el in gen_eval_loss]
            gen_eval_loss_total_loss = np.sum(gen_eval_loss)
            new_gen_eval_loss = [gen_eval_loss_total_loss]
            for i_weights in range(len(gen_eval_loss)):
                new_gen_eval_loss.append(gen_eval_loss[i_weights] / loss_weights[i_weights])
            gen_eval_loss = new_gen_eval_loss

            index +=1
            # evaluate discriminator loss           
            disc_test_loss.append(disc_eval_loss)
            # evaluate generator loss
            gen_test_loss.append(gen_eval_loss)

            #break


        #--------------------------------------------------------------------------------------------
        #------------------------------ Updates -----------------------------------------------------
        #--------------------------------------------------------------------------------------------


        # make loss dict 
        print('Total Test batches were {}'.format(index))
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
        discriminator_test_loss = np.mean(np.array(disc_test_loss), axis=0)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        generator_test_loss = np.mean(np.array(gen_test_loss), axis=0)
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        # print losses
        # print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s}'.format(
        #     'component', *discriminator.metrics_names))
        print(discriminator.metrics_names)
        print('-' * 65)
        ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}'
        print(ROW_FMT.format('generator (train)',
                                *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                                *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                                *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                                *test_history['discriminator'][-1]))

        # save weights every epoch                                                                                                                                                                                                                                                    
        generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                                overwrite=True)
        discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                    overwrite=True)

        epoch_time = time.time()-test_start
        print("The Testing for {} epoch took {} seconds. Weights are saved in {}".format(epoch, epoch_time, WeightsDir))
        test_time = epoch_time
        epoch_time = train_time + test_time

        time_history['train'].append(train_time)
        time_history['test'].append(test_time)
        time_history['epoch'].append(epoch_time)

        
        # save loss dict to pkl file
        pickle.dump({'train': train_history, 'test': test_history}, open(pklfile, 'wb'))
        pickle.dump(time_history, open(timefile, 'wb'))

        #--------------------------------------------------------------------------------------------
        #------------------------------ Analysis ----------------------------------------------------
        #--------------------------------------------------------------------------------------------

        
        # if a short analysis is to be performed for each epoch
        if analyse:
            print('analysing..........')
            atime = time.time()
            # load all test data
            for index, dtest in enumerate(Testfiles):
                if index == 0:
                   X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
                else:
                   if X_test.shape[0] < nb_Test:
                     X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=xscale, angscale=angscale, angtype=angtype, thresh=thresh, daxis=daxis)
                     X_test = np.concatenate((X_test, X_temp))
                     Y_test = np.concatenate((Y_test, Y_temp))
                     ang_test = np.concatenate((ang_test, ang_temp))
                     ecal_test = np.concatenate((ecal_test, ecal_temp))
            if X_test.shape[0] > nb_Test:
               X_test, Y_test, ang_test, ecal_test = X_test[:nb_Test], Y_test[:nb_Test], ang_test[:nb_Test], ecal_test[:nb_Test]
            else:
               nb_Test = X_test.shape[0] # the nb_test maybe different if total events are less than nEvents      
            var=gan.sortEnergy([np.squeeze(X_test), Y_test, ang_test], ecal_test, energies, ang=1)
            result = gan.OptAnalysisAngle(var, generator, energies, xpower = xpower, concat=2)
            print('{} seconds taken by analysis'.format(time.time()-atime))
            analysis_history['total'].append(result[0])
            analysis_history['energy'].append(result[1])
            analysis_history['moment'].append(result[2])
            analysis_history['angle'].append(result[3])
            print('Result = ', result)
            # write analysis history to a pickel file
            pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

# def Discriminator_Train_steps(optimizer_discriminator, optimizer_generator, discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Trainfiles, nb_train_batches, daxis, daxis2, combined, nb_epochs=30, batch_size=128, global_batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
#     print('Discriminator')
#     start = time.time()
#     # Get a single batch    
#     image_batch = dataset.get('X')#.numpy()
#     energy_batch = dataset.get('Y')#.numpy()
#     ecal_batch = dataset.get('ecal')#.numpy()
#     ang_batch = dataset.get('ang')#.numpy()
#     #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

#     # filefortests = '/data/redacost/filefortests.pkl'
#     # with open(filefortests, 'rb') as f:
#     #     x = pickle.load(f) 
#     # noise = np.asarray(x['noise'])
#     #tf.print(noise)

#     batch_size = energy_batch.get_shape().as_list()[0]#.numpy()[0]
#     #print(batch_size)
    
#     # Generate Fake events with same energy and angle as data batch
#     noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
#     generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
#     generated_images = generator(generator_ip, training=False)
#     #tf.print(generated_images) #same image

    

#     # Train discriminator first on real batch 
#     fake_batch = gan.BitFlip(np.ones(batch_size).astype(np.float32))
#     #fake_batch = x['ganflip1']
#     fake_batch = [[el] for el in fake_batch]
#     labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

#     print(time.time()-start)
#     time1 = time.time()

#     with tf.GradientTape() as tape:
#         predictions = discriminator(image_batch, training=True)
#         real_batch_loss = compute_global_loss(labels, predictions, global_batch_size, loss_weights=loss_weights)
    
#     # tf.print('Discriminator real')
#     # tf.print(predictions)
    
#     gradients = tape.gradient(real_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
#     optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights
  
#     print(time.time()-time1)
#     time2=time.time()

#     #tf.print(real_batch_loss)
#     #return real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3], real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3]

#     #Train discriminato on the fake batch
#     fake_batch = gan.BitFlip(np.zeros(batch_size).astype(np.float32))
#     #fake_batch = x['ganflip2']
#     fake_batch = [[el] for el in fake_batch]
#     labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

#     with tf.GradientTape() as tape:
#         predictions = discriminator(generated_images, training=True)
#         fake_batch_loss = compute_global_loss(labels, predictions, global_batch_size, loss_weights=loss_weights)
#     gradients = tape.gradient(fake_batch_loss, discriminator.trainable_variables) # model.trainable_variables or  model.trainable_weights
#     optimizer_discriminator.apply_gradients(zip(gradients, discriminator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

#     print(time.time()-time2)
#     print('begin 3')

#     # tf.print('Discriminator false')
#     # tf.print(predictions)

#     #tf.print(real_batch_loss)
#     # tf.print(real_batch_loss)
#     # tf.print(fake_batch_loss)

#     #return losses separatly for reduce op
#     return real_batch_loss[0], real_batch_loss[1], real_batch_loss[2], real_batch_loss[3], fake_batch_loss[0], fake_batch_loss[1], fake_batch_loss[2], fake_batch_loss[3]


def Generator_Train_steps(optimizer_discriminator, optimizer_generator, discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Trainfiles, nb_train_batches, daxis, daxis2, combined, nb_epochs=30, batch_size=128, global_batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
    print('Generator') 
    start = time.time()
    # Get a single batch    
    image_batch = dataset.get('X')#.numpy()
    energy_batch = dataset.get('Y')#.numpy()
    ecal_batch = dataset.get('ecal')#.numpy()
    ang_batch = dataset.get('ang')#.numpy()
    #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

    batch_size = energy_batch.get_shape().as_list()[0]

    # filefortests = '/data/redacost/filefortests.pkl'
    # with open(filefortests, 'rb') as f:
    #     x = pickle.load(f) 
    # noise = np.asarray(x['noise'])

    # generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
    # generated_images = generator(generator_ip, training=False)
    # tf.print(generated_images)

    trick = np.ones(batch_size).astype(np.float32)
    #trick = x['trick']

    fake_batch = [[el] for el in trick]
    labels = [fake_batch, tf.reshape(energy_batch, (-1,1)), ang_batch, ecal_batch]
    #labels = [fake_batch, energy_batch, ang_batch, ecal_batch]

    time1 = time.time() - start
    print(time1)
    start=time.time()

    gen_losses = []
    # Train generator twice using combined model
    for _ in range(2):
        start=time.time()
        #noise = x['noise']
        noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
        generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1) # sampled angle same as g4 theta
        time1 = time.time() - start
        print(time1)
        start=time.time()        

        with tf.GradientTape() as tape:
            #predictions = combined(generator_ip, training= True)
            generated_images = generator(generator_ip ,training= True)
            #tf.print(generated_images)
            predictions = discriminator(generated_images , training=False)
            loss = compute_global_loss(labels, predictions, global_batch_size, loss_weights=loss_weights)
        
        # tf.print('--Generator------------')
        # tf.print(predictions)
        # tf.print('---------------------------')

        gradients = tape.gradient(loss, generator.trainable_variables) # model.trainable_variables or  model.trainable_weights
        optimizer_generator.apply_gradients(zip(gradients, generator.trainable_variables)) # model.trainable_variables or  model.trainable_weights

        time1 = time.time() - start
        print(time1)

        for el in loss:
            gen_losses.append(el)


    return gen_losses[0], gen_losses[1], gen_losses[2], gen_losses[3], gen_losses[4], gen_losses[5], gen_losses[6], gen_losses[7]    
        

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


# def Test_steps(optimizer_discriminator, optimizer_generator, discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Testfiles, nb_test_batches, daxis, daxis2, combined, nb_epochs=30, batch_size=128, global_batch_size=128 ,latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):    
#     # Get a single batch    
#     image_batch = dataset.get('X')#.numpy()
#     energy_batch = dataset.get('Y')#.numpy()
#     ecal_batch = dataset.get('ecal')#.numpy()
#     ang_batch = dataset.get('ang')#.numpy()
#     #add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower, daxis2), axis=-1)

#     batch_size = energy_batch.get_shape().as_list()[0]

#     # Generate Fake events with same energy and angle as data batch
#     noise = np.random.normal(0, 1, (batch_size, latent_size-2)).astype(np.float32)
#     generator_ip = tf.concat((tf.reshape(energy_batch, (-1,1)), tf.reshape(ang_batch, (-1, 1)), noise),axis=1)
#     generated_images = generator(generator_ip, training=False)

#     # concatenate to fake and real batches
#     X = tf.concat((image_batch, generated_images), axis=0)
#     y = np.array([1] * batch_size + [0] * batch_size).astype(np.float32)
#     ang = tf.concat((ang_batch, ang_batch), axis=0)
#     ecal = tf.concat((ecal_batch, ecal_batch), axis=0)
#     aux_y = tf.concat((energy_batch, energy_batch), axis=0)
#     #add_loss= tf.concat((add_loss_batch, add_loss_batch), axis=0)

#     y = [[el] for el in y]

#     labels = [y, aux_y, ang, ecal]
#     disc_eval = discriminator(X, training=False)
#     disc_eval_loss = compute_global_loss(labels, disc_eval, global_batch_size, loss_weights=loss_weights)
    
#     trick = np.ones(batch_size).astype(np.float32) #original doest have astype
#     fake_batch = [[el] for el in trick]
#     labels = [fake_batch, energy_batch, ang_batch, ecal_batch]
#     gen_eval = combined(generator_ip, training=False)
    
#     gen_eval_loss = compute_global_loss(labels, gen_eval, global_batch_size, loss_weights=loss_weights)


#     #disc_eval_loss = discriminator.evaluate( X, [y, aux_y, ang, ecal, add_loss], verbose=False, batch_size=batch_size)
#     #gen_eval_loss = combined.evaluate(generator_ip, [np.ones(batch_size), energy_batch, ang_batch, ecal_batch, add_loss_batch], verbose=False, batch_size=batch_size)

#     return disc_eval_loss[0], disc_eval_loss[1], disc_eval_loss[2], disc_eval_loss[3], gen_eval_loss[0], gen_eval_loss[1], gen_eval_loss[2], gen_eval_loss[3]
#     #return disc_eval_loss, gen_eval_loss

# @tf.function
# def distributed_discriminator_train_step(optimizer_discriminator, optimizer_generator, strategy, discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Trainfiles, nb_train_batches, daxis, daxis2, combined, nb_epochs=30, batch_size=128, global_batch_size=128,latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
#     #X_train, epoch_disc_loss, epoch_gen_loss = 

#     #print('check100')

#     #fake, energy, ang, ecal
#     real_batch_loss_1, real_batch_loss_2, real_batch_loss_3, real_batch_loss_4, \
#     fake_batch_loss_1, fake_batch_loss_2, fake_batch_loss_3, fake_batch_loss_4 = strategy.run(Discriminator_Train_steps, args=(optimizer_discriminator, optimizer_generator, \
#                     discriminator, generator, dataset, nEvents, WeightsDir, pklfile, \
#                     Trainfiles, nb_train_batches, daxis, daxis2, combined, \
#                     nb_epochs, batch_size, global_batch_size, latent_size, loss_weights, lr, rho, decay, g_weights, d_weights, xscale, xpower, \
#                     angscale, angtype, yscale, thresh, analyse, resultfile, energies, dformat, particle, verbose, \
#                     warm, prev_gweights, prev_dweights))

    
#     real_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_1, axis=None)
#     real_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_2, axis=None)
#     real_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_3, axis=None)
#     real_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss_4, axis=None)
#     real_batch_loss = [real_batch_loss_1, real_batch_loss_2, real_batch_loss_3, real_batch_loss_4]

#     fake_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_1, axis=None)
#     fake_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_2, axis=None)
#     fake_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_3, axis=None)
#     fake_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss_4, axis=None)
#     fake_batch_loss = [fake_batch_loss_1, fake_batch_loss_2, fake_batch_loss_3, fake_batch_loss_4]

#     return real_batch_loss, fake_batch_loss 
    #strategy.reduce(tf.distribute.ReduceOp.SUM, real_batch_loss, axis=None), strategy.reduce(tf.distribute.ReduceOp.SUM, fake_batch_loss, axis=None)

@tf.function
def distributed_generator_train_step(optimizer_discriminator, optimizer_generator, strategy, discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Trainfiles, nb_train_batches, daxis, daxis2, combined, nb_epochs=30, batch_size=128, global_batch_size=128,latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
    #X_train, epoch_disc_loss, epoch_gen_loss = 

    #print('check100')

    gen_losses = []

    #fake, energy, ang, ecal
    gen_batch_loss_1, gen_batch_loss_2, gen_batch_loss_3, gen_batch_loss_4, \
    gen_batch_loss_5, gen_batch_loss_6, gen_batch_loss_7, gen_batch_loss_8 = strategy.run(Generator_Train_steps, args=(optimizer_discriminator, optimizer_generator, \
                    discriminator, generator, dataset, nEvents, WeightsDir, pklfile, \
                    Trainfiles, nb_train_batches, daxis, daxis2, combined, \
                    nb_epochs, batch_size, global_batch_size, latent_size, loss_weights, lr, rho, decay, g_weights, d_weights, xscale, xpower, \
                    angscale, angtype, yscale, thresh, analyse, resultfile, energies, dformat, particle, verbose, \
                    warm, prev_gweights, prev_dweights))

    
    gen_batch_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_1, axis=None)
    gen_batch_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_2, axis=None)
    gen_batch_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_3, axis=None)
    gen_batch_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_4, axis=None)
    gen_batch_loss = [gen_batch_loss_1, gen_batch_loss_2, gen_batch_loss_3, gen_batch_loss_4]

    gen_losses.append(gen_batch_loss)

    gen_batch_loss_5 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_5, axis=None)
    gen_batch_loss_6 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_6, axis=None)
    gen_batch_loss_7 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_7, axis=None)
    gen_batch_loss_8 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_batch_loss_8, axis=None)
    gen_batch_loss = [gen_batch_loss_5, gen_batch_loss_6, gen_batch_loss_7, gen_batch_loss_8]

    gen_losses.append(gen_batch_loss)

    return gen_losses

# @tf.function
# def distributed_test_step(optimizer_discriminator, optimizer_generator, strategy, discriminator, generator, dataset, nEvents, WeightsDir, pklfile, Testfiles, nb_test_batches, daxis, daxis2, combined, nb_epochs=30, batch_size=128, global_batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], dformat='channels_last', particle='Ele', verbose=False, warm=False, prev_gweights='', prev_dweights=''):
#     disc_test_loss_1, disc_test_loss_2, disc_test_loss_3, disc_test_loss_4, \
#     gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4 = strategy.run(Test_steps, args=(optimizer_discriminator, optimizer_generator, \
#         discriminator, generator, dataset, nEvents, WeightsDir, pklfile, \
#         Testfiles, nb_test_batches, daxis, daxis2, combined, \
#         nb_epochs, batch_size, global_batch_size, latent_size, loss_weights, lr, rho, decay, g_weights, d_weights, xscale, xpower, \
#         angscale, angtype, yscale, thresh, analyse, resultfile, energies, dformat, particle, verbose, \
#         warm, prev_gweights, prev_dweights,))

#     disc_test_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_1, axis=None)
#     disc_test_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_2, axis=None)
#     disc_test_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_3, axis=None)
#     disc_test_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, disc_test_loss_4, axis=None)
#     disc_test_loss = [disc_test_loss_1, disc_test_loss_2, disc_test_loss_3, disc_test_loss_4]

#     gen_test_loss_1 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_1, axis=None)
#     gen_test_loss_2 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_2, axis=None)
#     gen_test_loss_3 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_3, axis=None)
#     gen_test_loss_4 = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_test_loss_4, axis=None)
#     gen_test_loss = [gen_test_loss_1, gen_test_loss_2, gen_test_loss_3, gen_test_loss_4]

    
#     return disc_test_loss, gen_test_loss


if __name__ == '__main__':
    main()
