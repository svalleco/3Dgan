import argparse
import horovod as hvd
from dataset import NumpyPathDataset
import os
import time
import tensorflow as tf
import sys
import importlib
import numpy as np
import random
from metrics import (calculate_fid_given_batch_volumes, get_swd_for_volumes,
                     get_normalized_root_mse, get_mean_squared_error, get_psnr, get_ssim)
from utils import count_parameters, image_grid, parse_tuple, MPMap
# from mpi4py import MPI
from rectified_adam import RAdamOptimizer
from networks.loss import forward_simultaneous, forward_generator, forward_discriminator
import psutil
from networks.ops import num_filters
from tensorflow.data.experimental import AUTOTUNE
#import nvgpu

# DO WE NEED TO ADDRESS THE CHANNELS_FIRST (CPU) VS CHANNELS_LAST FORMATTING?

# TODO! preprocess data (np.arrays), address resolution concerns
def dataset():
    """TODO: Docstring for dataset.

    :function: TODO
    :returns: Return NumpyDataset

    """
    data_path = os.path.join(args.dataset_path, f'{args.size}x{args.size}/')
    npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0, is_correct_phase=phase >= args.starting_phase)
    return npy_data

# returns optimizers (Adam=default)
def optimizers():
    """TODO: Docstring for optimizers.

    :function: TODO
    :returns: TODO

    """
    if args.optimizer == 'Adam':
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=args.g_lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=args.d_lr, beta1=args.beta1, beta2=args.beta2)
    elif args.optimizer == 'RMSProp':
        optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=args.g_lr)
        optimizer_disc = tf.train.RMSPropOptimizer(learning_rate=args.d_lr)
    elif args.optimizer == 'GradientDescent':
        optimizer_gen = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    elif args.optimizer == '':     #not sure what this means!
        optimizer_gen = RAdamOptimizer(learning_rate=args.g_lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_disc = RAdamOptimizer(learning_rate=args.d_lr, beta1=args.beta1, beta2=args.beta2)


def get_args():
    global args
    
    parser = argparse.ArgumentParser(description='Arguments')
    
    parser.add_argument('--architecture', type=str, default='AngleArch3dGAN', choices=['AngleArch3dGAN', 'ProgressiveGAN'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSProp', 'GradientDescent', ''])

    # AngleArch3dGAN Arguments (defaults = caltech values)
    parser.add_argument('--nbepochs', action='store', type=int, default=60, help='Number of epochs to train for.')
    parser.add_argument('--batchsize', action='store', type=int, default=128, help='batch size per update')
    parser.add_argument('--latentsize', action='store', type=int, default=256, help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--datapath', action='store', type=str, default='/data/shared/gkhattak/*Measured3ThetaEscan/*.h5', help='HDF5 files to train from.')
    parser.add_argument('--nbEvents', action='store', type=int, default=200000, help='Total Number of events used for Training')
    parser.add_argument('--verbose', action='store_true', help='Whether or not to use a progress bar')
    parser.add_argument('--weightsdir', action='store', type=str, default='weights/3dgan_weights', help='Directory to store weights.')
    parser.add_argument('--pklfile', action='store', type=str, default='results/3dgan_history.pkl', help='Pickle file to store losses.')
    parser.add_argument('--xscale', action='store', type=int, default=1, help='Multiplication factor for ecal deposition')
    parser.add_argument('--xpower', action='store', type=float, default=0.85, help='pre processing of cell energies by raising to a power')
    parser.add_argument('--yscale', action='store', type=int, default=100, help='Division Factor for Primary Energy.')
    parser.add_argument('--ascale', action='store', type=int, default=1, help='Multiplication factor for angle input')
    parser.add_argument('--resultfile', action='store', type=str, default='results/3dgan_analysis.pkl', help='File to save losses.')
    parser.add_argument('--analyse', action='store_true', default=False, help='Whether or not to perform analysis')
    parser.add_argument('--energies', action='store', type=int, default=[0, 110, 150, 190], help='Energy bins for analysis')
    parser.add_argument('--lossweights', action='store', type=int, default=[3, 0.1, 25, 0.1, 0.1], help='loss weights =[gen_weight, aux_weight, ang_weight, ecal_weight, add loss weight]')
    parser.add_argument('--thresh', action='store', type=int, default=0, help='Threshold for cell energies')
    parser.add_argument('--angtype', action='store', type=str, default='mtheta', help='Angle to use for Training. It can be theta, mtheta or eta')
    parser.add_argument('--learningRate', '-lr', action='store', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--optimizer', action='store', type=str, default='RMSprop', help='Keras Optimizer to use.')
    parser.add_argument('--intraop', action='store', type=int, default=9, help='Sets onfig.intra_op_parallelism_threads and OMP_NUM_THREADS')
    parser.add_argument('--interop', action='store', type=int, default=1, help='Sets config.inter_op_parallelism_threads')
    parser.add_argument('--warmupepochs', action='store', type=int, default=5, help='No wawrmup epochs')
    parser.add_argument('--channel_format', action='store', type=str, default='channels_first', help='NCHW vs NHWC')
    parser.add_argument('--analysis', action='store', type=bool, default=False, help='Calculate optimisation function')
    
    # ProgressiveGAN Arguments
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--dataset_size', type=int, default=4)
    parser.add_argument('final_shape', type=str, help="'(c, z, y, x)', e.g. '(1, 64, 128, 128)'")
    parser.add_argument('--starting_phase', type=int, default=None, required=True)
    parser.add_argument('--ending_phase', type=int, default=None, required=True)
    parser.add_argument('--latent_dim', type=int, default=None, required=True)
    parser.add_argument('--network_size', default=None, choices=['xxs', 'xs', 's', 'm', 'l', 'xl', 'xxl'], required=True)
    parser.add_argument('--scratch_path', type=str, default=None, required=True)
    parser.add_argument('--base_batch_size', type=int, default=256, help='batch size used in phase 1')
    parser.add_argument('--max_global_batch_size', type=int, default=256)
    parser.add_argument('--mixing_nimg', type=int, default=2 ** 19)
    parser.add_argument('--stabilizing_nimg', type=int, default=2 ** 19)
    parser.add_argument('--g_lr', type=float, default=1e-3)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--loss_fn', default='logistic', choices=['logistic', 'wgan'])
    parser.add_argument('--gp_weight', type=float, default=1)
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horovod', default=False, action='store_true')
    parser.add_argument('--calc_metrics', default=False, action='store_true')
    parser.add_argument('--g_annealing', default=1, type=float, help='generator annealing rate, 1 -> no annealing.')
    parser.add_argument('--d_annealing', default=1, type=float, help='discriminator annealing rate, 1 -> no annealing.')
    parser.add_argument('--num_metric_samples', type=int, default=512)
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--ema_beta', type=float, default=0.99)
    parser.add_argument('--d_scaling', default='none', choices=['linear', 'sqrt', 'none'], help='How to scale discriminator learning rate with horovod size.')
    parser.add_argument('--g_scaling', default='none', choices=['linear', 'sqrt', 'none'], help='How to scale generator learning rate with horovod size.')
    parser.add_argument('--continue_path', default=None, type=str)
    parser.add_argument('--starting_alpha', default=1, type=float)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--use_adasum', default=False, action='store_true')
    parser.add_argument('--optim_strategy', default='simultaneous', choices=['simultaneous', 'alternate'])
    parser.add_argument('--num_inter_ops', default=4, type=int)
    parser.add_argument('--num_labels', default=None, type=int)
    parser.add_argument('--g_clipping', default=False, type=bool)
    parser.add_argument('--d_clipping', default=False, type=bool)
    parser.add_argument('--load_phase', default=None, type=int)
    
    args = parser.parse_args()
    return args


# calls: dataset(), optimizers(), builds generator and discriminator
def run(config):
    """TODO: Docstring for run.
    
    The main function, training done here 

    :a: TODO
    :returns: TODO

    """
    global local_rank
    
    # Get the discriminator and generator from the architecture of choice
    if args.architecture == 'AngleArch3dGAN':
        discriminator = importlib.import_module(f'AngleArch3dGAN.py').discriminator
        generator = importlib.import_module(f'AngleArch3dGAN.py').generator 
    elif args.architecture == 'ProgressiveGAN':
        discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
        generator = importlib.import_module(f'networks.{args.architecture}.generator').generator
    
    # horovod settings
    if args.horovod:
        verbose = hvd.rank() == 0
        global_size = hvd.size()
        global_rank = hvd.rank()
        local_rank = hvd.local_rank()
    else:
        verbose = True
        global_size = 1
        global_rank = 0
        local_rank = 0

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', args.architecture, timestamp)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    if verbose:
        writer = tf.summary.FileWriter(logdir=logdir)
        print("Arguments passed:")
        print(args)
        print(f"Saving files to {logdir}")

    else:
        pass

    
    final_shape = parse_tuple(args.final_shape)
    image_channels = final_shape[0]
    final_resolution = final_shape[-1]
    num_phases = int(np.log2(final_resolution) - 1)
    base_dim = num_filters(-num_phases + 1, num_phases, size=args.network_size)

    var_list = list()
    global_step = 0

    
    # -------------
    # Phasing Loop
    #-------------

    for phase in range(1, num_phases + 1):
        
        tf.reset_default_graph()
        npy_data = dataset() 
        
        
# calls get_args(), run(config), initializes horovod 
def main():
    get_args()
    config = ''
    if horovod:
        hvd.init()
    run(config)
        

# calls main()
if __name__ == "__main__":       
    main()