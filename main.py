import argparse
import time
import os
import sys
import numpy as np
import random
import h5py 
import math
import PIL
import importlib
import analysis.utils.GANutils as gan
import tensorflow as tf
import horovod as hvd #AngleGAN
import horovod.tensorflow as hvd #pgan
import horovod.keras as hvd

from rectified_adam import RAdamOptimizer
from networks.loss import forward_simultaneous, forward_generator, forward_discriminator
from networks.ops import num_filters
from collections import defaultdict
from six.moves import range
from dataset import NumpyPathDataset
from utils import count_parameters, image_grid, parse_tuple
from PIL import Image


import keras
import keras.backend as K
from keras.layers import Input
from keras.models  import Model
from keras.callbacks import CallbackList
from keras.optimizers import Adam, RMSprop
kv2 = keras.__version__.startswith('2')     # written in the tf1 file as a workaround for a keras 2 bug in Gan3DTrainingAngle()

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
os.environ['LD_LIBRARY_PATH'] = os.getcwd()
#import nvgpu


################################### EM TODO! ####################################
    # TIE TOGETHER PGAN AND ANGLEGAN!
    # REMOVE KERAS?
    # UNDERSTAND PGAN + REVIEW PGAN PAPER
    # DATA PROCESSING FUNCTION + GETDATAANGLE
    # INTEGRATE GAN3DTRAINANGLE() WITH PGAN BETTER



# Pasted from GetDataAngle() in AngleTrain
# get data for training - returns X, Y, ang, ecal; called in Gan3DTrainAngle
def GetDataAngle(datafile, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4):
    print ('Loading Data from .....', datafile)
    f = h5py.File(datafile,'r')            # load data into f variable
    ang = np.array(f.get(angtype))         # ang is an array of angle data from f
    X = np.array(f.get('ECAL'))* xscale    # x is an array of scaled ecal data from f
    Y = np.array(f.get('energy'))/yscale   # y is an array of scaled energy data from f
    X[X < thresh] = 0            # when X values are less than the threshold, they are reset to 0
    
    # set X, Y, and ang o float 32 datatypes
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ang = ang.astype(np.float32)
    
    X = np.expand_dims(X, axis=-1)         # insert a new axis at the beginning for X
    
    # check data format and sum along axis
    if K.image_data_format() !='channels_last':
       X = np.moveaxis(X, -1, 1)
       ecal = np.sum(X, axis=(2, 3, 4))
    else:
       ecal = np.sum(X, axis=(1, 2, 3))
     
    # X ^ xpower
    if xpower !=1.:
        X = np.power(X, xpower)
        
    # x=ecal data; y=energy data; ecal=summed ecal(x); ang=angle data
    return X, Y, ang, ecal

# Takes 51x51x25 image array --> 64x64x32 image array (so it is multiples of 2)
def resize(image_array):
    og_dims = [51, 51, 25]
    desired_dims = [64, 64, 32]
    img = PIL.Image.fromarray(image_array, mode=None)  #51x51x25 image
    
    #potential resizing option
    resized_img = tf.image.resize(img, desired_dims, method='bicubic', preserve_aspect_ratio=False, antialias=False, name=None)
    #resized_img = tf.image.resize(img, [64, 64, 32], method='lanczos3', preserve_aspect_ratio=False, antialias=False, name=None)
    #resized_img = tf.image.resize(img, [64, 64, 32], method='lanczos5', preserve_aspect_ratio=False, antialias=False, name=None)
    resized_image_array = np.asarray(resized_img)
    
    #scipy.misc.imresize(arr, size, interp='lanczos', mode=None) #deprecated in scipy 1.3?
    
    #generic padding function option
    resized_image_array = np.pad(image_array, ((7,6), (7,6), (3,4)), mode='minimum') # try other padding methods?
        
    return resized_image_array


# TODO! preprocess data (np.arrays), address resolution concerns
def dataset(phase, local_rank, global_size, verbose, final_shape, num_phases, image_channels):
        
    ####################### Pgan/main Dataset block of code ############################
    size = 2 * 2 ** phase
    data_path = os.path.join(args.dataset_path, f'{args.size}x{args.size}/')
    npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0, is_correct_phase=phase >= args.starting_phase)
    
    return npy_data

    # Get DataLoader
    batch_size = max(1, args.base_batch_size // (2 ** (phase - 1)))
    
    if phase >= args.starting_phase:
        assert batch_size * global_size <= args.max_global_batch_size
        if verbose:
            print(f"Using local batch size of {args.batch_size} and global batch size of {args.batch_size * args.global_size}")
                  
    zdim_base = max(1, final_shape[1] // (2 ** (num_phases - 1)))
    base_shape = (image_channels, zdim_base, 4, 4)
    current_shape = [batch_size, image_channels, *[size * 2 ** (phase - 1) for size in base_shape[1:]]]
    
    real_image_input = tf.placeholder(shape=current_shape, dtype=tf.float32)
    real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * .01
    real_label = None

    if real_label is not None:
        real_label = tf.one_hot(real_label, depth=args.num_labels)


# returns optimizers (Adam=default) and gen/disc learning rates, called in run() during phase loop, replaces optimizers block of code in pgan main()
def optimizers():
    if args.optimizer == 'Adam': # pgan default
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=args.g_lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=args.d_lr, beta1=args.beta1, beta2=args.beta2)
    elif args.optimizer == 'RMSProp':
        optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=args.g_lr)
        optimizer_disc = tf.train.RMSPropOptimizer(learning_rate=args.d_lr)
    elif args.optimizer == 'GradientDescent':
        optimizer_gen = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    elif args.optimizer == 'RAdam':     
        optimizer_gen = tf.train.RAdamOptimizer(learning_rate=args.g_lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_disc = tf.train.RAdamOptimizer(learning_rate=args.d_lr, beta1=args.beta1, beta2=args.beta2)
    
    # from lines 169-176 in saraGAN/main
    if args.horovod:
            if args.use_adasum:
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)#, op=hvd.Adasum)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc, op=hvd.Adasum)
            else:
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc)
   
    g_lr = args.g_lr
    d_lr = args.d_lr

    if args.horovod:
        if args.g_scaling == 'sqrt':
            g_lr = g_lr * np.sqrt(hvd.size())
        elif args.g_scaling == 'linear':
            g_lr = g_lr * hvd.size()
        elif args.g_scaling == 'none':
            pass
        else:
            raise ValueError(args.g_scaling)

        if args.d_scaling == 'sqrt':
            d_lr = d_lr * np.sqrt(hvd.size())
        elif args.d_scaling == 'linear':
            d_lr = d_lr * hvd.size()
        elif args.d_scaling == 'none':
            pass
        else:
            raise ValueError(args.d_scaling)

    g_lr = tf.Variable(g_lr, name='g_lr', dtype=tf.float32)
    d_lr = tf.Variable(d_lr, name='d_lr', dtype=tf.float32)
    
    lr_step = tf.Variable(0, name='step', dtype=tf.float32)
    update_step = lr_step.assign_add(1.0)

    with tf.control_dependencies([update_step]):
        update_g_lr = g_lr.assign(g_lr * args.g_annealing)
        update_d_lr = d_lr.assign(d_lr * args.d_annealing)

    return g_lr, d_lr, optimizer_gen, optimizer_disc


def networks():
    with tf.variable_scope('alpha'):
        alpha = tf.Variable(1, name='alpha', dtype=tf.float32)
        # Alpha init
        init_alpha = alpha.assign(1)

        # Specify alpha update op for mixing phase.
        num_steps = args.mixing_nimg // (args.batch_size * args.global_size)
        alpha_update = 1 / num_steps
        # noinspection PyTypeChecker
        update_alpha = alpha.assign(tf.maximum(alpha - alpha_update, 0))

        if args.optim_strategy == 'simultaneous':
            gen_loss, disc_loss, gp_loss, gen_sample = forward_simultaneous(
                args.generator,
                args.discriminator,
                args.real_image_input,
                args.latent_dim,
                alpha,
                args.phase,
                args.num_phases,
                args.base_dim,
                args.base_shape,
                args.activation,
                args.leakiness,
                args.network_size,
                args.loss_fn,
                args.gp_weight
            )
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            g_gradients, g_variables = zip(*args.optimizer_gen.compute_gradients(gen_loss,
                                                                            var_list=gen_vars))
            if args.g_clipping:
                g_gradients, _ = tf.clip_by_global_norm(g_gradients, 1.0)


            d_gradients, d_variables = zip(*args.optimizer_disc.compute_gradients(disc_loss,
                                                                             var_list=disc_vars))
            if args.d_clipping:
                d_gradients, _ = tf.clip_by_global_norm(d_gradients, 1.0)


            g_norms = tf.stack([tf.norm(grad) for grad in g_gradients if grad is not None])
            max_g_norm = tf.reduce_max(g_norms)
            d_norms = tf.stack([tf.norm(grad) for grad in d_gradients if grad is not None])
            max_d_norm = tf.reduce_max(d_norms)


        elif args.optim_strategy == 'alternate':

            disc_loss, gp_loss = forward_discriminator(
                args.generator,
                args.discriminator,
                args.real_image_input,
                args.latent_dim,
                alpha,
                args.phase,
                args.num_phases,
                args.base_dim,
                args.base_shape,
                args.activation,
                args.leakiness,
                args.network_size,
                args.loss_fn,
                args.gp_weight,
                conditioning=args.real_label
            )

            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            d_gradients = args.optimizer_disc.compute_gradients(disc_loss, var_list=disc_vars)
            d_norms = tf.stack([tf.norm(grad) for grad, var in d_gradients if grad is not None])
            max_d_norm = tf.reduce_max(d_norms)

            train_disc = args.optimizer_disc.apply_gradients(d_gradients)

            with tf.control_dependencies([train_disc]):
                gen_sample, gen_loss = forward_generator(
                    args.generator,
                    args.discriminator,
                    args.real_image_input,
                    args.latent_dim,
                    args.alpha,
                    args.phase,
                    args.num_phases,
                    args.base_dim,
                    args.base_shape,
                    args.activation,
                    args.leakiness,
                    args.network_size,
                    args.loss_fn,
                    is_reuse=True
                )

                gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                g_gradients = args.optimizer_gen.compute_gradients(gen_loss, var_list=gen_vars)
                g_norms = tf.stack([tf.norm(grad) for grad, var in g_gradients if grad is not None])
                max_g_norm = tf.reduce_max(g_norms)
                train_gen = args.optimizer_gen.apply_gradients(g_gradients)

        else:
            raise ValueError("Unknown optim strategy ", args.optim_strategy)

        if args.verbose:
            print(f"Generator parameters: {count_parameters('generator')}")
            print(f"Discriminator parameters:: {count_parameters('discriminator')}")

        # train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
        # train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

        ema = tf.train.ExponentialMovingAverage(decay=args.ema_beta)
        ema_op = ema.apply(gen_vars)
        # Transfer EMA values to original variables
        ema_update_weights = tf.group(
            [tf.assign(var, ema.average(var)) for var in gen_vars])

        with tf.name_scope('summaries'):
            # Summaries
            tf.summary.scalar('d_loss', disc_loss)
            tf.summary.scalar('g_loss', gen_loss)
            tf.summary.scalar('gp', tf.reduce_mean(gp_loss))

            for g in zip(g_gradients, g_variables):
                tf.summary.histogram(f'grad_{g[1].name}', g[0])

            for g in zip(d_gradients, d_variables):
                tf.summary.histogram(f'grad_{g[1].name}', g[0])

            # tf.summary.scalar('convergence', tf.reduce_mean(disc_real) - tf.reduce_mean(tf.reduce_mean(disc_fake_d)))

            tf.summary.scalar('max_g_grad_norm', max_g_norm)
            tf.summary.scalar('max_d_grad_norm', max_d_norm)

            real_image_grid = tf.transpose(real_image_input[0], (1, 2, 3, 0))
            shape = real_image_grid.get_shape().as_list()
            grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
            grid_rows = shape[0] // grid_cols
            grid_shape = [grid_rows, grid_cols]
            real_image_grid = image_grid(real_image_grid, grid_shape, image_shape=shape[1:3],
                                         num_channels=shape[-1])

            fake_image_grid = tf.transpose(gen_sample[0], (1, 2, 3, 0))
            fake_image_grid = image_grid(fake_image_grid, grid_shape, image_shape=shape[1:3],
                                         num_channels=shape[-1])

            fake_image_grid = tf.clip_by_value(fake_image_grid, -1, 2)

            tf.summary.image('real_image', real_image_grid)
            tf.summary.image('fake_image', fake_image_grid)

            tf.summary.scalar('fake_image_min', tf.math.reduce_min(gen_sample))
            tf.summary.scalar('fake_image_max', tf.math.reduce_max(gen_sample))

            tf.summary.scalar('real_image_min', tf.math.reduce_min(real_image_input[0]))
            tf.summary.scalar('real_image_max', tf.math.reduce_max(real_image_input[0]))
            tf.summary.scalar('alpha', alpha)

            tf.summary.scalar('g_lr', g_lr)
            tf.summary.scalar('d_lr', d_lr)

            merged_summaries = tf.summary.merge_all()

        # Other ops
        init_op = tf.global_variables_initializer()
        assign_starting_alpha = alpha.assign(args.starting_alpha)
        assign_zero = alpha.assign(0)
        broadcast = hvd.broadcast_global_variables(0)

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            trainable_variable_names = [v.name for v in tf.trainable_variables()]

            if var_list is not None and phase > args.starting_phase:
                print("Restoring variables from:", os.path.join(logdir, f'model_{phase - 1}'))
                var_names = [v.name for v in var_list]
                load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
                saver = tf.train.Saver(load_vars)
                saver.restore(sess, os.path.join(logdir, f'model_{phase - 1}'))
            elif var_list is not None and args.continue_path and phase == args.starting_phase:
                print("Restoring variables from:", args.continue_path)
                var_names = [v.name for v in var_list]
                load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
                saver = tf.train.Saver(load_vars)
                saver.restore(sess, os.path.join(args.continue_path))
            else:
                if verbose:
                     print("Not restoring variables.")
                     print("Variable List Length:", len(var_list))

            var_list = gen_vars + disc_vars

            if phase < args.starting_phase:
                continue

            if phase == args.starting_phase:
                sess.run(assign_starting_alpha)
            else:
                sess.run(init_alpha)

            if verbose:
                print(f"Begin mixing epochs in phase {phase}")
            if args.horovod:
                sess.run(broadcast)

            local_step = 0
            # take_first_snapshot = True

            while True:
                start = time.time()
                if local_step % 2048 == 0 and local_step > 1:
                    if args.horovod:
                        sess.run(broadcast)
                    saver = tf.train.Saver(var_list)
                    if verbose:
                        saver.save(sess, os.path.join(logdir, f'model_{phase}_ckpt_{global_step}'))

                batch_loc = np.random.randint(0, len(npy_data) - batch_size)
                batch_paths = npy_data[batch_loc: batch_loc + batch_size]
                batch = np.stack([np.load(path) for path in batch_paths])
                batch = batch[:, np.newaxis, ...].astype(np.float32) / 1024 - 1

                _, _, summary, d_loss, g_loss = sess.run(
                     [train_gen, train_disc, merged_summaries,
                      disc_loss, gen_loss], feed_dict={real_image_input: batch})
                global_step += batch_size * global_size
                local_step += 1

                end = time.time()
                img_s = global_size * batch_size / (end - start)
                if verbose:

                    if local_step % 32 == 0:
                        writer.add_summary(summary, global_step)
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='img_s', simple_value=img_s)]),
                                           global_step)

                    print(f"Step {global_step:09} \t"
                          f"img/s {img_s:.2f} \t "
                          f"d_loss {d_loss:.4f} \t "
                          f"g_loss {g_loss:.4f} \t "
                          # f"memory {memory_percentage:.4f} % \t"
                          f"alpha {alpha.eval():.2f}")


                if global_step >= ((phase - args.starting_phase)
                                   * (args.mixing_nimg + args.stabilizing_nimg)
                                   + args.mixing_nimg):
                    break

                sess.run(update_alpha)
                sess.run(ema_op)
                sess.run(update_d_lr)
                sess.run(update_g_lr)

                assert alpha.eval() >= 0

            if verbose:
                print(f"Begin stabilizing epochs in phase {phase}")

            sess.run(assign_zero)



# parses and returns arguments/params, called in run()
def get_args():
    global args
    
    parser = argparse.ArgumentParser(description='Arguments')
    
    parser.add_argument('--architecture', type=str, default='AngleArch3dGAN', choices=['AngleArch3dGAN', 'ProgressiveGAN'])
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSProp', 'GradientDescent', 'RAdam'])

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
    parser.add_argument('--g_lr', type=float, default=1e-3, help='Learning rate for the generator')
    parser.add_argument('--d_lr', type=float, default=1e-3, help='Learning rate for the discriminator')
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


# DO WE NEED TO ADDRESS THE CHANNELS_FIRST (CPU) VS CHANNELS_LAST FORMATTING?
# channels format: want channels_first for cpu
def set_format(channel_format):
    global daxis
    if channel_format == 'channels_first':
        print('Setting th channel ordering (NCHW)')
        K.set_image_dim_ordering('th')
        K.set_image_data_format('channels_first')
    else:
        print('Setting tf channel ordering (NHWC)')
        K.set_image_dim_ordering('tf')
        K.set_image_data_format('channels_last')
        
    if K.image_data_format() !='channels_last':
        daxis = (2,3,4)
    else:
       daxis = (1,2,3)
    


# Creates a list of lists, used in Gan3DTrainAngle()
def genbatches(a,n):
    for i in range(0, len(a), n):
        # Create an index range for l of n items:
        yield a[i:i+n]


# Shuffles 4 arrays, used in Gan3DTrainAngle(X_train, Y_train, ecal_train, ang_train)
def randomize(a, b, c, d):
    assert a.shape[0] == b.shape[0]
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    shuffled_c = c[permutation]
    shuffled_d = d[permutation]
    
    return shuffled_a, shuffled_b, shuffled_c, shuffled_d


# histogram count - sums 8 bins btwn [0.05, 0.03, 0.02, 0.0125, 0.008, 0.003, 0]**p
def hist_count(x, p=1):
    bin1 = np.sum(np.where(x>(0.05**p) , 1, 0), axis=daxis)
    bin2 = np.sum(np.where((x<(0.05**p)) & (x>(0.03**p)), 1, 0), axis=daxis)
    bin3 = np.sum(np.where((x<(0.03**p)) & (x>(0.02**p)), 1, 0), axis=daxis)
    bin4 = np.sum(np.where((x<(0.02**p)) & (x>(0.0125**p)), 1, 0), axis=daxis)
    bin5 = np.sum(np.where((x<(0.0125**p)) & (x>(0.008**p)), 1, 0), axis=daxis)
    bin6 = np.sum(np.where((x<(0.008**p)) & (x>(0.003**p)), 1, 0), axis=daxis)
    bin7 = np.sum(np.where((x<(0.003**p)) & (x>0.), 1, 0), axis=daxis)
    bin8 = np.sum(np.where(x==0, 1, 0), axis=daxis)
    bins = np.concatenate([bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8], axis=1)
    bins[np.where(bins==0)]=1
    return bins


# Pasted from Gan3DTrainAngle() in AngleTrain
# Training Function - build & compile discriminator, build & compile generator, run the generator and discriminator, unused callback list functions, read TrainFiles & TestFiles,
#                     run through epochs, train the generator & discriminator, collect discriminator losses, collect generator losses, test, save weights every epoch
def Gan3DTrainAngle(discriminator, generator, opt, datapath, nEvents, WeightsDir, pklfile, global_batch_size, nb_epochs=30, batch_size=128, latent_size=200, loss_weights=[3, 0.1, 25, 0.1, 0.1], lr=0.001, rho=0.9, decay=0.0, g_weights='params_generator_epoch_', d_weights='params_discriminator_epoch_', xscale=1, xpower=1, angscale=1, angtype='theta', yscale=100, thresh=1e-4, analyse=False, resultfile="", energies=[], warmup_epochs=0):
    start_init = time.time()
    verbose = False    
    particle='Ele'
    f = [0.9, 0.1]
    loss_ftn = hist_count
    
    if hvd.rank()==0:
        print('[INFO] Building discriminator')
    #discriminator.summary()
    discriminator.compile(
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )

    # build the generator
    if hvd.rank()==0:
        print('[INFO] Building generator')
    #generator.summary()
    generator.compile(
        optimizer=opt,
        loss='binary_crossentropy'
    )
 
    # build combined Model
    # generator: latent vector --> fake image
    latent = Input(shape=(latent_size, ), name='combined_z')   # random latent vector = generator input
    fake_image = generator(latent)     # fake image = generator output
    # discriminator: fake image --> fake, aux, ang, ecal, add_loss
    discriminator.trainable = False
    fake, aux, ang, ecal, add_loss= discriminator(fake_image)
    combined = Model(
        input=[latent],
        output=[fake, aux, ang, ecal, add_loss],
        name='combined_model'
    )
    combined.compile(
        #optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        optimizer=opt,
        loss=['binary_crossentropy', 'mean_absolute_percentage_error', 'mae', 'mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
        loss_weights=loss_weights
    )
    if kv2: 
        discriminator.trainable = True #workaround for keras 2 bug
        
    gcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    dcb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    ccb = CallbackList( \
        callbacks=[ \
        hvd.callbacks.BroadcastGlobalVariablesCallback(0), \
        hvd.callbacks.MetricAverageCallback(), \
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=warmup_epochs, verbose=1), \
        hvd.callbacks.LearningRateScheduleCallback(start_epoch=warmup_epochs, end_epoch=nb_epochs, multiplier=1.), \
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1) \
        ])

    gcb.set_model( generator )
    dcb.set_model( discriminator )
    ccb.set_model( combined )

    gcb.on_train_begin()
    dcb.on_train_begin()
    ccb.on_train_begin()

    # Getting Data: gan/DivideFiles() splits ~10 files into train and test files using [0.9, 0.1] probability
    Trainfiles, Testfiles = gan.DivideFiles(datapath, datasetnames=["ECAL"], Particles =[particle])
    if hvd.rank()==0:
        print(Trainfiles)
        print(Testfiles)
    nb_Test = int(nEvents * f[1]) # The number of test files calculated from fraction of nEvents
    nb_Train = int(nEvents * f[0]) # The number of train files calculated from fraction of nEvents
    
    # Bug check for reading the test file in
    if len(Testfiles) == 0:
       print('Error reading the Testfiles. The enumerated list will show up as empty. Check the GANutils.py file in 3Dgan/keras/analysis/utils.')
       
    # Read test data into an array with the size of nb_Test
    for index, dtest in enumerate(Testfiles):
       if index == 0:
           X_test, Y_test, ang_test, ecal_test = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
       else:
           if X_test.shape[0] < nb_Test:
              X_temp, Y_temp, ang_temp,  ecal_temp = GetDataAngle(dtest, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
              X_test = np.concatenate((X_test, X_temp))
              Y_test = np.concatenate((Y_test, Y_temp))
              ang_test = np.concatenate((ang_test, ang_temp))
              ecal_test = np.concatenate((ecal_test, ecal_temp))
    if X_test.shape[0] > nb_Test:
        X_test, Y_test, ang_test, ecal_test = X_test[:nb_Test], Y_test[:nb_Test], ang_test[:nb_Test], ecal_test[:nb_Test]
    else:
        nb_Test = X_test.shape[0] # the nb_test may be different if total events are less than nEvents
    
    # Read train data into an array, make sure it is the same length as nb_Train (The number of train files calculated from fraction of nEvents)
    for index, dtrain in enumerate(Trainfiles):
        if index == 0:
            X_train, Y_train, ang_train, ecal_train = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
        else:
            X_temp, Y_temp, ang_temp, ecal_temp = GetDataAngle(dtrain, xscale=xscale, xpower=xpower, angscale=angscale, angtype=angtype, thresh=thresh)
            X_train = np.concatenate((X_train, X_temp))
            Y_train = np.concatenate((Y_train, Y_temp))
            ang_train = np.concatenate((ang_train, ang_temp))
            ecal_train = np.concatenate((ecal_train, ecal_temp))

    nb_train = X_train.shape[0]    # Total events in training files
    total_batches = nb_train / global_batch_size
    
    if hvd.rank()==0:
        print('Total Training batches = {} with {} events'.format(total_batches, nb_train))

    if hvd.rank()==0:           # will throw an error if the number of epochs is not large enough
       print('Test Data loaded of shapes:')
       print(X_test.shape)
       print(Y_test.shape)
       print('*************************************************************************************')
       print('Ang varies from {} to {} with mean {}'.format(np.amin(ang_test), np.amax(ang_test), np.mean(ang_test)))
       print('Cell varies from {} to {} with mean {}'.format(np.amin(X_test[X_test>0]), np.amax(X_test[X_test>0]), np.mean(X_test[X_test>0])))
       
       if analyse:
          var = gan.sortEnergy(X_test, Y_test, ang_test, ecal_test, energies)
       train_history = defaultdict(list)
       test_history = defaultdict(list)
       analysis_history = defaultdict(list)
       init_time = time.time()- start_init
       print('Initialization time is {} seconds'.format(init_time))
    
    # run through epochs
    for epoch in range(nb_epochs):
        epoch_start = time.time()
        if hvd.rank()==0:
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
 
        epoch_gen_loss = []
        epoch_disc_loss = []
        randomize(X_train, Y_train, ecal_train, ang_train)

        epoch_gen_loss = []
        epoch_disc_loss = []
        
        image_batches = genbatches(X_train, batch_size)    # creates len(X_train) index ranges for len(batch_size) # of items
        energy_batches = genbatches(Y_train, batch_size)   # creates len(Y_train) index ranges for len(batch_size) # of items
        ecal_batches = genbatches(ecal_train, batch_size)  # creates len(ecal_train) index ranges for len(batch_size) # of items
        ang_batches = genbatches(ang_train, batch_size)    # creates len(ang_train) index ranges for len(batch_size) # of items
        
        # go through batches: train the generator and discriminator
        for index in range(int(total_batches)):
            start = time.time()         
            image_batch = next(image_batches) 
            energy_batch = next(energy_batches)
            ecal_batch = next(ecal_batches)
            ang_batch = next(ang_batches)
            add_loss_batch = np.expand_dims(loss_ftn(image_batch, xpower), axis=-1)
            noise = np.random.normal(0, 1, (batch_size, latent_size-2))
            generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1)
            generated_images = generator.predict(generator_ip, verbose=0)
  
            # collect the loss of the discriminator with real and fake images
            real_batch_loss = discriminator.train_on_batch(image_batch, [gan.BitFlip(np.ones(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])
            fake_batch_loss = discriminator.train_on_batch(generated_images, [gan.BitFlip(np.zeros(batch_size)), energy_batch, ang_batch, ecal_batch, add_loss_batch])

            # if ecal sum has 100% loss then end the training
            if fake_batch_loss[4] == 100.0 and index >10:
                if hvd.rank()==0:
                    print("Empty image with Ecal loss equal to 100.0 for {} batch".format(index))
                    generator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(g_weights), overwrite=True)
                    discriminator.save_weights(WeightsDir + '/{0}eee.hdf5'.format(d_weights), overwrite=True)
                    print ('real_batch_loss', real_batch_loss)
                    print ('fake_batch_loss', fake_batch_loss)
                sys.exit()
            epoch_disc_loss.append([
                (a + b) / 2 for a, b in zip(real_batch_loss, fake_batch_loss)
            ])
            trick = np.ones(batch_size)
            
            # collect generator losses in array
            gen_losses = []
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size-1))
                generator_ip = np.concatenate((energy_batch.reshape(-1, 1), ang_batch.reshape(-1, 1), noise), axis=1) # sampled angle same as g4 theta
                gen_losses.append(combined.train_on_batch(
                    [generator_ip],
                    [trick, energy_batch.reshape(-1, 1), ang_batch, ecal_batch, add_loss_batch]))
            generator_loss = [(a + b) / 2 for a, b in zip(*gen_losses)]
            epoch_gen_loss.append(generator_loss)
            #print ('generator_loss', generator_loss)
            index +=1

            # Used at design time for debugging
            #print('real_batch_loss', real_batch_loss)
            #print ('fake_batch_loss', fake_batch_loss)
            #disc_out = discriminator.predict(image_batch)
            #print('disc_out')
            #print(np.transpose(disc_out[4][:5].astype(int)))
            #print('add_loss_batch')
            #print(np.transpose(add_loss_batch[:5]))

        # Testing  
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        if hvd.rank()==0:
            if analyse:
                result = gan.OptAnalysisShort(var, generated_images, energies)
                print('Analysing............')
                analysis_history['total'].append(result[0])
                analysis_history['energy'].append(result[1])
                analysis_history['moment'].append(result[2])
                analysis_history['angle'].append(result[3])
                print('Result = ', result)
                pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))

            print('{0:<20s} | {1:6s} | {2:12s} | {3:12s}| {4:5s} | {5:8s} | {6:8s}'.format('component', *discriminator.metrics_names))
            print('-' * 65)
            ROW_FMT = '{0:<20s} | {1:<4.2f} | {2:<10.2f} | {3:<10.2f}| {4:<10.2f} | {5:<10.2f}| {6:<10.2f}'
            print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))

            # save weights every epoch
            generator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(g_weights, epoch),
                               overwrite=True)
            discriminator.save_weights(WeightsDir + '/{0}{1:03d}.hdf5'.format(d_weights, epoch),
                                   overwrite=True)
        
            epoch_time = time.time()-test_start
            pickle.dump({'train': train_history}, open(pklfile, 'wb'))
            
            
 # I made a configure function to try and organize the code, I think we could have it set up the config for run(config), but I don't really understand how to set this stuff up
def configure():
    ######## ANGLEGAN ###########
    # configure the session
    config = tf.ConfigProto(log_device_placement=True)
    config.intra_op_parallelism_threads = args.intraop
    config.inter_op_parallelism_threads = args.interop
    os.environ['KMP_BLOCKTIME'] = str(1)
    os.environ['KMP_SETTINGS'] = str(1)
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact'
    # os.environ['KMP_AFFINITY'] = 'balanced'
    # os.environ['OMP_NUM_THREADS'] = str(params.intraop)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)
    K.set_session(tf.Session(config=config))
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    ############ PGAN (main.py/from if __name__ == '__main__')##############
    
    if args.horovod:
        hvd.init()
        np.random.seed(args.seed + hvd.rank())
        tf.random.set_random_seed(args.seed + hvd.rank())
        random.seed(args.seed + hvd.rank())

        print(f"Rank {hvd.rank()}:{hvd.local_rank()} reporting!")

    else:
        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)
        random.seed(args.seed)
    
    if 'OMP_NUM_THREADS' not in os.environ:
        print("Warning: OMP_NUM_THREADS not set. Setting it to 1.")
        os.environ['OMP_NUM_THREADS'] = str(1)

    gopts = tf.GraphOptions(place_pruned_graph=True)
    config = tf.ConfigProto(graph_options=gopts, allow_soft_placement=True)
    # config = tf.ConfigProto()
    
    if args.gpu:
        config.gpu_options.allow_growth = True
        # config.inter_op_parallelism_threads = 1
        #config.gpu_options.per_process_gpu_memory_fraction = 0.96
        if args.horovod:
            config.gpu_options.visible_device_list = str(hvd.local_rank())

    else:
        config = tf.ConfigProto(graph_options=gopts,
                                intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                                inter_op_parallelism_threads=args.num_inter_ops,
                                allow_soft_placement=True,
                                device_count={'CPU': int(os.environ['OMP_NUM_THREADS'])})

       
# I made a function to organize horovod setup, I don't really understand how it works though so we may need to add stuff
def setup_horovod():
    hvd.init()

    
# calls: dataset(), optimizers(), builds generator and discriminator
# calls get_args(), sets the channels, initializes horovod (similar to main() in AngleTrain and saraGAN/main())
def run(config):
    # parse and return global arguments (params)
    get_args()
    
    # Get the discriminator and generator from the architecture of choice
    if args.architecture == 'AngleArch3dGAN':
        discriminator = importlib.import_module(f'AngleArch3dGAN.py').discriminator
        generator = importlib.import_module(f'AngleArch3dGAN.py').generator 
    elif args.architecture == 'ProgressiveGAN':
        discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
        generator = importlib.import_module(f'networks.{args.architecture}.generator').generator
      
        
    ########################## IMPLEMENT ANGLEGAN CODE ########################
    
    # sets channels format (want channels_first for cpu?)
    channel_format = args.channel_format
    set_format(channel_format)
    
    # configure the session
    configure()
    
    # if we are running on horovod, call the function to set stuff up
    if args.horovod:
        setup_horovod()
        
    global_batch_size = args.batch_size * hvd.size()
    print("Global batch size is: {0} / batch size is: {1}".format(args.global_batch_size, args.batch_size))
    
    # Building AngleGAN discriminator and generator
    gan.safe_mkdir(weightdir)
    d=discriminator(xpower)
    g=generator(latent_size)
    
    # train the generator and discriminator with the Gan3DTrainANgle() function
    Gan3DTrainAngle(d, g, opt, datapath, nEvents, weightdir, pklfile, global_batch_size=global_batch_size, nb_epochs=nb_epochs, batch_size=batch_size,
                    latent_size=latent_size, loss_weights=loss_weights, xscale = xscale, xpower=xpower, angscale=ascale,
                    yscale=yscale, thresh=thresh, angtype=angtype, analyse=analyse, resultfile=resultfile,
                    energies=energies, warmup_epochs=warmup_epochs)
    
    
    ######################### saraGAN/main.py code ####################### 
    
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

    
    # ------------------------------------------------------------------------
    # Phasing Loop
    #-------------------------------------------------------------------------

    for phase in range(1, num_phases + 1):
        
        tf.reset_default_graph()
        
        # call dataset() -- replaces dataset block of code in pgan main()
        npy_data = dataset(phase, local_rank, global_size, verbose, final_shape, num_phases, image_channels) 
        
        # call optimizers() -- replaces block of code in pgan main()
        # get the optimizers specified in the parameters
        g_lr, d_lr, optimizer_gen, optimizer_disc = optimizers()
        
        # call networks() -- replaces networks block of code in pgan main()
        networks()
        
        
# calls run()
def main():   
    # run the main code!
    config = configure()
    run(config)  

# calls main()
if __name__ == "__main__":       
    main()