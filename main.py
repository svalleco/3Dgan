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

# I moved extra functions to keras/analysis/utils/GANutils.py
from GANutils import hist_count, randomize, genbatches

# used for resizing -- I don't know if this will be too slow
from scipy.ndimage import zoom

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
os.environ['LD_LIBRARY_PATH'] = os.getcwd()
#import nvgpu


# Pasted from GetDataAngle() in AngleTrain
# get data for training - returns img3d, e_p, ang, ecal; called in Gan3DTrainAngle
def GetDataAngle(datafile, img3dscale =1, img3dpower=1, e_pscale = 100, angscale=1, angtype='theta', thresh=1e-4):
    print ('Loading Data from .....', datafile)
    f = h5py.File(datafile,'r')                    # load data into f variable
    ang = np.array(f.get(angtype))                 # ang is an array of angle data from f, one value is concatenated onto the latent vector
    3d_imgs = np.array(f.get('ECAL'))* img3dscale    # img3d is a 3d array, cut from the cylinder that the calorimeter produces (has 25 layers along z-axis)
    e_p = np.array(f.get('energy'))/e_pscale       # e_p is an array of scaled energy data from f, one value is concatenated onto the latent vector
    3d_imgs[3d_imgs < thresh] = 0        # when 3d_imgs values are less than the threshold, they are reset to 0
    
    # set 3d_imgs, e_p, and ang as float 32 datatypes
    3d_imgs = 3d_imgs.astype(np.float32)
    e_p = e_p.astype(np.float32)
    ang = ang.astype(np.float32)
    
    3d_imgs = np.expand_dims(3d_imgs, axis=-1)         # insert a new axis at the beginning for 3d_imgs
    
    # sum along axis
    ecal = np.sum(3d_imgs, axis=(1, 2, 3))    # summed 3d_imgs data, used for training the discriminator
     
    # 3d_imgs ^ 3d_imgspower
    if 3d_imgspower !=1.:
        3d_imgs = np.power(3d_imgs, 3d_imgspower)
            
    # 3d_imgs=ecal data; e_p=energy data; ecal=summed 3d_imgs (used to train the discriminator); ang=angle data
    return 3d_imgs, e_p, ang, ecal


# Takes [5000x51x51x25] image array and size parameter --> [sizexsizex25]
def resize(3d_imgs, size):
    resized_3d_imgs = np.array([]) # create an array to hold all 5000 resized 3d_imgs
    
    for num_img in np.arange(5000):     #index through the 5000 3d images packed in
        3d_img = 3d_imgs[num_img, :, :, :, 0]    #[5000,51,51,25,1] --> an individual [51,51,25] 3d image
        
        resized_3d_img = np.zeros(size, size, 25)   # create an empty 3d_image to store each of the 25 adjusted 2d images when we are done adjusting their size
        
        for cal_layer in np.arange(25):    #index through the 25 calorimeter layers
            2d_img = 3d_img[:, :, cal_layer]
            
            if size < 64:
                2d_img = PIL.Image.fromarray(2d_img, mode=None)  #51x51x25 image
                resized_2d_img = tf.image.resize(img, [size, size, 25], method='neares', preserve_aspect_ratio=False, antialias=False, name=None)
                #resized_img = tf.image.resize(img, [size, size, 25], method='lanczos3', preserve_aspect_ratio=False, antialias=False, name=None)
                #resized_img = tf.image.resize(img, [size, size, 25], method='lanczos5', preserve_aspect_ratio=False, antialias=False, name=None)
                #resized_img = tf.image.resize(img, [size, size, 25], method='bilinear', preserve_aspect_ratio=False, antialias=False, name=None)
                #resized_img = tf.image.resize(img, [size, size, 25], method='bicubic', preserve_aspect_ratio=False, antialias=False, name=None)
                resized_2d_img = np.asarray(resized_img)
            
            elif size == 64:    
                resized_2d_img = np.pad(2d_img, ((7,6), (7,6), (0,0)), mode='empty') #minimum') # try other padding methods?
                # pad to [64x64x25] - generic padding function option - Gul rukh prefers this to bicubic/lanczos (so no data disruption)
            else: 
                print('ERROR, size: '+str(size)+' passed is incompatible. Make sure the size is one of the following: [4,8,16,32,64]')
    
            resized_3d_img[:, :, cal_layer] = resized_2d_img   # save our resized_2d_img in the 3d_img corresponding to the calorimeter layer
                
        resized_3d_imgs = np.append(resized_3d_imgs, resized_3d_img)   # save our 3d image in the array holding all 5000 3d images
    
    return resized_3d_imgs   #returns an array of the 5000 resized 3d images


# dataset function from pgan code
def dataset(datafile, phase, local_rank, global_size, verbose, final_shape, num_phases, image_channels):
  
    3d_imgs, e_p, ang, ecal = GetDataAngle(datafile)
    resized_3d_imgs = resize(3d_imgs, size)
    
    # we need to decide later how to incorporate the e_p and ang variables (anglegan concatenates them to the latent vector)
    size = 2 * 2 ** phase   #[4,8,16,32,64] 
    
    data_path = os.path.join(args.dataset_path, f'{args.size}x{args.size}/')
    npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0, is_correct_phase=phase >= args.starting_phase)
    

    ####################### Pgan/main Dataset block of code ############################
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

    return npy_data, real_image_input, real_label, base_shape, batch_size

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

    return g_lr, d_lr, optimizer_gen, optimizer_disc, lr_step, update_step, update_g_lr, update_d_lr


def networks(config, phase, generator, discriminator, g_lr, d_lr, optimizer_gen, optimizer_disc, lr_step, update_step, update_g_lr, update_d_lr, npy_data, real_image_input, real_label, base_shape, batch_size, base_dim, num_phases, var_list, verbose, logdir, writer):
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
                generator,
                discriminator,
                real_image_input,
                args.latent_dim,
                alpha,
                phase,
                num_phases,
                base_dim,
                base_shape,
                args.activation,
                args.leakiness,
                args.network_size,
                args.loss_fn,
                args.gp_weight
            )
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

            g_gradients, g_variables = zip(*optimizer_gen.compute_gradients(gen_loss,
                                                                            var_list=gen_vars))
            if args.g_clipping:
                g_gradients, _ = tf.clip_by_global_norm(g_gradients, 1.0)


            d_gradients, d_variables = zip(*optimizer_disc.compute_gradients(disc_loss,
                                                                             var_list=disc_vars))
            if args.d_clipping:
                d_gradients, _ = tf.clip_by_global_norm(d_gradients, 1.0)


            g_norms = tf.stack([tf.norm(grad) for grad in g_gradients if grad is not None])
            max_g_norm = tf.reduce_max(g_norms)
            d_norms = tf.stack([tf.norm(grad) for grad in d_gradients if grad is not None])
            max_d_norm = tf.reduce_max(d_norms)


        elif args.optim_strategy == 'alternate':

            disc_loss, gp_loss = forward_discriminator(
                generator,
                discriminator,
                real_image_input,
                args.latent_dim,
                alpha,
                phase,
                num_phases,
                args.base_dim,
                base_shape,
                args.activation,
                args.leakiness,
                args.network_size,
                args.loss_fn,
                args.gp_weight,
                conditioning=real_label
            )

            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            d_gradients = optimizer_disc.compute_gradients(disc_loss, var_list=disc_vars)
            d_norms = tf.stack([tf.norm(grad) for grad, var in d_gradients if grad is not None])
            max_d_norm = tf.reduce_max(d_norms)

            train_disc = optimizer_disc.apply_gradients(d_gradients)

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
                train_gen = optimizer_gen.apply_gradients(g_gradients)

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


 # I made a configure function to try and organize the code, I think we could have it set up the config for run(config), but I don't really understand how to set this stuff up
def configure():
    
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
    # parse and return global arguments 
    get_args()
    global global_size, global_rank, global_step
    
    # Get the discriminator and generator from the architecture of choice
    if args.architecture == 'AngleArch3dGAN':
        discriminator = importlib.import_module(f'AngleArch3dGAN.py').discriminator
        generator = importlib.import_module(f'AngleArch3dGAN.py').generator 
    elif args.architecture == 'ProgressiveGAN':
        discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
        generator = importlib.import_module(f'networks.{args.architecture}.generator').generator
      
        
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
        npy_data, real_image_input, real_label, base_shape, batch_size = dataset(datafile, phase, local_rank, global_size, verbose, final_shape, num_phases, image_channels) 
        
        # call optimizers() -- replaces block of code in pgan main()
        # get the optimizers specified in the parameters
        g_lr, d_lr, optimizer_gen, optimizer_disc, lr_step, update_step, update_g_lr, update_d_lr = optimizers()
        
        # call networks() -- replaces networks block of code in pgan main()
        networks(config, phase, generator, discriminator, g_lr, d_lr, optimizer_gen, optimizer_disc, lr_step, update_step, update_g_lr, update_d_lr, npy_data, real_image_input, real_label, base_shape, batch_size, base_dim, num_phases, var_list, verbose, logdir, writer)
        
        
# calls run()
def main():   
    # run the main code!
    config = configure()
    run(config)  

# calls main()
if __name__ == "__main__":       
    main()