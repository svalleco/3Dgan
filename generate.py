import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import time
import random
from metrics import (calculate_fid_given_batch_volumes, get_swd_for_volumes,
                     get_normalized_root_mse, get_mean_squared_error, get_psnr, get_ssim)
from dataset import NumpyPathDataset
from utils import count_parameters, image_grid, parse_tuple
from mpi4py import MPI
import os
import importlib
from tqdm import tqdm
import imageio

from tensorflow.data.experimental import AUTOTUNE


def main(args, config):
    phase = args.phase
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

    if verbose:
        timestamp = time.strftime("%Y-%m-%d_%H:%M", time.gmtime())
        logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_samples', timestamp)
        os.makedirs(logdir)
    else:
        logdir = None

    if args.horovod:
        logdir = MPI.COMM_WORLD.bcast(logdir, root=0)

    if verbose:
        print("Arguments passed:")
        print(args)
        print(f"Saving files to {logdir}")

    tf.reset_default_graph()
    # Get Dataset.

    final_shape = parse_tuple(args.final_shape)
    final_resolution = final_shape[-1]
    num_phases = int(np.log2(final_resolution) - 1)
    size = 2 * 2 ** phase
    data_path = os.path.join(args.dataset_path, f'{size}x{size}/')
    npy_data = NumpyPathDataset(data_path, None,  copy_files=False, is_correct_phase=False)
    dataset = tf.data.Dataset.from_tensor_slices(npy_data.scratch_files)

    batch_size = 1

    if args.horovod:
        dataset.shard(hvd.size(), hvd.rank())

    def load(x):
        x = np.load(x.numpy().decode('utf-8'))[np.newaxis, ...]
        return x

    # Lay out the graph.
    dataset = dataset.shuffle(len(npy_data))
    dataset = dataset.map(lambda x: tf.py_function(func=load, inp=[x], Tout=tf.uint16), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x: tf.cast(x, tf.float32) / 1024 - 1, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = dataset.make_one_shot_iterator()
    real_image_input = dataset.get_next()
    real_image_input = tf.ensure_shape(real_image_input, [batch_size] + list(npy_data.shape))
    # real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * .01

    with tf.variable_scope('alpha'):
        alpha = tf.Variable(0, name='alpha', dtype=tf.float32)

    zdim_base = max(1, final_shape[1] // (2 ** (num_phases - 1)))
    base_shape = (1, zdim_base, 4, 4)

    noise_input_d = tf.random.normal(shape=[tf.shape(real_image_input)[0], args.latent_dim])
    gen_sample_d = generator(noise_input_d, alpha, phase, num_phases,
                             args.base_dim, base_shape, activation=args.activation, param=args.leakiness)

    if verbose:
        print(f"Generator parameters: {count_parameters('generator')}")

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    real_image_grid = tf.transpose(real_image_input[0], (1, 2, 3, 0))
    shape = real_image_grid.get_shape().as_list()
    grid_cols = int(2 ** np.floor(np.log(np.sqrt(shape[0])) / np.log(2)))
    grid_rows = shape[0] // grid_cols
    grid_shape = [grid_rows, grid_cols]
    real_image_grid = image_grid(real_image_grid, grid_shape, image_shape=shape[1:3],
                                 num_channels=shape[-1])

    fake_image_grid = tf.transpose(gen_sample_d[0], (1, 2, 3, 0))
    fake_image_grid = image_grid(fake_image_grid, grid_shape, image_shape=shape[1:3],
                                 num_channels=shape[-1])

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        trainable_variable_names = [v.name for v in tf.trainable_variables()]
        var_list = gen_vars + disc_vars
        var_names = [v.name for v in var_list]
        load_vars = [sess.graph.get_tensor_by_name(n) for n in var_names if n in trainable_variable_names]
        saver = tf.train.Saver(load_vars)
        saver.restore(sess, os.path.join(args.model_path))

        if args.horovod:
            sess.run(hvd.broadcast_global_variables(0))

        num_samples = args.num_samples // global_size

        calc_swds: bool = size >= 16
        calc_ssims: bool = min(npy_data.shape[1:]) >= 16

        fids_local = []
        swds_local = []
        psnrs_local = []
        mses_local = []
        nrmses_local = []
        ssims_local = []

        for i in tqdm(range(num_samples)):

            ix = (global_rank + i * global_size)
            real_batch, fake_batch, grid_real, grid_fake = sess.run(
                 [real_image_input, gen_sample_d, real_image_grid, fake_image_grid]
            )

            # fake_batch = sess.run(real_image_input)

            grid_real = np.squeeze(grid_real)
            grid_fake = np.squeeze(grid_fake)

            imageio.imwrite(os.path.join(logdir, f'grid_real_{ix}.png'), grid_real)
            imageio.imwrite(os.path.join(logdir, f'grid_fake_{ix}.png'), grid_fake)

            fake_batch = (np.clip(fake_batch, -1, 2) * 1024).astype(np.int16)
            real_batch = (np.clip(real_batch, -1, 2) * 1024).astype(np.int16)

            # fake_batch = real_batch

            assert real_batch.min() < -512
            assert fake_batch.min() < -512

            fids_local.append(calculate_fid_given_batch_volumes(real_batch, fake_batch, sess))
            if calc_swds:
                swds = get_swd_for_volumes(real_batch, fake_batch)
                swds_local.append(swds)

            psnr = get_psnr(real_batch, fake_batch)
            if calc_ssims:
                ssim = get_ssim(real_batch, fake_batch)
                ssims_local.append(ssim)

            mse = get_mean_squared_error(real_batch, fake_batch)
            nrmse = get_normalized_root_mse(real_batch, fake_batch)

            psnrs_local.append(psnr)
            mses_local.append(mse)
            nrmses_local.append(nrmse)

            save_path = os.path.join(logdir, f'{ix}.npy')
            np.save(save_path, fake_batch)

        fid_local = np.stack(fids_local).mean(0)
        psnr_local = np.mean(psnrs_local)
        ssim_local = np.mean(ssims_local)
        mse_local = np.mean(mses_local)
        nrmse_local = np.mean(nrmses_local)

        if args.horovod:
            fid = MPI.COMM_WORLD.allreduce(fid_local, op=MPI.SUM) / hvd.size()
            psnr = MPI.COMM_WORLD.allreduce(psnr_local, op=MPI.SUM) / hvd.size()
            mse = MPI.COMM_WORLD.allreduce(mse_local, op=MPI.SUM) / hvd.size()
            nrmse = MPI.COMM_WORLD.allreduce(nrmse_local, op=MPI.SUM) / hvd.size()
            if calc_ssims:
                ssim = MPI.COMM_WORLD.allreduce(ssim_local, op=MPI.SUM) / hvd.size()
        else:
            fid = fid_local
            psnr = psnr_local
            ssim = ssim_local
            mse = mse_local
            nrmse = nrmse_local

        if calc_swds:
            swds_local = np.array(swds_local)
            # Average over batches
            swds_local = swds_local.mean(axis=0)
            if args.horovod:
                swds = MPI.COMM_WORLD.allreduce(swds_local, op=MPI.SUM) / hvd.size()
            else:
                swds = swds_local

        summary_str = ""
        if verbose:
            summary_str += f"FIDS: {fid.tolist()} \n\n"
            summary_str += f"FID: {fid.mean():.4f} \n"
            summary_str += f"PSNR: {psnr:.4f} \n"
            summary_str += f"MSE: {mse:.4f} \n"
            summary_str += f"Normalized Root MSE: {nrmse:.4f} \n"
            if calc_swds:
                summary_str += f"SWDS: {swds} \n"
            if calc_ssims:
                summary_str += f"SSIM: {ssim} \n"

        if verbose:
            with open(os.path.join(logdir, 'summary.txt'), 'w') as f:
                f.write(summary_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('final_shape', type=str, help="'(c, z, y, x)', e.g. '(1, 64, 128, 128)'")
    parser.add_argument('--model_path', default=None, type=str, required=True)
    parser.add_argument('--num_samples', default=None, type=int, required=True)
    parser.add_argument('--phase', type=int, default=None, required=True)
    parser.add_argument('--base_dim', type=int, default=None, required=True)
    parser.add_argument('--latent_dim', type=int, default=None, required=True)
    parser.add_argument('--activation', type=str, default='leaky_relu')
    parser.add_argument('--leakiness', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horovod', default=False, action='store_true')
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    gopts = tf.GraphOptions(place_pruned_graph=True)
    config = tf.ConfigProto(graph_options=gopts,
                            intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                            inter_op_parallelism_threads=2,
                            allow_soft_placement=True,
                            device_count={'CPU': int(os.environ['OMP_NUM_THREADS'])})

    config.gpu_options.allow_growth = True

    discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
    generator = importlib.import_module(f'networks.{args.architecture}.generator').generator

    if args.horovod:
        hvd.init()
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        np.random.seed(args.seed + hvd.rank())
        tf.random.set_random_seed(args.seed + hvd.rank())
        random.seed(args.seed + hvd.rank())

        print(f"Rank {hvd.rank()}:{hvd.local_rank()} reporting!")

    else:
        np.random.seed(args.seed)
        tf.random.set_random_seed(args.seed)
        random.seed(args.seed)

    main(args, config)
