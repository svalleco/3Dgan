# pylint: disable=import-error
import argparse
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import time
import random
from metrics import (calculate_fid_given_batch_volumes, get_swd_for_volumes,
                     get_normalized_root_mse, get_mean_squared_error, get_psnr, get_ssim)
from dataset import NumpyPathDataset
from utils import count_parameters, image_grid, parse_tuple, MPMap
# from mpi4py import MPI
import os
import importlib
from rectified_adam import RAdamOptimizer
from networks.loss import forward_simultaneous, forward_generator, forward_discriminator
import psutil
from networks.ops import num_filters
from tensorflow.data.experimental import AUTOTUNE
import nvgpu



def main(args, config):

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
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # ??
    if verbose:
        writer = tf.summary.FileWriter(logdir=logdir)
        print("Arguments passed:")
        print(args)
        print(f"Saving files to {logdir}")

    else:
        pass
        # writer = None
    
    final_shape = parse_tuple(args.final_shape) # Fiinal shape (51,51,25) ?
    image_channels = final_shape[0]
    final_resolution = final_shape[-1]
    num_phases = int(np.log2(final_resolution) - 1)
    base_dim = num_filters(-num_phases + 1, num_phases, size=args.network_size)
    
    var_list = list()
    global_step = 0

    for phase in range(1, num_phases + 1):

        tf.reset_default_graph()
        
        # ------------------------------------------------------------------------------------------#
        # DATASET
        
        size = 2 * 2 ** phase

        data_path = os.path.join(args.dataset_path, f'{size}x{size}/')
        npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0,
                                    is_correct_phase=phase >= args.starting_phase)

        # # dataset = tf.data.Dataset.from_generator(npy_data.__iter__, npy_data.dtype, npy_data.shape)
        # dataset = tf.data.Dataset.from_tensor_slices(npy_data.scratch_files)

        # Get DataLoader
        batch_size = max(1, args.base_batch_size // (2 ** (phase - 1)))

        if phase >= args.starting_phase:
            assert batch_size * global_size <= args.max_global_batch_size
            if verbose:
                print(f"Using local batch size of {batch_size} and global batch size of {batch_size * global_size}")

        # if args.horovod:
        #     dataset.shard(hvd.size(), hvd.rank())
        #
        # def load(x):
        #     x = np.load(x.decode())[np.newaxis, ...].astype(np.float32) / 1024 - 1
        #     return x
        #
        # if args.gpu:
        #     parallel_calls = AUTOTUNE
        # else:
        #     parallel_calls = int(os.environ['OMP_NUM_THREADS'])
        #
        # dataset = dataset.shuffle(len(npy_data))
        # dataset = dataset.map(lambda x: tuple(tf.py_func(load, [x], [tf.float32])), num_parallel_calls=parallel_calls)
        # dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.repeat()
        # dataset = dataset.prefetch(AUTOTUNE)
        # dataset = dataset.make_one_shot_iterator()
        # data = dataset.get_next()
        # if len(data) == 1:
        #     real_image_input = data
        #     real_label = None
        # elif len(data) == 2:
        #     real_image_input, real_label = data
        # else:
        #     raise NotImplementedError()

        zdim_base = max(1, final_shape[1] // (2 ** (num_phases - 1)))
        base_shape = (image_channels, zdim_base, 4, 4)
        current_shape = [batch_size, image_channels, *[size * 2 ** (phase - 1) for size in
                                                       base_shape[1:]]]
        real_image_input = tf.placeholder(shape=current_shape, dtype=tf.float32)

        # real_image_input = tf.random.normal([1, batch_size, image_channels, *[size * 2 ** (phase -
        #                                                                                  1) for size in base_shape[1:]]])
        # real_image_input = tf.squeeze(real_image_input, axis=0)
        # real_image_input = tf.ensure_shape(real_image_input, [batch_size, image_channels, *[size * 2 ** (phase - 1) for size in base_shape[1:]]])
        real_image_input = real_image_input + tf.random.normal(tf.shape(real_image_input)) * .01
        real_label = None

        if real_label is not None:
            real_label = tf.one_hot(real_label, depth=args.num_labels)
            
        # ------------------------------------------------------------------------------------------#
        # OPTIMIZERS

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

        d_lr = tf.Variable(d_lr, name='d_lr', dtype=tf.float32)
        g_lr = tf.Variable(g_lr, name='g_lr', dtype=tf.float32)

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)
        #optimizer_gen = tf.train.RMSPropOptimizer(learning_rate=g_lr)
        #optimizer_disc = tf.train.RMSPropOptimizer(learning_rate=d_lr)
        # optimizer_gen = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        # optimizer_disc = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        # optimizer_gen = RAdamOptimizer(learning_rate=g_lr, beta1=args.beta1, beta2=args.beta2)
        # optimizer_disc = RAdamOptimizer(learning_rate=d_lr, beta1=args.beta1, beta2=args.beta2)

        lr_step = tf.Variable(0, name='step', dtype=tf.float32)
        update_step = lr_step.assign_add(1.0)

        with tf.control_dependencies([update_step]):
            update_g_lr = g_lr.assign(g_lr * args.g_annealing)
            update_d_lr = d_lr.assign(d_lr * args.d_annealing)

        if args.horovod:
            if args.use_adasum:
                # optimizer_gen = hvd.DistributedOptimizer(optimizer_gen, op=hvd.Adasum)
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc, op=hvd.Adasum)
            else:
                optimizer_gen = hvd.DistributedOptimizer(optimizer_gen)
                optimizer_disc = hvd.DistributedOptimizer(optimizer_disc)

        # ------------------------------------------------------------------------------------------#
        # NETWORKS

        with tf.variable_scope('alpha'):
            alpha = tf.Variable(1, name='alpha', dtype=tf.float32)
            # Alpha init
            init_alpha = alpha.assign(1)

            # Specify alpha update op for mixing phase.
            num_steps = args.mixing_nimg // (batch_size * global_size)
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

            # g_norms = tf.stack([tf.norm(grad) for grad, var in g_gradients if grad is not None])
            # max_g_norm = tf.reduce_max(g_norms)
            # d_norms = tf.stack([tf.norm(grad) for grad, var in d_gradients if grad is not None])
            # max_d_norm = tf.reduce_max(d_norms)

            # g_clipped_grads = [(tf.clip_by_norm(grad, clip_norm=128), var) for grad, var in g_gradients]
            # train_gen = optimizer_gen.apply_gradients(g_clipped_grads)
            train_gen = optimizer_gen.apply_gradients(zip(g_gradients, g_variables))
            train_disc = optimizer_disc.apply_gradients(zip(d_gradients, d_variables))

            # train_gen = optimizer_gen.apply_gradients(g_gradients)
            # train_disc = optimizer_disc.apply_gradients(d_gradients)

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
                    is_reuse=True
                )

                gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                g_gradients = optimizer_gen.compute_gradients(gen_loss, var_list=gen_vars)
                g_norms = tf.stack([tf.norm(grad) for grad, var in g_gradients if grad is not None])
                max_g_norm = tf.reduce_max(g_norms)
                train_gen = optimizer_gen.apply_gradients(g_gradients)

        else:
            raise ValueError("Unknown optim strategy ", args.optim_strategy)

        if verbose:
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
            # if args.gpu:
            #     assert tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
            # sess.graph.finalize()
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
                    # memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # if not args.gpu:
                    #     memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # else:
                    #     memory_percentage = nvgpu.gpu_info()[local_rank]['mem_used_percent']


                    # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='memory_percentage', simple_value=memory_percentage)]),
                    #                    global_step)

                    print(f"Step {global_step:09} \t"
                          f"img/s {img_s:.2f} \t "
                          f"d_loss {d_loss:.4f} \t "
                          f"g_loss {g_loss:.4f} \t "
                          # f"memory {memory_percentage:.4f} % \t"
                          f"alpha {alpha.eval():.2f}")

                #     # if take_first_snapshot:
                #     #     import tracemalloc
                #     #     tracemalloc.start()
                #     #     snapshot_first = tracemalloc.take_snapshot()
                #     #     take_first_snapshot = False

                #     # snapshot = tracemalloc.take_snapshot()
                #     # top_stats = snapshot.compare_to(snapshot_first, 'lineno')
                #     # print("[ Top 10 differences ]")
                #     # for stat in top_stats[:10]:
                #     #     print(stat)
                #     # snapshot_prev = snapshot

                if global_step >= ((phase - args.starting_phase)
                                   * (args.mixing_nimg + args.stabilizing_nimg)
                                   + args.mixing_nimg):
                    break

                sess.run(update_alpha)
                sess.run(ema_op)
                sess.run(update_d_lr)
                sess.run(update_g_lr)

                assert alpha.eval() >= 0

                # if verbose:
                #     writer.flush()

            if verbose:
                print(f"Begin stabilizing epochs in phase {phase}")

            sess.run(assign_zero)

            while True:
                start = time.time()
                assert alpha.eval() == 0
                if local_step % 2048 == 0 and local_step > 0:

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
                        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='img_s',
                                                                            simple_value   =img_s)]),
                                        global_step)
                    writer.add_summary(summary, global_step)
                    # memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # if not args.gpu:
                    #     memory_percentage = psutil.Process(os.getpid()).memory_percent()
                    # else:
                    #     gpu_info = nvgpu.gpu_info()
                    #     memory_percentage = nvgpu.gpu_info()[local_rank]['mem_used_percent']

                    # writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='memory_percentage', simple_value=memory_percentage)]),
                    #                    global_step)

                    print(f"Step {global_step:09} \t"
                          f"img/s {img_s:.2f} \t "
                          f"d_loss {d_loss:.4f} \t "
                          f"g_loss {g_loss:.4f} \t "
                          # f"memory {memory_percentage:.4f} % \t"
                          f"alpha {alpha.eval():.2f}")

                sess.run(ema_op)

                # if verbose:
                #     writer.flush()

                if global_step >= (phase - args.starting_phase + 1) * (args.stabilizing_nimg + args.mixing_nimg):
                    # if verbose:
                    #     run_metadata = tf.RunMetadata()
                    #     opts = tf.profiler.ProfileOptionBuilder.float_operation()
                    #     g = tf.get_default_graph()
                    #     flops = tf.profiler.profile(g, run_meta=run_metadata, cmd='op', options=opts)
                    #     writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='graph_flops',
                    #                                                           simple_value=flops.total_float_ops)]),
                    #                        global_step)
                    #
                    #     # Print memory info.
                    #     try:
                    #         print(nvgpu.gpu_info())
                    #     except subprocess.CalledProcessError:
                    #         pid = os.getpid()
                    #         py = psutil.Process(pid)
                    #         print(f"CPU Percent: {py.cpu_percent()}")
                    #         print(f"Memory info: {py.memory_info()}")

                    break

            # # Calculate metrics.
            # calc_swds: bool = size >= 16
            # calc_ssims: bool = min(npy_data.shape[1:]) >= 16
            #
            # if args.calc_metrics:
            #     fids_local = []
            #     swds_local = []
            #     psnrs_local = []
            #     mses_local = []
            #     nrmses_local = []
            #     ssims_local = []
            #
            #     counter = 0
            #     while True:
            #         if args.horovod:
            #             start_loc = counter + hvd.rank() * batch_size
            #         else:
            #             start_loc = 0
            #         real_batch = np.stack([npy_data[i] for i in range(start_loc, start_loc + batch_size)])
            #         real_batch = real_batch.astype(np.int16) - 1024
            #         fake_batch = sess.run(gen_sample).astype(np.float32)
            #
            #         # Turn fake batch into HUs and clip to training range.
            #         fake_batch = (np.clip(fake_batch, -1, 2) * 1024).astype(np.int16)
            #
            #         if verbose:
            #             print('real min, max', real_batch.min(), real_batch.max())
            #             print('fake min, max', fake_batch.min(), fake_batch.max())
            #
            #         fids_local.append(calculate_fid_given_batch_volumes(real_batch, fake_batch, sess))
            #
            #         if calc_swds:
            #             swds = get_swd_for_volumes(real_batch, fake_batch)
            #             swds_local.append(swds)
            #
            #         psnr = get_psnr(real_batch, fake_batch)
            #         if calc_ssims:
            #             ssim = get_ssim(real_batch, fake_batch)
            #             ssims_local.append(ssim)
            #         mse = get_mean_squared_error(real_batch, fake_batch)
            #         nrmse = get_normalized_root_mse(real_batch, fake_batch)
            #
            #         psnrs_local.append(psnr)
            #         mses_local.append(mse)
            #         nrmses_local.append(nrmse)
            #
            #         if args.horovod:
            #             counter = counter + global_size * batch_size
            #         else:
            #             counter += batch_size
            #
            #         if counter >= args.num_metric_samples:
            #             break
            #
            #     fid_local = np.mean(fids_local)
            #     psnr_local = np.mean(psnrs_local)
            #     ssim_local = np.mean(ssims_local)
            #     mse_local = np.mean(mses_local)
            #     nrmse_local = np.mean(nrmses_local)
            #
            #     if args.horovod:
            #         fid = MPI.COMM_WORLD.allreduce(fid_local, op=MPI.SUM) / hvd.size()
            #         psnr = MPI.COMM_WORLD.allreduce(psnr_local, op=MPI.SUM) / hvd.size()
            #         mse = MPI.COMM_WORLD.allreduce(mse_local, op=MPI.SUM) / hvd.size()
            #         nrmse = MPI.COMM_WORLD.allreduce(nrmse_local, op=MPI.SUM) / hvd.size()
            #         if calc_ssims:
            #             ssim = MPI.COMM_WORLD.allreduce(ssim_local, op=MPI.SUM) / hvd.size()
            #     else:
            #         fid = fid_local
            #         psnr = psnr_local
            #         ssim = ssim_local
            #         mse = mse_local
            #         nrmse = nrmse_local
            #
            #     if calc_swds:
            #         swds_local = np.array(swds_local)
            #         # Average over batches
            #         swds_local = swds_local.mean(axis=0)
            #         if args.horovod:
            #             swds = MPI.COMM_WORLD.allreduce(swds_local, op=MPI.SUM) / hvd.size()
            #         else:
            #             swds = swds_local
            #
            #     if verbose:
            #         print(f"FID: {fid:.4f}")
            #         writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='fid',
            #                                                               simple_value=fid)]),
            #                            global_step)
            #
            #         print(f"PSNR: {psnr:.4f}")
            #         writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='psnr',
            #                                                               simple_value=psnr)]),
            #                            global_step)
            #
            #         print(f"MSE: {mse:.4f}")
            #         writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='mse',
            #                                                               simple_value=mse)]),
            #                            global_step)
            #
            #         print(f"Normalized Root MSE: {nrmse:.4f}")
            #         writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='nrmse',
            #                                                               simple_value=nrmse)]),
            #                            global_step)
            #
            #         if calc_swds:
            #             print(f"SWDS: {swds}")
            #             for i in range(len(swds))[:-1]:
            #                 lod = 16 * 2 ** i
            #                 writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=f'swd_{lod}',
            #                                                                       simple_value=swds[
            #                                                                           i])]),
            #                                    global_step)
            #             writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=f'swd_mean',
            #                                                                   simple_value=swds[
            #                                                                       -1])]), global_step)
            #         if calc_ssims:
            #             print(f"SSIM: {ssim}")
            #             writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=f'ssim',
            #                                                                   simple_value=ssim)]), global_step)

            if verbose:
                print("\n\n\n End of phase.")

                # Save Session.
                sess.run(ema_update_weights)
                saver = tf.train.Saver(var_list)
                saver.save(sess, os.path.join(logdir, f'model_{phase}'))

            if args.ending_phase:
                if phase == args.ending_phase:
                    print("Reached final phase, breaking.")
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str)
    parser.add_argument('dataset_path', type=str)
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
    parser.add_argument('--g_annealing', default=1,
                        type=float, help='generator annealing rate, 1 -> no annealing.')
    parser.add_argument('--d_annealing', default=1,
                        type=float, help='discriminator annealing rate, 1 -> no annealing.')
    parser.add_argument('--num_metric_samples', type=int, default=512)
    parser.add_argument('--beta1', type=float, default=0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--ema_beta', type=float, default=0.99)
    parser.add_argument('--d_scaling', default='none', choices=['linear', 'sqrt', 'none'],
                        help='How to scale discriminator learning rate with horovod size.')
    parser.add_argument('--g_scaling', default='none', choices=['linear', 'sqrt', 'none'],
                        help='How to scale generator learning rate with horovod size.')
    parser.add_argument('--continue_path', default=None, type=str)
    parser.add_argument('--starting_alpha', default=1, type=float)
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--use_adasum', default=False, action='store_true')
    parser.add_argument('--optim_strategy', default='simultaneous', choices=['simultaneous', 'alternate'])
    parser.add_argument('--num_inter_ops', default=4, type=int)
    parser.add_argument('--num_labels', default=None, type=int)
    parser.add_argument('--g_clipping', default=False, type=bool)
    parser.add_argument('--d_clipping', default=False, type=bool)
    # parser.add_argument('--load_phase', default=None, type=int)
    args = parser.parse_args()

    # if args.coninue_path:
    #     assert args.load_phase is not None, "Please specify in which phase the weights of the " \
    #                                         "specified continue_path should be loaded."

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

    if args.architecture in ('stylegan2'):
        assert args.starting_phase == args.ending_phase

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

    discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
    generator = importlib.import_module(f'networks.{args.architecture}.generator').generator

    main(args, config)
