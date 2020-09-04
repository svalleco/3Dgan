import importlib
import numpy as np
import tensorflow as tf
from utils import count_parameters
import time
import nvgpu
import subprocess
import os
import psutil


def main(architecture, final_shape, real_image_input, latent_dim, base_dim, phase, loss_fn='logistic'):

    discriminator = importlib.import_module(f'networks.{architecture}.discriminator').discriminator
    generator = importlib.import_module(f'networks.{architecture}.generator').generator

    final_resolution = final_shape[-1]
    num_phases = int(np.log2(final_resolution) - 1)

    with tf.variable_scope('alpha'):
        alpha = tf.Variable(1, name='alpha', dtype=tf.float32)
        # Specify alpha update op for mixing phase.

    zdim_base = max(1, final_shape[1] // (2 ** (num_phases - 1)))
    base_shape = (1, zdim_base, 4, 4)

    z = tf.random.normal(shape=[tf.shape(real_image_input)[0], latent_dim])
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation='leaky_relu',
                           param=0.2)

    # Discriminator Training
    disc_fake_d = discriminator(tf.stop_gradient(gen_sample), alpha, phase, num_phases,
                                base_dim, latent_dim, activation='leaky_relu', param=0.2)
    disc_real = discriminator(real_image_input, alpha, phase, num_phases,
                              base_dim, latent_dim, activation='leaky_relu', param=0.2,
                              is_reuse=True)

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample)
    gradients = tf.gradients(discriminator(interpolates, alpha, phase,
                                           num_phases, base_dim, latent_dim,
                                           is_reuse=True, activation='leaky_relu',
                                           param=0.2), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3, 4)))

    # Generator training.
    disc_fake_g = discriminator(gen_sample, alpha, phase, num_phases, base_dim, latent_dim,
                                activation='leaky_relu', param=0.2, is_reuse=True)

    if loss_fn == 'wgan':
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = 10 * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = 1 * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real))
        disc_loss += gp_loss
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    print(f"Generator parameters: {count_parameters('generator')}")
    print(f"Discriminator parameters:: {count_parameters('discriminator')}")
    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # Build Optimizers
    with tf.variable_scope('optim_ops'):
        g_lr = 1e-3
        d_lr = 1e-3

        optimizer_gen = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=0, beta2=0.9)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=0, beta2=0.9)

        g_gradients = optimizer_gen.compute_gradients(gen_loss, var_list=gen_vars)
        d_gradients = optimizer_disc.compute_gradients(disc_loss, var_list=disc_vars)

        train_gen = optimizer_gen.apply_gradients(g_gradients)
        train_disc = optimizer_disc.apply_gradients(d_gradients)

    return train_gen, train_disc


if __name__ == '__main__':
    num_phases = 8
    base_dim = 512
    base_shape = [1, 1, 4, 4]
    latent_dim = 512
    final_shape = [1, 128, 512, 512]
    for phase in range(7, 8):
        tf.reset_default_graph()
        shape = [1, 1] + list(np.array(base_shape)[1:] * 2 ** (phase - 1))
        real_image_input = tf.random.normal(shape=shape)

        train_gen, train_disc = main('surfgan', final_shape, real_image_input, latent_dim, base_dim, phase)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.graph.finalize()
            sess.run(init_op)
            start = time.time()

            print("Phase ", phase)
            sess.run([train_gen, train_disc])
            # Print memory info.
            try:
                print(nvgpu.gpu_info())
            except subprocess.CalledProcessError:
                 pid = os.getpid()
                 py = psutil.Process(pid)
                 print(f"CPU Percent: {py.cpu_percent()}")
                 print(f"Memory info: {py.memory_info()}")

            end = time.time()

            print(f"{end - start} seconds")


