import tensorflow as tf
import numpy as np

channel_format = 'channels_first'

# set channel format - want channels_first for cpu    
if channel_format == 'channels_first':
   daxis = (2,3,4)
else:
   daxis = (1,2,3)
   
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


def forward_generator(generator,
                      discriminator,
                      real_image_input,
                      latent_dim,
                      alpha,
                      phase,
                      num_phases,
                      base_dim,
                      base_shape,
                      activation,
                      leakiness,
                      network_size,
                      loss_fn,
                      loss_weights,
                      e_p,
                      ang,
                      is_reuse=False
                      ):
    z_batch_size = tf.shape(real_image_input)[0]                 # this value should be an integer

    #Adel is passing e_p and ang as the correct batch sized numpy arrays
    e_p_tensor = tf.reshape(e_p, [z_batch_size,1])   # need z_batch_size x 1
    ang_tensor = tf.reshape(ang, [z_batch_size,1])   # need z_batch_size x 1
    
    z = tf.random.normal(shape=[z_batch_size, latent_dim-2])   
    z = tf.concat([z, e_p_tensor, ang_tensor], 1)    # shape = (z_batch_size, 256)
    
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, is_reuse=is_reuse)

    gen_sample = gen_sample + tf.random.normal(shape=tf.shape(gen_sample)) * 0.01

    # Generator training.
    disc_fake_g = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=is_reuse)
    if loss_fn == 'wgan':
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))
        
    elif loss_fn == 'anglegan':
        add_loss_batch = np.expand_dims(hist_count(image_batch, xpower), axis=-1)

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return gen_sample, gen_loss


def forward_discriminator(generator,
                          discriminator,
                          real_image_input,
                          latent_dim,
                          alpha,
                          phase,
                          num_phases,
                          base_dim,
                          base_shape,
                          activation,
                          leakiness,
                          network_size,
                          loss_fn,
                          loss_weights,
                          gp_weight,
                          e_p,
                          ang,
                          is_reuse=False,
                          ):
    z_batch_size = tf.shape(real_image_input)[0]                 # this value should be an integer

    #Adel is passing e_p and ang as the correct batch sized numpy arrays
    e_p_tensor = tf.reshape(e_p, [z_batch_size,1])   # need z_batch_size x 1
    ang_tensor = tf.reshape(ang, [z_batch_size,1])   # need z_batch_size x 1
    
    z = tf.random.normal(shape=[z_batch_size, latent_dim-2])   
    z = tf.concat([z, e_p_tensor, ang_tensor], 1)    # shape = (z_batch_size, 256)
    
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, is_reuse=is_reuse)

    gen_sample = gen_sample + tf.random.normal(shape=tf.shape(gen_sample)) * 0.01

    # Discriminator Training
    disc_fake_d = discriminator(tf.stop_gradient(gen_sample), alpha, phase, num_phases, base_shape, 
                                base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, )
    disc_real = discriminator(real_image_input, alpha, phase, num_phases, base_shape, 
                              base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, )

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample)
    gradients = tf.gradients(discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, ), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3, 4)))

    if loss_fn == 'wgan':
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = gp_weight * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real))
        disc_loss += gp_loss
    
    elif loss_fn == 'anglegan':
        add_loss_batch = np.expand_dims(hist_count(image_batch, xpower), axis=-1)

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return disc_loss, gp_loss


def forward_simultaneous(generator,
                         discriminator,
                         real_image_input,
                         latent_dim,
                         alpha,
                         phase,
                         num_phases,
                         base_dim,
                         base_shape,
                         activation,
                         leakiness,
                         network_size,
                         loss_fn,
                         loss_weights,
                         gp_weight,
                         e_p,
                         ang,
                         conditioning=None
                         ):
    z_batch_size = tf.shape(real_image_input)[0]                 # this value should be an integer

    #Adel is passing e_p and ang as the correct batch sized numpy arrays
    e_p_tensor = tf.reshape(e_p, [z_batch_size,1])   # need z_batch_size x 1
    ang_tensor = tf.reshape(ang, [z_batch_size,1])   # need z_batch_size x 1
    
    z = tf.random.normal(shape=[z_batch_size, latent_dim-2])   
    z = tf.concat([z, e_p_tensor, ang_tensor], 1)    # shape = (z_batch_size, 256)
    
    gen_sample = generator(z, alpha, phase, num_phases,
                           base_dim, base_shape, activation=activation,
                           param=leakiness, size=network_size, conditioning=conditioning)

    gen_sample = gen_sample + tf.random.normal(shape=tf.shape(gen_sample)) * 0.01
    # Discriminator Training
    disc_fake_d = discriminator(tf.stop_gradient(gen_sample), alpha, phase, num_phases, base_shape, 
                                base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, conditioning=conditioning)
    disc_real = discriminator(real_image_input, alpha, phase, num_phases, base_shape, 
                              base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, conditioning=conditioning)

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample)

    gradients = tf.gradients(discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, conditioning=conditioning), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3)))

    # Generator training.
    disc_fake_g = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=True, conditioning=conditioning)

    if loss_fn == 'wgan':
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = gp_weight * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real))
        disc_loss += gp_loss
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))
        
    elif loss_fn == 'anglegan':
        add_loss_batch = np.expand_dims(hist_count(image_batch, xpower), axis=-1)

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return gen_loss, disc_loss, gp_loss, gen_sample
