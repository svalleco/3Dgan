import tensorflow as tf
import numpy as np
from networks.pgan.loss_utils import bce, mae, mape  # loss functions used for loss_fn='anglegan'
from networks.pgan.loss_utils import ecal_sum, ecal_angle # physics functions used for training the discriminator (should be in conditional lambda layer)


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
    disc_fake_g, fake_ecal, fake_ang = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=is_reuse)
    if loss_fn == 'wgan':
        gen_loss = -tf.reduce_mean(disc_fake_g)

    elif loss_fn == 'logistic':
        gen_loss = tf.reduce_mean(tf.nn.softplus(-disc_fake_g))
        
    elif loss_fn == 'anglegan':
        gen_loss = tf.reduce_mean(-disc_fake_g) # TODO
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
    disc_fake_d, fake_ecal, fake_ang = discriminator(tf.stop_gradient(gen_sample), alpha, phase, num_phases, base_shape, 
                                base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, )
    disc_real, real_ecal, real_ang = discriminator(real_image_input, alpha, phase, num_phases, base_shape, 
                              base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, )

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample)
    disc_fake_d2, fake_ecal2, fake_ang2 = discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, )
    gradients = tf.gradients(disc_fake_d2, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3, 4)))

    # wasserstein gan
    if loss_fn == 'wgan':
        # has the real/fake activation layer built in, as a critic that rates. Allows you to move far away and still get a good gradient
        gradient_penalty = (slopes - 1) ** 2
        gp_loss = gp_weight * gradient_penalty
        disc_loss = disc_fake_d - disc_real
        drift_loss = 1e-3 * disc_real ** 2
        disc_loss = tf.reduce_mean(disc_loss + gp_loss + drift_loss)

    elif loss_fn == 'logistic':
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        gp_loss = gp_weight * gradient_penalty
        disc_loss = tf.reduce_mean(tf.nn.softplus(disc_fake_d)) + tf.reduce_mean(
            tf.nn.softplus(-disc_real)) ## TODO - To check 
        disc_loss += gp_loss
    
    elif loss_fn == 'anglegan':   # NEEDS TO BE TESTED + DEBUGGED!!
        fake_loss = bce(disc_real, disc_fake_d)     # NOT SURE!! CHECK!
        # need to use ecal_angle() in conditional lambda layer (in d) to find ang_output/ang_target
        ang_loss = mae(real_ang, fake_ang)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ANGS?!!
        # need to use ecal_sum() in conditional lambda layer (in d) to find ecal_output/ecal_target
        ecal_loss = mape(real_ecal, fake_ecal)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ECALS?!!
        
        losses = np.array([fake_loss, ang_loss, ecal_loss])   # calculate the losses and store in an array
        loss_weights = loss_weights.numpy() # make sure pgan weight vector is a np.array
        disc_loss = np.dot(loss_weights, losses)  # weight and sum the losses
        
        gp_loss = gp_weight * gradient_penalty   #HOW TO IMPLEMENT THIS???
        disc_loss += gp_loss    #HOW TO IMPLEMENT THIS???
    
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
    disc_fake_d, fake_ecal, fake_ang = discriminator(tf.stop_gradient(gen_sample), alpha, phase, num_phases, base_shape, 
                                base_dim, latent_dim, activation=activation, param=leakiness,
                                size=network_size, conditioning=conditioning)
    disc_real, real_ecal, real_ang = discriminator(real_image_input, alpha, phase, num_phases, base_shape, 
                              base_dim, latent_dim, activation=activation, param=leakiness,
                              is_reuse=True, size=network_size, conditioning=conditioning)

    gamma = tf.random_uniform(shape=[tf.shape(real_image_input)[0], 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates = gamma * real_image_input + (1 - gamma) * tf.stop_gradient(gen_sample)

    
    disc_fake_d3, fake_ecal3, fake_ang3 = discriminator(interpolates, alpha, phase,
                                           num_phases, base_shape, base_dim, latent_dim,
                                           is_reuse=True, activation=activation,
                                           param=leakiness, size=network_size, conditioning=conditioning) # ToDO Renaming 
    gradients = tf.gradients(disc_fake_d3, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=(1, 2, 3)))

    # Generator training.
    disc_fake_g, fake_ecal_g, fake_ang_g = discriminator(gen_sample, alpha, phase, num_phases, base_shape, base_dim, latent_dim,
                                activation=activation, param=leakiness, size=network_size, is_reuse=True, conditioning=conditioning)
    # wasserstein gan
    if loss_fn == 'wgan':
        # has the real/fake activation layer built in, as a critic that rates. Allows you to move far away and still get a good gradient
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
        
    elif loss_fn == 'anglegan':   # NEEDS TO BE TESTED + DEBUGGED!!
        fake_loss = bce(disc_real, disc_fake_d)     # NOT SURE!! CHECK!
        # need to use ecal_angle() in conditional lambda layer (in d) to find ang_output/ang_target
        ang_loss = mae(real_ang, fake_ang)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ANGS?!!
        # need to use ecal_sum() in conditional lambda layer (in d) to find ecal_output/ecal_target
        ecal_loss = mape(real_ecal, fake_ecal)  # DO I NEED TO SWITCH THE ORDER OF THE REAL AND FAKE ECALS?!!
        
        losses = np.array([fake_loss, ang_loss, ecal_loss])   # calculate the losses and store in an array
        loss_weights = loss_weights.numpy() # make sure pgan weight vector is a np.array
        disc_loss = np.dot(loss_weights, losses)  # weight and sum the losses
        
        gp_loss = gp_weight * gradient_penalty   #HOW TO IMPLEMENT THIS???
        disc_loss += gp_loss    #HOW TO IMPLEMENT THIS???

    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    return gen_loss, disc_loss, gp_loss, gen_sample
