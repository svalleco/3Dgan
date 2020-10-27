import tensorflow as tf
import numpy as np

def k(x):
    if x < 3:
        return 1
    else:
        return 3

def calculate_gain(activation, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if activation in linear_fns or activation == 'sigmoid':
        return 1
    elif activation == 'tanh':
        return 5.0 / 3
    elif activation == 'relu':
        return np.sqrt(2.0)
    elif activation == 'leaky_relu':
        assert param is not None
        if not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return np.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(activation))


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def get_weight(shape, activation, lrmul=1, use_eq_lr=True, use_spectral_norm=False, param=None):
    fan_in = np.prod(shape[:-1])
    gain = calculate_gain(activation, param)
    he_std = gain / np.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul

    w = tf.get_variable('weight', shape=shape,
                        initializer=tf.initializers.random_normal(0, init_std))

    if use_eq_lr:
        w = w * runtime_coef

    if use_spectral_norm:
        w = spectral_norm(w)

    return w


def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1, 1])


def dense(x, fmaps, activation, lrmul=1, param=None):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], activation, lrmul=lrmul, param=param)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


def conv3d(x, fmaps, kernel, activation, param=None, lrmul=1):
    w = get_weight([*kernel, x.shape[1].value, fmaps], activation, param=param, lrmul=lrmul)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NCDHW')


def group_conv3d(x, filter, groups):

    inputs = tf.split(x, groups, axis=1)
    filters = tf.split(filter, groups, axis=-2)
    output = tf.concat(
        [tf.nn.conv3d(i, f,
                      strides=[1, 1, 1, 1, 1],
                      padding='SAME',
                      data_format='NCDHW')
         for i, f in zip(inputs, filters)], axis=1)

    return output


def leaky_relu(x, alpha_lr=0.2):
    with tf.variable_scope('leaky_relu'):
        alpha_lr = tf.constant(alpha_lr, dtype=x.dtype, name='alpha_lr')

        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha_lr)

            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha_lr)
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha_lr)

            return y, grad

        return func(x)


def act(x, activation, param=None):
    if activation == 'leaky_relu':
        assert param is not None
        return leaky_relu(x, alpha_lr=param)
    elif activation == 'linear':
        return x
    else:
        raise ValueError(f"Unknown activation {activation}")


# def num_filters(phase, num_phases, base_dim):
#     num_downscales = int(np.log2(base_dim / 32))
#     filters = min(base_dim // (2 ** (phase - num_phases + num_downscales)), base_dim)
#     return filters


def num_filters(phase, num_phases, base_shape, base_dim=None, size=None):
    if size == 'xxs':
        filter_list = [256, 256, 64, 32, 16, 8, 4, 2]
    elif size == 'xs':
        filter_list = [256, 256, 64, 64, 32, 16, 8, 4]
    elif size == 's':
        filter_list = [512, 512, 128, 128, 64, 32, 16, 8]
    elif size == 'm':
        filter_list = [1024, 1024, 256, 256, 128, 64, 32, 16]
#        filter_list = [256, 256, 128, 64, 32, 16]
    elif size == 'l':
        filter_list = [2048, 2048, 512, 512, 256, 128, 64, 32]
    elif size == 'xl':
        filter_list = [4096, 4096, 1024, 1024, 512, 256, 128, 64]
    elif size == 'xxl':
        filter_list = [8192, 8192, 2048, 1024, 1024, 512, 256, 128]
    else:
        raise ValueError(f"Unknown size: {size}")
    assert len(filter_list) == 8, "Filter lists are built for LIDC-IDRI dataset."
    # filter_list = filter_list[-num_phases:]
    # Take base_shape[1:] to cut of the number of input channels:
    # We want to determine number of filters based on spatial number of voxels; channels are irrelevant
    current_dim = [2 ** (phase - 1) * dim for dim in base_shape[1:]]
    print(f"DEBUG: base_shape={base_shape}, phase={phase}, current_dim={current_dim}")
    log_product = np.log2(np.product(current_dim))
    # Filter lists were designed for dimensions where the 2-log is [4, 7, 10, ...]
    reference_log = [4 + n * 3 for n in range(0,7)]
    # Map the index to the nearest reference log
    # E.g. for dimension [16, 16, 5] the product is 1280, log2(1280) = 10.32 which is closest
    # to 10, thus I get the third element from filter_list as the number of filters.
    index = np.argmin(np.abs(np.array(reference_log)-log_product))
    filters = filter_list[index]
    print(f"DEBUG: log_product={log_product}, index={index}, filters={filters}")
    # filters = filter_list[phase - 1]
# print(f"DEBUG: returning num_filters: {filters}")
    return filters


def to_rgb(x, channels=1):
    return apply_bias(conv3d(x, channels, (1, 1, 1), activation='linear'))


def from_rgb(x, filters_out, activation, param=None):
    x = conv3d(x, filters_out, (1, 1, 1), activation, param)
    x = apply_bias(x)
    x = act(x, activation, param=param)
    return x


def avg_unpool3d(x, factor=2, gain=1):
    if gain != 1:
        x = x * gain

    if factor == 1:
        return x

    x = tf.transpose(x, [2, 3, 4, 1, 0])  # [B, C, D, H, W] -> [D, H, W, C, B]
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [factor ** 3, 1, 1, 1, 1, 1])
    x = tf.batch_to_space_nd(x, [factor, factor, factor], [[0, 0], [0, 0], [0, 0]])
    x = tf.transpose(x[0], [4, 3, 0, 1, 2])  # [D, H, W, C, B] -> [B, C, D, H, W]
    return x


def avg_pool3d(x, factor=2, gain=1):
    if gain != 1:
        x *= gain

    if factor == 1:
        return x

    ksize = [1, 1, factor, factor, factor]
    return tf.nn.avg_pool3d(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCDHW')


def upscale3d(x, factor=2):
    with tf.variable_scope('upscale_3d'):
        @tf.custom_gradient
        def func(x):
            y = avg_unpool3d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = avg_pool3d(dy, factor, gain=factor ** 3)
                return dx, lambda ddx: avg_unpool3d(ddx, factor)

            return y, grad

        return func(x)


def downscale3d(x, factor=2):
    with tf.variable_scope('downscale_3d'):
        @tf.custom_gradient
        def func(x):
            y = avg_pool3d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = avg_unpool3d(dy, factor, gain=1 / factor ** 3)
                return dx, lambda ddx: avg_pool3d(ddx, factor)

            return y, grad

        return func(x)


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('pixel_norm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('minibatch_std'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])
        s = x.shape
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3], s[4]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[1, 2, 3, 4], keepdims=True)
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3], s[4]])
        return tf.concat([x, y], axis=1)


def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 5  # NCDHW
    with tf.variable_scope('instance_norm'):
        x -= tf.reduce_mean(x, axis=[2, 3, 4], keepdims=True)
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3, 4], keepdims=True) + epsilon)
        return x


def apply_noise(x):
    assert len(x.shape) == 5  # NCDHW
    with tf.variable_scope('apply_noise'):
        noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3], x.shape[4]])
        noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        return x + noise * noise_strength


def style_mod(x, dlatent, activation, param=None):
    with tf.variable_scope('style_mod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[1] * 2, activation=activation, param=param))
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:, 0] + 1) + style[:, 1]


def conv3d_depthwise(x, f, strides, padding):
    x = tf.split(x, x.shape[-1], axis=-1)
    filters = tf.split(f, f.shape[-2], axis=-2)
    x = tf.concat([tf.nn.conv3d(i, f, strides=strides, padding=padding) for i, f in zip(x, filters)], axis=-1)
    return x

