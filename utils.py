import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import ast
from multiprocessing import Pool
import os
import horovod.tensorflow as hvd

# Op to update the learning rate according to a schedule
def lr_update(lr, intra_phase_step, steps_per_phase, lr_max, lr_increase, lr_decrease, lr_rise_niter, lr_decay_niter):
    """Update the learning rate according to a schedule.
    Args:
      lr: Tensor which contains the current learning rate that needs to be updated
      intra_phase_step: Step counter representing the number of images processed since the start of the current phase
      steps_per_phase: total number of steps in a phase
      lr_max: learning rate after increase (and before decrease) segments
      lr_increase: type of increase function to use (e.g. None, linear, exponential)
      lr_decrease: type of decrease function to use (e.g. None, linear, exponential)
      lr_rise_niter: number of iterations over which the increase from the minimum to the maximum value should happen
      lr_decay_niter: number of iterations over which the decrease from the maximum to the minumum value should happen.
    Returns: an Op that can be passed to session.run to update the learning (lr) Tensor
    """

    # Default starting point is that update_lr = lr_max. If there are no lr_increase or lr_decrease
    # functions specified, it stays like this.
    lr_update = lr_max

    # Is a learning rate schedule defined at all? (otherwise, immediately return a constant)
    if (lr_increase or lr_decrease):
        # Rather than if-else statements, the way to define a piecewiese function is through tf.cond

        # Prepare some variables:
        a = tf.cast(tf.math.divide(lr_max, 100), tf.float32)
        b_rise = tf.cast(tf.math.divide(np.log(100), lr_rise_niter), tf.float32)
        b_decay = tf.cast(tf.math.divide(np.log(100), lr_decay_niter), tf.float32)
        step_decay_start = tf.subtract(steps_per_phase, lr_decay_niter)
        remaining_steps = tf.subtract(steps_per_phase, intra_phase_step)

        # Define the different functions
        def update_increase_lin ():
            return tf.multiply(
                               tf.cast(tf.truediv(intra_phase_step, lr_rise_niter), tf.float32),
                               lr_max
                               )
        def update_increase_exp():
            return tf.multiply(
                                a,
                                tf.math.exp(tf.multiply(b_rise, tf.cast(intra_phase_step, tf.float32)))
                                )

        def update_decrease_lin():
            return tf.multiply(
                               tf.cast(tf.truediv(remaining_steps, lr_decay_niter), tf.float32),
                               lr_max
                               )

        def update_decrease_exp():
            return tf.multiply(
                                a,
                                tf.math.exp(tf.multiply(b_decay, tf.cast(remaining_steps, tf.float32)))
                                )

        def no_op():
            return lr_update

        if lr_increase == 'linear':
            # Are we in the increasing part? Return update_increase_lin function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step < lr_rise_niter, update_increase_lin, no_op)
        elif lr_increase == 'exponential':
            # Are we in the increasing part? Return update_increase_exp function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step < lr_rise_niter, update_increase_exp, no_op)
            
        if lr_decrease == 'linear':
            # Are we in the decreasing part? Return return update_decrease function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step > step_decay_start, update_decrease_lin, no_op) 
        elif lr_decrease == 'exponential':
            # Are we in the decreasing part? Return return update_decrease function (else, keep update_lr unchanged)
            lr_update = tf.cond(intra_phase_step > step_decay_start, update_decrease_exp, no_op) 
 
    return lr.assign(lr_update)

# Op to update the learning rate according to a schedule
def lr_update_numpy(lr, intra_phase_step, steps_per_phase, lr_max, lr_increase, lr_decrease, lr_rise_niter, lr_decay_niter):
    """Update the learning rate according to a schedule.
    Args:
      lr: Tensor which contains the current learning rate that needs to be updated
      intra_phase_step: Step counter representing the number of images processed since the start of the current phase
      steps_per_phase: total number of steps in a phase
      lr_max: learning rate after increase (and before decrease) segments
      lr_increase: type of increase function to use (e.g. None, linear, exponential)
      lr_decrease: type of decrease function to use (e.g. None, linear, exponential)
      lr_rise_niter: number of iterations over which the increase from the minimum to the maximum value should happen
      lr_decay_niter: number of iterations over which the decrease from the maximum to the minumum value should happen.
    Returns: an Op that can be passed to session.run to update the learning (lr) Tensor
    """
    # Is a learning rate schedule defined at all? (otherwise, immediately return a constant)
    if not (lr_increase or lr_decrease):
        return lr.assign(lr_max)
    else:
        # Are we in the increasing part?
        if intra_phase_step < lr_rise_niter:
            if not lr_increase:
                updated_lr = lr_max
            elif lr_increase == 'linear':
                updated_lr = (intra_phase_step / lr_rise_niter) * lr_max
            elif lr_increase == 'exponential':
                # Define lr at step 0 to be 1% of lr_max
                a = lr_max / 100
                # Make sure then when intra_phase_step = lr_rise_niter, the lr = lr_max
                b = np.log(100 / lr_rise_niter) 
                update_lr = a*np.exp(b*intra_phase_step)  
            else:
                raise NotImplementedError("Unsupported learning rate increase type: %s" % lr_increase)
        # Are we in the decreasing part?
        elif intra_phase_step > (steps_per_phase - lr_decay_niter):
            if not lr_decrease:
                updated_lr = lr_max
            if lr_decrease == 'linear':
                updated_lr = ((steps_per_phase - intra_phase_step) / lr_decay_niter) * lr_max
            elif lr_increase == 'exponential':
                # Define lr at the last step to be 1% of lr_max
                a = lr_max / 100
                # Make sure that when intra_phase_step == steps_per_phase - lr_decay_niter, lr_max is returned
                b = np.log(100 / lr_decay_niter)
                update_lr = a*np.exp(b*(steps_per_phase - intra_phase_step))
            else:
                raise NotImplementedError("Unsupported learning rate decrease type: %s" % lr_decrease)
        # Are we in the flat part?
        else:
            updated_lr = lr_max
            
        return lr.assign(updated_lr)


# log0 only logs from hvd.rank() == 0
def log0(string):
    if hvd.rank() == 0:
        print(string)

def parse_tuple(string):
    s = ast.literal_eval(str(string))
    return s


def count_parameters(scope):
    return sum(np.product(p.shape) for p in tf.trainable_variables(scope))


def image_grid(input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
    """Arrange a minibatch of images into a grid to form a single image.
    Args:
      input_tensor: Tensor. Minibatch of images to format, either 4D
          ([batch size, height, width, num_channels]) or flattened
          ([batch size, height * width * num_channels]).
      grid_shape: Sequence of int. The shape of the image grid,
          formatted as [grid_height, grid_width].
      image_shape: Sequence of int. The shape of a single image,
          formatted as [image_height, image_width].
      num_channels: int. The number of channels in an image.
    Returns:
      Tensor representing a single image in which the input images have been
      arranged into a grid.
    Raises:
      ValueError: The grid shape and minibatch size don't match, or the image
          shape and number of channels are incompatible with the input tensor.
    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
        raise ValueError("Grid shape %s incompatible with minibatch size %i." %
                         (grid_shape, int(input_tensor.shape[0])))
    if len(input_tensor.shape) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.shape[1]) != num_features:
            raise ValueError("Image shape and number of channels incompatible with "
                             "input tensor.")
    elif len(input_tensor.shape) == 4:
        if (int(input_tensor.shape[1]) != image_shape[0] or
                int(input_tensor.shape[2]) != image_shape[1] or
                int(input_tensor.shape[3]) != num_channels):
            raise ValueError("Image shape and number of channels incompatible with "
                             "input tensor.")
    else:
        raise ValueError("Unrecognized input tensor format.")
    height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = array_ops.reshape(
        input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
    input_tensor = array_ops.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = array_ops.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = array_ops.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = array_ops.reshape(
        input_tensor, [1, height, width, num_channels])

    return input_tensor


def uniform_box_sampler(arr, min_width, max_width):
    """
    Extracts a sample cut from `arr`.

    Parameters:
    -----------
    arr : array
        The numpy array to sample a box from
    min_width : int or tuple
        The minimum width of the box along a given axis.
        If a tuple of integers is supplied, it my have the
        same length as the number of dimensions of `arr`
    max_width : int or tuple
        The maximum width of the box along a given axis.
        If a tuple of integers is supplied, it my have the
        same length as the number of dimensions of `arr`

    Returns:
    --------
    (slices, x) : A tuple of the slices used to cut the sample as well as
    the sampled subsection with the same dimensionality of arr.
        slice :: list of slice objects
        x :: array object with the same ndims as arr
    """
    if isinstance(min_width, (tuple, list)):
        assert len(min_width) == arr.ndim, 'Dimensions of `min_width` and `arr` must match'

    else:
        min_width = (min_width,) * arr.ndim
    if isinstance(max_width, (tuple, list)):
        assert len(max_width) == arr.ndim, 'Dimensions of `max_width` and `arr` must match'
    else:
        max_width = (max_width,) * arr.ndim

    slices = []
    for dim, mn, mx in zip(arr.shape, min_width, max_width):
        start = int(np.random.uniform(0, dim))
        stop = start + int(np.random.uniform(mn, mx + 1))
        slices.append(slice(start, stop))
    return slices, arr[slices]


class MPMap:
    def __init__(self, f):
        self.pool = Pool(int(os.environ['OMP_NUM_THREADS']))
        self.f = f

    def map(self, l: list):
        return self.pool.map_async(self.f, l)

    def close(self):
        self.pool.close()
