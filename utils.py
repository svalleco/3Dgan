import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import ast
from multiprocessing import Pool
import os


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
