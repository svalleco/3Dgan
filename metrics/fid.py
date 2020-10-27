"""
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/
python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3,
    HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary.
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
"""

import tensorflow as tf
import os
import functools
import numpy as np
from tensorflow.python.ops import array_ops
from utils import uniform_box_sampler
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import linalg_ops
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.contrib.layers.python.layers import layers
from six.moves import urllib
import tarfile
import sys

from dataset import NumpyDataset


def _validate_images(images, image_size):
    images = ops.convert_to_tensor(images)
    images.shape.with_rank(4)
    images.shape.assert_is_compatible_with([None, image_size, image_size, None])
    return images


def _symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.

    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.

    Also note that this method **only** works for symmetric matrices.

    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.

    Returns:
      Matrix square root of mat.
    """
    # Unlike numpy, tensorflow's return order is (s, u, v)
    s, u, v = linalg_ops.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = array_ops.where(math_ops.less(s, eps), s, math_ops.sqrt(s))
    # Note that the v returned by Tensorflow is v = V
    # (when referencing the equation A = U S V^T)
    # This is unlike Numpy which returns v = V^T
    return math_ops.matmul(
        math_ops.matmul(u, array_ops.diag(si)), v, transpose_b=True)


def trace_sqrt_product(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.

    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
       => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
       => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
       => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                     = sum(sqrt(eigenvalues(A B B A)))
                                     = sum(eigenvalues(sqrt(A B B A)))
                                     = trace(sqrt(A B B A))
                                     = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.

    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma

    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = _symmetric_matrix_square_root(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = math_ops.matmul(sqrt_sigma,
                                      math_ops.matmul(sigma_v, sqrt_sigma))

    return math_ops.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):
    """Classifier distance for evaluating a generative model.

    This methods computes the Frechet classifier distance from activations of
    real images and generated images. This can be used independently of the
    frechet_classifier_distance() method, especially in the case of using large
    batches during evaluation where we would like precompute all of the
    activations before computing the classifier distance.

    This technique is described in detail in https://arxiv.org/abs/1706.08500.
    Given two Gaussian distribution with means m and m_w and covariance matrices
    C and C_w, this function calculates

                  |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

    which captures how different the distributions of real images and generated
    images (or more accurately, their visual features) are. Note that unlike the
    Inception score, this is a true distance and utilizes information about real
    world images.

    Note that when computed using sample means and sample covariance matrices,
    Frechet distance is biased. It is more biased for small sample sizes. (e.g.
    even if the two distributions are the same, for a small sample size, the
    expected Frechet distance is large). It is important to use the same
    sample size to compute frechet classifier distance when comparing two
    generative models.

    Args:
      real_activations: 2D Tensor containing activations of real data. Shape is
        [batch_size, activation_size].
      generated_activations: 2D Tensor containing activations of generated data.
        Shape is [batch_size, activation_size].

    Returns:
     The Frechet Inception distance. A floating-point scalar of the same type
     as the output of the activations.

    """
    real_activations.shape.assert_has_rank(2)
    generated_activations.shape.assert_has_rank(2)

    activations_dtype = real_activations.dtype
    if activations_dtype != dtypes.float64:
        real_activations = math_ops.cast(real_activations, dtypes.float64)
        generated_activations = math_ops.cast(generated_activations, dtypes.float64)

    # Compute mean and covariance matrices of activations.
    m = math_ops.reduce_mean(real_activations, 0)
    m_w = math_ops.reduce_mean(generated_activations, 0)
    num_examples_real = math_ops.cast(
        array_ops.shape(real_activations)[0], dtypes.float64)
    num_examples_generated = math_ops.cast(
        array_ops.shape(generated_activations)[0], dtypes.float64)

    # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
    real_centered = real_activations - m
    sigma = math_ops.matmul(
        real_centered, real_centered, transpose_a=True) / (
                num_examples_real)  # Fix BS 1

    gen_centered = generated_activations - m_w
    sigma_w = math_ops.matmul(
        gen_centered, gen_centered, transpose_a=True) / (
                  num_examples_generated)  # Fix BS 1
    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = math_ops.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    # Next the distance between means.
    mean = math_ops.reduce_sum(
        math_ops.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
    fid = trace + mean
    if activations_dtype != dtypes.float64:
        fid = math_ops.cast(fid, activations_dtype)

    return fid



INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
INCEPTION_INPUT = 'Mul:0'
INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299
BATCH_SIZE = 64


def get_graph_def_from_url_tarball(url, filename, tar_filename=None):
    """Get a GraphDef proto from a tarball on the web.

    Args:
      url: Web address of tarball
      filename: Filename of graph definition within tarball
      tar_filename: Temporary download filename (None = always download)

    Returns:
      A GraphDef loaded from a file in the downloaded tarball.
    """
    if not (tar_filename and os.path.exists(tar_filename)):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (url,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        tar_filename, _ = urllib.request.urlretrieve(url, tar_filename, _progress)
    with tarfile.open(tar_filename, 'r:gz') as tar:
        proto_str = tar.extractfile(filename).read()
    return graph_pb2.GraphDef.FromString(proto_str)


def _default_graph_def_fn():
    return get_graph_def_from_url_tarball(INCEPTION_URL, INCEPTION_FROZEN_GRAPH,
                                          os.path.basename(INCEPTION_URL))


def run_image_classifier(tensor,
                         graph_def,
                         input_tensor,
                         output_tensor,
                         scope='RunClassifier'):
    """Runs a network from a frozen graph.

    Args:
      tensor: An Input tensor.
      graph_def: A GraphDef proto.
      input_tensor: Name of input tensor in graph def.
      output_tensor: A tensor name or list of tensor names in graph def.
      scope: Name scope for classifier.

    Returns:
      Classifier output if `output_tensor` is a string, or a list of outputs if
      `output_tensor` is a list.

    Raises:
      ValueError: If `input_tensor` or `output_tensor` aren't in the graph_def.
    """
    input_map = {input_tensor: tensor}
    is_singleton = isinstance(output_tensor, str)
    if is_singleton:
        output_tensor = [output_tensor]
    classifier_outputs = importer.import_graph_def(
        graph_def, input_map, output_tensor, name=scope)
    if is_singleton:
        classifier_outputs = classifier_outputs[0]

    return classifier_outputs


def run_inception(images,
                  graph_def=None,
                  default_graph_def_fn=_default_graph_def_fn,
                  image_size=INCEPTION_DEFAULT_IMAGE_SIZE,
                  input_tensor=INCEPTION_INPUT,
                  output_tensor=INCEPTION_OUTPUT):
    """Run images through a pretrained Inception classifier.

    Args:
      images: Input tensors. Must be [batch, height, width, channels]. Input shape
        and values must be in [-1, 1], which can be achieved using
        `preprocess_image`.
      graph_def: A GraphDef proto of a pretrained Inception graph. If `None`,
        call `default_graph_def_fn` to get GraphDef.
      default_graph_def_fn: A function that returns a GraphDef. Used if
        `graph_def` is `None. By default, returns a pretrained InceptionV3 graph.
      image_size: Required image width and height. See unit tests for the default
        values.
      input_tensor: Name of input Tensor.
      output_tensor: Name or list of output Tensors. This function will compute
        activations at the specified layer. Examples include INCEPTION_V3_OUTPUT
        and INCEPTION_V3_FINAL_POOL which would result in this function computing
        the final logits or the penultimate pooling layer.

    Returns:
      Tensor or Tensors corresponding to computed `output_tensor`.

    Raises:
      ValueError: If images are not the correct size.
      ValueError: If neither `graph_def` nor `default_graph_def_fn` are provided.
    """
    images = _validate_images(images, image_size)

    if graph_def is None:
        if default_graph_def_fn is None:
            raise ValueError('If `graph_def` is `None`, must provide '
                             '`default_graph_def_fn`.')
        graph_def = default_graph_def_fn()

    activations = run_image_classifier(images, graph_def, input_tensor,
                                       output_tensor)
    if isinstance(activations, list):
        for i, activation in enumerate(activations):
            if array_ops.rank(activation) != 2:
                activations[i] = layers.flatten(activation)
    else:
        if array_ops.rank(activations) != 2:
            activations = layers.flatten(activations)

    return activations


def inception_activations(images, num_splits=1):

    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.compat.v1.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    inc_activations = tf.map_fn(
        fn=functools.partial(run_inception, output_tensor='pool_3:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=8,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    inc_activations = array_ops.concat(array_ops.unstack(inc_activations), 0)
    return inc_activations


def get_inception_activations(session, activations, inception_images, inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE: i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(
            activations, feed_dict={inception_images: inp})

    return act


def activations2distance(session, activations1, activations2, act1, act2):
    assert act1.ndim == 2
    assert act2.ndim == 2
    fcd = frechet_classifier_distance_from_activations(activations1, activations2)
    print(act1.shape, act2.shape, act1.min(), act2.min(), act1.max(), act2.max(), act1.dtype,
          act2.dtype)
    d = session.run(fcd, feed_dict={activations1: act1, activations2: act2})
    print(d)
    return d


def get_fid(session, activations, inception_images, images1, images2):
    assert (type(images1) == np.ndarray)
    assert (len(images1.shape) == 4)
    assert (images1.shape[1] == 3)
    if not (np.min(images1) >= 0 and np.max(
            images1) > 10):
        print(f"WARNING: Image values should be in the range [0, 255], "
              f"got {images1.min(), images1.max()}")
    assert (type(images2) == np.ndarray)
    assert (len(images2.shape) == 4)
    assert (images2.shape[1] == 3)
    if not (np.min(images2) >= 0 and np.max(
            images2) > 10):
        print(
            f"WARNING: Image values should be in the range [0, 255], "
            f"got {images2.min(), images2.max()}")
    assert (images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    # print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    # start_time = time.time()
    activations1 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations1')
    activations2 = tf.compat.v1.placeholder(tf.float32, [None, None], name='activations2')

    act1 = get_inception_activations(session, activations, inception_images, images1)
    act2 = get_inception_activations(session, activations, inception_images, images2)
    fid = activations2distance(session, activations1, activations2, act1, act2)
    # print(f'FID {fid} calculation time: %f s' % (time.time() - start_time))
    return fid


def get_fid_for_volumes(session, activations, inception_images, volumes1, volumes2, normalize_op=None):
    if volumes1.shape[1] == 1:
        volumes1 = np.repeat(volumes1, 3, axis=1)
        volumes2 = np.repeat(volumes2, 3, axis=1)

    if normalize_op:
        volumes1 = normalize_op(volumes1)
        volumes2 = normalize_op(volumes2)

    # print('volumes1', volumes1.min(), volumes1.max(), volumes1.shape, volumes1.dtype)
    # print('volumes2', volumes2.min(), volumes2.max(), volumes2.shape, volumes2.dtype)

    fids = np.mean([get_fid(session, activations, inception_images, volumes1[:, :, i, ...], volumes2[:, :, i, ...]) for i in range(
        volumes1.shape[2])])

    print(fids)

    return fids


def test():
    # hvd.init()

    def norm_op(x):
        return (x * 255).astype(np.int16)

    shape = (128, 1, 16, 64, 64)
    const_batch1 = np.full(shape=shape, fill_value=.05).astype(np.float32)
    const_batch2 = np.full(shape=shape, fill_value=.05).astype(np.float32)
    rand_batch1 = np.random.rand(*shape)
    rand_batch2 = np.random.rand(*shape)
    black_noise1 = const_batch1 + np.random.randn(*const_batch1.shape) * .01
    black_noise2 = const_batch1 + np.random.randn(*const_batch1.shape) * .01
    noise_black_patches1 = rand_batch1.copy()
    noise_black_patches2 = rand_batch2.copy()

    for i in range(shape[0]):
        for _ in range(16):
            arr_slices = uniform_box_sampler(noise_black_patches1, min_width=(1, 1, 4, 8, 8,),
                                             max_width=(1, 1, 8, 16, 16))[0]
            noise_black_patches1[arr_slices] = 0

    for i in range(shape[0]):
        for _ in range(16):
            arr_slices = uniform_box_sampler(noise_black_patches2, min_width=(1, 1, 4, 8, 8,),
                                             max_width=(1, 1, 8, 16, 16))[0]
            noise_black_patches2[arr_slices] = 0

    # print("black/black", get_fid_for_volumes(const_batch1, const_batch2, normalize_op=norm_op))
    # print("rand/rand", get_fid_for_volumes(rand_batch1, rand_batch2, normalize_op=norm_op))
    # print("blacknoise/blacknoise", get_fid_for_volumes(black_noise1, black_noise2, normalize_op=norm_op))
    # print("blackpatches/blackpatches", get_fid_for_volumes(noise_black_patches1, noise_black_patches2, normalize_op=norm_op))

    # print('black/rand', get_fid_for_volumes(const_batch1, rand_batch1, normalize_op=norm_op))
    # print('rand/black+noise', get_fid_for_volumes(rand_batch1, black_noise1, normalize_op=norm_op))
    # print('black+patches/black+noise', get_fid_for_volumes(noise_black_patches1, black_noise1, normalize_op=norm_op))

    # print('black/black+noise', get_fid_for_volumes(const_batch1, black_noise1, normalize_op=norm_op))
    with tf.Session() as sess:

        inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None])
        activations = inception_activations(inception_images)

        # print('rand/rand+blackpatches',
        #     get_fid_for_volumes(sess, activations, inception_images, rand_batch1, noise_black_patches1, normalize_op=norm_op))

        # print('black/rand+blackpatches',
        #     get_fid_for_volumes(const_batch1, activations, inception_images, noise_black_patches1, normalize_op=norm_op))

        dataset = NumpyDataset('/lustre4/2/managed_datasets/LIDC-IDRI/npy/average/32x32/',
                               scratch_dir='/scratch-local/', copy_files=True)


        npy_files = np.stack([dataset[i] for i in range(128)])




if __name__ == '__main__':
    test()
