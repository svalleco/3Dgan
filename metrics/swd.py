import numpy as np
import scipy.ndimage
from utils import uniform_box_sampler

filter_1d = [1, 4, 6, 4, 1]
f = np.array(filter_1d, dtype=np.float32)
f = f[:, np.newaxis, np.newaxis] * f[np.newaxis, np.newaxis, :] * f[np.newaxis, :, np.newaxis]
gaussian_filter = f / f.sum()
_GAUSSIAN_FILTER = gaussian_filter.reshape(5, 5, 5)


# ----------------------------------------------------------------------------
def get_descriptors_for_minibatch(minibatch, nhood_size, nhoods_per_image):
    S = minibatch.shape  # (minibatch, channel, depth, height, width)
    assert len(S) == 5
    N = nhoods_per_image * S[0]
    D = nhood_size[0] // 2
    H = nhood_size[1] // 2
    W = nhood_size[2] // 2
    nhood, chan, d, x, y = np.ogrid[0:N, 0:S[1], -D:D + 1, -H:H + 1, -W:W + 1]
    img = nhood // nhoods_per_image
    d = d + np.random.randint(D, S[2] - D, size=(N, 1, 1, 1, 1))
    x = x + np.random.randint(W, S[4] - W, size=(N, 1, 1, 1, 1))
    y = y + np.random.randint(H, S[3] - H, size=(N, 1, 1, 1, 1))
    idx = (((img * S[1] + chan) * S[2] + d) * S[3] + y) * S[4] + x
    return minibatch.flat[idx]


# ----------------------------------------------------------------------------

def finalize_descriptors(desc):
    if isinstance(desc, list):
        desc = np.concatenate(desc, axis=0)
    assert desc.ndim == 5  # (neighborhood, channel, depth, height, width)
    if desc.shape[1] > 1:
        desc -= np.mean(desc, axis=(0, 2, 3, 4), keepdims=True)
        desc /= np.std(desc, axis=(0, 2, 3, 4), keepdims=True)
    desc = desc.reshape(desc.shape[0], -1)
    return desc


# ----------------------------------------------------------------------------

def sliced_wasserstein(a, b, dir_repeats, dirs_per_repeat):
    assert a.ndim == 2 and b.shape == b.shape  # (neighborhood, descriptor_component)
    results = []
    for repeat in range(dir_repeats):
        dirs = np.random.randn(a.shape[1], dirs_per_repeat)  # (descriptor_component, direction)
        dirs /= np.sqrt(
            np.sum(np.square(dirs), axis=0, keepdims=True))  # normalize descriptor components for each direction
        dirs = dirs.astype(np.float32)
        projA = np.matmul(a, dirs)  # (neighborhood, direction)
        projB = np.matmul(b, dirs)
        projA = np.sort(projA, axis=0)  # sort neighborhood projections for each direction
        projB = np.sort(projB, axis=0)
        dists = np.abs(projA - projB)  # pointwise wasserstein distances
        results.append(np.mean(dists))  # average over neighborhoods and directions
    return np.mean(results)  # average over repeats


# ----------------------------------------------------------------------------

def pyr_down(minibatch):  # matches cv2.pyrDown()
    assert minibatch.ndim == 5
    return scipy.ndimage.convolve(
        minibatch, _GAUSSIAN_FILTER[np.newaxis, np.newaxis, ...], mode='mirror')[:, :, ::2, ::2, ::2]


def pyr_up(minibatch):  # matches cv2.pyrUp()
    assert minibatch.ndim == 5
    S = minibatch.shape
    res = np.zeros((S[0], S[1], S[2] * 2, S[3] * 2, S[4] * 2), minibatch.dtype)
    res[:, :, ::2, ::2, ::2] = minibatch
    return scipy.ndimage.convolve(res, _GAUSSIAN_FILTER[np.newaxis, np.newaxis, ...] * 8.0, mode='mirror')


def generate_laplacian_pyramid(minibatch, num_levels):
    pyramid = [np.float32(minibatch)]
    for i in range(1, num_levels):
        pyramid.append(pyr_down(pyramid[-1]))
        pyramid[-2] -= pyr_up(pyramid[-1])
    return pyramid


def reconstruct_laplacian_pyramid(pyramid):
    minibatch = pyramid[-1]
    for level in pyramid[-2::-1]:
        minibatch = pyr_up(minibatch) + level
    return minibatch


def get_swd_for_volumes(images1, images2, nhood_size=(2, 8, 8), nhoods_per_image=512, dir_repeats=8,
                        dirs_per_repeat=512):
    resolutions = []
    res = images1.shape[-1]

    while res >= 16:
        resolutions.append(res)
        res //= 2

    descriptors_real = [[] for _ in resolutions]
    descriptors_fake = [[] for _ in resolutions]

    if len(descriptors_real) == 0:
        print("No descriptors, probably resolution is too small. Returning None")
        return None

    for lod, level in enumerate(generate_laplacian_pyramid(images1, len(resolutions))):
        desc = get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image)
        descriptors_real[lod].append(desc)

    for lod, level in enumerate(generate_laplacian_pyramid(images2, len(resolutions))):
        desc = get_descriptors_for_minibatch(level, nhood_size, nhoods_per_image)
        descriptors_fake[lod].append(desc)

    descriptors_real = [finalize_descriptors(np.concatenate(d)) for d in descriptors_real]
    descriptors_fake = [finalize_descriptors(np.concatenate(d)) for d in descriptors_fake]

    dist = [sliced_wasserstein(dreal, dfake, dir_repeats, dirs_per_repeat) for dreal, dfake in
            zip(descriptors_real, descriptors_fake)]

    dist = dist + [np.mean(dist)]

    return dist


if __name__ == '__main__':
    # ----------------------------------------------------------------------------

    shape = (128, 1, 32, 128, 128)
    const_batch1 = np.full(shape=shape, fill_value=-1000).astype(np.int16)
    const_batch2 = np.full(shape=shape, fill_value=-1000).astype(np.int16)
    rand_batch1 = (np.clip(np.random.rand(*shape), -1, 2) * 1024).astype(np.int16)
    rand_batch2 = (np.clip(np.random.rand(*shape), -1, 2) * 1024).astype(np.int16)
    black_noise1 = const_batch1 + np.random.randn(*const_batch1.shape)
    black_noise2 = const_batch1 + np.random.randn(*const_batch1.shape)
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

    # for i in range(shape[2]):
    #     img = noise_black_patches1[0, 0, i]
    #     plt.imshow(img)
    #     plt.savefig(f'test{i}.png')
    #     plt.close()

    print("black/black", get_swd_for_volumes(const_batch1, const_batch2))
    print("rand/rand", get_swd_for_volumes(rand_batch1, rand_batch2))
    print("black_noise1/black_noise2", get_swd_for_volumes(black_noise1, black_noise2))
    print("patches1/patches2", get_swd_for_volumes(noise_black_patches1, noise_black_patches2))

    print('black/rand', get_swd_for_volumes(const_batch1, rand_batch1, ))
    print('rand/black+noise', get_swd_for_volumes(rand_batch1, black_noise1, ))
    print('patches/black+noise', get_swd_for_volumes(noise_black_patches1, black_noise1))

    print('rand/rand+blackpatches', get_swd_for_volumes(rand_batch1, noise_black_patches1, ))
    print('black/black+noise', get_swd_for_volumes(const_batch1, black_noise1, ))

    print('black/rand+blackpatches', get_swd_for_volumes(const_batch1, noise_black_patches1, ))
