from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity
import numpy as np


def get_mean_squared_error(real, fake):
    return mean_squared_error(real, fake)


def get_normalized_root_mse(real, fake):
    return normalized_root_mse(real, fake, normalization='min-max')


def get_psnr(real, fake, data_range=3072):
    return peak_signal_noise_ratio(real, fake, data_range=data_range)


def get_ssim(real, fake, data_range=3):
    real = np.transpose(real, [0, 2, 3, 4, 1])
    fake = np.transpose(fake, [0, 2, 3, 4, 1])
    if real.shape[0] == 1:
        real = real[0, ...]
    if fake.shape[0] == 1:
        fake = fake[0, ...]
    return structural_similarity(real, fake, data_range=data_range, multichannel=True, gaussian_weights=True)


if __name__ == '__main__':

    volume1 = (np.clip(np.random.normal(size=(1, 1, 16, 64, 64)), -1, 2) * 1024).astype(np.int16)
    volume2 = (np.clip(np.random.normal(size=(1, 1, 16, 64, 64)), -1, 2) * 1024).astype(np.int16)

    print(volume1.min(), volume2.max(), volume1.min(), volume2.max())

    print(get_mean_squared_error(volume1, volume2))
    print(get_normalized_root_mse(volume1, volume2))
    print(get_psnr(volume1, volume2))
    print(get_ssim(volume1, volume2))
