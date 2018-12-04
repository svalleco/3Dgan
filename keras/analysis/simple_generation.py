
import argparse
import os, h5py
import numpy as np
import context
#from context import EcalEnergyGan
from EcalEnergyGan import discriminator as build_discriminator
from EcalEnergyGan import generator as build_generator

def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate 3D images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )   
    parser.add_argument('--nevents', '-n', action='store', type=int,
                        default='1000', help='Number of events to simulate')
    parser.add_argument('--latent-size', action='store', type=int, default=200,
                        help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--nrun', action='store', type=int, default=1,
                        help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--gweights', '-g', action='store', type=str,
                        default='params_generator_epoch_029.hdf5', help='path to generator weights file')
    parser.add_argument('--dweights', '-d', action='store', type=str,
                        default='params_discriminator_epoch_029.hdf5', help='path to discriminator weights file')


    return parser

parser = get_parser()
results = parser.parse_args()
n_events = results.nevents
latent_space = results.latent_size
gen_weights = results.gweights
disc_weights = results.dweights
#gen_weights='/data/svalleco/GAN/weights_gk/params_generator_epoch_029.hdf5'
#disc_weights='/data/svalleco/GAN/weights_gk/params_discriminator_epoch_029.hdf5'
outfile_name = 'generation{0:03d}.hdf5'.format(results.nrun)
#if not ((os.path.isfile(gen_weights)) & (os.path.isfile(disc_weights))):
    # download from somewhere
    # but for now just load mine
#    gen_weights='params_generator_epoch_048.hdf5'
#    disc_weights='params_discriminator_epoch_048.hdf5'

np.random.seed()
g = build_generator(latent_space, return_intermediate=False)
g.load_weights(gen_weights)

noise = np.random.normal(0, 1, (n_events, latent_space))
sampled_energies = np.random.uniform(1, 5, (n_events, 1))
generator_ip = np.multiply(sampled_energies, noise)

generated_images = g.predict(generator_ip, verbose=False, batch_size=128)
print(generated_images.shape)
def safe_mkdir(path):
    '''
    Safe mkdir (i.e., don't create if already exists, 
    and no violation of race conditions)
    '''
    from os import makedirs
    from errno import EEXIST
    try:
        makedirs(path)
    except OSError as exception:
        if exception.errno != EEXIST:
            raise exception
outdir = 'plots'
safe_mkdir(outdir)

#f = plt.figure(figsize=(6, 6))
bins = np.linspace(0, 1, 30)

#_ = plt.hist(isreal, bins=bins, histtype='step', label='GAN', color='green')

#plt.legend(ncol=2, mode='expand')
#plt.xlabel('P(real)')
#plt.ylabel('Number of events')
#plt.ylim(ymax=4000)

#plt.tight_layout()
#plt.savefig(os.path.join('..', outdir, 'prob_real.pdf'))



generated_images = np.squeeze(generated_images)

with h5py.File(outfile_name,'w') as outfile:
    outfile.create_dataset('ECAL',data=generated_images)
