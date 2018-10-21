
import argparse
import os, h5py
import numpy as np

from gan3D import discriminator as build_discriminator
from gan3D import generator as build_generator

def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate 3D images'
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )   
    parser.add_argument('--nevents', '-n', action='store', type=int,
                        default='1000', help='Number of events to simulate')
    parser.add_argument('--latent-size', action='store', type=int, default=200,
                        help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--nrun', action='store', type=int, default=0,
                        help='size of random N(0, 1) latent space to sample')


    return parser

parser = get_parser()
results = parser.parse_args()
n_events = results.nevents
latent_space = results.latent_size

#gen_weights = 'gen_weights.hdf5'
#disc_weights = 'disc_weights.hdf5'
gen_weights='/data/svalleco/GAN/weights_gk/params_generator_epoch_029.hdf5'
disc_weights='/data/svalleco/GAN/weights_gk/params_discriminator_epoch_029.hdf5'
outfile_name = '/data/svalleco/GAN/generatedImages/generation{1:03d}.hdf5'.format(results.nrun)
outfileE_name = '/data/svalleco/GAN/generatedImages/generationE{1:03d}.hdf5'.format(results.nrun)
outfileG_name = '/data/svalleco/GAN/generatedImages/generationG{1:03d}.hdf5'.format(results.nrun)
#if not ((os.path.isfile(gen_weights)) & (os.path.isfile(disc_weights))):
    # download from somewhere
    # but for now just load mine
#    gen_weights='params_generator_epoch_048.hdf5'
#    disc_weights='params_discriminator_epoch_048.hdf5'

np.random.seed()
g = build_generator(latent_space, return_intermediate=False)
g.load_weights(gen_weights)

noise = np.random.normal(0, 1, (n_events, latent_space))
sampled_labels = np.random.randint(0, 2, n_events)


generated_images = g.predict(
    [noise, sampled_labels.reshape(-1, 1)], verbose=False, batch_size=100)
print generated_images.shape
d = build_discriminator()
d.load_weights(disc_weights)

isreal, aux_out = np.array(
    d.predict(generated_images, verbose=False, batch_size=100)
)
#isreal_pythia, aux_out_pythia = np.array(
#    d.predict(np.expand_dims(real_images / 100, -1), verbose=False, batch_size=100)
#)
print aux_out.shape
print isreal.shape
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



generated_images = (generated_images) * 100.
generated_images = np.squeeze(generated_images)
np.float64(generated_images)
print generated_images.dtype
fakeHCAL = np.zeros((n_events,5,5,60))
fakeTARGET = np.ones((n_events,1,5))
#print isreal
print np.where(aux_out>0.8000)

with h5py.File(outfile_name,'w') as outfile:
    outfile.create_dataset('ECAL',data=generated_images)
    outfile.create_dataset('HCAL',data=fakeHCAL)
    outfile.create_dataset('target',data=fakeTARGET)

with h5py.File(outfileE_name,'w') as outfileE:
    outfileE.create_dataset('ECAL',data=generated_images[np.where(aux_out>0.8000)[0]])
    outfileE.create_dataset('HCAL',data=fakeHCAL)
    outfileE.create_dataset('target',data=fakeTARGET)

with h5py.File(outfileG_name,'w') as outfileG:
    outfileG.create_dataset('ECAL',data=generated_images[np.where(aux_out<0.2000)[0]])
    outfileG.create_dataset('HCAL',data=fakeHCAL)
    outfileG.create_dataset('target',data=fakeTARGET)

