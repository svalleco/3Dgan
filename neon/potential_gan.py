import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
#from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.initializers import Gaussian
from neon.layers import GeneralizedGANCost, Affine, Linear, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm
from neon.layers.layer import Linear, Reshape
from neon.layers.container import GenerativeAdversarial
from neon.models.model import GAN, Model
from neon.transforms import Rectlin, Logistic, GANCost, Tanh
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from neon.layers.layer import Dropout
from neon.data.hdf5iterator import HDF5Iterator
from neon.optimizers import GradientDescentMomentum, RMSProp, Adam
from gen_data_norm import gen_rhs
from neon.backends import gen_backend
from temporary_utils import temp_3Ddata
import numpy as np
# import matplotlib.pyplot as plt
import h5py

# load up the data set

print 'starting HDF5Iterator'
train_set = HDF5Iterator('EGshuffled_train.h5')
valid_set = HDF5Iterator('EGshuffled_test.h5')

print 'train_set OK'
#tate=lt.plot(X_train[0, 12])
#plt.savefigure('data_img.png')

# setup weight initialization function
init = Gaussian(scale=0.01)

# discriminiator using convolution layers
lrelu = Rectlin(slope=0.1)  # leaky relu for discriminator
# sigmoid = Logistic() # sigmoid activation function
conv1 = dict(init=init, batch_norm=False, activation=lrelu, bias=init)
conv2 = dict(init=init, batch_norm=False, activation=lrelu, padding=2, bias=init)
conv3 = dict(init=init, batch_norm=False, activation=lrelu, padding=1, bias=init)
D_layers = [
            Conv((5, 5, 5, 32), **conv1),
            Dropout(keep = 0.8),
            Conv((5, 5, 5, 8), **conv2),
            BatchNorm(),
            Dropout(keep = 0.8),
            Conv((5, 5, 5, 8), **conv2),
            BatchNorm(),
            Dropout(keep = 0.8),
            Conv((5, 5, 5, 8), **conv3),
            BatchNorm(),
            Dropout(keep = 0.8),
            Pooling((2, 2, 2)),
            Affine(1024, init=init, activation=lrelu),
            BatchNorm(),
            Affine(1024, init=init, activation=lrelu),
            BatchNorm(),
            Affine(1, init=init, bias=init, activation=Logistic())
            ]

# generator using convolution layers
init_gen = Gaussian(scale=0.001)
relu = Rectlin(slope=0)  # relu for generator
pad1 = dict(pad_h=2, pad_w=2, pad_d=2)
str1 = dict(str_h=2, str_w=2, str_d=2)
conv1 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad1, strides=str1, bias=init_gen)
pad2 = dict(pad_h=2, pad_w=2, pad_d=2)
str2 = dict(str_h=2, str_w=2, str_d=2)
conv2 = dict(init=init_gen, batch_norm=False, activation=lrelu, padding=pad2, strides=str2, bias=init_gen)
pad3 = dict(pad_h=0, pad_w=0, pad_d=0)
str3 = dict(str_h=1, str_w=1, str_d=1)
conv3 = dict(init=init_gen, batch_norm=False, activation=Tanh(), padding=pad3, strides=str3, bias=init_gen)
G_layers = [
            Affine(1024, init=init_gen, bias=init_gen, activation=relu),
            BatchNorm(),
            Affine(8 * 7 * 7 * 7, init=init_gen, bias=init_gen),
            Reshape((8, 7, 7, 7)),
            Deconv((6, 6, 6, 6), **conv1), #14x14x14
            BatchNorm(),
            # Linear(5 * 14 * 14 * 14, init=init),
            # Reshape((5, 14, 14, 14)),
            Deconv((5, 5, 5, 64), **conv2), #27x27x27
            BatchNorm(),
            Conv((3, 3, 3, 1), **conv3)
           ]

layers = GenerativeAdversarial(generator=Sequential(G_layers, name="Generator"),
                               discriminator=Sequential(D_layers, name="Discriminator"))
print 'layers defined'
# setup optimizer
# optimizer = RMSProp(learning_rate=1e-4, decay_rate=0.9, epsilon=1e-8)
optimizer = GradientDescentMomentum(learning_rate=1e-3, momentum_coef = 0.9)
#optimizer = Adam(learning_rate=1e-3)

# setup cost function as Binary CrossEntropy
cost = GeneralizedGANCost(costfunc=GANCost(func="wasserstein"))

nb_epochs = 15
latent_size = 200
inb_classes = 2
nb_test = 100

# initialize model
noise_dim = (latent_size)
gan = GAN(layers=layers, noise_dim=noise_dim, k=5, wgan_param_clamp=0.9)

# configure callbacks
callbacks = Callbacks(gan, eval_set=valid_set)
callbacks.add_callback(GANCostCallback())
#callbacks.add_save_best_state_callback("./best_state.pkl")

print 'starting training'
# run fit
gan.fit(train_set, num_epochs=nb_epochs, optimizer=optimizer,
        cost=cost, callbacks=callbacks)

# gan.save_params('our_gan.prm')

x_new = np.random.randn(100, latent_size) 
inference_set = HDF5Iterator(x_new, None, nclass=2, lshape=(latent_size))
my_generator = Model(gan.layers.generator)
my_generator.save_params('our_gen.prm')
my_discriminator = Model(gan.layers.discriminator)
my_discriminator.save_params('our_disc.prm')
test = my_generator.get_outputs(inference_set)
test = np.float32(test*max_elem + mean)
test =  test.reshape((100, 25, 25, 25))

print(test.shape, 'generator output')

#plt.plot(test[0, :, 12, :])
# plt.savefigure('output_img.png')

h5f = h5py.File('output_data.h5', 'w')
h5f.create_dataset('dataset_1', data=test)

