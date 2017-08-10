#!/usr/bin/env python
"""
A GAN using 3D conv layers
"""
import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.initializers import Gaussian
from neon.layers import GeneralizedGANCost, Sequential, Conv, Deconv, Dropout, Pooling
from neon.layers.layer import Linear, Reshape
from neon.layers.container import GenerativeAdversarial
from neon.models.model import GAN
from neon.transforms import Rectlin, Logistic, GANCost
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from neon.layers.layer import Dropout

# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.add_argument('--kbatch', type=int, default=1,
                    help='number of data batches per noise batch in training')
args = parser.parse_args()

# load up the data set

# setup weight initialization function
init = Gaussian()

# discriminiator using convolution layers
lrelu = Rectlin(slope=0.1)  # leaky relu for discriminator
# sigmoid = Logistic() # sigmoid activation function
conv1 = dict(init=init, batch_norm=False, activation=lrelu) # what's about BatchNorm Layer and batch_norm parameter?
conv2 = dict(init=init, batch_norm=True, activation=lrelu, padding=2)
conv3 = dict(init=init, batch_norm=True, activation=lrelu, padding=1)
D_layers = [Conv((5, 5, 5, 32), **conv1),
            Dropout(keep = 0.8),

            Conv((5, 5, 5, 8), **conv2),
            Dropout(keep = 0.8),

            Conv((5, 5, 5, 8), **conv2),
            Dropout(keep = 0.8),

            Conv((5, 5, 5, 8), **conv3),
            Dropout(keep = 0.8),

            Pooling((2, 2, 2)),
            # what's about the Flatten Layer?
            Linear(1, init=init)] #what's about the activation function?


# generator using covolution layers
latent_size = 200
relu = Rectlin(slope=0)  # relu for generator
conv4 = dict(init=init, batch_norm=True, activation=lrelu, dilation=dict(dil_h=2, dil_w=2, dil_d=2))
conv5 = dict(init=init, batch_norm=True, activation=lrelu, padding=dict(pad_h=2, pad_w=2, pad_d=0), dilation=dict(dil_h=2, dil_w=2, dil_d=3))
conv6 = dict(init=init, batch_norm=False, activation=lrelu, padding=dict(pad_h=1, pad_w=0, pad_d=3))
G_layers = [Linear(64 * 7 * 7, init=init), # what's about the input volume
            Reshape((7, 7, 8, 8)),
            Conv((6, 6, 8, 64), **conv4),
            Conv((6, 5, 8, 6), **conv5),
            Conv((3, 3, 8, 6), **conv6),
            Conv((2, 2, 2, 1), init=init, batch_norm=False, activation=relu)]
            # what's about Embedding

layers = GenerativeAdversarial(generator=Sequential(G_layers, name="Generator"),
                               discriminator=Sequential(D_layers, name="Discriminator"))

# setup cost function as CrossEntropy
cost = GeneralizedGANCost(costfunc=GANCost(func="modified"))
