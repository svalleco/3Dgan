import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.initializers import Gaussian
from neon.layers import GeneralizedGANCost, Sequential, Conv, Deconv, Dropout, Polling
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
conv1 = dict(init=init, batch_norm=False, activation=lrelu)
conv2 = dict(init=init, batch_norm=True, activation=lrelu, padding=2)
conv3 = dict(init=init, batch_norm=True, activation=lrelu, padding=1)
D_layers = [Conv((5, 5, 5, 32), name="D11", **conv1),
            Dropout(keep = 0.8),

            Conv((5, 5, 5, 8), name="D12", **conv2),
            Dropout(keep = 0.8),

            Conv((5, 5, 5, 8), name="D21", **conv2),
            Dropout(keep = 0.8),

            Conv((5, 5, 5, 8), name="D22", **conv3),
            Dropout(keep = 0.8),

            #Polling((2, 2, 2)),
            
            Linear(1, init=init, name="D_out")]


# generator using "decovolution" layers
latent_size = 200
relu = Rectlin(slope=0)  # relu for generator
conv4 = dict(init=init, batch_norm=True, activation=lrelu, dilation=[2, 2, 2])
conv5 = dict(init=init, batch_norm=True, activation=lrelu, padding=[2, 2, 0], dilation=[2, 2, 3])
conv6 = dict(init=init, batch_norm=False, activation=lrelu, padding=[1, 0, 3])
G_layers = [Linear(latent_size, init=init, name="G11"),
            Reshape((7, 7, 8, 8)), 
            Deconv((6, 6, 8, 64), **conv4),
            Deconv((6, 5, 8, 6), **conv5), 
            Deconv((3, 3, 8, 6), **conv6), 
            # Deconv((3, 3, 8, 6), **conv6),
            Deconv((2, 2, 2, 1), init=init, batch_norm=False, activation=relu)]

layers = GenerativeAdversarial(generator=Sequential(G_layers, name="Generator"),
                               discriminator=Sequential(D_layers, name="Discriminator"))

# setup cost function as CrossEntropy
cost = GeneralizedGANCost(costfunc=GANCost(func="modified"))



