import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
#from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.initializers import Gaussian
from neon.layers import GeneralizedGANCost, Affine, Linear, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm
from neon.layers.layer import Linear, Reshape
from neon.layers.container import Tree, Multicost
from neon.models.model import Model
from neon.transforms import Rectlin, Logistic, GANCost, Tanh
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from neon.layers.layer import Dropout
from neon.data.dataiterator import HDF5Iterator
from neon.optimizers import GradientDescentMomentum, RMSProp, Adam
from gen_data_norm import gen_rhs
from neon.backends import gen_backend
from temporary_utils import temp_3Ddata
import numpy as np
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import h5py
# new definitions

import logging
main_logger = logging.getLogger('neon')
main_logger.setLevel(10)

class myGenerativeAdversarial(Tree):
    """
    Container for Generative Adversarial Net (GAN). It contains the Generator
    and Discriminator stacks as sequential containers.

    Arguments:
        layers (list): A list containing two Sequential containers
    """
    def __init__(self, generator, discriminator, name=None):
        super(Tree, self).__init__(name)

        self.generator = generator
        self.discriminator = discriminator
        self.layers = self.generator.layers + self.discriminator.layers

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level.

        Arguments:
            level (int, optional): indentation level

        Returns:
            str: layer info at the given indentation level
        """
        padstr = '\n' + '  ' * level
        ss = '  ' * level + self.classnm + padstr
        ss += '  ' * level + 'Generator:\n'
        ss += padstr.join([l.nested_str(level + 1) for l in self.generator.layers])
        ss += '\n' + '  ' * level + 'Discriminator:\n'
        ss += padstr.join([l.nested_str(level + 1) for l in self.discriminator.layers])
        return ss


class myGAN(Model):
    """
    Model for Generative Adversarial Networks.

    Arguments:
        layers: Generative Adversarial layer container
        noise_dim (Tuple): Dimensionality of the noise feeding the generator
        noise_type (Str): Noise distribution, 'normal' (default) or 'uniform'
        weights_only (bool): set to True if you do not want to recreate layers
                             and states during deserialization from a serialized model
                             description.  Defaults to False.
        name (str): Model name.  Defaults to "model"
        optimizer (Optimizer): Optimizer object which defines the learning rule for updating
                               model parameters (i.e., GradientDescentMomentum, Adadelta)
        k (int): Number of data batches per noise batch
        wgan_param_clamp (float or None): In case of WGAN weight clamp value, None for others
        wgan_train_sched (bool): Whether to use the FAIR WGAN training schedule of critics
    """
    def __init__(self, layers, noise_dim, noise_type='normal', weights_only=False,
                 name="model", optimizer=None, k=1,
                 wgan_param_clamp=None, wgan_train_sched=False):
        self.noise_dim = noise_dim
        self.noise_type = noise_type
        self.k = k
        self.wgan_param_clamp = wgan_param_clamp
        self.wgan_train_sched = wgan_train_sched
        self.nbatches = 0
        self.ndata = 0
        super(GAN, self).__init__(layers, weights_only=weights_only, name=name,
                                  optimizer=optimizer)

    @staticmethod
    def clip_param_in_layers(layer_list, abs_bound=None):
        """
        Element-wise clip all parameter tensors to between
        ``-abs_bound`` and ``+abs_bound`` in a list of layers.

        Arguments:
            layer_list (list): List of layers
            be (Backend object): Backend in which the tensor resides
            abs_bound (float, optional): Value to element-wise clip gradients
                                         or parameters. Defaults to None.
        """
        param_list = get_param_list(layer_list)
        for (param, grad), states in param_list:
            if abs_bound:
                param[:] = param.backend.clip(param, -abs(abs_bound), abs(abs_bound))

    def fill_noise(self, z, normal=True):
        """
        Fill z with either uniform or normally distributed random numbers
        """
        if normal:
            # Note fill_normal is not deterministic
            self.be.fill_normal(z)
        else:
            z[:] = 2 * self.be.rand() - 1.

    def initialize(self, dataset, cost=None):
        """
        Propagate shapes through the layers to configure, then allocate space.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform initialization on
            cost (Cost): Defines the function which the model is minimizing based
                         on the output of the last layer and the input labels.
        """
        if self.initialized:
            return

        # Propagate shapes through the layers to configure
        prev_input = dataset
        #prev_input = self.layers.configure(self.noise_dim)
        prev_input = self.layers.configure(prev_input)

        if cost is not None:
            cost.initialize(prev_input)
            self.cost = cost

        # Now allocate space
        self.layers.generator.allocate(accumulate_updates=False)
        self.layers.discriminator.allocate(accumulate_updates=True)
        self.layers.allocate_deltas()
        self.initialized = True

        self.zbuf = self.be.iobuf(self.noise_dim)
        self.ybuf = self.be.iobuf((1,))
        self.z0 = self.be.iobuf(self.noise_dim)  # a fixed noise buffer for generating images
        self.z0 = prev_input
        #self.fill_noise(self.z0, normal=(self.noise_type == 'normal'))
        self.cost_dis = np.empty((1,), dtype=np.float32)
        self.current_batch = self.gen_iter = self.last_gen_batch = 0

    def get_k(self, giter):
        """
        WGAN training schedule for generator following Arjovsky et al. 2017

        Arguments:
            giter (int): Counter for generator iterations
        """
        if self.wgan_train_sched and (giter < 25 or giter % 500 == 0):
            return 100
        else:
            return self.k

    def _epoch_fit(self, dataset, callbacks):
        """
        Helper function for fit which performs training on a dataset for one epoch.

        Arguments:
            dataset (NervanaDataIterator): Dataset iterator to perform fit on
        """
        epoch = self.epoch_index
        self.total_cost[:] = 0
        last_gen_iter = self.gen_iter
        z, y_temp = self.zbuf, self.ybuf

        # iterate through minibatches of the dataset
        for mb_idx, (x, _) in enumerate(dataset):
            callbacks.on_minibatch_begin(epoch, mb_idx)
            self.be.begin(Block.minibatch, mb_idx)

            # clip all discriminator parameters to a cube in case of WGAN
            if self.wgan_param_clamp:
                self.clip_param_in_layers(self.layers.discriminator.layers_to_optimize,
                                          self.wgan_param_clamp)

            # train discriminator on noise
            self.fill_noise(z, normal=(self.noise_type == 'normal'))
            #z = self.z0
            Gz = self.fprop_gen(z)
            y_noise = self.fprop_dis(Gz)
            y_temp[:] = y_noise
            delta_noise = self.cost.costfunc.bprop_noise(y_noise)
            self.bprop_dis(delta_noise)
            self.layers.discriminator.set_acc_on(True)

            # train discriminator on data
            y_data = self.fprop_dis(x)
            delta_data = self.cost.costfunc.bprop_data(y_data)
            self.bprop_dis(delta_data)
            self.optimizer.optimize(self.layers.discriminator.layers_to_optimize, epoch=epoch)
            self.layers.discriminator.set_acc_on(False)

            # keep GAN cost values for the current minibatch
            # abuses get_cost(y,t) using y_noise as the "target"
            self.cost_dis[:] = self.cost.get_cost(y_data, y_temp, cost_type='dis')

            # train generator
            if self.current_batch == self.last_gen_batch + self.get_k(self.gen_iter):
                self.fill_noise(z, normal=(self.noise_type == 'normal'))
                Gz = self.fprop_gen(z)
                y_temp[:] = y_data
                y_noise = self.fprop_dis(Gz)
                delta_noise = self.cost.costfunc.bprop_generator(y_noise)
                delta_dis = self.bprop_dis(delta_noise)
                self.bprop_gen(delta_dis)
                self.optimizer.optimize(self.layers.generator.layers_to_optimize, epoch=epoch)
                # keep GAN cost values for the current minibatch
                self.cost_dis[:] = self.cost.get_cost(y_temp, y_noise, cost_type='dis')
                # accumulate total cost.
                self.total_cost[:] = self.total_cost + self.cost_dis
                self.last_gen_batch = self.current_batch
                self.gen_iter += 1

            self.be.end(Block.minibatch, mb_idx)
            callbacks.on_minibatch_end(epoch, mb_idx)
            self.current_batch += 1

        # now we divide total cost by the number of generator iterations,
        # so it was never total cost, but sum of averages
        # across all the minibatches we trained on the generator
        assert self.gen_iter > last_gen_iter, \
            "at least one generator iteration is required for total cost estimation in this epoch"
        self.total_cost[:] = self.total_cost / (self.gen_iter - last_gen_iter)

        # Package a batch of data for plotting
        self.data_batch, self.noise_batch = x, self.fprop_gen(self.z0)

    def fprop_gen(self, x, inference=False):
        """
        fprop the generator layer stack
        """
        return self.layers.generator.fprop(x, inference)

    def fprop_dis(self, x, inference=False):
        """
        fprop the discriminator layer stack
        """
        return self.layers.discriminator.fprop(x, inference)

    def bprop_dis(self, delta):
        """
        bprop the discriminator layer stack
        """
        return self.layers.discriminator.bprop(delta)

    def bprop_gen(self, delta):
        """
        bprop the generator layer stack
        """
        return self.layers.generator.bprop(delta)


# load up the data set
X, y = temp_3Ddata()
X[X < 1e-6] = 0
mean = np.mean(X, axis=0, keepdims=True)
max_elem = np.max(np.abs(X))
print(np.max(np.abs(X)),'max abs element')
print(np.min(X),'min element')
X = (X- mean)/max_elem
print(X.shape, 'X shape')
print(np.max(X),'max element after normalisation')
print(np.min(X),'min element after normalisation')
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
print(X_train.shape, 'X train shape')
print(y_train.shape, 'y train shape')

gen_backend(backend='gpu', batch_size=100)


# setup datasets
train_set = EnergyData(X=X_train, Y=y_train, lshape=(1,25,25,25))

# grab one iteration from the train_set
iterator = train_set.__iter__()
(X, Y) = iterator.next()
print X  # this should be shape (N, 25,25, 25)
print Y  # this should be shape (Y1,Y2) of shapes (1)(1)
assert X.is_contiguous
assert Y.is_contiguous

in_set.reset()

# generate test set
valid_set =EnergyData(X=X_test, Y=y_test, lshape=(1,25,25,25))


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
b1 = BranchNode("b1")
b2 = BranchNode("b2")
branch1 = [   
            b1
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
            b2,
            Affine(1, init=init, bias=init, activation=Logistic())
            ] #real/fake
branch2 = [b2, 
           Affine(1, init=init, bias=init, activation=Linear())] #E primary
branch3 = [b1,Linear(1, init=Constant(1.0))] #SUM ECAL

D_layers = Tree([branch1, branch2, branch3], name="Discriminator") #keep weight between branches equal to 1. for now (alphas=(1.,1.,1.) as by default )
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

layers = GenerativeAdversarial(generator=Tree(G_layers, name="Generator"),
                               D_layers)
                               #discriminator=Sequential(D_layers, name="Discriminator"))
print 'layers defined'
# setup optimizer
# optimizer = RMSProp(learning_rate=1e-4, decay_rate=0.9, epsilon=1e-8)
optimizer = GradientDescentMomentum(learning_rate=1e-3, momentum_coef = 0.9)
#optimizer = Adam(learning_rate=1e-3)

# setup cost function as Binary CrossEntropy
#cost = GeneralizedGANCost(costfunc=GANCost(func="wasserstein"))
cost = Multicost([GANCost(func="wasserstein"), MeanSquared, MeanSquared])
nb_epochs = 15
latent_size = 200

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

