import numpy as np 
import os
from datetime import datetime
from neon.callbacks.callbacks import Callbacks, GANCostCallback
#from neon.callbacks.plotting_callbacks import GANPlotCallback
from neon.initializers import Gaussian
from neon.layers import GeneralizedGANCost, Affine, Sequential, Conv, Deconv, Dropout, Pooling, BatchNorm
from neon.layers.layer import Linear, Reshape
from neon.layers.container import GenerativeAdversarial
from neon.models.model import GAN, Model
from neon.transforms import Rectlin, Logistic, GANCost, Tanh
from neon.util.argparser import NeonArgparser
from neon.util.persist import ensure_dirs_exist
from neon.layers.layer import Dropout
from neon.data.dataiterator import ArrayIterator
from neon.optimizers import GradientDescentMomentum, RMSProp
from gen_data_norm import gen_rhs
from neon.backends import gen_backend
from temporary_utils import temp_3Ddata
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import h5py

#h5f = h5py.File('/afs/cern.ch/work/e/eorlova/3Dgan/neon/output_data.h5','r')
f = h5py.File("/afs/cern.ch/work/e/eorlova/Ele_Fixed100_total2.h5","r")
data = f.get('ECAL')
print(data)
xtr = np.array(data)

print(xtr.shape)
#xtr = xtr.reshape((xtr.shape[0], 25 * 25 * 25)).astype(np.float32)
print(xtr[9,:,:,12])
