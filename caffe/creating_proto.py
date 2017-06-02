#######    This file creates the prototext for the net and solvers  ########################################################################################

import sys
caffe_root='/data/caffe/'
sys.path.insert(0, caffe_root+'python')

import caffe
import numpy as np
import time
import os
import sys

import h5py
from h5py import File as HDF5File

from caffe import layers as L, params as P

latent = 200 # latent vector dimension
img = 25 # image size
gpu_id = 0
batch_size = 100
num_clas = 2
sub_nets = ('generator2', 'discriminator2', 'data2')

############   creating the data net     #############################
data = caffe.NetSpec()
data.ECAL, data.TAG = L.HDF5Data(batch_size = batch_size, source = "train.txt", ntop = 2)    # train.txt is a text file containing the path to the training data folder
with open('data2.prototxt', 'w') as f:
          f.write(str(data.to_proto()))

############    creating the generator net     ########################
n = caffe.NetSpec()
n.feat = L.Input(shape=dict(dim=[batch_size, latent]))      # random array
n.clas = L.Input(shape=dict(dim=[batch_size,1]))         # array with classes
n.embed = L.Embed(n.clas, input_dim=num_clas, num_output=latent, weight_filler=dict(type='xavier'))  # class dependant embedding   (xavier for glorot_normal in keras)
n.flat = L.Flatten(n.embed)
n.mult = L.Eltwise(n.flat, n.feat, operation=0)  #    0 = multiplication mode
n.Dense = L.InnerProduct(n.mult, num_output=7*7*8*8, weight_filler=dict(type='msra'))  #    3136
n.resh = L.Reshape(n.Dense, reshape_param ={'shape':{'dim':[100, 7, 7, 8, 8]}})
n.conv5 = L.Convolution(n.resh, num_output=64, kernel_size= [6, 6, 8],  pad=[2, 2, 3], engine=1)  # (not working for nd) weight_filler=dict(type='msra') => keras he_uniform
n.relu5 = L.ReLU(n.conv5, negative_slope=0.3, engine=1)
n.bn5 = L.BatchNorm(n.relu5, in_place=True)
n.upsmpl5 = L.Deconvolution(n.bn5, convolution_param=dict(num_output=1, group=1, kernel_size=4, stride = 2, pad=1)) #f=2, kernel_size:{{2*f- f%2}} stride:{{f}} num_output:{{C}} group:{{C}} pad:{{ceil((f-1)/2.)}} (gives error for nd) weight_filler: "bilinear" 
n.conv4 = L.Convolution(n.upsmpl5, num_output=6, kernel_size= [6, 5, 8],  pad=[2, 2, 0], engine=1)# (not working for nd) weight_filler=dict(type='msra')  => keras he_uniform 
n.relu4 = L.ReLU(n.conv4, negative_slope=0.3, engine=1)
n.bn4 = L.BatchNorm(n.relu4, in_place=True)
n.upsmpl4 = L.Deconvolution(n.bn4, convolution_param=dict(num_output=1, group=1, kernel_size=[4, 4, 5], stride = [2, 2, 3], pad=1)) # f = [2, 2, 3]
n.conv3 = L.Convolution(n.upsmpl4, num_output=6, kernel_size= [3, 3, 8],  pad=[1, 0, 3], engine=1) # (not working for nd) weight_filler=dict(type='msra')  => keras he_uniform 
n.relu3 = L.ReLU(n.conv3, negative_slope=0.3, engine=1)
n.conv2 = L.Convolution(n.relu3, num_output=1, kernel_size= [2, 2, 2],pad = [2, 0, 3], engine=1)   # (not working for nd) weight_filler=dict(type='xavier')
n.generated = L.ReLU(n.conv2, negative_slope=0.3, engine=1)
with open('generator2.prototxt', 'w') as f:               # for GAN implementation
          f.write('force_backward:true\n')
with open('generator2.prototxt', 'a') as f:
          f.write(str(n.to_proto()))
############    creating the discriminator net #######################
n = caffe.NetSpec()
n.ECAL = L.Input(shape=dict(dim=[batch_size, 1, img, img, img]))
n.TAG = L.Input(shape=dict(dim=[batch_size, 1]))        #  class
n.event = L.Input(shape=dict(dim=[batch_size, 1]))      #  If real or gen
n.conv1 = L.Convolution(n.ECAL, num_output=32, kernel_size= 5,  pad=2, engine=1)
n.relu1 = L.ReLU(n.conv1, negative_slope=0.3, engine=1)
n.drpout1 = L.Dropout(n.relu1, dropout_ratio= 0.2)
n.conv2 = L.Convolution(n.drpout1, num_output=8, kernel_size= 5,  pad=2, engine=1)
n.relu2 = L.ReLU(n.conv2, negative_slope=0.3, engine=1)
n.bn2 = L.BatchNorm(n.relu2, in_place=True)
n.drpout2 = L.Dropout(n.relu2, dropout_ratio= 0.2)
n.conv3 = L.Convolution(n.drpout2, num_output=8, kernel_size= 5,  pad=2, engine=1)
n.relu3 = L.ReLU(n.conv3, negative_slope=0.3, engine=1)
n.bn3 = L.BatchNorm(n.relu3, in_place=True)
n.drpout3 = L.Dropout(n.bn3, dropout_ratio= 0.2)
n.conv4 = L.Convolution(n.drpout3, num_output=8, kernel_size= 5, pad=1, stride= 2, engine=1)  # used stride instead of average pooling
n.relu4 = L.ReLU(n.conv4, negative_slope=0.3, engine=1)
n.bn4 = L.BatchNorm(n.relu4, in_place=True)
n.drpout4 = L.Dropout(n.bn4, dropout_ratio= 0.2)
#n.pool4 = L.Pooling(n.drpout4, kernel_size=2, stride=1, pool=P.Pooling.AVE)
n.tag = L.InnerProduct(n.drpout4, num_output = 1)
n.loss_tag = L.SigmoidCrossEntropyLoss(n.tag, n.TAG, loss_weight=0.5)
n.aux = L.InnerProduct(n.drpout4, num_output = 1)
n.loss_aux = L.SigmoidCrossEntropyLoss(n.aux, n.event, loss_weight=0.5)

with open('discriminator2.prototxt', 'w') as f:
          f.write(str(n.to_proto()))


#############################################################################  

#make solvers
with open ("solver_template2.prototxt", "r") as myfile:
  solver_template=myfile.read()
  
for curr_net in sub_nets:
  with open("solver_%s.prototxt" % curr_net, "w") as myfile:
    myfile.write(solver_template.replace('@NET@', curr_net))                 

#initialize the nets
#caffe.set_device(gpu_id)
#caffe.set_mode_gpu()
#generator = caffe.RMSPropSolver('solver_generator2.prototxt')
#discriminator = caffe.RMSPropSolver('solver_discriminator2.prototxt')
#data_reader = caffe.RMSPropSolver('solver_data2.prototxt')



