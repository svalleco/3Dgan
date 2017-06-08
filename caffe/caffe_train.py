########    This file creates the prototext for the net and solvers  ########################################################################################
########    The dcgan implementation is used from" https://github.com/samson-wang/dcgan.caffe " 
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

def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

if len(sys.argv) == 1:
  start_snapshot = 0
batch_size = 100
latent = 200 # latent vector dimension
max_iter = 1161  # maximum number of iterations
display_every = 100 # show losses every so many iterations
snapshot_every = 500 # snapshot every so many iterations
snapshot_folder = 'snapshots_test' # where to save the snapshots (and load from)
snapshot_at_iter = -1
snapshot_at_iter_file = 'snapshot_at_iter.txt'
gpu_id = 0
num_clas = 2
lr = 0.9
sub_nets = ('generator2', 'discriminator2', 'data2')

#initialize the nets
caffe.set_device(gpu_id)
caffe.set_mode_gpu()
generator = caffe.RMSPropSolver('solver_generator2.prototxt')
discriminator = caffe.RMSPropSolver('solver_discriminator2.prototxt')
data_reader = caffe.RMSPropSolver('solver_data2.prototxt')

#load from snapshot                                                             
if start_snapshot:
  curr_snapshot_folder = snapshot_folder +'/' + str(start_snapshot)
  print >> sys.stderr, '\n === Starting from snapshot ' + curr_snapshot_folder + ' ===\n'
  generator_caffemodel = curr_snapshot_folder +'/' + 'generator.caffemodel'
  if os.path.isfile(generator_caffemodel):
    generator.net.copy_from(generator_caffemodel)
  else:
    raise Exception('File %s does not exist' % generator_caffemodel)
  discriminator_caffemodel = curr_snapshot_folder +'/' + 'discriminator.caffemodel'
  if os.path.isfile(discriminator_caffemodel):
    discriminator.net.copy_from(discriminator_caffemodel)
  else:
    raise Exception('File %s does not exist' % discriminator_caffemodel)


#read weights of losses                                                         
#discr_loss_weight = discriminator.net._blob_loss_weights   #  he used[discriminator.net._blob_names_index['discr_loss']]

#do training                                                                    
start = time.time()
#do training                                                                    
start = time.time()
for it in range(start_snapshot,max_iter):

  # read the data
  data_reader.net.forward()

  # feed the data to the generator and Get the generated image 
  noise = np.random.normal(0, 1, (batch_size, latent)).astype(np.float32)
  sampled_labels = np.random.randint(0, num_clas, batch_size)
  generator.net.blobs['feat'].data[...] = noise
  generator.net.blobs['clas'].data[...] = sampled_labels.reshape((-1, 1))
  generator.net.forward()
  generated_img = generator.net.blobs['generated'].data

  # run the discriminator on real data                                         
  discriminator.net.blobs['ECAL'].data[...] = data_reader.net.blobs['ECAL'].data
  discriminator.net.blobs['TAG'].data[...] = data_reader.net.blobs['TAG'].data
  discriminator.net.blobs['event'].data[...] = bit_flip(np.ones((batch_size,1), dtype='float32'))          
  discriminator.net.forward()
  discr_real_loss = np.copy(discriminator.net.blobs['loss'].data)
 # discriminator.increment_iter()
  # discriminator.net.clear_param_diffs()
  # discriminator.net.backward()
  discriminator.step(1)             # train the discriminator

  # run the discriminator on generated data 
  discriminator.net.blobs['ECAL'].data[...] = generated_img
  discriminator.net.blobs['TAG'].data[...] = sampled_labels #.reshape((-1, 1))
  discriminator.net.blobs['event'].data[...] = bit_flip(np.zeros((batch_size,1), dtype='float32'))
  discriminator.net.forward()
  discr_fake_loss = np.copy(discriminator.net.blobs['loss'].data)
  #discriminator.net.backward()
  discriminator.step(1)
  #discriminator.apply_update()

# run the discriminator on generated data with opposite labels, to get the gradient for the generator
  for i in range(2):                                                        
     discriminator.net.blobs['event'].data[...] =bit_flip( np.ones((batch_size,1), dtype='float32'))
     discriminator.net.forward()
     discr_fake_for_generator_loss = np.copy(discriminator.net.blobs['loss'].data)
 # generator.increment_iter()
     generator.net.clear_param_diffs()    
     discriminator.net.backward()
                                        
# Train the generator
     generator.net.blobs['generated'].diff[...] = discriminator.net.blobs['ECAL'].diff
     generator.net.backward()
  #generator.apply_update()
  #generator.update()
     for layer in generator.net.layers:
        for blob in layer.blobs:
            blob.data[...] -= lr * blob.diff
# add by samson                                                                                                         
#display                                                                      
  if it % display_every == 0:
    print >> sys.stderr, "[%s] Iteration %d: %f seconds" % (time.strftime("%c"), it, time.time()-start)
    print >> sys.stderr, "  discr real loss:  %f" % (discr_real_loss)
    print >> sys.stderr, "  discr fake loss:  %f" % (discr_fake_loss)
    print >> sys.stderr, "  discr fake loss for generator: %f" % (discr_fake_for_generator_loss)
    start = time.time()
    if os.path.isfile(snapshot_at_iter_file):
      with open (snapshot_at_iter_file, "r") as myfile:
        snapshot_at_iter = int(myfile.read())

  #snapshot                                                                     
  if it % snapshot_every == 0 or it == snapshot_at_iter:
    curr_snapshot_folder = snapshot_folder +'/' + str(it)
    print >> sys.stderr, '\n === Saving snapshot to ' + curr_snapshot_folder + ' ===\n'
    if not os.path.exists(curr_snapshot_folder):
      os.makedirs(curr_snapshot_folder)
    generator_caffemodel = curr_snapshot_folder + '/' + 'generator.caffemodel'
    generator.net.save(generator_caffemodel)
    discriminator_caffemodel = curr_snapshot_folder + '/' + 'discriminator.caffemodel'
    discriminator.net.save(discriminator_caffemodel)
