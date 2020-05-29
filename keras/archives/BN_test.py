import keras
from keras.layers.normalization import BatchNormalization as BN1
from BN2 import BatchNormalization as BN2
from keras.models import Sequential, Model
from keras import backend as K
import numpy as np
from keras.layers import Input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def test_that_trainable_disables_updates():
    val_a = np.random.random((10, 1))
    val_out = np.random.random((10, 1))
    print('Mean=', np.mean(val_a))
    print('std=', np.std(val_a))
    a = Input(shape=(1,))
    layer = BN2(input_shape=(1,))
    b = layer(a)
    model = Model(a, b)

    model.trainable = False
    print('Starting............................................................................................................')
    print('Model.trainable=false')
    print('Model Updates')
    print(model.updates)
    print('Layer Updates')
    print(layer.updates)
    print('Compiling...........')
    model.compile('sgd', 'mse')
    print('weights')
    weights = model.get_weights()
    print('model', weights)
    weights = layer.get_weights()
    print('layer', weights)

    print('Training with non trainable model.....')
    model.train_on_batch(val_a, val_out)
    print('After training trainable model')
    print('weights')
    weights = model.get_weights()
    print('model', weights)
    weights = layer.get_weights()
    print('layer', weights)
   
    model.trainable = True
    print('Model.trainable=true')
    print('Compiling...........')
    model.compile('sgd', 'mse')
    print('Model Updates')
    print(model.updates)
    print('Training with trainable model.....')
    model.train_on_batch(val_a, val_out)
    print('After training trainable model')
    print('weights')
    weights = model.get_weights()
    print('model', weights)
    weights = layer.get_weights()
    print('layer', weights)

    layer.trainable=False
    print('Compiling...........')
    model.compile('sgd', 'mse')
    print('Layer not trainable')
    print('trainable weights')
    print(layer.trainable_weights)
    print('untrainable weights')
    print(layer.non_trainable_weights)

    print('Layer Updates')
    print(layer.updates)
    print('Training with trainable model.....')
    model.train_on_batch(val_a, val_out)
    print('After training with untrainable layer')
    print('weights')
    weights = model.get_weights()
    print('model', weights)
    weights = layer.get_weights()
    print('layer', weights)
    model.save_weights('weights.hdf5')
    model.load_weights('weights.hdf5')
    #layer.trainable=True
    #print('Compiling...........')
    #model.compile('sgd', 'mse')
    print('Training...........')
    model.train_on_batch(val_a, val_out)
    print('After training')
    print('weights')
    weights = model.get_weights()
    print('model', weights)
    weights = layer.get_weights()
    print('layer', weights)
    print('trainable weights')
    print(layer.trainable_weights)
    print('untrainable weights')
    print(layer.non_trainable_weights)

    print('Layer Updates')
    print(layer.updates)


if __name__ == '__main__':
    test_that_trainable_disables_updates()
