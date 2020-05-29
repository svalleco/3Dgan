import keras
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Sequential, Model
from keras import backend as K
import numpy as np
from keras.layers import Input

def test_that_trainable_disables_updates():
    val_a = np.random.random((10, 1))
    val_out = np.random.random((10, 1))
    print('Mean=', np.mean(val_a))
    print('std=', np.std(val_a))
    a = Input(shape=(1,))
    layer = BN(input_shape=(1,))
    b = layer(a)
    model = Model(a, b)
    print('Starting............................................................................................................')
    print('Model.trainable=true')
    print('Compiling...........')
    model.compile('sgd', 'mse')
    c = Input(shape=(1,))
    model.trainable = False
    d = model(c)
    frozen_model = Model(input=c, output=d)
    print('Model.trainable=false')
    print('Compiling...........')
    frozen_model.compile('sgd', 'mse')
    print('weights')
    weights = model.get_weights()
    print('model', weights)
    for i in np.arange(10):
      print('iteration {}'.format(i))
      print('Training with trainable model.....')
      model.train_on_batch(val_a, val_out)
      print('After training trainable model')
      print('weights')
      weights = model.get_weights()
      print('model', weights)
    
      print('Training with frozen model.....')
      frozen_model.train_on_batch(2 * val_a, val_out)
      print('After training trainable model')
      print('weights')
      weights = frozen_model.get_weights()
      print('model', weights)
      weights = layer.get_weights()
    

if __name__ == '__main__':
    test_that_trainable_disables_updates()
