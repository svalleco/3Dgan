import numpy as np
import h5py
from sklearn.model_selection import train_test_split

def temp_3Ddata():

   f = h5py.File("/Users/svalleco/GAN/data/EGshuffled.h5","r")
   data = f.get('ECAL')
   #dtag =f.get('TAG')
   xtr = np.array(data)
   #print (xtr.shape)
   labels = np.ones(xtr.shape[0])
   #tag=numpy.array(dtag)
   #xtr=xtr[...,numpy.newaxis]
   #xtr=numpy.rollaxis(xtr,4,1)
   #print xtr.shape
   
   return xtr.reshape((xtr.shape[0], 25 * 25 * 25)).astype(np.float32), labels.astype(np.float32)

def get_output():
   f = h5py.File("/Users/svalleco/GAN/data/output_data.h5","r")
   data = f.get("dataset_1")
   x = np.array(data)
   return x.astype(np.float32)

def make_hdf5iterator_files():
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

    gen_backend(backend='cpu', batch_size=100)

    # generate the HDF5 file
    datsets = {'train': (X_train, y_train),
               'test': (X_test, y_test)}

    for ky in ['train', 'test']:
        df = h5py.File('EGshuffled_%s.h5' % ky, 'w')

        # input images
        in_dat = datsets[ky][0]
        df.create_dataset('input', data=in_dat)
        df['input'].attrs['lshape'] = (1, 25, 25, 25)  # (C, H, W)

        target = datsets[ky][1].reshape((-1, 1))  # make it a 2D array
        df.create_dataset('output', data=target)
        df['output'].attrs['nclass'] = 1
        df.close()
