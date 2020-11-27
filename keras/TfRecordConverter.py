import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import h5py

import glob

import analysis.utils.GANutils as gan


def get_parser():
    parser = argparse.ArgumentParser(description='TFRecord dataset Params' )
    parser.add_argument('--datapath', action='store', type=str, default='', help='HDF5 files to convert.')
    parser.add_argument('--outpath', action='store', type=str, default='', help='Dir to save the tfrecord files.')
    return parser



#convert to diferent features types
def convert_int_feature(val):
    if not isinstance(val, list):
        print('error')
        val = [val]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def convert_float_feature(val):
    if not isinstance(val, list):
        print('not enter')
        val = [val]
    return tf.train.Feature(float_list=tf.train.FloatList(value=val))

def convert_byte_feature(val):
    # if not isinstance(val, list):
    #     print('error')
    #     val = [val]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))

def convert_ECAL(ecalarray, xscale=1):
    featurelist = np.array(ecalarray)
    flat = featurelist.flatten()
    print('finished ECAL')
    return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

def convert_floats(featdataset, feature):
    featarray = np.array(featdataset)
    print('finished ', feature)
    return tf.train.Feature(float_list=tf.train.FloatList(value=featarray))

#receives a tensor and reshapes it
def retrieve_ECAL(ecalarray, xscale=1, originalarraydim=[5000,51,51,25]):
    #sizes: 5000, 51, 51, 25
    return tf.reshape(ecalarray, originalarraydim)

def GetDataAngleParallel(dataset, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1):
    #print ('Loading Data from .....', dataset)
    
    X=np.array(dataset.get('ECAL'))* xscale
    Y=np.array(dataset.get('energy'))/yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X=X[indexes]
    Y=Y[indexes]
    if angtype in dataset:
      ang = np.array(dataset.get(angtype))[indexes]
    else:
      ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal=ecal[indexes]
    ecal=np.expand_dims(ecal, axis=daxis)
    if xpower !=1.:
        X = np.power(X, xpower)

    # Y = [[el] for el in Y]
    # ang = [[el] for el in ang]
    # ecal = [[el] for el in ecal]

    final_dataset = {'X': X,'Y': Y, 'ang': ang, 'ecal': ecal}

    return final_dataset

#convert dataset with preprocessing
def ConvertH5toTFRecordPreprocessing(datafile,filenumber,datadirectory):
    # read file
    print('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    
    dataset = GetDataAngleParallel(f)

    # for k in f.keys():
    #     print(f.get(k))

    # print(f.get('ECAL'))

    # return

    #should I save only the necessary labes?
    #features 'ECAL', 'bins', 'energy', 'eta', 'mtheta', 'sum', 'theta'
    finaldata = tf.train.Example(
        features=tf.train.Features( 
            feature={
                'X': convert_ECAL(dataset.get('ECAL')), #float32
                'ecalsize': convert_int_feature(list(dataset.get('ECAL').shape)), #needs size of ecal so it can reconstruct the array
                'Y': convert_floats(dataset.get('energy'), 'energy'), #float32
                'ang': convert_floats(dataset.get('eta'), 'eta'), #float32
                'ecal': convert_floats(dataset.get('mtheta'), 'mtheta'), #float64
            }
        )
    )

    filename = datadirectory + '/Ele_VarAngleMeas_100_200_{0:03}.tfrecords'.format(filenumber)
    print('Writing data in .....', filename)
    writer = tf.io.TFRecordWriter(str(filename))
    writer.write(finaldata.SerializeToString())

    return finaldata.SerializeToString()

def RetrieveTFRecordpreprocessing(recorddatapaths, batch_size):
    recorddata = tf.data.TFRecordDataset(recorddatapaths)

    #print(type(recorddata))

    
    retrieveddata = {
        'X': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'ecalsize': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        'Y': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'ang': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'ecal': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        dataset = tf.io.parse_single_example(example_proto, retrieveddata)
        dataset['ECAL'] = tf.reshape(dataset['ECAL'], dataset['ecalsize'])
        dataset['Y'] = tf.reshape(dataset['Y'], [5000,1])
        dataset['ang'] = tf.reshape(dataset['ang'], [5000,1])
        dataset['ecal'] = tf.reshape(dataset['ecal'], [5000,1])
        return dataset

    parsed_dataset = recorddata.map(_parse_function).batch(batch_size, drop_remainder=True)


    return parsed_dataset

#main convert function
def ConvertH5toTFRecord(datafile,filenumber,datadirectory):
    # read file
    print('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    

    # for k in f.keys():
    #     print(f.get(k))

    # print(f.get('ECAL'))

    # return

    #should I save only the necessary labes?
    #features 'ECAL', 'bins', 'energy', 'eta', 'mtheta', 'sum', 'theta'
    finaldata = tf.train.Example(
        features=tf.train.Features( 
            feature={
                'ECAL': convert_ECAL(f.get('ECAL')), #float32
                'ecalsize': convert_int_feature(list(f.get('ECAL').shape)), #needs size of ecal so it can reconstruct the array
                #'bins': convert_int_feature(f.get('bins')), #int64
                'energy': convert_floats(f.get('energy'), 'energy'), #float32
                'eta': convert_floats(f.get('eta'), 'eta'), #float32
                'mtheta': convert_floats(f.get('mtheta'), 'mtheta'), #float64 (???)
                'sum': convert_floats(f.get('sum'), 'sum'), #float32
                'theta': convert_floats(f.get('theta'), 'theta'), #float32<---------
                
            }
        )
    )

    filename = datadirectory + '/Ele_VarAngleMeas_100_200_{0:03}.tfrecords'.format(filenumber)
    print('Writing data in .....', filename)
    writer = tf.io.TFRecordWriter(str(filename))
    writer.write(finaldata.SerializeToString())

    return finaldata.SerializeToString()

def RetrieveTFRecord(recorddatapaths):
    recorddata = tf.data.TFRecordDataset(recorddatapaths)

    #print(type(recorddata))

    
    retrieveddata = {
        'ECAL': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'ecalsize': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        #'bins': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'energy': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'eta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'mtheta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'sum': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'theta': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, retrieveddata)

    parsed_dataset = recorddata.map(_parse_function)

    #return parsed_dataset

    #print(type(parsed_dataset))

    for parsed_record in parsed_dataset:
        dataset = parsed_record

    dataset['ECAL'] = tf.reshape(dataset['ECAL'], dataset['ecalsize'])

    dataset.pop('ecalsize')

    return dataset


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    datapath = params.datapath# Data path
    outpath = params.outpath # training output
    Files = sorted( glob.glob(datapath))
    print ("Found {} files. ".format(len(Files)))

    filenumber = 0

    for f in Files:
        print(filenumber)
        ConvertH5toTFRecord(f,filenumber,outpath)
        filenumber += 1

       

    #finaldata = ConvertH5toTFRecord("../data/Ele_VarAngleMeas_100_200_000.h5",0,"datadirectory")
    #print(finaldata)
    #feat = RetrieveTFRecord(finaldata)
    #feat = RetrieveTFRecord(['./datadirectory/Ele_VarAngleMeas_100_200_000.tfrecords'])
    #print(feat.get('ECAL'))
    #ecal = retrieve_ECAL(feat.get('ECAL'), originalarraydim=feat.get('ecalsize'))
    #print(ecal)
    #print(np.array(ecal))
    #print(tf.reshape(feat.get('ECAL'), feat.get('ecalsize')))
    #print(feat)
    #main()