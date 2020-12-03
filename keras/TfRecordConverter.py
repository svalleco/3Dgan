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
    #print('finished ECAL')
    return tf.train.Feature(float_list=tf.train.FloatList(value=flat))

def convert_floats(featdataset, feature):
    #featarray = np.array(featdataset)
    #print('finished ', feature)
    return tf.train.Feature(float_list=tf.train.FloatList(value=[featdataset]))

#receives a tensor and reshapes it
def retrieve_ECAL(ecalarray, xscale=1, originalarraydim=[5000,51,51,25]):
    #sizes: 5000, 51, 51, 25
    return tf.reshape(ecalarray, originalarraydim)

def GetDataAngleParallel(dataset, xscale =1, xpower=1, yscale = 100, angscale=1, angtype='theta', thresh=1e-4, daxis=1):
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

    #Y = [[el] for el in Y]
    #ang = [[el] for el in ang]
    #ecal = [[el] for el in ecal]
    ecal = ecal.flatten()

    final_dataset = {'X': X,'Y': Y, 'ang': ang, 'ecal': ecal}

    return final_dataset

#convert dataset with preprocessing
def ConvertH5toTFRecordPreprocessing(datafile,filenumber,datadirectory):
    # read file
    print('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    
    dataset = GetDataAngleParallel(f)

    dataset = tf.data.Dataset.from_tensor_slices((dataset.get('X'),dataset.get('Y'),dataset.get('ang'),dataset.get('ecal')))#.batch(128)

    tf.print(dataset)

    #for f0, f1, f2, f3 in dataset.take(1):
    #    print(f0)
    #    print(f1)
    #    print(f2)
    #    print(f3)

    #return

    #b = 0
    #for batch in dataset:
    #    print(b)         
    #    b += 1

    # for k in f.keys():
    #     print(f.get(k))

    # print(f.get('ECAL'))

    # return
    #print(dataset)
    #print(dataset.get('X'))
    #return

    #should I save only the necessary labes?
    #features 'ECAL', 'bins', 'energy', 'eta', 'mtheta', 'sum', 'theta'
    #seri = 0
    print('Start')
    def serialize(feature1,feature2,feature3,feature4):
        finaldata = tf.train.Example(
            features=tf.train.Features( 
                feature={
                    'X': convert_ECAL(feature1), #float32
                    #'ecalsize': convert_int_feature(list(dataset.get('X').shape)), #needs size of ecal so it can reconstruct the array
                    'Y': convert_floats(feature2, 'Y'), #float32
                    'ang': convert_floats(feature3, 'ang'), #float32
                    'ecal': convert_floats(feature4, 'ecal'), #float64
                }
            )
        )
        #seri += 1
        #print(seri)
        return finaldata.SerializeToString()

    def serialize_example(f0,f1,f2,f3):
        tf_string = tf.py_function(serialize,(f0,f1,f2,f3),tf.string)
        return tf.reshape(tf_string, ())

    #for f0, f1, f2, f3 in dataset.take(1):
    #    print(serialize(f0,f1,f2,f3))

    serialized_dataset = dataset.map(serialize_example)
    print(serialized_dataset) 
        

    #def generator():
    #    for features in dataset:
    #        yield dataset(*features)

    #serialized_dataset = tf.data.Dataset.from_generator(generator, output_types=tf.string, output_shapes=())

    #print(finaldata)

    filename = datadirectory + '/Ele_VarAngleMeas_100_200_{0:03}.tfrecords'.format(filenumber)
    print('Writing data in .....', filename)
    writer = tf.data.experimental.TFRecordWriter(str(filename))
    writer.write(serialized_dataset)

    return serialized_dataset

def RetrieveTFRecordpreprocessing(recorddatapaths, batch_size):
    recorddata = tf.data.TFRecordDataset(recorddatapaths)

    #print('Start')
    #size = recorddata.cardinality().numpy()
    #print(size)

    ds_size = sum(1 for _ in recorddata)

    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    #print(type(recorddata))

    #for rec in recorddata.take(10):
    #    print(repr(rec))

    retrieveddata = {
        'X': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        #'ecalsize': tf.io.FixedLenSequencFeature((), dtype=tf.int64, allow_missing=True), #needs size of ecal so it can reconstruct the narray
        'Y': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0), #float32
        'ang': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
        'ecal': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        data['X'] = tf.reshape(data['X'],[1,51,51,25])
        #print(tf.shape(data['Y']))
        data['Y'] = tf.reshape(data['Y'],[1])
        data['ang'] = tf.reshape(data['ang'],[1])
        data['ecal'] = tf.reshape(data['ecal'],[1])
        #print(tf.shape(data['Y']))
        return data

    parsed_dataset = recorddata.map(_parse_function).batch(batch_size, drop_remainder=True).repeat().with_options(options)
    #print(parsed_dataset)
   
    #b = 0
    #for batch in parsed_dataset:
    #    b += 1
    #    print(b)
    #    print(batch.get('Y'))

    #for par in parsed_dataset.take(10):
    #    print(repr(par))

    #return parsed_dataset

    #print(type(parsed_dataset))

    return parsed_dataset, ds_size

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
        finaldata = ConvertH5toTFRecordPreprocessing(f,filenumber,outpath)
        #print(finaldata)
        filenumber += 1
        #feat = RetrieveTFRecordpreprocessing(outpath)
        #print(feat.get('X'))
       

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
