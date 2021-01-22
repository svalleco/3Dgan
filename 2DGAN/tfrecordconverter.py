import argparse
import os
import sys

import tensorflow as tf
import numpy as np
import h5py

import glob

#import analysis.utils.GANutils as gan


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
def retrieve_ECAL(ecalarray, xscale=1, originalarraydim=[5000,25, 25, 25]): #missing entries per file
    #sizes: 5000, 25, 25, 25
    return tf.reshape(ecalarray, originalarraydim)

def data_preperation(X, y, keras_dformat, batch_size, percent=100):      #data preperation

    # Missing Loading data

    X[X < 1e-6] = 0  #remove unphysical values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)

    #take just a percentage form the data to make fast tests
    # X_train=X_train[:int(len(X_train)*percent/100),:]
    # y_train=y_train[:int(len(y_train)*percent/100)]
    # X_test=X_test[:int(len(X_test)*percent/100),:]
    # y_test=y_test[:int(len(y_test)*percent/100)]

    # tensorflow ordering - do after
    X_train =np.expand_dims(X_train, axis=-1)  #macht jeden Eintrag in der Liste zu einer Unterliste [1,2,3]->[[1],[2],[3]]
    # X_test = np.expand_dims(X_test, axis=-1)

    #print("X_train_shape (reordered): ", X_train.shape)
    if keras_dformat !='channels_last':
       X_train =np.moveaxis(X_train, -1, 1)    #Dreht die Matrix, damit die Dimension passt
       # X_test = np.moveaxis(X_test, -1,1)

    y_train= y_train/100     #Normalisieren
    # y_test=y_test/100
    """
    print("X_train_shape: ", X_train.shape)
    print("X_test_shape: ", X_test.shape)
    print("y_train_shape: ", y_train.shape)
    print("y_test_shape: ", y_test.shape)
    print('*************************************************************************************')
    """
    #####################################################################################
    nb_train = X_train.shape[0]
    #nb_test =  X_test.shape[0]
    if nb_train < batch_size:
        print("\nERROR: batch_size is larger than trainings data")
        print("batch_size: ", batch_size)
        print("trainings data: ", nb_train, "\n")

    X_train = X_train.astype(np.float32)  #alle Werte in Floats umwandeln
    #X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    #y_test = y_test.astype(np.float32)
    if keras_dformat =='channels_last':
        ecal_train = np.sum(X_train, axis=(1, 2, 3))
        #ecal_test = np.sum(X_test, axis=(1, 2, 3))
    else:
        ecal_train = np.sum(X_train, axis=(2, 3, 4))
        #ecal_test = np.sum(X_test, axis=(2, 3, 4))
    return {'X': X_train,'Y': y_train, 'ecal': ecal_train}
    #return X_train, X_test, y_train, y_test, ecal_train, ecal_test, nb_train, nb_test

#convert dataset with preprocessing
def ConvertH5toTFRecordPreprocessing(datafile,filenumber,datadirectory):
    # read file
    print('Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    
    #dataset = GetDataAngleParallel(f)
    dataset = data_preperation()

    dataset = tf.data.Dataset.from_tensor_slices((dataset.get('X'),dataset.get('Y'),dataset.get('ecal')))#.batch(128)

    tf.print(dataset)

    print('Start')
    def serialize(feature1,feature2,feature3):
        finaldata = tf.train.Example(
            features=tf.train.Features( 
                feature={
                    'X': convert_ECAL(feature1), #float32
                    'Y': convert_floats(feature2, 'Y'), #float32
                    'ecal': convert_floats(feature3, 'ecal'), #float64
                }
            )
        )
        return finaldata.SerializeToString()

    def serialize_example(f0,f1,f2,f3):
        tf_string = tf.py_function(serialize,(f0,f1,f2),tf.string)
        return tf.reshape(tf_string, ())


    serialized_dataset = dataset.map(serialize_example)
    print(serialized_dataset) 
        

    #Rewrite file names
    filename = datadirectory + '/Ele_VarAngleMeas_100_200_{0:03}.tfrecords'.format(filenumber)
    print('Writing data in .....', filename)
    writer = tf.data.experimental.TFRecordWriter(str(filename))
    writer.write(serialized_dataset)

    return serialized_dataset

def RetrieveTFRecordpreprocessing(recorddatapaths, batch_size):
    recorddata = tf.data.TFRecordDataset(recorddatapaths, num_parallel_reads=tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    #print(type(recorddata))

    #for rec in recorddata.take(10):
    #    print(repr(rec))

    retrieveddata = {
        'X': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True), #float32
        'Y': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0), #float32
        'ecal': tf.io.FixedLenFeature((), dtype=tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        data = tf.io.parse_single_example(example_proto, retrieveddata)
        data['X'] = tf.reshape(data['X'],[1, 25, 25, 25]) #need to get size [1, 25, 25, 25] for channel first [25, 25, 25, 1] for channel last
        #print(tf.shape(data['Y']))
        data['Y'] = tf.reshape(data['Y'],[1])
        #data['ang'] = tf.reshape(data['ang'],[1])
        data['ecal'] = tf.reshape(data['ecal'],[1])
        #print(tf.shape(data['Y']))
        return data

    parsed_dataset = recorddata.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(124987).repeat().batch(batch_size, drop_remainder=True).with_options(options)

    return parsed_dataset


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
