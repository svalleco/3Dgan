import os
import sys
import re
import glob
import h5py
import numpy as np

class GANGen:
     '''
    Data generator class for directory of h5 files, for Image Simulation through GAN
    '''
    def __init__( self, batch_size,train_split=0.9, test_split=0.1, data_dir, file_no):
        self.batch_size = batch_size
        Files = glob.glob(data_dir)
        self.train_split = train_split 
        self.validation_split = validation_split
        self.test_split = test_split
        self.fileindex = 0
        self.filesize = 0
        self.position = 0
    #function to call when generating data for training

  
    def train(self,modeltype=3):
        '''
        Generate data for training only
        '''
        length = len(self.filelist)
        #deleting the validation and test set filenames from the filelist
        del self.filelist[np.floor(((self.train_split))*length).astype(int):]
        return self.batches(modeltype)
    #function to call when generating data for testing

    #function to call when generating data for validating
    def validation(self,modeltype=3):
        '''
        Generate data for validation only
        '''
        length = len(self.filelist)
        #modifying the filename list to only include files for validation set
        self.filelist = self.filelist[np.floor(self.train_split*length+1).astype(int):np.floor((self.train_split+self.validation_split)*length+1).astype(int)]
        return self.batches(modeltype)


        
    #The function which reads files to gather data until batch size is satisfied
    def batch_helper(self, fileindex, position, batch_size):
        '''
        Reads files to gather data until batch size is satisfied, then yeilds
        '''
        f = h5py.File(self.filelist[fileindex],'r')
        self.filesize = np.array(f['ECAL']).shape[0]


        if (position + batch_size < self.filesize):
            data_ECAL = np.array(f['ECAL'][position : position + batch_size])
            data_HCAL = np.array(f['HCAL'][position : position + batch_size])
            target = np.array(f['target'][position : position + batch_size][:,:,0])
            position += batch_size
            f.close()
            return data_ECAL,data_HCAL, target, fileindex, position
        
        else:

            data_ECAL = np.array(f['ECAL'][position : ])
            data_HCAL = np.array(f['HCAL'][position : ])
            target = np.array(f['target'][position:][:,:,0])
            #target = np.delete(target,0,1)
            f.close()
            

            if (fileindex+1 < len(self.filelist)):
                if(self.batch_size-data_ECAL.shape[0]>0):
                    while(self.batch_size-data_ECAL.shape[0]>0):
                        if(int(np.floor((self.batch_size-data_ECAL.shape[0])/self.filesize))==0):
                            number_of_files=1
                        else:
                            number_of_files=int(np.ceil((self.batch_size-data_ECAL.shape[0])/self.filesize))
                        for i in xrange(0,number_of_files):

                            if fileindex + i + 1 > len(self.filelist):
                                fileindex = -1 - i

                            f = h5py.File(self.filelist[fileindex+i+1],'r')

                            if (self.batch_size-data_ECAL.shape[0]<self.filesize):
                                position = self.batch_size-data_ECAL.shape[0]
                                data_temp_ECAL = np.array(f['ECAL'][ : position])
                                data_temp_HCAL = np.array(f['HCAL'][: position])
                                target_temp = np.array(f['target'][:position][:,:,0])

                            else:
                                data_temp_ECAL = np.array(f['ECAL'])
                                data_temp_HCAL = np.array(f['HCAL'])
                                target_temp = np.array(f['target'][:,:,0])

                            f.close()
                            data_ECAL = np.concatenate((data_ECAL, data_temp_ECAL), axis=0)
                            data_HCAL = np.concatenate((data_HCAL, data_temp_HCAL), axis=0)
                            target = np.concatenate((target, target_temp), axis=0)

                    if (fileindex +i+1<len(self.filelist)):
                        fileindex = fileindex +i+1
                    else:
                        fileindex = 0
                else:
                    position = 0
                    fileindex=fileindex+1
            else:
                fileindex = 0
                position = 0
            
            return data_ECAL,data_HCAL, target, fileindex, position
    #The function which loops indefinitely and continues to return data of the specified batch size

    def batches(self, modeltype):
        '''
        Loops indefinitely and continues to return data of specified batch size
        '''
        while (self.fileindex < len(self.filelist)):
            data_ECAL,data_HCAL, target, self.fileindex, self.position = self.batch_helper(self.fileindex, self.position, self.batch_size)
            if data_ECAL.shape[0]!=self.batch_size:
                continue

            if modeltype==3:
                data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(1, 24, 24, 25))
                data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(1, 4, 4, 60))

            elif modeltype==2:
                data_ECAL = data_ECAL.reshape((data_ECAL.shape[0],)+(24, 24, 25))
                data_ECAL = np.swapaxes(data_ECAL, 1, 3)
                data_HCAL = data_HCAL.reshape((data_HCAL.shape[0],)+(4, 4, 60))
                data_HCAL = np.swapaxes(data_HCAL, 1, 3)

            elif modeltype==1:
                data_ECAL= np.reshape(data_ECAL,(self.batch_size,-1))
                data_HCAL= np.reshape(data_HCAL,(self.batch_size,-1))

            yield ([data_ECAL,data_HCAL],target)
self.fileindex = 0
