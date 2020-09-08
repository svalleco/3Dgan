import os
import sys
import numpy as np
import h5py
import cv2



"""
Specific Platform settings
"""
os.environ['LD_LIBRARY_PATH'] = os.getcwd()
output='/home/achaibi/scratch/CERN_anglegan/numpy'
root='/home/achaibi/scratch/CERN_anglegan/dataset'


def load_files(path):
    """Short summary.

    Parameters
    ----------
    path : type
        Description of parameter `path`.

    Returns
    -------
    type
        Description of returned object.

    """
    print(f"Dataset output shape")
    filenames = os.listdir(path)
    abs_filenames = [os.path.join(path, filename) for filename in filenames]
    #print("file naime : ", abs_filenames)
    assert all(filename for filename in abs_filenames)
    #for filename in abs_filenames:
    #    print(filename)
    return abs_filenames



def GetDataAngle(datafile, imgs3dscale =1, imgs3dpower=1, e_pscale = 100, angscale=1, angtype='theta', thresh=1e-4):
    """Short summary.
    Pasted from GetDataAngle() in AngleTrain
    get data for training - returns imgs3d, e_p, ang, ecal; called in Gan3DTrainAngle

    Parameters
    ----------
    datafile : type
        Description of parameter `datafile`.
    imgs3dscale : type
        Description of parameter `imgs3dscale`.
    imgs3dpower : type
        Description of parameter `imgs3dpower`.
    e_pscale : type
        Description of parameter `e_pscale`.
    angscale : type
        Description of parameter `angscale`.
    angtype : type
        Description of parameter `angtype`.
    thresh : type
        Description of parameter `thresh`.

    Returns
    -------
    type
        Description of returned object.

    """
    print ('Loading Data from .....', datafile)
    f = h5py.File(datafile,'r')                    # load data into f variable
    ang = np.array(f.get(angtype))                 # ang is an array of angle data from f, one value is concatenated onto the latent vector
    imgs3d = np.array(f.get('ECAL'))* imgs3dscale    # imgs3d is a 3d array, cut from the cylinder that the calorimeter produces -has 25 layers along z-axis
    e_p = np.array(f.get('energy'))/e_pscale       # e_p is an array of scaled energy data from f, one value is concatenated onto the latent vector
    imgs3d[imgs3d < thresh] = 0        # when imgs3d values are less than the threshold, they are reset to 0

    # set imgs3d, e_p, and ang as float 32 datatypes
    imgs3d = imgs3d.astype(np.float32)
    e_p = e_p.astype(np.float32)
    ang = ang.astype(np.float32)

    imgs3d = np.expand_dims(imgs3d, axis=-1)         # insert a new axis at the beginning for imgs3d

    # sum along axis
    ecal = np.sum(imgs3d, axis=(1, 2, 3))    # summed imgs3d data, used for training the discriminator

    # imgs3d ^ imgs3dpower
    if imgs3dpower !=1.:
        imgs3d = np.power(imgs3d, imgs3dpower)

    # imgs3d=ecal data; e_p=energy data; ecal=summed imgs3d (used to train the discriminator); ang=angle data
    return imgs3d, e_p, ang, ecal



# note that all the images will be a bit off center because 51 --> 64 is an odd number (unequal central padding)
def resize(imgs3d, size, mode='rectangle'):
    imgs3d = imgs3d[:, :, :, :, 0]    # drop the channels dimension

    if size == 51:   # return the unchanged image [51,51,25]
        return imgs3d
     
    if mode == 'rectangle':    
        resized_imgs3d = np.zeros((5000, size, size, int(size/2)))    # create an array to hold all 5000 resized imgs3d
        
        for num_img in np.arange(5000):         # index through the 5000 3d images packed in
            img3d = imgs3d[num_img, :, :, :]    # grab an individual [51,51,25] 3d image
            
            # pad centrally with zeroes to [64x64x32], do this step first so that the framing is the same for all images
            resized_img3d = np.pad(img3d, ((7,6), (7,6), (4,3)), mode='minimum')  
        
            if size == 64:   # put in the padded image [64,64,32]
                resized_imgs3d[num_img, :, :, :] = resized_img3d  
            
            else:   # size < 64: we need to zoom out to lower the resolution
                # resize XY-plane to (size x size)
                xy_resized_img3d = np.zeros((size, size, 25))   # create an empty 3d_image to store changes
                for z_index in np.arange(25):    # index through the 25 calorimeter layers of the z-axis
                    img2d = img3d[:, :, z_index]   # grab a 2d image from the xy plane
                    resized_img2d = cv2.resize(img2d, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
                    xy_resized_img3d[:, :, z_index] = resized_img2d   # save our resized_img2d in the img3d corresponding to the calorimeter layer

                # resize YZ-plane to (size x size/2)        
                resized_img3d = np.zeros((size, size, int(size/2)))   # create an empty 3d_image to store changes            # resize YZ-plane to (size,size)=square or (size,size/2)=rectangle
                for x_index in np.arange(size):    # index through the x-axis
                    img2d = xy_resized_img3d[x_index, :, :]
                    resized_img2d = cv2.resize(img2d, dsize=(int(size/2), size), interpolation=cv2.INTER_NEAREST)
                    resized_img3d[x_index, :, :] = resized_img2d   # save our resized_img2d in the img3d corresponding to the x layer
                
            # save the resized 3d image in the matrix holding all 5000 3d images
            resized_imgs3d[num_img, :, :, :] = resized_img3d   
           
    elif mode == 'square':
        if size == 64:
            print('ERROR - Square mode is not compatible with size 64! The max size for square mode is 32.')
        else:
            resized_imgs3d = np.zeros((5000, size, size, size)) # create an array to hold all 5000 resized imgs3d
        
            for num_img in np.arange(5000):     # index through the 5000 3d images packed in
                img3d = imgs3d[num_img, :, :, :]    # grab an individual [51,51,25] 3d image

                img3d = np.pad(img3d, ((0,0), (0,0), (4,3)), mode='minimum') # pad centrally with zeroes to [51x51x32]

                # resize XY-plane to (size x size)
                xy_resized_img3d = np.zeros((size, size, 25))   # create an empty 3d_image to store changes
                for z_index in np.arange(25):    # index through the 25 calorimeter layers of the z-axis
                    img2d = img3d[:, :, z_index]   # grab a 2d image from the xy plane
                    resized_img2d = cv2.resize(img2d, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
                    xy_resized_img3d[:, :, z_index] = resized_img2d   # save our resized_img2d in the img3d corresponding to the calorimeter layer
                    
                # resize YZ-plane to (size x size)        
                resized_img3d = np.zeros((size, size, size))   # create an empty 3d_image to store changes
                for x_index in np.arange(size):    # index through the 51 values of x-axis
                    img2d = xy_resized_img3d[x_index, :, :]
                    resized_img2d = cv2.resize(img2d, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
                    resized_img3d[x_index, :, :] = resized_img2d   # save our resized_img2d in the img3d corresponding to the x layer
                    
                # save our 3d image in the matrix holding all 5000 3d images
                resized_imgs3d[num_img, :, :, :] = resized_img3d   
                
    return resized_imgs3d   # returns a [5000, size, size, size||size/2] np.array matrix that is 5000 3d images [size, size, size||size/2]


def createNumpyFiles(imgs3d, size):
    """Short summary.
        Creating Numpy array by file by size
    Parameters
    ----------
    imgs3d : type
        Description of parameter `imgs3d`.
    size : type
        Description of parameter `size`.

    Returns
    -------
    type
        Description of returned object.

    """
    output_folder = os.path.join(output, f'{size}x{size}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, tensor in enumerate(imgs3d):
        #array = tensor.numpy()
        print("saving : ", tensor.shape)
        filename = os.path.join(output_folder, f'{i:04}.npy')
        np.save(filename, tensor)


# calls run()
def convert():
    """Short summary.
        Converting the HDF5 files to a resized Numpy array
    Returns
    -------
    type
        Numpy file for each resized image

    """
    filespath = load_files(root)
    allimages = np.empty((5000,51,51,25,1))
    for afile in filespath:
        imgs3d, e_p, ang, ecal = GetDataAngle(afile)
        allimages = np.concatenate((allimages, imgs3d))

    print("allimages shape", allimages)
    imgs3d_resized = resize(allimages, 4)
    createNumpyFiles(allimages, 4)

    imgs3d_resized = resize(allimages, 8)
    createNumpyFiles(allimages, 8)

    imgs3d_resized = resize(allimages, 16)
    createNumpyFiles(imgs3d_resized, 16)

    imgs3d_resized = resize(allimages, 32)
    createNumpyFiles(imgs3d_resized, 32)

    imgs3d_resized = resize(allimages, 64)
    createNumpyFiles(imgs3d_resized, 64)

# calls convert()
if __name__ == "__main__":
    convert()
