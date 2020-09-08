import numpy as np
import tensorflow as tf
from PIL import Image
import h5py
from numpy import asarray
import cv2

# Pasted from GetDataAngle() in AngleTrain
# get data for training - returns imgs3d, e_p, ang, ecal; called in Gan3DTrainAngle
def GetDataAngle(datafile, imgs3dscale =1, imgs3dpower=1, e_pscale = 100, angscale=1, angtype='theta', thresh=1e-4):
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
    channel = imgs3d[0,0,0,0,:]
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
    
    # reorganize dimensions: (num_imgs, x,y,z) --> (num_imgs, z,x,y)
    resized_imgs3d = np.moveaxis(resized_imgs3d, 3, 1)
    
    # put the channel back in: channels_first
    resized_imgs3d = np.expand_dims(resized_imgs3d, axis=0)
    resized_imgs3d[:,0,0,0,0] = channel
            
    return resized_imgs3d   # returns a [5000, size, size, size||size/2] np.array matrix that is 5000 3d images [size, size, size||size/2]


imgs3d, e_p, ang, ecal = GetDataAngle('Ele_VarAngleMeas_100_200_005.h5')
print(imgs3d.shape)

resized_imgs3d = resize(imgs3d, 32)#, mode='square')
print(resized_imgs3d.shape)


def get_pics(datafile):
    imgs3d, e_p, ang, ecal = GetDataAngle(datafile)
    pics_res4 = resize(imgs3d, 4, mode='rectangle')
    pics_res8 = resize(imgs3d, 8, mode='rectangle')
    pics_res16 = resize(imgs3d, 16, mode='rectangle')
    pics_res32 = resize(imgs3d, 32, mode='rectangle')
    pics_res64 = resize(imgs3d, 64, mode='rectangle')
    
    return pics_res4, pics_res8, pics_res16, pics_res32, pics_res64


def get_img2d(size, num_img, z_val):
    if size != 51:
        sized_pics = resize(imgs3d, size)
    else:
        sized_pics = imgs3d
    img2d = sized_pics[num_img, :, :, z_val]
    #print('GETTING IMAGE2D')
    #print(img2d.shape)
    return img2d
   
 
def show_img2d(img2d_array):    
    pic = Image.fromarray(img2d_array)
    pic.show()
      
#pic = get_img2d(64, 2500, 12)
#show_img2d(pic)
    

def visualize_shower(size, num_to_see, axis='y', mode='rectangle'):
    pics = imgs3d
    pics = resize(pics, size, mode)
    count = 0
    for num_img in np.arange(5000):
        for index in np.arange(int(size/2)):
            if axis=='z':
                pic = pics[num_img, :, :, index]   #index through z axis, max 25
            if axis=='x':
                pic = pics[num_img, index, :, :]    #index through x axis, max 51
            if axis=='y':
                pic = pics[num_img, :, index, :]    #index through y axis, max 51
            
            # look for high content images
            if np.any(pic>size/10):
                pic = pic*500 #to accentuate the color differences
                show_img2d(pic)
                #print(pic.tolist())
                
                count += 1
                if count == num_to_see:
                    return


    
    
    
    
    