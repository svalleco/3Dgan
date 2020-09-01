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

      
# Takes [5000x51x51x25] image array and size parameter --> returns [5000 x size x size x size or size/2] np.array that is 5000 3d images
def resize(imgs3d, size, mode='rectangle'):
    if mode == 'square':
        resized_imgs3d = np.zeros((5000, size, size, size)) # create an array to hold all 5000 resized imgs3d
    else:    # mode == 'rectangle'
        resized_imgs3d = np.zeros((5000, size, size, int(size/2))) # create an array to hold all 5000 resized imgs3d


    for num_img in np.arange(5000):     # index through the 5000 3d images packed in
        img3d = imgs3d[num_img, :, :, :, 0]    # grab an individual [51,51,25] 3d image
        #print('img3ds shape: ')
        #print(img3d.shape)
        if size < 64:
            
            xy_resized_img3d = np.zeros((size, size, 25))   # create an empty 3d_image to store changes
            # resize XY-plane to (size x size)
            for z_index in np.arange(25):    # index through the 25 calorimeter layers of the z-axis
                #print('entered z_index loop: '+str(z_index))
                img2d = img3d[:, :, z_index]   # grab a 2d image from the xy plane
                #print('img2ds shape: ')
                #print(img2d.shape)
                resized_img2d = cv2.resize(img2d, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
                #print('resized_img2ds shape: ')
                #print(resized_img2d.shape)
                xy_resized_img3d[:, :, z_index] = resized_img2d   # save our resized_img2d in the img3d corresponding to the calorimeter layer
            
            # resize YZ-plane to (size x size or size/2)
            if mode == 'square':
                resized_img3d = np.zeros((size, size, size))   # create an empty 3d_image to store changes
            else:    # mode == 'rectangle'
                resized_img3d = np.zeros((size, size, int(size/2)))   # create an empty 3d_image to store changes            # resize YZ-plane to (size,size)=square or (size,size/2)=rectangle
            for x_index in np.arange(size):    # index through the 51 values of x-axis
                img2d = xy_resized_img3d[x_index, :, :]
                #print('x_index img2d.shape')
                #print(img2d.shape)   # (size, 25)
                if mode == 'square':
                    resized_img2d = cv2.resize(img2d, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
                else:    # mode == 'rectangle'
                    #print('mode is rectangle')
                    resized_img2d = cv2.resize(img2d, dsize=(int(size/2), size), interpolation=cv2.INTER_NEAREST)
                    #print('successful img2resizing')
                    #print(resized_img2d.shape)   #(16,25)
                resized_img3d[x_index, :, :] = resized_img2d   # save our resized_img2d in the img3d corresponding to the calorimeter layer
            
        elif size == 64: # NOTE: WON'T WORK WELL TO USE SIZE 64 IF YOU ARE USING THE SQUARE MODE (STRETCHING 25 --> 64), STOP AT SIZE 32
            if mode == 'rectangle':
                resized_img3d = np.pad(img3d, ((7,6), (7,6), (4,3)), mode='minimum')  # pad centrally with zeroes to [64x64x32] 
            elif mode == 'square':
                resized_img3d = np.pad(img3d, ((7,6), (7,6), (19, 20)), mode='minimum')  # pad centrally with zeroes to [64x64x64] -- MIGHT BE MESSY!
        
        elif size == 51:   # unchanged
            return resized_imgs3d
        else: 
                print('ERROR, size: '+str(size)+' passed is incompatible. Make sure the size is one of the following: [4,8,16,32,64]')
    
        resized_imgs3d[num_img, :, :, :] = resized_img3d   # save our 3d image in the matrix holding all 5000 3d images
    return resized_imgs3d   # returns a [5000, size, size, size or size/2] np.array matrix that is 5000 3d images [size, size, size or size/2]


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
    

def find_pics_with_stuff(size, num_to_see):
    pics = imgs3d
    pics = resize(pics, size)
    #print('PICS SHAPE')
    #print(pics.shape)
    count = 0
    for num_img in np.arange(5000):
        for index in np.arange(int(size/2)):
            #pic = pics[num_img, :, :, index]   #index through z axis, max 25
            #pic = pics[num_img, index, :, :]    #index through x axis, max 51
            pic = pics[num_img, :, index, :]    #index through y axis, max 51
            #print('PIC DIMS')
            #print(pic.shape)
            if np.any(pic>size/10):
                pic = pic*500 #to accentuate the color differences
                show_img2d(pic)
                #print(pic.tolist())
                
                count += 1
                if count == num_to_see:
                    return

find_pics_with_stuff(64, 1)
find_pics_with_stuff(32, 1)
find_pics_with_stuff(16, 1)
find_pics_with_stuff(8, 1)

    
    
    
    
    