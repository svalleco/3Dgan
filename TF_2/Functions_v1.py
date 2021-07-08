from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse
import os
#os.environ['LD_LIBRARY_PATH'] = os.getcwd()
#print(os.environ)
from six.moves import range
import sys
import h5py 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #I need this for my plot


def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x



def func_for_gen(nb_test, latent_size):
    noise =            np.random.normal(0, 1, (nb_test, latent_size))  #input for bit_flip() to generate true/false values for discriminator
    gen_aux =          np.random.uniform(1, 5,(nb_test,1 ))   #generates aux for dicriminator
    gen_ecal =         np.multiply(2, gen_aux)                          #generates ecal for discriminator
    generator_input =  np.multiply(gen_aux, noise)                      #generates input for generator
    return noise, gen_aux, generator_input, gen_ecal


def check_GPU():          #check if GPUs are available (not used)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("GPUs: ", gpus)
    if gpus:
      # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    return
        
 
        
def train_from_folder(train_data_path):   #load all trainingsdata files within this folder, also subfolders
    #stack train data from more files
    print("Stack Trainingsdata:")
    train_file_folder = train_data_path

    folders = os.listdir(train_file_folder)
    X=np.zeros((1,25,25,25))
    y=np.zeros((1))

    for folder in folders:
        print("Folder: ", folder)
        listing = os.listdir(train_file_folder+folder)
        for infile in listing:   
            print("File: ", infile)
            file = h5py.File(train_file_folder + folder + "//" + infile,'r')
            e_file = file.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal
            X_file = np.array(file.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild
            y_file = np.array(e_file[:,1])
            file.close()

            #print("X_file_Shape: ",X_file.shape)   
            #print("y_file_shape: ", y_file.shape)
            X = np.concatenate((X,X_file))
            y = np.concatenate((y,y_file))


    X = np.delete(X, 0,0)   #heißt Lösche Element 0 von Spalte 0
    y = np.delete(y, 0,0)   #heißt Lösche Element 0 von Spalte 0
    print("\nConcatenate:")
    print("X_Shape: ",X.shape)   
    print("y_Shape: ", y.shape)
    return X, y



def create_folder(folder_name):        #create folders in which the trainings progress will be saved
    dirName=folder_name
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    return


def plot_loss(train_history, test_history):      #plot the losses as graph
    #generator train loss
    gen_loss=[]
    gen_generation_loss=[]
    gen_auxiliary_loss=[]
    gen_lambda5_loss=[]
    x=[]
    for epoch in range(len(train_history["generator"])):
        x.append(epoch+1)
        gen_loss.append(train_history["generator"][epoch][0])
        gen_generation_loss.append(train_history["generator"][epoch][1])
        gen_auxiliary_loss.append(train_history["generator"][epoch][2])
        gen_lambda5_loss.append(train_history["generator"][epoch][3])

    #generator test loss
    gen_test_loss=[]
    gen_test_generation_loss=[]
    gen_test_auxiliary_loss=[]
    gen_test_lambda5_loss=[]
    for epoch in range(len(test_history["generator"])):
        gen_test_loss.append(test_history["generator"][epoch][0])
        gen_test_generation_loss.append(test_history["generator"][epoch][1])
        gen_test_auxiliary_loss.append(test_history["generator"][epoch][2])
        gen_test_lambda5_loss.append(test_history["generator"][epoch][3])


    #discriminator train loss
    disc_loss=[]
    disc_generation_loss=[]
    disc_auxiliary_loss=[]
    disc_lambda5_loss=[]
    x=[]
    for epoch in range(len(train_history["discriminator"])):
        x.append(epoch+1)
        disc_loss.append(train_history["discriminator"][epoch][0])
        disc_generation_loss.append(train_history["discriminator"][epoch][1])
        disc_auxiliary_loss.append(train_history["discriminator"][epoch][2])
        disc_lambda5_loss.append(train_history["discriminator"][epoch][3])

    #discriminator test loss
    disc_test_loss=[]
    disc_test_generation_loss=[]
    disc_test_auxiliary_loss=[]
    disc_test_lambda5_loss=[]
    for epoch in range(len(test_history["discriminator"])):
        disc_test_loss.append(test_history["discriminator"][epoch][0])
        disc_test_generation_loss.append(test_history["discriminator"][epoch][1])
        disc_test_auxiliary_loss.append(test_history["discriminator"][epoch][2])
        disc_test_lambda5_loss.append(test_history["discriminator"][epoch][3])    

    
    #loss
    plt.title("Generator Loss")
    plt.plot(x, gen_test_loss, label = "Generator Test", color ="green", linestyle="dashed")
    plt.plot(x, gen_loss, label = "Generator Train", color ="green")

    #discriminator loss
    plt.plot(x, disc_test_loss, label = "Discriminator Test", color ="red", linestyle="dashed")
    plt.plot(x, disc_loss, label = "Discriminator Train", color ="red")
    

    plt.legend()
    plt.ylim(0,40)
    #plt.yscale("log")
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')    
    plt.show()
    return


def loss_table(train_history,test_history):        #print the loss table during training
    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
            'component', "total_loss", "fake/true_loss", "AUX_loss", "ECAL_loss"))
    print('-' * 65)

    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f}'
    print(ROW_FMT.format('generator (train)',
                         *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                         *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                         *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                         *test_history['discriminator'][-1]))
    return



def import_data(train_on_folder, train_data_path, train_folder_path):     #function which imports the trainingsdata
    if train_on_folder == False:
        #read Trainingsfile
        d=train_data_path
        d=h5py.File(d,'r')
        e=d.get('target')  
        X=np.array(d.get('ECAL'))
        y=(np.array(e[:,1]))
        print("X_Shape (Imported Data): ",X.shape)
        print("Länge y: ", len(y))
    elif train_on_folder == True:
        X, y = train_from_folder(train_folder_path)  
    return X, y
 


def data_preperation(X, y, keras_dformat, batch_size, percent=100):      #data preperation
    X[X < 1e-6] = 0  #remove unphysical values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1)

    #take just a percentage form the data to make fast tests
    X_train=X_train[:int(len(X_train)*percent/100),:]
    y_train=y_train[:int(len(y_train)*percent/100)]
    X_test=X_test[:int(len(X_test)*percent/100),:]
    y_test=y_test[:int(len(y_test)*percent/100)]

    # tensorflow ordering
    X_train =np.expand_dims(X_train, axis=-1)  #macht jeden Eintrag in der Liste zu einer Unterliste [1,2,3]->[[1],[2],[3]]
    X_test = np.expand_dims(X_test, axis=-1)

    #print("X_train_shape (reordered): ", X_train.shape)
    if keras_dformat !='channels_last':
       X_train =np.moveaxis(X_train, -1, 1)    #Dreht die Matrix, damit die Dimension passt
       X_test = np.moveaxis(X_test, -1,1)

    y_train= y_train/100     #Normalisieren?
    y_test=y_test/100
    """
    print("X_train_shape: ", X_train.shape)
    print("X_test_shape: ", X_test.shape)
    print("y_train_shape: ", y_train.shape)
    print("y_test_shape: ", y_test.shape)
    print('*************************************************************************************')
    """
    #####################################################################################
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]
    if nb_train < batch_size:
        print("\nERROR: batch_size is larger than trainings data")
        print("batch_size: ", batch_size)
        print("trainings data: ", nb_train, "\n")

    X_train = X_train.astype(np.float32)  #alle Werte in Floats umwandeln
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    if keras_dformat =='channels_last':
        ecal_train = np.sum(X_train, axis=(1, 2, 3))
        ecal_test = np.sum(X_test, axis=(1, 2, 3))
    else:
        ecal_train = np.sum(X_train, axis=(2, 3, 4))
        ecal_test = np.sum(X_test, axis=(2, 3, 4))
    return X_train, X_test, y_train, y_test, ecal_train, ecal_test, nb_train, nb_test
        
        
def plot_images(image_tensor, epoch, save_folder, save=False, number=1):    #plot images of trainingsdata or generator
    xx = np.linspace(1,25,25)
    yy = np.linspace(1,25,25)
    XX, YY = np.meshgrid(xx, yy)
    
    
    for i in range(number):#len(image_tensor)):
        dat=image_tensor[i]
        #print(dat.shape)
        ZZ =dat[:][:][13]
        #print(ZZ.shape)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(YY, XX, ZZ, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        #ax.plot_wireframe(xx, yy, ZZ)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Energy');
        ax.set_ylim(25, 0)    #invert y axes, that the particle enters form the front side
        number_epoch = str(epoch)
        number_epoch = number_epoch.zfill(4)
        plt.title("Epoch "+number_epoch)
        if save==True:
            plt.savefig(save_folder+"/Save_Images/plot_" + number_epoch + ".png")
        plt.show()         


def reset_Session():     #a function, which resets the connection to the GPU at the end of the run
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print('---GPU memory reseted') # if it's done something you should see a number being outputted        
        
