from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse
import os
from six.moves import range
import sys
import h5py 
import numpy as np
#from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #I need this for my plot



#v1_5_8: Some optimizations in the validation functions

def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

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

    y_train= y_train/100     #Normalisieren
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
        

  
def create_files_list(train_data_path):     #all files have to be within one folder
    train_file_folder = train_data_path
    files = os.listdir(train_file_folder)
    return files
    
    
def train_file_by_file_import(train_data_path, files, file_number):
    train_file_folder = train_data_path
    infile = files[file_number]
    #print("File: ", infile)
    file = h5py.File(train_file_folder + "//" + infile,'r')
    e_file = file.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal
    X_file = np.array(file.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild
    y_file = np.array(e_file[:,1])
    file.close()

    #print("X_Shape: ",X_file.shape)   
    #print("y_Shape: ", y_file.shape)
    return X_file, y_file
    
        
def train_from_folder(train_data_path):   #load all trainingsdata files within this folder, also subfolders
    #stack train data from more files
    print("Stack Trainingsdata:")
    train_file_folder = train_data_path

    folders = os.listdir(train_file_folder)
   # folders = natsorted(folders)
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



def create_folder(folder_name, print_outputs = True):        #create folders in which the trainings progress will be saved
    dirName=folder_name
    try:
        # Create target Directory
        os.mkdir(dirName)
        if print_outputs == True:
            print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        if print_outputs == True:
            print("Directory " , dirName ,  " already exists")
    return


def plot_loss(train_history, test_history, save_folder, save=False):      #plot the losses as graph
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

    
    #generator loss
    plt.title("Total Loss")
    plt.plot(x, gen_test_loss, label = "Generator Test", color ="green", linestyle="dashed")
    plt.plot(x, gen_loss, label = "Generator Train", color ="green")

    #discriminator loss
    plt.plot(x, disc_test_loss, label = "Discriminator Test", color ="red", linestyle="dashed")
    plt.plot(x, disc_loss, label = "Discriminator Train", color ="red")
    

    plt.legend()
    if len(x) >=5:
        plt.ylim(0,40)
    #plt.yscale("log")
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    if save == True:
        plt.savefig(save_folder + "/lossplot.png")    
    plt.show()
    
    ######################################################
    #second plot
    plt.title("Single Losses")
    #plt.plot(x, gen_loss, label = "Generator Total", color ="green")
    plt.plot(x, gen_generation_loss, label = "Gen True/Fake", color ="green")
    plt.plot(x, gen_auxiliary_loss, label = "Gen AUX", color ="lime")
    plt.plot(x, gen_lambda5_loss, label = "Gen ECAL", color ="aquamarine")
    
    #plt.plot(x, disc_loss, label = "Discriminator Train", color ="red")
    plt.plot(x, disc_generation_loss, label = "Disc True/Fake", color ="red")
    plt.plot(x, disc_auxiliary_loss, label = "Disc AUX", color ="orange")
    plt.plot(x, disc_lambda5_loss, label = "Disc ECAL", color ="lightsalmon")
    
    plt.legend()
    if len(x) >=5:
        plt.ylim(0,20)
    #plt.yscale("log")
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    if save == True:
        plt.savefig(save_folder + "/single_losses.png")    
    plt.show()
    
    return

#plot for validation metric
def plot_validation(train_history, save_folder):
    validation_loss =[]
    epoch=[]
    for i in range(len(train_history["validation"])):
        epoch.append(i+1)
        validation_loss.append(train_history["validation"][i])
        
    plt.title("Validation Metric")
    plt.plot(epoch, validation_loss, label = "Validation Metric", color ="green")
    plt.legend()
    if len(epoch) >=5:
        plt.ylim(0,1)
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Metric') 
    plt.savefig(save_folder + "/Validation_Plot.png")    
    plt.show()
    return

#gromov-wasserstein-distance
def plot_gromov_w_distance(train_history, save_folder):
    validation_loss =[]
    epoch=[]
    for i in range(len(train_history["Gromov_Wasserstein_validation"])):
        epoch.append(i+1)
        validation_loss.append(train_history["Gromov_Wasserstein_validation"][i])
        
    plt.title("Gromov Wasserstein Distance")
    plt.plot(epoch, validation_loss, label = "Gromov Wasserstein Distance", color ="green")
    plt.legend()
    if len(epoch) >=5:
        plt.ylim(0,0.1)
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Gromov Wasserstein Distance') 
    plt.savefig(save_folder + "/Gromov Wasserstein Distance.png")    
    plt.show()
    return


def loss_table(train_history,test_history, save_folder, epoch = 0, validation_metric = 0,  save=False, timeforepoch=0):        #print the loss table during training
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
    if save == True: 
        if epoch == 0:
            f= open(save_folder + "/loss_table.txt","w")
        else:
            f= open(save_folder + "/loss_table.txt","a")
        str_epoch = "Epoch: " + str(epoch)
        f.write(str_epoch) 
        f.write("\n")
        f.write('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format('component', "total_loss", "fake/true_loss", "AUX_loss", "ECAL_loss"))
        f.write("\n")
        f.write('-' * 65) 
        f.write("\n")
        f.write(ROW_FMT.format('generator (train)', *train_history['generator'][-1])) 
        f.write("\n")
        f.write(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))         
        f.write("\n")
        f.write(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1])) 
        f.write("\n")
        f.write(ROW_FMT.format('discriminator (test)', *test_history['discriminator'][-1])) 
        e = timeforepoch
        f.write('\nTime for Epoch: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        f.write("\nValidarion Metric: " + str(validation_metric))
        f.write("\nGromov Wasserstein Distance: " + str(train_history['Gromov_Wasserstein_validation'][-1]))
        f.write("\n\n")
        f.close()                    
    return




        
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
        ax.set_ylabel('z')
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
        





        
        
        
        
        
        
        
        
        
        
#############################################################################
#functions for gulrukhs validation function

import argparse
import os
from six.moves import range
import sys
import h5py 
import numpy as np
#from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy.core.umath_tests as umath


#functions which can be saved seperately

# get sums along different axis
def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

# get moments
def get_moments(sumsx, sumsy, sumsz, totalE, m, x=51, y=51, z=25):
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    totalE = np.squeeze(totalE)
    index = sumsx.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
        relativeIndices = np.tile(np.arange(x), (index,1))
        moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
        ECAL_momentX = np.divide(umath.inner1d(sumsx, moments) ,totalE)
        if i==0: ECAL_midX = ECAL_momentX.transpose()
        momentX[:,i] = ECAL_momentX
    for i in range(m):
        relativeIndices = np.tile(np.arange(y), (index,1))
        moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
        ECAL_momentY = np.divide(umath.inner1d(sumsy, moments), totalE)
        if i==0: ECAL_midY = ECAL_momentY.transpose()
        momentY[:,i]= ECAL_momentY
    for i in range(m):
        relativeIndices = np.tile(np.arange(z), (index,1))
        moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
        ECAL_momentZ = np.divide(umath.inner1d(sumsz, moments), totalE)
        if i==0: ECAL_midZ = ECAL_momentZ.transpose()
        momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

#Optimization metric
def metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1):
    metricp = 0
    metrice = 0
    metrica = 0
    for energy in energies:
        #Relative error on mean moment value for each moment and each axis
        x_act= np.mean(var["momentX_act"+ str(energy)], axis=0)
        x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
        y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
        y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
        z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
        z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
        var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act
        var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
        var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
        #Taking absolute of errors and adding for each axis then scaling by 3
        var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)])+ np.absolute(var["posz_error"+ str(energy)]))/3
        #Summing over moments and dividing for number of moments
        var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
        metricp += var["pos_total"+ str(energy)]
        #Take profile along each axis and find mean along events
        sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis=0), np.mean(var["sumsz_act" + str(energy)], axis=0)
        sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
        var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
        var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
        var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
        #Take absolute of error and mean for all events                                                           
        var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)]) + np.absolute(var["posz_error"+ str(energy)]))/3
        #Summing over moments and dividing for number of moments
        var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
        metricp += var["pos_total"+ str(energy)]
        #Take profile along each axis and find mean along events
        sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis= 0), np.mean(var["sumsz_act" + str(energy)], axis=0)
        sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
        var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
        var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
        var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
        #Take absolute of error and mean for all events
        var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)]))/x
        var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)]))/y
        var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/z

        var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
        metrice += var["eprofile_total"+ str(energy)]
        if ang:
            var["angle_error"+ str(energy)] = np.mean(np.absolute((var[angtype + "_act" + str(energy)] - var[angtype + "_gan" + str(energy)])/var[angtype + "_act" + str(energy)]))
            metrica += var["angle_error"+ str(energy)]
    metricp = metricp/len(energies)
    metrice = metrice/len(energies)
    if ang:metrica = metrica/len(energies)
    tot = metricp + metrice
    if ang:tot = tot +metrica
    result = [tot, metricp, metrice]
    if ang: result.append(metrica)
    return result

# short version of analysis                                                                                                                      
def OptAnalysisShort(var, generated_images, energies, ang=1):
    m=2
    
    x = generated_images.shape[1]
    y = generated_images.shape[2]
    z = generated_images.shape[3]
    for energy in energies:
        if energy==0:
            var["events_gan" + str(energy)]=generated_images
        else:
            var["events_gan" + str(energy)]=generated_images[var["indexes" + str(energy)]]
        #print(var["events_gan" + str(energy)].shape)
        var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
        var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
        var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
        var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
        var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
        if ang: var["angle_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])
    return metric(var, energies, m, angtype='angle', x=x, y=y, z=z, ang=ang)



#validation script
#keras_dformat= "channels_first"
def validate(generator, percent=20, keras_dformat='channels_first', data_path="/eos/home-f/frehm/TF2/Data/EleEscan_1_1/"):
    X=np.zeros((1,25,25,25))
    y=np.zeros((1))
    file = h5py.File(data_path + "EleEscan_1_2.h5",'r')   #file_1 does not work and gives nan values
    e_file = file.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal
    X_file = np.array(file.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild
    y_file = np.array(e_file[:,1])
    file.close()
    file2 = h5py.File(data_path + "EleEscan_1_3.h5",'r')   #file_1 does not work and gives nan values
    e_file2 = file2.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal
    X_file2 = np.array(file2.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild
    y_file2 = np.array(e_file2[:,1])
    file2.close()
    X = np.concatenate((X_file, X_file2))
    y = np.concatenate((y_file, y_file2))

    X[X < 1e-6] = 0  #remove unphysical values

    X = np.delete(X, 0,0)   #heißt Lösche Element 0 von Spalte 0
    y = np.delete(y, 0,0)   #heißt Lösche Element 0 von Spalte 0
    
    X_val = X
    y_val = y

    X_val=X_val[:int(len(X_val)*percent/100),:]
    y_val=y_val[:int(len(y_val)*percent/100)]

    # tensorflow ordering
    X_val = np.expand_dims(X_val, axis=-1)


    if keras_dformat !='channels_last':
        X_val = np.moveaxis(X_val, -1,1)

    y_val=y_val/100

    nb_val = X_val.shape[0]


    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    if keras_dformat =='channels_last':
        ecal_val = np.sum(X_val, axis=(1, 2, 3))
    else:
        ecal_val = np.sum(X_val, axis=(2, 3, 4))

    X_val = np.squeeze(X_val)
    var={}
    tolerance = 5
    energies = [0, 50, 100, 200, 250, 300, 400, 500]
    data0 = X_val  #the generated data
    data1 = y_val    #aux
    ecal = ecal_val
    ang=0
    for energy in energies:
        if energy==0:
            var["events_act" + str(energy)]=data0
            var["energy" + str(energy)]=data1
            if ang: var["angle_act" + str(energy)]=data[2]
            var["ecal_act" + str(energy)]=ecal
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
        else:
            var["indexes" + str(energy)] = np.where((data1 > (energy - tolerance)/100. ) & ( data1 < (energy + tolerance)/100.))
            var["events_act" + str(energy)]=data0[var["indexes" + str(energy)]]
            var["energy" + str(energy)]=data1[var["indexes" + str(energy)]]
            if ang:  var["angle_act" + str(energy)]=data[2][var["indexes" + str(energy)]]
            var["ecal_act" + str(energy)]=ecal[var["indexes" + str(energy)]]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]


    #validation

    #var = sortEnergy([np.squeeze(X_test), Y_test], np.squeeze(ecal_test), energies, ang=0)
    nb_test = len(y_val); latent_size =200
    noise = np.random.normal(0.1, 1, (nb_test, latent_size))
    generator_ip = np.multiply(data1.reshape((-1, 1)), noise)

    #sess = tf.compat.v1.Session(graph = infer_graph)
    generated_images = generator.predict(generator_ip,batch_size=128)
    #generated_images = sess.run(l_output, feed_dict = {l_input:generator_ip})
    generated_images= np.squeeze(generated_images)

    #print("X_test_shape: ", X_val.shape)
    #print("X gen shape : ", generated_images.shape)
    #print("ecal_val_shape: ", ecal_val.shape)

    analysis_history = defaultdict(list)
    #generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
    result = OptAnalysisShort(var, generated_images, energies, ang=0)
    print('Analysing............')
    # All of the results correspond to mean relative errors on different quantities
    analysis_history['total'].append(result[0]) 
    analysis_history['energy'].append(result[1])   #this is the number to optimize
    analysis_history['moment'].append(result[2])
    print('Result = ', result[1]) #optimize over result[0]
    #print("Value to optimize: ", np.round(result[1],3))
    #pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))
    return result[1]














############################################################################################
#wasserstein Validation
#Gromov-Wasserstein Validation
#https://github.com/svalleco/3Dgan/blob/Anglegan/keras/misc/GromovWass.py#L287

def preproc(n, scale=1):
    return n * scale

def postproc(n, scale=1):
    return n/scale

def load_sorted(sorted_path, energies, ang=0):
    sorted_files = sorted(glob.glob(sorted_path))
    srt = {}
    for f in sorted_files:
        energy = int(filter(str.isdigit, f)[:-1])
        if energy in energies:
            srtfile = h5py.File(f,'r')
            srt["events_act" + str(energy)] = np.array(srtfile.get('ECAL'))
            srt["energy" + str(energy)] = np.array(srtfile.get('Target'))
            if ang:
                srt["angle" + str(energy)] = np.array(srtfile.get('Angle'))
            print( "Loaded from file", f)
    return srt
import glob
# sort data for fixed angle
def get_sorted(datafiles, energies, flag=False, num_events1=10000, num_events2=2000, tolerance=5, thresh=0):
    srt = {}
    for index, datafile in enumerate(datafiles):
        data = GetData(datafile, thresh)
        X = data[0]
        sumx = np.sum(np.squeeze(X), axis=(1, 2, 3))
        indexes= np.where(sumx>0)
        X=X[indexes]
        Y = data[1]
        Y=Y[indexes]
        for energy in energies:
            if index== 0:
                if energy == 0:
                    srt["events_act" + str(energy)] = X # More events in random bin
                    srt["energy" + str(energy)] = Y
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                    srt["events_act" + str(energy)] = X[indexes]
                    srt["energy" + str(energy)] = Y[indexes]
            else:
                if energy == 0:
                   if flag:
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X, axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y, axis=0)
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    if srt["events_act" + str(energy)].shape[0] < num_events2:
                        indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                        srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X[indexes], axis=0)
                        srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y[indexes], axis=0)
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events2]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events2]
    return srt

def save_sorted(srt, energies, srtdir, ang=0):
    safe_mkdir(srtdir)
    for energy in energies:
        srtfile = os.path.join(srtdir, "events_{:03d}.h5".format(energy))
        with h5py.File(srtfile ,'w') as outfile:
            outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
            outfile.create_dataset('Target',data=srt["energy" + str(energy)])
            if ang:
                outfile.create_dataset('Angle',data=srt["angle" + str(energy)])
        print ("Sorted data saved to {}".format(srtfile))

    
#Divide files in train and test lists     
def DivideFiles(FileSearch="/data/LCD/*/*.h5",
                Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
    #print ("Searching in :",FileSearch)
    Files =sorted( glob.glob(FileSearch))
    #print ("Found {} files. ".format(len(Files)))
    FileCount=0
    Samples={}
    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0].replace("Escan","")
        if ParticleName in Particles:
            try:
                Samples[ParticleName].append(F)
            except:
                Samples[ParticleName]=[(F)]
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break
    out=[]
    for j in range(len(Fractions)):
        out.append([])
    SampleI=len(Samples.keys())*[int(0)]
    for i,SampleName in enumerate(Samples):
        Sample=Samples[SampleName]
        NFiles=len(Sample)
        for j,Frac in enumerate(Fractions):
            EndI=int(SampleI[i]+ round(NFiles*Frac))
            out[j]+=Sample[SampleI[i]:EndI]
            SampleI[i]=EndI
    return out    

# sort data for fixed angle
def get_sorted(datafiles, energies, flag=False, num_events1=10000, num_events2=2000, tolerance=5, thresh=0):
    srt = {}
    for index, datafile in enumerate(datafiles):
        data = GetData(datafile, thresh)
        X = data[0]
        sumx = np.sum(np.squeeze(X), axis=(1, 2, 3))
        indexes= np.where(sumx>0)
        X=X[indexes]
        Y = data[1]
        Y=Y[indexes]
        for energy in energies:
            if index== 0:
                if energy == 0:
                    srt["events_act" + str(energy)] = X # More events in random bin
                    srt["energy" + str(energy)] = Y
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                    srt["events_act" + str(energy)] = X[indexes]
                    srt["energy" + str(energy)] = Y[indexes]
            else:
                if energy == 0:
                   if flag:
                    srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X, axis=0)
                    srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y, axis=0)
                    if srt["events_act" + str(energy)].shape[0] > num_events1:
                        srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
                        srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
                        flag=False
                else:
                    if srt["events_act" + str(energy)].shape[0] < num_events2:
                        indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
                        srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X[indexes], axis=0)
                        srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y[indexes], axis=0)
                    srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events2]
                    srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events2]
    return srt

# get data for fixed angle
def GetData(datafile, thresh=0, num_events=10000):
   #get data for training
    #print( 'Loading Data from .....', datafile)
    f=h5py.File(datafile,'r')
    y=f.get('target')[:num_events]
    x=np.array(f.get('ECAL')[:num_events])
    y=(np.array(y[:,1]))
    if thresh>0:
        x[x < thresh] = 0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

# generate images
def generate(g, index, cond, latent=256, concat=1, batch_size=50):
    energy_labels=np.expand_dims(cond[0], axis=1)
    if len(cond)> 1: # that means we also have angle
        angle_labels = cond[1]
        if concat==1:
            noise = np.random.normal(0, 1, (index, latent-1))  
            noise = energy_labels * noise
            gen_in = np.concatenate((angle_labels.reshape(-1, 1), noise), axis=1)
        elif concat==2:
            noise = np.random.normal(0, 1, (index, latent-2))
            gen_in = np.concatenate((energy_labels, angle_labels.reshape(-1, 1), noise), axis=1)
        else:  
            noise = np.random.normal(0, 1, (index, 2, latent))
            angle_labels=np.expand_dims(angle_labels, axis=1)
            gen_in = np.concatenate((energy_labels, angle_labels), axis=1)
            gen_in = np.expand_dims(gen_in, axis=2)
            gen_in = gen_in * noise
    else:
        noise = np.random.normal(0, 1, (index, latent))
        #energy_labels=np.expand_dims(energy_labels, axis=1)
        gen_in = energy_labels * noise
    generated_images = g.predict(gen_in, verbose=False, batch_size=batch_size)
    return generated_images

import scipy as sp
#import ot
def Gromov_metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1):
    metricp = 0
    metrice = 0
    metrica = 0
    metrics = 0
         
    for i, energy in enumerate([energies[0]]):
        if i==0:
            moment_act=np.hstack((var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]))
            shapes_act = np.hstack((var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)]))
            if ang: angles_act= np.reshape(var["mtheta_act" + str(energy)], (-1, 1))
            
            #print(var["sf_act"+ str(energy)].shape)
            sampfr_act = np.reshape(var["sf_act"+ str(energy)], (-1, 1))
            moment_gan=np.hstack((var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)]))
            shapes_gan = np.hstack((var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)]))
            if ang: angles_gan= np.reshape(var["mtheta_gan" + str(energy)], (-1, 1))
            #print(var["sf_gan"+ str(energy)].shape)
            sampfr_gan = np.reshape(var["sf_gan"+ str(energy)], (-1, 1))
            #sampfr_gan = var["sf_gan"+ str(energy)]
            #print(sampfr_gan.shape)
         
         
        #print(moment_act.shape)
        #print(shapes_act.shape)
        #print(sampfr_act.shape)
        #print(moment_gan.shape)
        #print(shapes_gan.shape)
        #print(sampfr_gan.shape)
        metric_act = np.hstack((moment_act, shapes_act, sampfr_act))
        metric_gan = np.hstack((moment_gan, shapes_gan, sampfr_gan))
        #print("gan shape ", metric_gan.shape)
        #print("act shape ", metric_act.shape)

        metric_a = np.transpose(metric_act)
        metric_g = np.transpose(metric_gan)
        #print("a shape ", metric_a.shape)
        #print("g shape ", metric_g.shape)
        a = (0.25 /127.)* np.ones(74)  #127
        b = (0.25/6.) *np.ones(6)
        c = 0.25 *np.ones(2)
    
        p = np.concatenate((a, b, c))
        q = np.concatenate((a, b, c))
        #print("p shape ", p.shape)
        #print("q shape ", q.shape)
        C1 = sp.spatial.distance.cdist(metric_a, metric_a, 'correlation')
        #print("c1 shape ", C1.shape)
        C2 = sp.spatial.distance.cdist(metric_g, metric_g, 'correlation')
        #print("c2 shape ", C2.shape)
        C1 = C1/np.amax(C1)
        C2 = C2/np.amax(C2)
        #gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=True, log=True)
        #gw, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'kl_loss', epsilon=5e-2, log=True, verbose=True)
        #print("gw")
        gw, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=5e-2, log=True, verbose=True)
        print('Gromov-Wasserstein distances: ' + str(log['gw_dist']))
    return log['gw_dist']



def analyse(g, read_data, save_data, gen_weights , sorted_path, optimizer, data_path= "/eos/home-f/frehm/TF2/Data/EleEscan_1_1/", xscale=1, power=1, particle="Ele", 
            thresh=1e-6, ang=0, concat=1, preproc=preproc, postproc=postproc):
    #print ("Started")
    num_events=2000
    num_data = 140000
    ascale = 1
    Test = False
    latent= 200
    m = 2
    var = {}
    energies = [0]#, 150, 190]
    #energies = [50, 100, 200, 300, 400]
    sorted_path= sorted_path 
    #g =generator(latent)
    if read_data:
        start = time.time()
        var = load_sorted(sorted_path + "/*.h5", energies, ang = ang)
        sort_time = time.time()- start
        #print ("Events were loaded in {} seconds".format(sort_time))
    else:
        Trainfiles, Testfiles = DivideFiles(data_path + "EleEscan_1_1.h5", Fractions=[.9,.1], datasetnames=["ECAL"], Particles =[particle])
        if Test:
            data_files = Testfiles
        else:
            data_files = Trainfiles + Testfiles
        start = time.time()
        #energies = [50, 100, 200, 250, 300, 400, 500]
        var = get_sorted(data_files, energies, flag=False, num_events1=10000, num_events2=2000, thresh=thresh)
        data_time = time.time() - start
        #print ("{} events were loaded in {} seconds".format(num_data, data_time))
        if save_data:
            save_sorted(var, energies, sorted_path, ang=ang)        
    
    
    total = 0
    for energy in energies:
        var["index" + str(energy)]= var["energy" + str(energy)].shape[0]
        total += var["index" + str(energy)]
        data_time = time.time() - start
    #print ("{} events were put in {} bins".format(total, len(energies)))
    #g.load_weights(gen_weights)
                
    start = time.time()
    for energy in energies:
        if ang:
            var["events_gan" + str(energy)] = generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100], concat=concat, latent=latent)
        else:
            var["events_gan" + str(energy)] = generate(g, var["index" + str(energy)], [var["energy" + str(energy)]/100], concat=concat, latent=latent)
            var["events_gan" + str(energy)] = np.squeeze(var["events_gan" + str(energy)])
        var["events_gan" + str(energy)] = postproc(var["events_gan" + str(energy)], scale=xscale)
    gen_time = time.time() - start
    #print ("{} events were generated in {} seconds".format(total, gen_time))
    calc={}
    #print("Weights are loaded in {}".format(gen_weights))
    for energy in energies:
        x = var["events_act" + str(energy)].shape[1]
        y = var["events_act" + str(energy)].shape[2]
        z = var["events_act" + str(energy)].shape[3]
        var["ecal_act"+ str(energy)] = np.sum(var["events_act" + str(energy)], axis = (1, 2, 3))
        var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
        calc["sumsx_act"+ str(energy)], calc["sumsy_act"+ str(energy)], calc["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
        calc["sumsx_gan"+ str(energy)], calc["sumsy_gan"+ str(energy)], calc["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
        #print(calc["sumsx_act"+ str(energy)].shape)
        calc["momentX_act" + str(energy)], calc["momentY_act" + str(energy)], calc["momentZ_act" + str(energy)]= get_moments(calc["sumsx_act"+ str(energy)], calc["sumsy_act"+ str(energy)], calc["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
        calc["momentX_gan" + str(energy)], calc["momentY_gan" + str(energy)], calc["momentZ_gan" + str(energy)] = get_moments(calc["sumsx_gan"+ str(energy)], calc["sumsy_gan"+ str(energy)], calc["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
        if ang:
            calc["mtheta_act"+ str(energy)]= measPython(var["events_act" + str(energy)])
            calc["mtheta_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])
        #print(var["ecal_act"+ str(energy)].shape)
        var["ecal_gan"+ str(energy)] = np.expand_dims(var["ecal_gan"+ str(energy)],-1)
        #print(var["ecal_gan"+ str(energy)].shape)
        calc["sf_act" + str(energy)] = np.divide(var["ecal_act"+ str(energy)], np.reshape(var["energy"+ str(energy)], (-1, 1)))
        calc["sf_gan" + str(energy)] = np.divide(var["ecal_gan"+ str(energy)], np.reshape(var["energy"+ str(energy)], (-1, 1)))
    return optimizer(calc, energies, m, x=x, y=y, z=z, ang=ang)
