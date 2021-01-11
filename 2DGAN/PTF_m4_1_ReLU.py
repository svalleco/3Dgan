#file which contains neural network models and loss functions

#v4_1: includes a second network with ReLU function as last layer

import tensorflow as tf
import numpy as np
import time
import sys


def generator_LeakyReLU(latent_size=200, keras_dformat='channels_first'):
    
    #keras_dformat='channels_first'
    if keras_dformat == 'channels_first':
        axis = 1  #if channels first, -1 if channels last
        dshape = [1, 25, 25, 25]
    else:
        axis = -1
        dshape = [25, 25, 25, 1]
    dim = (5,5,5)

    latent = tf.keras.Input(shape=(latent_size ), dtype="float32")  #define Input     
    x = tf.keras.layers.Dense(5*5*5, input_dim=latent_size)(latent)   #shape (none, 625) #none is batch size
    x = tf.keras.layers.Reshape(dim) (x)  #shape after (none, 5,5,5)  
    
    x1 = x
    x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([2,3,1])(x)   #permute starts indexing with 1
    
    def path(x):
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #2.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(3,3), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #3.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (8, 8), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #4. Conv Block
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #5. Conv Block
        x = tf.keras.layers.Conv2D(32, (4, 4), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #6. Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2)(x)
        #7.Conv Block
        x = tf.keras.layers.Conv2D(25, (2, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2)(x)
        return x

    x1 = path(x1)
    x2 = path(x2)
    x3 = path(x3)

    x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1

    x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis) #i stack them on the channels axis
    
    
#    print(x.shape)   
    x = tf.keras.layers.Conv2D(25, (3,3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
    
    x = tf.keras.layers.Reshape(dshape)(x)

    return tf.keras.Model(inputs=[latent], outputs=x)   #Model mit Input und Output zurückgeben


def generator_ReLU(latent_size=200, keras_dformat='channels_first', epoch=0):
    
    #keras_dformat='channels_first'
    if keras_dformat == 'channels_first':
        axis = 1  #if channels first, -1 if channels last
        dshape = [1, 25, 25, 25]
    else:
        axis = -1
        dshape = [25, 25, 25, 1]
    dim = (5,5,5)

    latent = tf.keras.Input(shape=(latent_size ), dtype="float32")  #define Input     
    x = tf.keras.layers.Dense(5*5*5, input_dim=latent_size)(latent)   #shape (none, 625) #none is batch size
    x = tf.keras.layers.Reshape(dim) (x)  #shape after (none, 5,5,5)  
    
    x1 = x
    x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([2,3,1])(x)   #permute starts indexing with 1
    
    def path(x):
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #2.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(3,3), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #3.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (8, 8), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #4. Conv Block
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #5. Conv Block
        x = tf.keras.layers.Conv2D(32, (4, 4), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #6. Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2)(x)
        #7.Conv Block
        x = tf.keras.layers.Conv2D(25, (2, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2)(x)
        return x

    x1 = path(x1)
    x2 = path(x2)
    x3 = path(x3)

    x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1

    x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis) #i stack them on the channels axis
    
    
#    print(x.shape)   
    x = tf.keras.layers.Conv2D(25, (3,3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
    
    x = tf.keras.layers.Reshape(dshape)(x)
    x = tf.keras.layers.ReLU() (x)
    return tf.keras.Model(inputs=[latent], outputs=x)   #Model mit Input und Output zurückgeben
#generator().summary()


def discriminator(keras_dformat='channels_first'):
    #keras_dformat='channels_first'
    if keras_dformat =='channels_last':
        dshape=(25, 25, 25,1)
        daxis=(1,2,3)
        axis = -1 
    else:
        dshape=(1, 25, 25, 25)
        daxis=(2,3,4)
        axis = 1 
    #keras_dformat='channels_first'   #i need this when I train gen with ch last and keras with ch first
    #dshape=(25, 25, 25, 1)    
    image = tf.keras.layers.Input(shape=dshape, dtype="float32")     #Input Image
    x = image
    x = tf.keras.layers.Reshape([25,25,25])(x)
    
    x1 = x
    x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([2,3,1])(x)   #permute starts indexing with 1
    
    def path(x):
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(64, (8, 8), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #2.Conv Block
        x = tf.keras.layers.Conv2D(32, (6, 6), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #x = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2), data_format=keras_dformat)(x)
        #3.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #4. Conv Block
        x = tf.keras.layers.Conv2D(32, (4, 4), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #x = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2), data_format=keras_dformat)(x)
        #6. Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #7. Conv Block
        x = tf.keras.layers.Conv2D(9, (3, 3), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #print(x.shape)
        return x

    x1 = path(x1)
    x2 = path(x2)
    x3 = path(x3)
    print(x1.shape)
    
    x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1
    
    x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis) #i stack them on the channels axis

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization(axis=axis)(x)

    #Takes Network outputs as input
    fake = tf.keras.layers.Dense(1, activation='sigmoid', name='generation')(x)   #Klassifikator true/fake
    aux = tf.keras.layers.Dense(1, activation='linear', name='auxiliary')(x)       #Soll sich an E_in (Ep) annähern
    #Takes image as input
    ecal = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=daxis))(image)    #Energie, die im Netzwerk steckt, Summe über gesamtes NEtzwerk
    return tf.keras.Model(inputs=image, outputs=[fake, aux, ecal])

#discriminator().summary()


# return a fit for Ecalsum/Ep for Ep 
#https://github.com/svalleco/3Dgan/blob/0c4fb6f7d47aeb54aae369938ac2213a2cc54dc0/keras/EcalEnergyTrain.py#L288
def GetEcalFit(sampled_energies, particle='Ele', mod=0, xscale=1): #the smaller the energy, the closer is the factor to 2, the bigger the energy, the smaller is the factor
    if mod==0:
       return np.multiply(2, sampled_energies)
    elif mod==1:
       if particle == 'Ele':
         root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale
       elif particle == 'Pi0':
         root_fit = [0.0085, -0.094, 2.051]
         ratio = np.polyval(root_fit, sampled_energies)
         return np.multiply(ratio, sampled_energies) * xscale

        
def func_for_gen(nb_test, latent_size=200, epoch=10):
    noise =            np.random.normal(0, 1, (nb_test, latent_size))  #input for bit_flip() to generate true/false values for discriminator
    if epoch<3:
        gen_aux =          np.random.uniform(1, 4,(nb_test,1 ))   #generates aux for dicriminator
    else:
        gen_aux =          np.random.uniform(0.02, 5,(nb_test,1 ))   #generates aux for dicriminator
    #gen_ecal =         np.multiply(2, gen_aux)                          #generates ecal for discriminator
    generator_input =  np.multiply(gen_aux, noise)                      #generates input for generator
    gen_ecal_func =    GetEcalFit(gen_aux, mod=1)
    return noise, gen_aux, generator_input, gen_ecal_func


def bit_flip_tf(x, prob = 0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1* np.logical_not(x[selection])
    x = tf.constant(x)
    return x


def disc_loss(generator, discriminator,image_batch, energy_batch, ecal_batch, batch_size, label, latent_size=200, wtf=6.0, wa=0.2, we=0.1, epoch=0):
    discriminate = discriminator(image_batch)

    #true/fake loss
    if label == "ones":
        labels = bit_flip_tf(tf.ones_like(discriminate[0])*0.9)     #true=1    
    elif label == "zeros":
        labels = bit_flip_tf(tf.zeros_like(discriminate[0])*0.1)    #fake=0
    loss_true_fake = tf.reduce_mean(- labels * tf.math.log(discriminate[0] + 2e-7) - (1 - labels) * tf.math.log(1 - discriminate[0] + 2e-7))

    #aux loss
    loss_aux = tf.reduce_mean(tf.math.abs((energy_batch - discriminate[1])/(energy_batch + 2e-7))) *100
        
    #ecal loss
    loss_ecal = tf.reduce_mean(tf.math.abs((ecal_batch - discriminate[2])/(ecal_batch + 2e-7))) *100
   
    #total loss
    weight_true_fake = wtf
    weight_aux = wa
    weight_ecal = we
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
    return total_loss, loss_true_fake, loss_aux, loss_ecal


def gen_loss(generator, discriminator, batch_size=128, latent_size=200, epoch=10, wtf=6.0, wa=0.2, we=0.1):
    noise, gen_aux, generator_input, gen_ecal = func_for_gen(nb_test=batch_size, latent_size=latent_size, epoch=epoch) 
    generated_images = generator(generator_input)
    discriminator_fake = discriminator(generated_images)
    
    #true/fake
    label_fake = bit_flip_tf(tf.ones_like(discriminator_fake[0])*0.9)   #ones = true
    loss_true_fake = tf.reduce_mean(- label_fake * tf.math.log(discriminator_fake[0] + 2e-7) - 
                               (1 - label_fake) * tf.math.log(1 - discriminator_fake[0] + 2e-7))
    
    #aux
    loss_aux = tf.reduce_mean(tf.math.abs((gen_aux - discriminator_fake[1])/(gen_aux + 2e-7))) *100
    
    #ecal
    loss_ecal = tf.reduce_mean(tf.math.abs((gen_ecal - discriminator_fake[2])/(gen_ecal + 2e-7))) *100
    
    #total loss
    weight_true_fake = wtf
    weight_aux = wa
    weight_ecal = we
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
    return total_loss, loss_true_fake, loss_aux, loss_ecal
