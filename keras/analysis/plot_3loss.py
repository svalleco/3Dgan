import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

try:
    import cPickle as pickle
except ImportError:
    import pickle
#with open('weights.pkl', 'rb') as f:
with open('dcgan-history.pkl', 'rb') as f:
    x = pickle.load(f)
gen_test = np.asarray(x['test']['generator'])
gen_train = np.asarray(x['train']['generator'])
disc_test = np.asarray(x['test']['discriminator'])
disc_train = np.asarray(x['train']['discriminator'])
gen_weight=10
ecal_weight=0.1
aux_weight=0.1

plt.figure(1)
plt.subplot(221)
plt.title('Training loss for Discriminator')
plt.plot(disc_train[:,0], label='tot')
plt.plot(disc_train[:,1], label='gen')
plt.plot(disc_train[:,2], label='aux')
plt.plot(disc_train[:,3], label='ecal')
plt.legend()                                                                   
plt.ylim(0, 100)                                                                                                                           

plt.subplot(222)
plt.title('\nTraining loss for Generator')
plt.plot(gen_train[:,0], label='tot')
plt.plot(gen_train[:,1], label='gen')
plt.plot(gen_train[:,2], label='aux')
plt.plot(gen_train[:,3], label='ecal')
plt.legend()                                                                   
plt.ylim(0, 100)                                                                                              

plt.subplot(223)
plt.title('\nTesting loss for Discriminator')
plt.plot(disc_test[:,0], label='tot')
plt.plot(disc_test[:,1], label='gen')
plt.plot(disc_test[:,2], label='aux')
plt.plot(disc_test[:,3], label='ecal')
plt.legend()
plt.ylim(0, 100)  
    
plt.subplot(224)
plt.title('\nTesting loss for Generator')
plt.plot(gen_test[:,0], label='tot')
plt.plot(gen_test[:,1], label='gen')
plt.plot(gen_test[:,2], label='aux')
plt.plot(gen_test[:,3], label='ecal')
plt.legend()
plt.ylim(0, 100)  
plt.savefig('losses.pdf') 

plt.figure(2)
plt.title('Training losses for GAN: Loss weights = (%0.2f, %.2f, %.2f)'%(gen_weight, aux_weight, ecal_weight))
plt.plot(disc_train[:,0], label='Disc tot', color='red')
plt.plot(disc_train[:,1], label='Disc gen (Binary Cross Entropy)', color='green')
plt.plot(disc_train[:,2], label='Disc aux (Mean Absolute Percentage Error)', color='blue')
plt.plot(disc_train[:,3], label='Disc ecal(Mean Absolute Percentage Error)', color='magenta')
plt.plot(gen_train[:,0], label='Gen tot', color='red', linestyle='--')
plt.plot(gen_train[:,1], label='Gen gen (Binary Cross Entropy)', color='green', linestyle='--')
plt.plot(gen_train[:,2], label='Gen aux (Mean Absolute Percentage Error)', color='blue', linestyle='--')
plt.plot(gen_train[:,3], label='Gen ecal(Mean Absolute Percentage Error)', color='magenta', linestyle='--')
plt.legend()
plt.xlabel('Epochs')  
plt.ylabel('Loss')                            
plt.ylim(0, 100)  
plt.savefig('Combined_losses.pdf')
