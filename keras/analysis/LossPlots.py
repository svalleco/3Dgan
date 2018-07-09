import os
import numpy as np
import matplotlib.pyplot as plt
from GANutils import safe_mkdir
plt.switch_backend('Agg')
try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
   lossfile = 'dcgan-ch-pion-history.pkl'
   weights = [2, 0.1, 0.1]
   #defining limits for different plots. Varies with result
   ymax = [30, 4, 3, 100, 1.25, 2.] #[weighted loss for all, BCE_test loss, BCE_train_loss, Aux unweighted, BCE-train for generator only]
   outdir = 'loss_plots_ch_pions_1000'
   safe_mkdir(outdir)
   plot_loss(lossfile, weights, ymax, outdir)
   print('The plots are saved in {}'.format(outdir))

def plot_loss(lossfile, weights, ymax, lossdir, fig=1):
   #Getting losses in arrays
   with open(lossfile, 'rb') as f:
    			x = pickle.load(f)
   gen_test = np.asarray(x['test']['generator'])
   gen_train = np.asarray(x['train']['generator'])
   disc_test = np.asarray(x['test']['discriminator'])
   disc_train = np.asarray(x['train']['discriminator'])
   
   #Plots for Testing and Training Losses
   plt.figure(fig)
   plt.subplot(221)
   plt.title('Disc. Weighted Train loss')
   plt.plot(disc_train[:,0], label='tot')
   plt.plot(weights[0] * disc_train[:,1], label='gen')
   plt.plot(weights[1] * disc_train[:,2], label='aux')
   plt.plot(weights[2] * disc_train[:,3], label='ecal')
   plt.legend()                                                                   
   plt.ylim(0, ymax[0])                                                                                                                           

   plt.subplot(222)
   plt.title('Gen. Weighted Train loss')
   plt.plot(gen_train[:,0], label='tot')
   plt.plot(weights[0] * gen_train[:,1], label='gen')
   plt.plot(weights[1] * gen_train[:,2], label='aux')
   plt.plot(weights[2] * gen_train[:,3], label='ecal')
   plt.legend()                                                                   
   plt.ylim(0, ymax[0])                                                                                              

   plt.subplot(223)
   plt.title('Testing loss for Discriminator')
   plt.plot(disc_test[:,0], label='tot')
   plt.plot(weights[0] * disc_test[:,1], label='gen')
   plt.plot(weights[1] * disc_test[:,2], label='aux')
   plt.plot(weights[2] * disc_test[:,3], label='ecal')
   plt.legend()
   plt.ylim(0, ymax[0])  
    
   plt.subplot(224)
   plt.title('\nTesting loss for Generator')
   plt.plot(gen_test[:,0], label='tot')
   plt.plot(weights[0] * gen_test[:,1], label='gen')
   plt.plot(weights[1] * gen_test[:,2], label='aux')
   plt.plot(weights[2] *disc_test[:,3], label='ecal')
   plt.legend()
   plt.ylim(0, ymax[0])  
   plt.savefig(os.path.join(lossdir,'losses.pdf')) 
   
   #Training losses
   fig = fig + 1
   plt.figure(fig)
   plt.title('Weighted Training losses for GAN: Loss weights = (%0.2f, %.2f, %.2f)'%(weights[0], weights[1], weights[2]))
   plt.plot(disc_train[:,0], label='Disc tot', color='red')
   plt.plot(weights[0] * disc_train[:,1], label='Disc gen (Binary Cross Entropy)', color='green')
   plt.plot(weights[1] * disc_train[:,2], label='Disc aux (Mean Absolute Percentage Error)', color='blue')
   plt.plot(weights[2] * disc_train[:,3], label='Disc ecal(Mean Absolute Percentage Error)', color='magenta')
   plt.plot(gen_train[:,0], label='Gen tot', color='red', linestyle='--')
   plt.plot(weights[0] * gen_train[:,1], label='Gen gen (Binary Cross Entropy)', color='green', linestyle='--')
   plt.plot(weights[1] * gen_train[:,2], label='Gen aux (Mean Absolute Percentage Error)', color='blue', linestyle='--')
   plt.plot(weights[2] * gen_train[:,3], label='Gen ecal(Mean Absolute Percentage Error)', color='magenta', linestyle='--')
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[0])  
   plt.savefig(os.path.join(lossdir, 'Combined_losses.pdf'))

   #training losses for Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('Binary Training losses for GAN')
   plt.plot(disc_train[:,1], label='Disc gen (Binary Cross Entropy)', color='green')
   plt.plot(gen_train[:,1], label='Gen gen (Binary Cross Entropy)', color='blue')
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_losses.pdf'))

   #training losses for Real/fake Magnified
   fig = fig + 1
   start_epoch = 10
   plt.figure(fig)
   plt.title('Binary Training losses for Generator')
   #plt.plot(disc_train[:,1], label='Disc gen (Binary Cross Entropy)', color='green')
   plt.plot(gen_train[start_epoch:,1], label='Gen gen (Binary Cross Entropy)', color='blue')
   plt.legend()
   plt.xlabel('Epochs starting from epoch' + str(start_epoch))  
   plt.ylabel('Loss')                            
   plt.ylim(ymax[4], ymax[5])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_gen_losses.pdf'))

   #testing losses for Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('Binary Testing losses for GAN')
   plt.plot(disc_test[:,1], label='Disc gen (Binary Cross Entropy)', color='green')
   plt.plot(gen_test[:,1], label='Gen gen (Binary Cross Entropy)', color='blue')
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[2])  
   plt.savefig(os.path.join(lossdir, 'BCE_test_losses.pdf'))

   #Training losses for auxlilliary losses
   fig = fig + 1
   plt.figure(fig)
   plt.title('Training losses for Auxilliary outputs')
   plt.plot(disc_train[:,2], label='Disc aux (Mean Absolute Percentage Error)', color='blue')
   plt.plot(disc_train[:,3], label='Disc ecal(Mean Absolute Percentage Error)', color='magenta')
   plt.plot(gen_train[:,2], label='Gen aux (Mean Absolute Percentage Error)', color='blue', linestyle='--')
   plt.plot(gen_train[:,3], label='Gen ecal(Mean Absolute Percentage Error)', color='magenta', linestyle='--')
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[3])  
   plt.savefig(os.path.join(lossdir, 'aux_training_losses.pdf'))

   #Diff. for training losses Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('Binary Training losses for GAN')
   plt.plot(gen_train[:,1] - disc_train[:,1], label='Gen loss - Disc loss (Binary Cross Entropy)')
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_losses_diff.pdf'))

if __name__ == "__main__":
   main()
