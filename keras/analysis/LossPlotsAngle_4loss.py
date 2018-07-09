import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from GANutilsANG import safe_mkdir
plt.switch_backend('Agg')
try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
   lossfile = 'dcgan-history-angle-concat-2loss.pkl'
   weights = [1, 2, 0.1, 10, 15, 0.1]
   #defining limits for different plots. Varies with result
   ymax = [25, 5, 5, 40, 1.8, 2.4, 20] #[combined loss, Gen train loss, Gen test loss, Aux training loss, lower limit for generator BCE only, upper limit for generator BCE] 
   outdir = 'loss_plots_2ang'
   ploss= 'Mean percentage error'
   aloss= 'Mean absolute error'
   bloss= 'Binary cross entropy'
   losstypes = ['Weighted sum', bloss, ploss, aloss, aloss, ploss]
   angtype = 'theta'
   lossnames = ['tot', 'gen', 'aux', '{}1'.format(angtype), '{}2'.format(angtype), 'ecal sum']

   safe_mkdir(outdir)
   plot_loss(lossfile, weights, ymax, outdir, lossnames, losstypes)
   print('Loss Plots are saved in {}'.format(outdir))
   
def plot_loss(lossfile, weights, ymax, lossdir, lossnames, losstype, fig=1):
   #Getting losses in arrays
   with open(lossfile, 'rb') as f:
    			x = pickle.load(f)
   gen_test = np.asarray(x['test']['generator'])
   gen_train = np.asarray(x['train']['generator'])
   disc_test = np.asarray(x['test']['discriminator'])
   disc_train = np.asarray(x['train']['discriminator'])
   color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                         '#bcbd22', '#17becf']
   loop = np.arange(len(lossnames))
   #Plots for Testing and Training Losses
   plt.figure(fig)
   plt.subplot(221)
   plt.title('Discriminator loss')
   for i in loop:
      plt.plot(weights[i] * disc_train[:,i], label='{} (train)'.format(lossnames[i]))
   plt.legend(fontsize='x-small')                                                                   
   plt.ylim(0, ymax[6])                                                                                                                           

   plt.subplot(222)
   plt.title('Generator loss')
   for i in loop:
      plt.plot(weights[i] * gen_train[:,i], label='{} (train)'.format(lossnames[i]))
   plt.legend(fontsize='x-small')                                                                   
   plt.ylim(0, ymax[0])                                                                                              

   plt.subplot(223)
   #plt.title('Testing loss for Discriminator')
   for i in loop:
      plt.plot(weights[i] * disc_test[:,i], label='{} (test)'.format(lossnames[i]))
   plt.legend(fontsize='x-small')
   plt.ylim(0, ymax[6])  
    
   plt.subplot(224)
  # plt.title('\nTesting loss for Generator')
   for i in loop:
      plt.plot(weights[i] * gen_test[:,i], label='{} (test)'.format(lossnames[i]))
   plt.legend(fontsize='x-small')
   plt.ylim(0, ymax[0])  
   plt.savefig(os.path.join(lossdir,'losses.pdf')) 
   
   #Training losses
   fig = fig + 1
   plt.figure(fig)
   plt.title('Weighted Training losses: Loss weights = (%0.2f, %.2f, %.2f,  %.2f)'%(weights[1], weights[2], weights[3],  weights[4]))
   for i in loop:
       plt.plot(weights[i] * disc_train[:,i], label='Disc {} ({})'.format(lossnames[i], losstype[i]), color=color[i])
       plt.plot(weights[i] * gen_train[:,i], label='Gen {} ({})'.format(lossnames[i], losstype[i]), color=color[i], linestyle='--')
   plt.legend(fontsize='x-small')
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[0])  
   plt.savefig(os.path.join(lossdir, 'Combined_losses.pdf'))

   #training losses for Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('{} losses for GAN'.format(losstype[1]))
   plt.plot(gen_train[:,1], label='Gen {} ({})'.format(lossnames[1], losstype[1]))
   plt.plot(disc_train[:,1], label='Disc {} ({})'.format(lossnames[1], losstype[1]))
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_losses.pdf'))

   #training losses for Real/fake Magnified
   fig = fig + 1
   start_epoch = 10
   plt.figure(fig)
   plt.title('{} losses for Generator'.format(losstype[1]))
   plt.plot(gen_train[start_epoch:,1], label='Gen {} ({})'.format(lossnames[1], losstype[1]))
   plt.legend()
   plt.xlabel('Epochs starting from epoch' + str(start_epoch))  
   plt.ylabel('Loss')                            
   plt.ylim(ymax[4], ymax[5])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_gen_losses.pdf'))

   #testing losses for Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('{} Testing losses for GAN'.format(losstype[1]))
   plt.plot(gen_test[:,1], label='Gen {} ({})'.format(lossnames[1], losstype[1]))
   plt.plot(disc_test[:,1], label='Disc {} ({})'.format(lossnames[1], losstype[1]))
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[2])  
   plt.savefig(os.path.join(lossdir, 'BCE_test_losses.pdf'))

   #Training losses for auxlilliary losses
   fig = fig + 1
   plt.figure(fig)
   plt.title('Training losses for Auxilliary outputs (unweighted)')
   for i in np.arange(2, len(lossnames)):
      plt.plot(disc_train[:,i], label='Disc {} ({})'.format(lossnames[i], losstype[i]), color=color[i])
      plt.plot(gen_train[:,i], label='Gen {} ({})'.format(lossnames[i], losstype[i]), color=color[i], linestyle='--')
   plt.legend(fontsize='x-small')
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[3])  
   plt.savefig(os.path.join(lossdir, 'aux_training_losses.pdf'))

   #Diff. for training losses Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('Diff between Genenerator loss and Discriminator loss for GAN')
   plt.plot(gen_train[:,1] - disc_train[:,1], label='Gen loss - Disc loss ({})'.format(losstype[1]))
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_losses_diff.pdf'))

if __name__ == "__main__":
   main()
