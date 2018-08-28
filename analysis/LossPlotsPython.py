import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils.GANutils import safe_mkdir
plt.switch_backend('Agg')
try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
   #pkl file name and plots dir
   lossfile =  '/nfshome/gkhattak/3Dgan/3dgan-history-1loss-50weight.pkl'
   outdir = 'results/loss_plots_sqrt'

   # limits for plots. Adjust according to current plots
   ymax = [20, 5, 5, 40, 0.85, 1.5, 10.] #[combined loss, Gen train loss, Gen test loss, Aux training loss, lower limit for generator BCE only, upper limit for generator BCE, Disc. Losses]

   start_epoch =7 # removing initial epochs to check for overfitting of Generation loss
   
   num_ang_losses = 1 # number of angle losses

   safe_mkdir(outdir)
   plot_loss(lossfile, ymax, outdir, start_epoch, num_ang_losses)
   print('Loss Plots are saved in {}'.format(outdir))
   
def plot_loss(lossfile, ymax, lossdir, start_epoch, num_ang_losses, fig=1):
   #Getting losses in arrays
   ploss= 'Mean percentage error'
   aloss= 'Mean absolute error'
   bloss= 'Binary cross entropy'
   angtype = 'theta'
            
   if num_ang_losses==1:
      weights = [1, 3, 0.1, 50, 0.1]
      losstype = ['Weighted sum', bloss, ploss, aloss, ploss]
      lossnames = ['tot', 'gen', 'aux', '{}'.format(angtype), 'ecal sum']

   elif num_ang_losses==2:
      weights = [1, 3, 0.1, 10, 15, 0.1]
      losstype = ['Weighted sum', bloss, ploss, aloss, aloss, ploss]
      lossnames = ['tot', 'gen', 'aux', '{}1'.format(angtype), '{}2'.format(angtype), 'ecal sum']
                             
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
   plt.figure(fig)
   plt.title('{} losses for Generator'.format(losstype[1]))
   epochs = np.arange(start_epoch, gen_train.shape[0])
   plt.plot(epochs, gen_train[start_epoch:,1], label='Gen {} ({})'.format(lossnames[1], losstype[1]))
   order = 3
   fit = np.polyfit(epochs, gen_train[start_epoch:,1], order)
   plt.plot(epochs, np.polyval(fit, epochs), label='Gen fit ({} degree polynomial)'.format(order), linestyle='--')
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

   #Training losses for ang losses
   fig = fig + 1
   plt.figure(fig)
   plt.title('Training losses for Angles (loss weightes ={})'.format(weights[3:3 + num_ang_losses]))
   for i in np.arange(3, 3 + num_ang_losses):
      plt.plot(disc_train[:,i] * weights[i], label='Disc {} ({})'.format(lossnames[i], losstype[i]), color=color[i])
      plt.plot(gen_train[:,i] * weights[i], label='Gen {} ({})'.format(lossnames[i], losstype[i]), color=color[i], linestyle='--')
      
   plt.legend(fontsize='x-small')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   #plt.ylim(0, 0.2 * 15)
   plt.savefig(os.path.join(lossdir, 'ang_losses.pdf'))
                                              

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
