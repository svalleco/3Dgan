import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils.GANutils import safe_mkdir
import argparse
if '.cern.ch' in os.environ.get('HOSTNAME'): # Here a check for host can be used to set defaults accordingly
    tlab = True
else:
    tlab= False

try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
   parser = get_parser()
   params = parser.parse_args()
   if tlab:
      historyfile = '/gkhattak/results/' + params.historyfile
   else:
      historyfile = '../results/' + params.historyfile
   outdir = 'results/' + params.outdir
   ylim = [params.ylim1, params.ylim2, params.ylim3, params.ylim4, params.ylim5, params.ylim6, params.ylim7]
   start_epoch =params.start_epoch
   fit_order = params.fit_order
   num_ang_losses = params.num_ang_losses
   num_add_loss = params.num_add_loss
   gen_weight = params.gen_weight
   aux_weight = params.aux_weight
   ecal_weight = params.ecal_weight
   ang_weight = params.ang_weight
   add_weight = params.add_weight
   leg = params.leg
   #Loss names 
   ploss= 'Mean percentage error'
   aloss= 'Mean absolute error'
   bloss= 'Binary cross entropy'
   # angle name
   angtype = 'theta'
   
   if num_ang_losses==1:
       weights = [1, gen_weight, aux_weight, ang_weight, ecal_weight]
       losstypes = ['Weighted sum', bloss, ploss, aloss, ploss]
       lossnames = ['tot', 'gen', 'aux', '{}'.format(angtype), 'ecal sum']
      
   elif num_ang_losses==2:
       weights = [1, gen_weight, aux_weight, ang_weight, ecal_weight]
       losstypes = ['Weighted sum', bloss, ploss, aloss, aloss, ploss]
       lossnames = ['tot', 'gen', 'aux', '{}1'.format(angtype), '{}2'.format(angtype), 'ecal sum']
   if num_add_loss==1:
       weights.append(add_weight)
       losstypes.append(ploss)
       lossnames.append('bin')

   safe_mkdir(outdir)
   plot_loss(historyfile, ylim, outdir, start_epoch, weights, losstypes, lossnames, num_ang_losses, order=fit_order, leg=leg)
   print('Loss Plots are saved in {}'.format(outdir))
   
def get_parser():
    parser = argparse.ArgumentParser(description='Loss plots' )
    parser.add_argument('--historyfile', action='store', type=str, default='3dgan_history_gan_training_2D.pkl', help='Pickle file for loss history')
    parser.add_argument('--outdir', action='store', type=str, default='loss_plots_gan_training_2D', help='directory for results')
    parser.add_argument('--ylim1', type=float, default=20, help='y max for combined train loss')
    parser.add_argument('--ylim2', type=float, default=2, help='y max for BCE train loss')
    parser.add_argument('--ylim3', type=float, default=2, help='y max for BCE test loss')
    parser.add_argument('--ylim4', type=float, default=40, help='y max for Axilliary training losses')
    parser.add_argument('--ylim5', type=float, default=0.85, help='y min for Generator BCE only')
    parser.add_argument('--ylim6', type=float, default=0.95, help='y max for Generator BCE only')
    parser.add_argument('--ylim7', type=float, default=10., help='y max for combined test & train losses')
    parser.add_argument('--start_epoch', type=int, default=0, help='can be used to remove initial epochs')
    parser.add_argument('--fit_order', type=int, default=3, help='order of polynomial used for fit')
    parser.add_argument('--num_ang_losses', type=int, default=1, help='number of losses used for angle')
    parser.add_argument('--num_add_loss', type=int, default=1, help='number of additional losses')
    parser.add_argument('--gen_weight', type=float, default=3, help='weight of GAN loss')
    parser.add_argument('--aux_weight', type=float, default=0.1, help='weight of auxilliary energy regression loss')
    parser.add_argument('--ecal_weight', type=float, default=0.1, help='weight of ecal sum loss')
    parser.add_argument('--ang_weight', type=float, default=25, help='weight of angle loss')
    parser.add_argument('--add_weight', type=float, default=0.1, help='weight of bin count loss')
    parser.add_argument('--leg', default=True, help='draw legend')    
    return parser

def plot_loss(lossfile, ymax, lossdir, start_epoch, weights, losstype, lossnames, num_ang_losses, fig=1, order=3, leg=True):
                             
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
   if leg: plt.legend(fontsize='x-small')                                                                   
   plt.ylim(0, ymax[6])                                                                                                                           

   plt.subplot(222)
   plt.title('Generator loss')
   for i in loop:
      plt.plot(weights[i] * gen_train[:,i], label='{} (train)'.format(lossnames[i]))
   if leg: plt.legend(fontsize='x-small')                                                                   
   plt.ylim(0, ymax[0])                                                                                              

   plt.subplot(223)
   #plt.title('Testing loss for Discriminator')
   for i in loop:
      plt.plot(weights[i] * disc_test[:,i], label='{} (test)'.format(lossnames[i]))
   if leg: plt.legend(fontsize='x-small')
   plt.ylim(0, ymax[6])  
    
   plt.subplot(224)
  # plt.title('\nTesting loss for Generator')
   for i in loop:
      plt.plot(weights[i] * gen_test[:,i], label='{} (test)'.format(lossnames[i]))
   if leg: plt.legend(fontsize='x-small')
   plt.ylim(0, ymax[0])  
   plt.savefig(os.path.join(lossdir,'losses.pdf')) 
   
   #Training losses
   fig = fig + 1
   plt.figure(fig)
   plt.title('Weighted Training losses: Loss weights = (%0.2f, %.2f, %.2f,  %.2f)'%(weights[1], weights[2], weights[3],  weights[4]))
   for i in loop:
       plt.plot(weights[i] * disc_train[:,i], label='Disc {} ({})'.format(lossnames[i], losstype[i]), color=color[i])
       plt.plot(weights[i] * gen_train[:,i], label='Gen {} ({})'.format(lossnames[i], losstype[i]), color=color[i], linestyle='--')
   if leg: plt.legend(fontsize='x-small')
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
   if leg: plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_losses.pdf'))

   #training losses for Real/fake Magnified
   fig = fig + 1
   plt.figure(fig)
   plt.title('{} losses for Generator'.format(losstype[1]))
   epochs = np.arange(start_epoch, gen_train.shape[0])
   min_loss = np.amin(gen_train[start_epoch:,1])
   plt.plot(epochs, gen_train[start_epoch:,1], label='Gen {} ({}) min={:.4f}'.format(lossnames[1], losstype[1], min_loss))
   fit = np.polyfit(epochs, gen_train[start_epoch:,1], order)
   plt.plot(epochs, np.polyval(fit, epochs), label='Gen fit ({} degree polynomial)'.format(order), linestyle='--')
   if leg: plt.legend()
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
   if leg: plt.legend()
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
   if leg: plt.legend(fontsize='x-small')
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
      
   if leg: plt.legend(fontsize='x-small')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   #plt.ylim(0, 0.2 * 15)
   plt.savefig(os.path.join(lossdir, 'ang_losses.pdf'))
                                              

   #Diff. for training losses Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('Diff between Genenerator loss and Discriminator loss for GAN')
   plt.plot(gen_train[:,1] - disc_train[:,1], label='Gen loss - Disc loss ({})'.format(losstype[1]))
   if leg: plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Loss')                            
   plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_losses_diff.pdf'))

if __name__ == "__main__":
   main()
