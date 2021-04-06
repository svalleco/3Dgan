# Plot losses using python
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
sys.path.insert(0,'../')
from analysis.utils.GANutils import safe_mkdir
import argparse
from matplotlib.pyplot import cm

try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
   parser = get_parser()
   params = parser.parse_args()
   
   historyfiles = params.historyfiles if isinstance(params.historyfiles, list) else [params.historyfiles]  #pkl file with loss history
   labels = params.labels if isinstance(params.labels, list) else [params.labels]  #pkl file with loss history
   outdir = params.outdir #dir for plots
   ylim = [params.ylim1, params.ylim2, params.ylim3, params.ylim4, params.ylim5, params.ylim6, params.ylim7] #limits for different plots
   start_epoch =params.start_epoch #starting from epoch
   end_epoch = params.end_epoch
   fit_order = params.fit_order #order of fit
   num_ang_losses = params.num_ang_losses # number of angle losses
   num_add_loss = params.num_add_loss # additional losses
   gen_weight = params.gen_weight # weight of GAN loss
   aux_weight = params.aux_weight # weight for auxilliary regression loss
   ecal_weight = params.ecal_weight # weight for sum of pixel intensities
   ang_weight = params.ang_weight # weight for angle loss
   add_weight = params.add_weight # weight for additional loss
   leg = params.leg # legend
   ang = params.ang # angle
   test = params.test # if test losses
   #Loss names 
   ploss= 'Mean percentage error'
   aloss= 'Mean absolute error'
   bloss= 'Binary cross entropy'
   # angle name
   angtype ='theta'
   if not gen_weight:
      gen_weight=3.0 if ang else 2.0
   if not aux_weight:
       aux_weight=0.1
   if not ecal_weight:
       ecal_weight=0.1
   if not ang_weight:
       ang_weight=25
   if not add_weight:
       add_weight=0.1

   if not ang:
       weights = [1, gen_weight, aux_weight, ecal_weight]
       losstypes = ['Weighted sum', bloss, ploss, ploss]
       lossnames = ['tot', 'gen', 'aux', 'ecal sum']
   else:
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
   plot_loss(historyfiles, labels, ylim, outdir, start_epoch, end_epoch, weights, losstypes, lossnames, num_ang_losses, order=fit_order, leg=leg, ang=ang, test=test)
   print('Loss Plots are saved in {}'.format(outdir))
   
def get_parser():
    parser = argparse.ArgumentParser(description='Loss plots' )
    parser.add_argument('--historyfiles', action='store', type=str, nargs='+', default='../results/3dgan_history.pkl', help='Pickle file for loss history')
    parser.add_argument('--labels', action='store', type=str, nargs='+', default='', help='labels for training')
    parser.add_argument('--outdir', action='store', type=str, default='results/loss_plots/', help='directory for results')
    parser.add_argument('--ylim1', type=float, default=30, help='y max for combined train loss')
    parser.add_argument('--ylim2', type=float, default=4, help='y max for BCE train loss')
    parser.add_argument('--ylim3', type=float, default=4, help='y max for BCE test loss')
    parser.add_argument('--ylim4', type=float, default=50, help='y max for Axilliary training losses')
    parser.add_argument('--ylim5', type=float, default=0.75, help='y min for Generator BCE only')
    parser.add_argument('--ylim6', type=float, default=2.25, help='y max for Generator BCE only')
    parser.add_argument('--ylim7', type=float, default=50., help='y max for combined test & train losses')
    parser.add_argument('--start_epoch', type=int, default=0, help='can be used to remove initial epochs')
    parser.add_argument('--end_epoch', type=int, default=60, help='can be used to remove later epochs')
    parser.add_argument('--fit_order', type=int, default=3, help='order of polynomial used for fit')
    parser.add_argument('--num_ang_losses', type=int, default=1, help='number of losses used for angle')
    parser.add_argument('--num_add_loss', type=int, default=1, help='number of additional losses')
    parser.add_argument('--gen_weight', type=float, help='weight of GAN loss')
    parser.add_argument('--aux_weight', type=float, help='weight of auxilliary energy regression loss')
    parser.add_argument('--ecal_weight', type=float, help='weight of ecal sum loss')
    parser.add_argument('--ang_weight', type=float, help='weight of angle loss')
    parser.add_argument('--add_weight', type=float, help='weight of bin count loss')
    parser.add_argument('--leg', type=int, default=1, help='draw legend') 
    parser.add_argument('--ang', type=int, default=1, help='if variable angle')
    parser.add_argument('--test', type=int, default=1, help='if test losses available')   
    return parser

def plot_loss(lossfiles, labels, ymax, lossdir, start_epoch, end_epoch, weights, losstype, lossnames, num_ang_losses, fig=1, order=3, leg=True, ang=1, test=1):

   for index, lossfile in enumerate(lossfiles):
     with open(lossfile, 'rb') as f:
           x = pickle.load(f)
     if index==0:
      gen_train = [np.asarray(x['train']['generator'])]
      disc_train = [np.asarray(x['train']['discriminator'])]
      if test:
        gen_test = [np.asarray(x['test']['generator'])]
        disc_test = [np.asarray(x['test']['discriminator'])]
     else:
        gen_train.append(np.asarray(x['train']['generator']))
        disc_train.append(np.asarray(x['train']['discriminator']))
        if test:
           gen_test.append(np.asarray(x['test']['generator']))
           disc_test.append(np.asarray(x['test']['discriminator']))

   #color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
   #        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
   #                      '#bcbd22', '#17becf']
   color= ['#1f77b4', '#ff7f0e', 'r', '#2ca02c', '#d62728',                                                                                                                                                    
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',                                                                                                                                                                 '#bcbd22', '#17becf']
   loop = np.arange(len(lossnames))
   index = len(lossfiles)
   #Plots for Testing and Training Losses
   for h in np.arange(index):
     plt.figure(fig)
     if test:
       plots = 221
     else:
       plots = 121
     plt.subplot(plots)
     plt.title('Discriminator loss')
     for i in loop:
        plt.plot(weights[i] * disc_train[h][:,i], label='{} (train)'.format(lossnames[i]))
     if leg: plt.legend(fontsize='x-small')                                                                   
     plt.ylim(0, ymax[6])                                                                                                                           
     plots+=1
     plt.subplot(plots)
     plt.title('Generator loss')
     for i in loop:
        plt.plot(weights[i] * gen_train[h][:,i], label='{} (train)'.format(lossnames[i]))
     if leg: plt.legend(fontsize='x-small')                                                                   
     plt.ylim(0, ymax[0])
     plots+=1                                                                                              
     if test:
       plt.subplot(plots)
   
       for i in loop:
          plt.plot(weights[i] * disc_test[h][:,i], label='{} (test)'.format(lossnames[i]))
       if leg: plt.legend(fontsize='x-small')
       plt.ylim(0, ymax[6])  
       plots+=1
       plt.subplot(plots)
  
       for i in loop:
         plt.plot(weights[i] * gen_test[h][:,i], label='{} (test)'.format(lossnames[i]))
       if leg: plt.legend(fontsize='x-small')
       plt.ylim(0, ymax[0])  
     plt.savefig(os.path.join(lossdir, labels[h]+'_losses.pdf')) 
     #Training losses
     fig = fig + 1
     plt.figure(fig)
     plt.title('Weighted Training losses: Loss weights = ({})'.format(weights))
     for i in loop:
       plt.plot(weights[i] * disc_train[h][:,i], label=labels[h] +'Disc {} ({})'.format(lossnames[i], losstype[i]), color=color[i])
       plt.plot(weights[i] * gen_train[h][:,i], label=labels[h] +'Gen {} ({})'.format(lossnames[i], losstype[i]), color=color[i], linestyle='--')
     if leg: plt.legend(fontsize='x-small')
     plt.xlabel('Epochs')  
     plt.ylabel('Loss')                            
     plt.ylim(0, ymax[0])  
     plt.savefig(os.path.join(lossdir, labels[h]+'_Combined_losses.pdf'))
     fig = fig + 1
     #training losses for Real/fake
   
     plt.figure(fig)
     plt.title('{} losses for GAN'.format(losstype[1]))
     plt.plot(gen_train[h][:,1], label=labels[h] +'Gen {} ({})'.format(lossnames[1], losstype[1]))
     plt.plot(disc_train[h][:,1], label=labels[h] +'Disc {} ({})'.format(lossnames[1], losstype[1]))
     if leg: plt.legend()
     plt.xlabel('Epochs')  
     plt.ylabel('Loss')                            
     plt.ylim(0, ymax[1])  
     plt.savefig(os.path.join(lossdir, labels[h] +'_BCE_train_losses.pdf'))

     #training losses for Real/fake Magnified
                                                                                                                                                              
     fig = fig + 1

   plt.figure(fig)
   plt.title('{} losses for Generator'.format(losstype[1]))
   epochs = np.arange(start_epoch, end_epoch)
   min_loss = []
   
   for h in np.arange(index):
     loss = np.zeros((end_epoch))
     loss[:gen_test[h][:,1].shape[0]] = gen_test[h][:end_epoch,1]
     min_loss.append( np.amin(gen_test[h][start_epoch:,1]))
     plt.plot(epochs, loss[start_epoch:end_epoch], label=labels[h] +'Gen {} ({}) min={:.4f}'.format(lossnames[1], losstype[1], min_loss[h]))
   if leg: plt.legend()
   plt.xlabel('Epochs starting from epoch' + str(start_epoch))
   plt.ylabel('Loss')
   plt.ylim(ymax[4], ymax[5])
   plt.savefig(os.path.join(lossdir, 'BCE_test_gen_losses.pdf'))


   #training losses for Real/fake Magnified
   fig = fig + 1
   plt.figure(fig)
   plt.title('{} losses for Generator'.format(losstype[1]))
   epochs = np.arange(start_epoch, end_epoch)
   min_loss = []
   for h in np.arange(index):
     loss = np.zeros((end_epoch))
     loss[:gen_train[h][:,1].shape[0]] = gen_train[h][:end_epoch,1]
     min_loss.append( np.amin(gen_train[h][start_epoch:,1]))
     plt.plot(epochs, loss[start_epoch:end_epoch], label=labels[h] +'Gen {} ({}) min={:.4f}'.format(lossnames[1], losstype[1], min_loss[h]))
     #fit = np.polyfit(epochs, gen_train[start_epoch:,1], order)
     #plt.plot(epochs, np.polyval(fit, epochs), label='Gen fit ({} degree polynomial)'.format(order), linestyle='--')
   if leg: plt.legend()
   plt.xlabel('Epochs starting from epoch' + str(start_epoch))  
   plt.ylabel('Loss')                            
   plt.ylim(ymax[4], ymax[5])  
   plt.savefig(os.path.join(lossdir, 'BCE_train_gen_losses.pdf'))

   #training losses for ecal loss                                                                                                                                                              
   fig = fig + 1
   plt.figure(fig)
   plt.title('{} losses for Generator'.format(lossnames[-1]))
   epochs = np.arange(start_epoch, end_epoch)
   min_loss = []
   for h in np.arange(index):
     loss = np.zeros((end_epoch))
     loss[:gen_train[h][:,-1].shape[0]] = gen_train[h][:end_epoch,-1]
     min_loss.append( np.amin(gen_train[h][start_epoch:end_epoch,-1]))
     plt.plot(epochs, loss[start_epoch:end_epoch], label=labels[h] +'Gen {} ({}) min={:.4f}'.format(lossnames[-1], losstype[-1], min_loss[h]))
   if leg: plt.legend()
   plt.xlabel('Epochs starting from epoch' + str(start_epoch))
   plt.ylabel('Loss')
   plt.ylim(0, 30)
   plt.savefig(os.path.join(lossdir, 'ecal_train_gen_losses.pdf'))

   #training losses for ecal loss
   fig = fig + 1
   plt.figure(fig)
   plt.title('{} losses for Generator'.format(lossnames[-1]))
   epochs = np.arange(start_epoch, end_epoch)
   min_loss = []
   for h in np.arange(index):
     loss = np.zeros((end_epoch))
     loss[:gen_test[h][:,-1].shape[0]] = gen_test[h][:end_epoch,-1]
     min_loss.append( np.amin(gen_test[h][start_epoch:,-1]))
     plt.plot(epochs, loss[start_epoch:end_epoch], label=labels[h] +'Gen {} ({}) min={:.4f}'.format(lossnames[-1], losstype[-1], min_loss[h]))
   if leg: plt.legend()
   plt.xlabel('Epochs starting from epoch' + str(start_epoch))
   plt.ylabel('Loss')
   plt.ylim(0, 30)
   plt.savefig(os.path.join(lossdir, 'ecal_test_gen_losses.pdf'))

   #testing losses for Real/fake
   if test:
     for h in np.arange(index):
       fig = fig + 1
       plt.figure(fig)
       plt.title('{} Testing losses for GAN'.format(losstype[1]))
       epochs = np.arange(start_epoch, min(end_epoch, gen_test[h][:,1].shape[0]))
       plt.plot(epochs, gen_test[h][:,1][start_epoch:end_epoch], label=labels[h] +'Gen {} ({})'.format(lossnames[1], losstype[1]))
       plt.plot(epochs, disc_test[h][:,1][start_epoch:end_epoch], label=labels[h] +'Disc {} ({})'.format(lossnames[1], losstype[1]))
       if leg: plt.legend()
       plt.xlabel('Epochs')  
       plt.ylabel('Loss')                            
       plt.ylim(0, ymax[2])  
       plt.savefig(os.path.join(lossdir, labels[h]+'_BCE_test_losses.pdf'))

     fig = fig + 1
     plt.figure(fig)
     plt.title('{} Testing losses for GAN'.format(losstype[1]))
     epochs = np.arange(start_epoch, end_epoch)
     #colors=iter(cm.rainbow(np.linspace(0,1,index)))
     for h in np.arange(index):
       epochs = np.arange(start_epoch, min(end_epoch, gen_test[h][:,1].shape[0]))
       plt.plot(epochs, gen_test[h][:,1][start_epoch:end_epoch], color= color[h], label=labels[h] +'Gen {} {} ({})'.format(labels[h], lossnames[1], losstype[1]))
       plt.plot(epochs, disc_test[h][:,1][start_epoch:end_epoch], color= color[h], linestyle='dashed', label=labels[h] +'Disc {} {} ({})'.format(labels[h], lossnames[1], losstype[1]))
     if leg: plt.legend()
     plt.xlabel('Epochs')
     plt.ylabel('Loss')
     plt.ylim(0, ymax[2])
     plt.savefig(os.path.join(lossdir, 'BCE_test_losses_multi.pdf'))

   for h in np.arange(index):
     #Training losses for auxlilliary losses
     fig = fig + 1
     plt.figure(fig)
     plt.title('Training losses for Auxilliary outputs (unweighted)')
     for i in np.arange(2, len(lossnames)):
        plt.plot(disc_train[h][:,i], label=labels[h] +'Disc {} ({})'.format(lossnames[i], losstype[i]), color=color[i])
        plt.plot(gen_train[h][:,i], label=labels[h] +'Gen {} ({})'.format(lossnames[i], losstype[i]), color=color[i], linestyle='--')
     if leg: plt.legend(fontsize='x-small')
     plt.xlabel('Epochs')  
     plt.ylabel('Loss')                            
     plt.ylim(0, ymax[3])  
     plt.savefig(os.path.join(lossdir, labels[h] +'_aux_training_losses.pdf'))

   if ang:
     #Training losses for ang losses
     for h in np.arange(index):
       fig = fig + 1
       plt.figure(fig)
       plt.title('Training losses for Angles (loss weightes ={})'.format(weights[3:3 + num_ang_losses]))
       for i in np.arange(3, 3 + num_ang_losses):
          plt.plot(disc_train[h][:,i] * weights[i], label=labels[h] +'Disc {} ({})'.format(lossnames[i], losstype[i]), color=color[i])
          plt.plot(gen_train[h][:,i] * weights[i], label=labels[h] +'Gen {} ({})'.format(lossnames[i], losstype[i]), color=color[i], linestyle='--')
      
       if leg: plt.legend(fontsize='x-small')
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       #plt.ylim(0, 0.2 * 15)
       plt.savefig(os.path.join(lossdir, 'ang_losses.pdf'))
   for h in np.arange(index):                                           
     #Diff. for training losses Real/fake
     fig = fig + 1
     plt.figure(fig)
     plt.title('Diff between Genenerator loss and Discriminator loss for GAN')
     plt.plot(gen_train[h][:,1] - disc_train[h][:,1], label='Gen loss - Disc loss ({})'.format(losstype[1]))
     if leg: plt.legend()
     plt.xlabel('Epochs')  
     plt.ylabel('Loss')                            
     plt.ylim(0, ymax[1])  
     plt.savefig(os.path.join(lossdir, labels[h] +'_BCE_train_losses_diff.pdf'))

if __name__ == "__main__":
   main()
