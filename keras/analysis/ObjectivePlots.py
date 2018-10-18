import os
import numpy as np
import matplotlib.pyplot as plt
from utils.GANutils import safe_mkdir
plt.switch_backend('Agg')
try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
   infile = '3dgan_analysis.pkl'
   #defining limits for different plots. Varies with result
   ymax = [30, 4, 3, 100, 1.25, 2.] #[weighted loss for all, BCE_test loss, BCE_train_loss, Aux unweighted, BCE-train for generator only]
   plot_obj(infile)
   print('The plots are saved')

def plot_loss(infile, fig=1):
   #Getting losses in arrays
   with open(infile, 'rb') as f:
    			x = pickle.load(f)
   total = np.asarray(x['total'])
   energy = np.asarray(x['energy'])
   moments = np.asarray(x['momentum'])
   
   #Plots for Testing and Training Losses
   plt.figure(fig)
   #plt.subplot(221)
   plt.title('Objective function')
   plt.plot(total[:], label='tot')
   plt.plot(energy[:], label='energy')
   plt.plot(moments[:], label='moments')
   plt.legend()                                                                   
   #plt.ylim(0, ymax[0])                                                                                                                           
   plt.savefig('objective.pdf') 
   

if __name__ == "__main__":
   main()
