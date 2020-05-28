import os
import numpy as np
import matplotlib.pyplot as plt
from LCDutils import safe_mkdir
plt.switch_backend('Agg')
try:
    import cPickle as pickle
except ImportError:
    import pickle

def main():
   lossfile = 'dcgan-pion-history2.pkl'
   resultfile = 'pions2p1p2scale500_4result.txt'
   outdir = 'diff_opt_plots'
   safe_mkdir(outdir)
   plot_loss(lossfile, resultfile, outdir)

def plot_loss(lossfile, resultfile, outdir, fig=1):
   filename = 'lossdiffVSopt_'
   scatterfile= filename + 'scatter.pdf'
   plotfile = filename + 'plot.pdf'
   #Getting losses in arrays
   with open(lossfile, 'rb') as f:
    			x = pickle.load(f)
   gen_test = np.asarray(x['test']['generator'])
   gen_train = np.asarray(x['train']['generator'])
   disc_test = np.asarray(x['test']['discriminator'])
   disc_train = np.asarray(x['train']['discriminator'])
   epochs = np.arange(len(x['train']['generator']))
   total, energy, position=np.loadtxt(resultfile, unpack=True)
   print total.shape
   #Diff. for training losses Real/fake
   fig = fig + 1
   plt.figure(fig)
   plt.title('Binary Training loss Difference vs. Optimization Function ')
   plt.scatter(gen_train[:,1] - disc_train[:,1], total, label='Gen loss - Disc loss (Binary Cross Entropy)')
   plt.legend()
   plt.xlabel('Loss Diff')  
   plt.ylabel('Opt. Metric')                            
   #plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(outdir, scatterfile))
   
   fig = fig + 1
   plt.figure(fig)
   plt.title('Binary Training loss Difference and Optimization Function vs. epochs')
   plt.plot(gen_train[:,1] - disc_train[:,1], label='Gen loss - Disc loss')
   plt.plot(total, label='Optimization metric')
   plt.legend()
   plt.xlabel('Epochs')  
   plt.ylabel('Value')                            
   #plt.ylim(0, ymax[1])  
   plt.savefig(os.path.join(outdir, plotfile))

if __name__ == "__main__":
   main()
