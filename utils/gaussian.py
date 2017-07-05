# This program generates a dataset of guassian(only along x-axis) with some noise  
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg') 
from matplotlib import cm 
import numpy as np
from scipy.stats import norm
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D
import math
mean1 = 10
sigma1 = 2
nevents =5 # events of one class. Total will be double
mean2 = 10
sigma2 = 4

# Normalized guassian functions. Both give same value 
#def gaussian(x, mu, sig):
#    return 1./(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/sigma, 2.)/2.)

x = np.arange(0, 25, 1)
y = np.arange(0, 25, 1)
z = np.arange(0, 25, 1)

Z = np.zeros(shape=(nevents*2, 25, 25, 25))
T = np.zeros(shape=(nevents*2,1))
index = 0
for i in range(nevents):
    noise = np.random.normal(0, 0.01, 25)
    B = gaussian(x, mean1, sigma1)+ noise
    B = 80 * B
    Z[index,:,12,12]= B
    fig=plt.figure(index)
    fig=plt.plot(x, B)
    plt.title("Training data class1 along x_axis")
    plt.xlabel("Cell location along x_axis")
    plt.ylabel("Value")
    plt.savefig("figs/demo" + str(index) +".png")
    noise = np.random.normal(0, 0.01, 25)
    B = gaussian(x, mean2, sigma2)+ noise
    B = 80 * B
    index= index + 1
    T[index]= 1
    Z[index,:,12,12]= B
    fig=plt.figure(index)
    fig=plt.plot(x, B)                                                             
    plt.title("Training data class 2 along x_axis") 
    plt.xlabel("Cell location along x_axis")
    plt.ylabel("Value")
    plt.savefig("figs/demo" + str(index) +".png")
    index= index+1
print "mean of 1 = %f" %np.mean(Z[:,:,12,12])
print "sigma of 1= %f" %np.std(Z[:,:,12,12])

print(Z.shape)
print(T.shape)
   
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('ECAL', data=Z, dtype='f8')
h5f.create_dataset('TAG', data=T, dtype='i8')
h5f.close()
#fig = plt.figure(nevents1*2)
#ax = fig.add_subplot(111, projection='3d')
#a, b, c, d = Z.nonzero()
#sp = ax.scatter(b, c, d, c=Z[0,b,c,d])
#plt.colorbar(sp)
#plt.savefig("figs/3d2.png")
