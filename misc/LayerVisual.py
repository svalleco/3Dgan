import h5py
import numpy as np
import setGPU
import math
import sys
import os
sys.path.insert(0,'/nfshome/gkhattak/3Dgan')
sys.path.insert(0,'/nfshome/gkhattak/3Dgan/analysis')
import utils.GANutils as gan
import utils.ROOTutils as my
import keras
import keras.backend as K
import ROOT

def main():
   #Architecture
   from AngleArch3dGAN_sqrt_vis import generator, discriminator

   #Weights
   disc_weight="../weights/params_discriminator_epoch_005.hdf5"
   gen_weight= "../weights/params_generator_epoch_005.hdf5"

   #Path to store results
   plotsdir = "results/sqrt_scale10_ep5_visualization/"
   gan.safe_mkdir(plotsdir)
   #Parameters
   latent = 256 # latent space     
   num_images = 10 # images to generate

   thetamin = np.radians(60)  # min theta
   thetamax = np.radians(120) # maximum theta
   pmin =100/100
   pmax =200/100

   g = generator(latent)
   g.load_weights(gen_weight)

   d = discriminator()
   d.load_weights(disc_weight)

   # input for generator
   sampled_energies =np.random.uniform(pmin, pmax, num_images)
   sampled_thetas = np.random.uniform(thetamin, thetamax, num_images)
   noise = np.random.normal(0, 1, (num_images, latent-1))
   noise = sampled_energies.reshape(-1, 1) * noise
   gen_in = np.concatenate((sampled_thetas.reshape(-1, 1), noise), axis=1)
   
   out = {} # dict for layer outputs
   plot=0
   for i, layer in enumerate(g.layers[1].layers):
     func = K.function([g.layers[1].layers[0].input, K.learning_phase()], [g.layers[1].layers[i].output])# function to get output of particular layer
     name = g.layers[1].layers[i].name
     out[name] = func([gen_in, 0])[0] # apply function to store output in dict
     layerdir = os.path.join(plotsdir + 'layer_{}'.format(name))
     gan.safe_mkdir(layerdir)# make dir for each layer
     if out[name].ndim==5: # if output is 3d
       filt= out[name].shape[4]
       for f in np.arange(filt):
         convlayer_to_visualize(np.mean(out[name], axis=0), name, f, os.path.join(layerdir, 'layer{}_filt{}.pdf'.format(name, f)))
         plot+=1
     elif out[name].ndim==2: # if output is 1d
       denselayer_to_visualize(np.mean(out[name], axis=0), name, os.path.join(layerdir, 'layer{}.pdf'.format(name)))
       plot+=1
     
   print('{} plots were saved in {} directory'.format(plot, plotsdir))

# Plot dense layer output as graph
def denselayer_to_visualize(layer_out, layer_name, out_file):
   canvas = ROOT.TCanvas("canvas" ,"Visual" ,200 ,10 ,700 ,500) #make
   layer_out= np.squeeze(layer_out)
   n = layer_out.shape[0]
   x = np.arange(n)
   graph = ROOT.TGraph()
   my.fill_graph(graph, x, layer_out)
   graph.GetXaxis().SetTitle('neuron')
   graph.GetYaxis().SetTitle('output')
   graph.SetTitle('Visualization of layer {}'.format(layer_name))
   graph.Draw()
   canvas.Update()
   canvas.Print(out_file)
      
# Plot conv output to 2D histograms   
def convlayer_to_visualize(layer_out, layer_name, filt, out_file):
   layer_out = layer_out[:, :, :, filt]
   canvas = ROOT.TCanvas("canvas" ,"Visual" ,200 ,10 ,700 ,500) #make
   canvas.Divide(2,2)
   x = layer_out.shape[0]
   y = layer_out.shape[1]
   z = layer_out.shape[2]
   opt='colz'
   leg = ROOT.TLegend(0.1,0.4,0.8,0.9)
   hx = ROOT.TH2F('x_{}_{}'.format(layer_name, filt), '', y, 0, y, z, 0, z)
   hy = ROOT.TH2F('y_{}_{}'.format(layer_name, filt), '', x, 0, x, z, 0, z)
   hz = ROOT.TH2F('z_{}_{}'.format(layer_name, filt), '', x, 0, x, y, 0, y)
   hx.SetStats(0)
   hy.SetStats(0)
   hz.SetStats(0)
   
   ROOT.gStyle.SetPalette(1)
   layer_out = np.expand_dims(layer_out, axis=0)
   layer_out = np.expand_dims(layer_out, axis=-1)
   print(layer_out.shape)
   my.FillHist2D_wt(hx, np.sum(layer_out, axis=1))
   my.FillHist2D_wt(hy, np.sum(layer_out, axis=2))
   my.FillHist2D_wt(hz, np.sum(layer_out, axis=3))
   canvas.cd(1)
   
   hx.Draw(opt)
   hx.GetXaxis().SetTitle("Y axis")
   hx.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   hx.SetTitle("Summed along X axis")
   canvas.Update()
   
   canvas.Update()
   canvas.cd(2)
   hy.Draw(opt)
   hy.GetXaxis().SetTitle("X axis")
   hy.GetYaxis().SetTitle("Z axis")
   hx.GetYaxis().CenterTitle()
   hy.SetTitle("Summed along Y axis")
   canvas.Update()
   
   canvas.Update()
   canvas.cd(3)
   hz.Draw(opt)
   hz.GetXaxis().SetTitle("X axis")
   hz.GetYaxis().SetTitle("Y axis")
   hx.GetYaxis().CenterTitle()
   hz.SetTitle("Summed along Z axis")
   canvas.Update()
   canvas.cd(4)
   leg.SetHeader("Visualization of layer {} for filter {}".format(layer_name, filt), 'C')
   leg.Draw()
   canvas.Update()
   canvas.Print(out_file)
   
if __name__ == "__main__":
   main()
       
                                                                            
