# 3Dgan
3Dgan implementation using keras2 with tensorflow backend can be found in the keras folder. 

AngleArch3dGAN.py is the architecture and AngleTrain3dGAN.py is the training script. 

The weights dir is used to store weights from training. If weights for different trainings are to be saved then --name can be used at command line to identify the training.

analysis/LossPlotsPython.py : Plots Train and Test losses. If training has been named different from default then at command line --historyfile and --outdir must be used to specify the location of loss history generated from a training and the dir where results should be stored respectively.

analysis/RootAnalysis.py : Physics Evaluation results. If training has been named different from default then paths for weights(--dweights & --gweights) and output(--plotdir) must be provided.

Tested with:
keras: 2.2.4
tensorflow : 1.14.1

Analysis scripts also uses Root and matplotlib