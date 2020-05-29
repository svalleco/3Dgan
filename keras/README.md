# Training with fixed angle data
EcalEnergyGan.py is the arhcitecture and EcalEnergyGan.py is the training script

# Training with variable angle data 
AngleArch3dGAN.py is the architecture and AngleTrain3dGAN.py is the training script.

The weights dir is used to store weights from training. If weights for different trainings are to \
be saved then --name can be used at command line to identify the training.

## Tensorboard

To plot the loss over time for the generator and discriminator, run the following
command:

tensorboard --logdir=/path/to/logs

This will output a URL that you go to in your browser (e.g. http://myserver:6006)
