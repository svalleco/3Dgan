# Analysis

The analysis compares the GAN generated images to G4 data events. All of the scripts in this section, take a sample of G4 events and then generate GAN events with similar input conditions (primary particle energy / primary particleincident  angle). Where events are selected in bins: the primary energy bins have a +/- 5 GeV tolerance and the incident angle bins have a tolerance of +/- 0.1 rad (5.73 degree). The [utils](utils) directory contains a set of files with frequently used utility functions. Most of the scripts except the LossPlotsPython.py require [ROOT software](https://root.cern.ch/) to be installed. Following is a brief description and a set of instructions for all scripts in this folder:
A common feature for all the scripts is the ang (1: variable angle version 0: fixed angle version). The default is variable angle. The instructions will include only most useful parameters. Other options can be explored from parser help. 

## 2Dprojections.py

This scripts compares 2D projections for all the three planes for events from the G4 data with corressponding GAN generated events with same input values. The script can be submitted as:

python 2Dprojections.py --gweight *generator_weights* --outdir *results/your_result_dir*

## LossPlotsPython.py

This script takes the loss history generated from each training, saved as a pickle file. The plots are generated using matplotlib. The script can be submitted as:

python LossPlotsPython.py --historyfile *path_to_loss_history* --outdir *path_to_save_results*

## RootAnalysisAngle.py 

The scripts compares in detail different features of G4 and GAN events. The script can be submitted as:

python RootAnalysisAngle.py --gweights *generator_weight1 generator_weight2* --dweights *discriminator_weight1  discriminator_weight2* --labels label1 label2 --outdir *results/your_result_dir*

## SelectEpoch.py

This script select the best epoch among a predefined number of epochs. The plots also provides the epoch to epoch progress based on the sampling fraction. The script can be submitted as:

python SelectEpoch.py --gweightsdir *path_to_weights_directory* --outdir *path_to_save_results* 

## SimpleAnalysis.py

This scripts compares the two most crucial features of the generated events: the sampling fraction and the shower shapes. The script can be submitted as:

python SimpleAnalysis.py --gweights *weight1 weight2* --labels label1 label2 --outdir *results/your_result_dir*

