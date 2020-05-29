# Analysis

The analysis compares the GAN generated images to G4 data events. All of the scripts in this section, takes a sample of G4 events and then generate GAN events with similar input conditions (primary particle energy / primary particleincident  angle). Where events are selected in bins: the primary energy bins have a +/- 5 GeV tolerance and the incident angle bins have a tolerance of +/- 0.1 rad (5.73 degree). The utils directory contains a set of files with frequently used utility functions. Most of the scripts except the LossPlotsPython.py require [ROOT software](https://root.cern.ch/) to be installed. Following is a description and a set of instructions for all scripts in this folder:
## 2Dprojections.py

This scripts compares 2D projections for all the three planes for events from the G4 data with corressponding GAN generated events with same input values.

## LossPlotsPython.py

This script takes the loss history generated from each training, saved as a pickle file. The plots are generated using matplotlib.

## RootAnalysisAngle.py 

The scripts compares in detail different features of G4 and GAN events.

## SelectEpoch.py

This script select the best epoch among a predefined number of epochs. The plots also provides the epoch to epoch progress based on the sampling fraction.

## SimpleAnalysis.py

This scripts compares the two most crucial features of the generated events: the sampling fraction and the shower shapes. 