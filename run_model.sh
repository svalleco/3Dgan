#!/bin/bash
mkdir Data
mkdir Results
pip3 install --user awscli
export PATH=$PATH:$HOME/.local/bin
aws s3 cp --no-sign-request --endpoint-url=https://s3.cern.ch s3://gan-bucket/ Data/. --recursive --exclude "EleEscan_EleEscan*"
cd keras
python3 AngleTrain3dGAN_tf.py --datapath="$HOME/3Dgan/Data/*.h5" --name='tpu_training' --nbepochs 1 --tpu=$TPU_NAME
