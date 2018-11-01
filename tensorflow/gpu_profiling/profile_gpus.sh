#!/bin/bash

while true; 
do nvidia-smi --query-gpu=utilization.gpu --format=csv >> gpu_utillization.log; sleep 0.1;
done
