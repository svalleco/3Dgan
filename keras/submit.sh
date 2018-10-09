python EcalEnergyTrain_hvd.py \
		--datapath=/home/linux/EleEscan/EleEscan_1_*.h5 \
		--weightsdir=$RUNDIR \
		--batchsize 16 \
		--optimizer=Adam \
		--latentsize 200 \
		--intraop 11 --interop 1 \
		--warmup 0 --nbepochs 1
