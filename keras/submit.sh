python EcalEnergyTrain_hvd.py \
		--datapath=/afs/cern.ch/work/s/svalleco/EleEscan/EleEscan_1_*.h5 \
		--weightsdir=$RUNDIR \
		--batchsize 128 \
		--optimizer=Adam \
		--latentsize 200 \
                --analysis=True \
		--intraop 11 --interop 1 \
		--warmup 0 --nbepochs 1
