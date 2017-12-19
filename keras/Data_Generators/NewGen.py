from __future__ import print_function
from CaloDNN.NeuralNets.LoadData import * 
ECALShape= None, 25, 25, 25
HCALShape= None, 5, 5, 60
FileSearch="/bigdata/shared/LCD/NewV1/*/*.h5"
Particles=["Ele"]
MaxEvents=int(5.e6)
NTestSamples=100000
NClasses=len(Particles)
BatchSize=100
NSamples=BatchSize*10
ECAL=True
HCAL=False
delivery_function = None
ECALNorm='NonLinear'
HCALNorm='NonLinear'
multiplier=2
n_threads=3
NTest = NTestSamples=10000
NTrain = 100000
Train_genC,Test_genC,Norms,shapes,TrainSampleList,TestSampleList= SetupData(FileSearch,
                                                          ECAL,
                                                          HCAL,
                                                          True,
                                                          NClasses,
                                                          [0.9, 0.1],
                                                          Particles,
                                                          BatchSize,
                                                          multiplier,
                                                          ECALShape,
                                                          HCALShape,
                                                          ECALNorm,
                                                          HCALNorm,
							  delivery_function,
     						          n_threads,
              						  NTrain,
							  NTest)
#print (len(Train_genC))
#print (len(Test_genC))
print (len(Norms))
print (len(shapes))
print (len(TrainSampleList))
print (len(TestSampleList))
sample_spec_train = list()
for item in TrainSampleList:
   sample_spec_train.append((item[0], item[1] , item[2], 1))

q_multipler = 2
read_multiplier = 1
n_buckets = 1
from adlkit.data_provider.data_providers import H5FileDataProvider
from adlkit.data_provider.cached_data_providers import *
   
Train_genC = H5FileDataProvider(sample_spec_train,
                                    batch_size=BatchSize,
                                    max=int(NSamples/BatchSize),
                                    process_function=LCDN(Norms),
                                    #delivery_function=unpack,
                                    n_readers=n_threads,
                                    q_multipler=q_multipler,
                                    n_buckets=n_buckets,
                                    read_multiplier=multiplier,
                                    make_one_hot=True,
                                    sleep_duration=1,
                                    wrap_examples=True)

print ("Class Index Map:", Train_genC.config.class_index_map)

print ("Starting Training Generators...")
sys.stdout.flush()
Train_genC.start()
Train_gen=Train_genC.first().generate()
print ("Done.")

GC=GeneratorCacher(Train_gen,BatchSize,max=NSamples,
                       wrap=True,
                       delivery_function=None,
                       cache_filename=None,   
                       delete_cache_file=True )
    
gen=GC.DiskCacheGenerator()

