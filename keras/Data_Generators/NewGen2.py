from __future__ import print_function
from CaloDNN.NeuralNets.LoadData import * 
ECALShape= None, 25, 25, 25
HCALShape= None, 5, 5, 60
FileSearch="/bigdata/shared/LCD/NewV1/*/*.h5"
Particles=["Ele"]
MaxEvents=int(8.e5)
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
NTest = NTestSamples=5000
NTrain = 500000
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

from adlkit.data_provider.data_providers import *
from adlkit.data_provider.cached_data_providers import *

def GANNormalization(Norms):
        def NormalizationFunction(Ds):
            Ds=Ds[0]
            Ds[0] = np.expand_dims(Ds[0], axis=-1)
            return Ds
	return NormalizationFunction
   
"""Train_genC = MakeGenerator(ECAL, HCAL, TrainSampleList, NTrain, GANNormalization(Norms), 
				    batchsize = BatchSize, 
				    shapes = shapes, 
				    n_threads=4, 
				    multiplier=multiplier, 
				    cachefile="tmp/Gulrukh_Train_Cache.h5")
print ('Train Set')
Test_genC = MakeGenerator(ECAL, HCAL, TestSampleList, NTest, LCDNormalization(Norms),
                                    batchsize=BatchSize,
                                    shapes = shapes,
                                    n_threads=4,
                                    multiplier=multiplier,
                                    cachefile="tmp/Gulrukh_Test_Cache.h5")
"""
#Train_gen=Train_genC.DiskCacheGenerator(4)
Train_genC.start()
Train_gen= Train_genC.first().generate().next()
Train_cache = GeneratorCacher(Train_gen, BatchSize, max=NSamples,
                          wrap=True,
                          delivery_function=None,
                          cache_filename=None,
                          delete_cache_file=False)

Traingen = Train_cache.DiskCacheGenerator()
print (Traingen.next())
"""N = 1
count = 0
start = time.time()
for tries in xrange(2):
  print ("*********************Try:", tries)
  for D in Traingen:
    Delta = (time.time() - start)
    print (count, ":", Delta, ":", Delta / float(N))
    sys.stdout.flush()
    N += 1
    for d in D:
      print (d.shape)
      NN = d.shape[0]
      # print d[0]
      pass
      count += NN"""
"""Test_gen =Test_genC.PreloadGenerator()
X_test, y_test=Test_genC.D
print(X_test.shape)"""
