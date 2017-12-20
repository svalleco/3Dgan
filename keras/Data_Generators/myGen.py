from __future__ import print_function
from CaloDNN.NeuralNets.LoadData import * 
from adlkit.data_provider.cached_data_providers import GeneratorCacher
#from adlkit.data_provider.data_providers import *
import time

def myNorm(Norms):
    def NormalizationFunction(Ds):
        # converting the data from an ordered-dictionary format to a list
        Ds = [Ds[item] for item in Ds]
        out = []
        # print('DS', Ds)
        # TODO replace with zip function
        for i,D in enumerate(Ds):
            Norm=Norms[i]
	    if i == 0:
                D[D < 1e-666666] = 0
            if i == 1:
	        D = D[:,1]
	    if Norm != 0.:
                if isinstance(Norm, float):
                    D /= Norm
                if isinstance(Norm, str) and Norm.lower() == "nonlinear":
                    D = np.tanh(
                        np.sign(Ds[i]) * np.log(np.abs(Ds[i]) + 1.0) / 2.0)
                out.append(D)
        return out
    return NormalizationFunction

def mySetupData(FileSearch,
              ECAL,HCAL,target,
              NClasses,f,Particles,
              BatchSize,
              multiplier,
              ECALShape,
              HCALShape,
              ECALNorm,
              HCALNorm,
	      targetNorm,
              delivery_function,
              n_threads,
              NTrain,
              NTest):
    datasets=[]
    shapes=[]
    Norms=[]

    if ECAL:
        datasets.append("ECAL")
        shapes.append((BatchSize*multiplier,)+ECALShape[1:])
        Norms.append(ECALNorm)
    if HCAL:
        datasets.append("HCAL")
        shapes.append((BatchSize*multiplier,)+HCALShape[1:])
        Norms.append(HCALNorm)
    if target:
        datasets.append("target")
#        shapes.append((BatchSize*multiplier,)+(1,5))
        shapes.append((BatchSize*multiplier,)+(1,))
        Norms.append(targetNorm)

    # This is for the OneHot    
    #shapes.append((BatchSize*multiplier, NClasses))
    #Norms.append(1.)

    TrainSampleList,TestSampleList=DivideFiles(FileSearch,f,
                                               datasetnames=datasets,
                                               Particles=Particles)
    sample_spec_train = list()
    for item in TrainSampleList:
        sample_spec_train.append((item[0], item[1] , item[2], 1))

    sample_spec_test = list()
    for item in TestSampleList:
        sample_spec_test.append((item[0], item[1] , item[2], 1))

    q_multipler = 2
    read_multiplier = 1
    n_buckets = 1

    Train_genC = H5FileDataProvider(sample_spec_train,
                                    max=math.ceil(float(NTrain)/BatchSize),
                                    batch_size=BatchSize,
                                    process_function=myNorm(Norms),
                                    delivery_function=delivery_function,
                                    n_readers=n_threads,
                                    q_multipler=q_multipler,
                                    n_buckets=n_buckets,
                                    read_multiplier=read_multiplier,
                                    #make_one_hot=True,
                                    sleep_duration=1,
                                    wrap_examples=True)

    Test_genC = H5FileDataProvider(sample_spec_test,
                                   max=math.ceil(float(NTest)/BatchSize),
                                   batch_size=BatchSize,
                                   process_function=myNorm(Norms),
                                   delivery_function=delivery_function,
                                   n_readers=n_threads,
                                   q_multipler=q_multipler,
                                   n_buckets=n_buckets,
                                   read_multiplier=read_multiplier,
                                   #make_one_hot=True,
                                   sleep_duration=1,
                                   wrap_examples=False)

    print ("Class Index Map:", Train_genC.config.class_index_map)

    return Train_genC,Test_genC,Norms,shapes,TrainSampleList,TestSampleList

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
target=True
delivery_function = None
ECALNorm= None 
HCALNorm= None
targetNorm=100.
multiplier=1
n_threads=3
NTest = NTestSamples=5000
NTrain = 500000
#This function will set up different values
Train_genC,Test_genC,Norms,shapes,TrainSampleList,TestSampleList= mySetupData(FileSearch,
                                                          ECAL,
                                                          HCAL,
                                                          target,
                                                          NClasses,
                                                          [0.9, 0.1],
                                                          Particles,
                                                          BatchSize,
                                                          multiplier,
                                                          ECALShape,
                                                          HCALShape,
                                                          ECALNorm,
                                                          HCALNorm,
							  targetNorm,
							  delivery_function,
     						          n_threads,
              						  NTrain,
							  NTest)

print ('Training Files = ', len(TrainSampleList))
print ('Testing Files = ', len(TestSampleList))
print ('Norms are =', Norms)
print ('Shapes are =', shapes)

Train_genC.start()
Train_gen1= Train_genC.first().generate().next()

print ('The data size is ', len(Train_gen1))
print ('Sum of 10 ecal images is', np.sum(Train_gen1[0][:10], axis=(1, 2, 3)))
print ('Corressponding energy is', Train_gen1[1][:10])

print ("Starting Training Generators...",)
sys.stdout.flush()
#Train_genC.start()
print ("Start.")
Train_gen = Train_genC.first().generate()
print ("Done.")

Test_cache = GeneratorCacher(Test_genC, BatchSize, max=NSamples,
                          wrap=True,
                          delivery_function=None,
                          cache_filename=None,
                          delete_cache_file=False)


Train_cache = GeneratorCacher(Train_gen, BatchSize, max=NSamples,
                          wrap=True,
                          delivery_function=None,
                          cache_filename=None,
                          delete_cache_file=False)

Traingen = Train_cache.DiskCacheGenerator()
Testgen = Test_cache.PreloadGenerator()

X_test, y_test= next(Testgen)
"""for D in Traingen:
  Data= D
  print ('The data size is ', len(Data))
  print (Data[0].shape)
  print (Data[1].shape)
"""
