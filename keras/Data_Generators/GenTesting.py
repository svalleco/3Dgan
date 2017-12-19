from DLTools.ThreadedGenerator import DLh5FileGenerator
import glob
import numpy as np

# Load the Data
#from CaloDNN.LoadData import * 

def ConstantNormalization(Norms):
    def NormalizationFunction(Ds):
        out = []
        for i,Norm in enumerate(Norms):
            Ds[i]/=Norm
            out.append(Ds[i])
        return out
    return NormalizationFunction

def MergeInputs():
    def f(X):
        return [X[0],X[1]],X[2]
    return f

def MakePreMixGenerator(Files,BatchSize,Norms=[1.,1.],  Max=3e6,Skip=0, 
                        ECAL=True, HCAL=False, Energy=True, 
			Type = False, **kwargs):
    datasets=[]

    if ECAL:
        datasets.append("ECAL")
    if HCAL:
        datasets.append("HCAL")
    if Type:
    	datasets.append("OneHot")
    if Energy:
        datasets.append("target")
    
    if ECAL and HCAL:
        post_f=MergeInputs()
    else:
        post_f=False
        
    pre_f=ConstantNormalization(Norms)
    
    G=DLh5FileGenerator(files=Files, datasets=datasets,
                        batchsize=BatchSize,
                        max=Max, skip=Skip, 
                        postprocessfunction=post_f,
                        preprocessfunction=pre_f,
                        **kwargs)
    
    return G

Data_dir = '/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_*.h5'
DataFiles = glob.glob(Data_dir)
print(len(DataFiles))

MyGen = MakePreMixGenerator(DataFiles, 200, [1.,100.])
TheGen= MyGen.Generator()
Data= TheGen.next()
print (len(Data))
print (np.sum(Data[0][:10], axis=(1, 2, 3)))
print (Data[1][:10, 1])
