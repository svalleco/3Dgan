from DLTools.ThreadedGenerator import DLh5FileGenerator
import glob
import numpy as np

def ConstantNormalization(Norms):
    def NormalizationFunction(Ds):
        out = []
        for i,Norm in enumerate(Norms):
            Ds[i]/=Norm
            out.append(Ds[i])
        return out
    return NormalizationFunction

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
    
    post_f=False
        
    pre_f= ConstantNormalization(Norms)
    
    G=DLh5FileGenerator(files=Files, datasets=datasets,
                        batchsize=BatchSize,
                        max=Max, skip=Skip, 
                        postprocessfunction=post_f,
                        preprocessfunction=pre_f,
                        **kwargs)
    
    return G

Data_dir = '/bigdata/shared/LCD/NewV1/EleEscan/EleEscan_*.h5'
DataFiles = glob.glob(Data_dir)
print 'Data is in ', len(DataFiles), 'files.'

MyGen = MakePreMixGenerator(DataFiles, 200, [1.,100.])
TheGen= MyGen.Generator()
Data= TheGen.next()
print 'The data size is ', len(Data)
print 'Sum of 10 ecal images is', np.sum(Data[0][:10], axis=(1, 2, 3))
print 'Corressponding energy is', Data[1][:10, 1]
