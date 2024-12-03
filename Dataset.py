import torch 
from torch.utils.data import Dataset,DataLoader
#使用librosa提取特征
import librosa
import numpy as np

#加载音频文件
path = './data'
audio,sr = librosa.load(path,sr=16000)

def features_exteract():
    #提取梅尔频率倒谱系数(mfcc特征数量13个)
    mfccs=librosa.feature.mfcc(y=audio, sr=sr,n_mfcc=13)
    #取平均值（mfccs的转置axis=0对列求平均值，返回1*n矩阵）
    return np.mean(mfccs.T,axis=0)

#定义Dataset和DataLoader
class MIMIIDataset(Dataset):
    def __init__(self,files,labels):
        super().__init__(self,files,labels)
        self.files = files
        self.labels = labels
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        audio = torch.load(self.files[idx])
        features = features_exteract(audio)
        label = self.labels[idx]
        return torch.tensor(features),torch.tensor(label)
