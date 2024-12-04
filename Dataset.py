import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class MIMIIDataset(Dataset):    
    def __init__(self, pkl_file_path):
        # 加载.pkl文件中的数据
        with open(pkl_file_path, 'rb') as file:
            self.data = pickle.load(file)
        self.merged_data =[]
        for snr_data in self.data:
            self.merged_data.extend(snr_data)
    
    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.merged_data)
    
    def __getitem__(self, idx):
        # 根据索引idx返回一个样本的特征和标签
        mfcc_features = self.merged_data[idx][0]
        device_index = self.merged_data[idx][1]
        label_index =   self.merged_data[idx][2]
        # 将MFCC特征转换为Tensor
        mfcc_features = torch.tensor(mfcc_features, dtype=torch.float32)
        
        # 将设备和标签索引转换为Tensor
        device_index = torch.tensor(device_index, dtype=torch.long)
        label_index = torch.tensor(label_index, dtype=torch.long)
        
        return mfcc_features, device_index, label_index
