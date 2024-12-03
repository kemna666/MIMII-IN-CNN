#导入transformer处理音频的库
from transformers import Wav2Vec2Processor
import torch.nn as nn
import torch
import torch.optim as optim

#Transformer模型
class TransAudio(nn.Module):
    #heads_num指的是多头注意力机制的头数
    def __init__(self,input_dim,heads_num,dim_feedforward,num_layers,num_classes): 
        super().__init__()
        #输入层
        self.input_layer = nn.Linear(input_dim,256)
        #编码器部分
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,                        #输入的特征维度
            nhead=heads_num,                    #多头注意力机制的头数（注意整除d_model）
            dim_feedforward=dim_feedforward,    #前向传播隐藏层的维度
            dropout=0.1,                        #默认0.1
            )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,                 #上文定义的编码器层
            num_layers=num_layers               #编码器层数
            )
        #输出层
        self.output_layer=nn.Linear(256,num_classes)
    
    def forward(self,x):
        x = self.input_layer(x)
        output = self.transformer_encoder(x)
        return output
    
#GAN
#GAN的生成器
class Generator(TransAudio):
    def __init__(self,input_dim,heads_num,dim_feedforward,num_layers,num_classes):
        super().__init__(self,input_dim,heads_num,dim_feedforward,num_layers,num_classes)
        self.output_layer=nn.Linear(256,num_classes)
    
    
    #利用随机噪声生成假数据
    def forward(self,z):
        x=self.input_layer(z)
        output = self.transformer_encoder(x)
        audio = self.output_layer(output)
        return audio
#定义判别器
class Discriminator(TransAudio):
    def __init__(self,input_dim,heads_num,dim_feedforward,num_layers,num_classes):
        super().__init__(self,input_dim,heads_num,dim_feedforward,num_layers,num_classes)
        #输出层输出是否为真，故输出维度为1
        self.output_layer=nn.Linear(256,1)
    
    def forward(self,x):
        x=self.input_layer(x)
        output = self.transformer_encoder(x)
        #torch.sigmoid(x)作用是将x转换为0-1之间的数
        vaildity =  torch.sigmoid(self.output_layer(output))
        return vaildity
    
