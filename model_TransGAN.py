import torch.nn as nn
import torch

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
            activation='relu',
            )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,                 #上文定义的编码器层
            num_layers=num_layers,               #编码器层数
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
        super().__init__(input_dim,heads_num,dim_feedforward,num_layers,num_classes)
        self.output_layer=nn.Linear(256,num_classes)
    
    
    #利用随机噪声生成假数据
    def forward(self,z):
        x=self.input_layer(z)
        print("Generator input layer output shape:", x.shape)
        output = self.transformer_encoder(x)
        
        audio = self.output_layer(output)
        print("Generator transformer encoder output shape:", output.shape)
        print("Generator final output shape:", audio.shape)
        return audio
#定义判别器
class Discriminator(TransAudio):
    def __init__(self,input_dim,heads_num,dim_feedforward,num_layers,num_classes):
        super().__init__(input_dim,heads_num,dim_feedforward,num_layers,num_classes)
        #输出层输出是否为真，故输出维度为1
        self.audio_output_layer = nn.Linear(256, 4)
        self.device_output_layer = nn.Linear(256, 4)  
        self.label_output_layer = nn.Linear(256, 4)  

    def forward(self,x):
        x=self.input_layer(x)
        output = self.transformer_encoder(x)
        #torch.sigmoid(x)作用是将x转换为0-1之间的数
        validity =  torch.sigmoid(self.output_layer(output))
        device_pred = self.device_output_layer(output)
        label_pred = self.label_output_layer(output)
        return validity, device_pred, label_pred
    
#正式定义TransGAN
class TransGAN(nn.Module):
    def __init__(self,generator,discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def train(self, noise, real_audio, real_device, real_label,optimizer_generator, optimizer_discriminator, device):
        #训练生成器
        criterion = nn.CrossEntropyLoss()
        noise = noise.to(device)
        real_audio = real_audio.to(device)
        real_device = real_device.to(device)
        real_label = real_label.to(device)
        # 梯度归零
        self.generator.zero_grad()
        # 生成假数据
        fake_audio = self.generator(noise)
        # detach()作用是防止梯度回传
        fake_discriminator, fake_device, fake_label = self.discriminator(fake_audio.detach())
        # 损失函数
        loss_generator_audio = criterion(fake_discriminator, torch.ones_like(fake_discriminator))
        loss_generator_device = criterion(fake_device, real_device)
        loss_generator_label = criterion(fake_label, real_label)
        loss_generator = loss_generator_audio + loss_generator_device + loss_generator_label
        # 反向传播
        loss_generator.backward()
        optimizer_generator.step()
        #训练判别器
        self.discriminator.zero_grad()
        print("Discriminator input shape:", real_audio.shape)
        real_discriminator, real_device_pred, real_label_pred = self.discriminator(real_audio)
        loss_discriminator_real_audio = criterion(real_discriminator, torch.ones_like(real_discriminator))
        loss_discriminator_real_device = criterion(real_device_pred, real_device)
        loss_discriminator_real_label = criterion(real_label_pred, real_label)
        fake_discriminator, fake_device, fake_label = self.discriminator(fake_audio)
        loss_discriminator_fake_device = criterion(fake_device, real_device)
        loss_discriminator_fake_label = criterion(fake_label, real_label)
        loss_discriminator_fake_audio = criterion(fake_discriminator, torch.zeros_like(fake_discriminator))
        loss_discriminator = (loss_discriminator_real_audio + loss_discriminator_real_device + loss_discriminator_real_label +
                            loss_discriminator_fake_audio + loss_discriminator_fake_device + loss_discriminator_fake_label) / 6
        loss_discriminator.backward()
        optimizer_discriminator.step()

        return loss_generator.item(), loss_discriminator.item()
