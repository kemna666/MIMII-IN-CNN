from model_TransGAN import Generator,Discriminator,TransGAN
from Dataset import MIMIIDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn 
import torch
import os
#定义模型
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
generator = Generator(input_dim=100,heads_num=4,dim_feedforward=2048,num_layers=2,num_classes=256)
discriminator = Discriminator(input_dim=256,heads_num=4,dim_feedforward=2048,num_layers=2,num_classes=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TransGAN(generator,discriminator).to(device)
dataset =MIMIIDataset('./data/data.pkl')
train_loader =DataLoader(dataset, batch_size=4, shuffle=True)
#定义训练参数
#训练轮数
num_epochs = 100

#开始训练
optimizer_generator = optim.Adam(model.generator.parameters(), lr=0.0001)
optimizer_discriminator = optim.Adam(model.discriminator.parameters(), lr=0.0001)
for epoch in range(num_epochs):
    for batch in train_loader:
        real_audio, real_device, real_label = batch
        noise = torch.randn(real_audio.size(0), 100)  # 生成与真实音频相同形状的噪声

        # 将数据移动到设备上
        noise = noise.to(device)
        real_audio = real_audio.to(device)
        real_device = real_device.to(device)
        real_label = real_label.to(device)

        loss_generator, loss_discriminator = model.train(noise, real_audio, real_device, real_label, optimizer_generator, optimizer_discriminator, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss Generator: {loss_generator}, Loss Discriminator: {loss_discriminator}')