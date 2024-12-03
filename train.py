from model_TransGAN import Generator,Discriminator,TransGAN
from Dataset import MIMIIDataset
import torch.optim as optim
import torch.nn as nn 
import torch
import pickle

#定义模型
generator = Generator(input_dim=100,heads_num=4,dim_feedforward=2048,num_layers=2,num_classes=256)
discriminator = Discriminator(input_dim=256,heads_num=4,dim_feedforward=2048,num_layers=2,num_classes=1)
model = TransGAN(generator,discriminator)
train_loader,val_loader,test_loader = MIMIIDataset.features_exteract()
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#定义训练参数
#训练轮数
num_epochs = 100

#开始训练
for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:  
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')