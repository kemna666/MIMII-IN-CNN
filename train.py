from model_TransGAN import Generator,Discriminator,TransGAN
from Dataset import MIMIIDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn 
import torch
from sklearn.model_selection import train_test_split
#定义模型
generator = Generator(input_dim=100,heads_num=4,dim_feedforward=2048,num_layers=2,num_classes=153)
discriminator = Discriminator(input_dim=153,heads_num=4,dim_feedforward=2048,num_layers=2,num_classes=4)
device = torch.device("cpu")
model = TransGAN(generator,discriminator).to(device)
dataset =MIMIIDataset('./data/data.pkl')
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader =DataLoader(dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
#定义训练参数
#训练轮数
num_epochs = 100

#开始训练
optimizer_generator = optim.Adam(model.generator.parameters(), lr=0.0001)
optimizer_discriminator = optim.Adam(model.discriminator.parameters(), lr=0.0001)
for epoch in range(num_epochs):
    loss_generator = 0.0
    loss_discriminator = 0.0
    for batch in train_loader:
        real_audio, real_device, real_label = batch
        noise = torch.randn(real_audio.size(0), 100)  # 生成与真实音频相同形状的噪声

        # 将数据移动到设备上
        noise = noise.to(device)
        real_audio = real_audio.to(device)
        real_device = real_device.to(device)
        real_label = real_label.to(device)
        
        print("Real device shape:", real_device.shape)
        print("Real device max value:", real_device.max())
        loss_generator, loss_discriminator = model.train(noise, real_audio, real_device, real_label, optimizer_generator, optimizer_discriminator, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss Generator: {loss_generator}, Loss Discriminator: {loss_discriminator}')

model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
        for batch_features, batch_device_nums, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
print(f"Final test loss: {test_loss / len(test_loader)}")
print(f"Final test accuracy: {100 * correct / total}%")