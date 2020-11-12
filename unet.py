#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:34:50 2020

@author: latente
"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True))

def trans_conv(in_c, out_c):
    return nn.ConvTranspose2d(in_c,out_c, kernel_size=2, stride=2)
    
def crop_tensor(tensor,target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size -target_size
    delta = delta//2
    return tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        
        self.transpose_1 = trans_conv(1024,512)
        self.up_conv_1 = double_conv(1024,512)
        self.transpose_2 = trans_conv(512,256)
        self.up_conv_2 = double_conv(512,256)
        self.transpose_3 = trans_conv(256,128)
        self.up_conv_3 = double_conv(256,128)
        self.transpose_4 = trans_conv(128,64)
        self.up_conv_4 = double_conv(128,64)
        self.last_conv = nn.Conv2d(64,1,kernel_size=1)
    
    def forward(self, image):
        #encoder
        x1 = self.down_conv_1(image) #passed to second part of net
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) #passed to second part of net
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) #passed to second part of net
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) #passed to second part of net
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        # print(x9.size())
        #decoder
        x10 = self.transpose_1(x9)
        x11 = self.up_conv_1(torch.cat([crop_tensor(x7,x10),x10],1))
        x12 = self.transpose_2(x11)
        x13 = self.up_conv_2(torch.cat([crop_tensor(x5,x12),x12],1))
        x14 = self.transpose_3(x13)
        x15 = self.up_conv_3(torch.cat([crop_tensor(x3,x14),x14],1))
        x16 = self.transpose_4(x15)
        x17 = self.up_conv_4(torch.cat([crop_tensor(x1,x16),x16],1))
        x18 = self.last_conv(x17)
        # print(x18.size(),(x18.shape[0],x18.shape[2],x18.shape[3]))
        return nn.Sigmoid()(x18)#.reshape((x18.shape[0],x18.shape[2],x18.shape[3]))
        

#Generate fake data
x=torch.zeros((572,572))
a, s = 100, 50
for i in range(6):
    x[a*i:a*i+s,a*i:a*i+s] = 1
    x[a*i:a*i+s,572-a*i:572-a*i+s] = 1
y=x[92:572-92,92:572-92].clone().long()
x+=(0.5*torch.rand((572,572)))
x = x-x.min()
x /= (x.max()-x.min())
plt.imshow(x);plt.title('X');plt.show()
plt.imshow(y);plt.title('Y');plt.show()

#Prepare dimensions for network      
x.unsqueeze_(0).unsqueeze_(0);
y.unsqueeze_(0);

#Initialize model, loss and learning rate
model = UNet()
loss_fn = nn.BCELoss()
learning_rate = 1e-5

#Train
for t in range(10):
    y_pred = model(x)
    plt.imshow(y_pred.detach().numpy()[0][0]);plt.title('Epoch %s Class 0'%t);plt.show()
    plt.imshow(y_pred.detach().numpy()[0][1]);plt.title('Epoch %s Class 1'%t);plt.show()
    
    loss = loss_fn(y_pred, y)
    print('Epoch', t, ':', loss.item())

    model.zero_grad()
    
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

plt.imshow(y_pred.detach().numpy()[0][0]>0.5);plt.show()
plt.imshow(y_pred.detach().numpy()[0][1]>0.5);plt.show()
