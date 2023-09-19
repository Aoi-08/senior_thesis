# -*- coding: utf-8 -*-

"""
このファイルではネットワークの構造を決めています
"""

from __future__ import division
import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_pfn_extras as ppe

from pfrl import action_value
from pfrl.initializers import init_chainer_default
from pfrl.nn.mlp import MLP
from pfrl.q_function import StateQFunction

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)  # seed値を設定
import tkinter
import cv2

class MLP(nn.Module):
    def __init__(self, n_actions=3,n_input=1296,n_added_input=0):
        self.n_actions = n_actions
        self.n_input = n_input
        self.n_added_input = n_added_input
        super(MLP, self).__init__()
        self.al1 = nn.Linear(n_input + n_added_input, n_actions)
        nn.init.kaiming_normal_(self.al1.weight)

        # self.al2 = nn.Linear(512, 100)
        # nn.init.kaiming_normal_(self.al2.weight)

        #self.al3 = nn.Linear(250, 100)
        #nn.init.kaiming_normal_(self.al3.weight)

        # self.al4 = nn.Linear(100, 10)
        # nn.init.kaiming_normal_(self.al4.weight)

        # self.al5 = nn.Linear(10, n_actions)
        # nn.init.kaiming_normal_(self.al5.weight)
    
    def forward(self, state):#state:入力情報
        # ha = F.relu(self.al1(state))
        # ha = F.relu(self.al2(ha))
        #ha = F.relu(self.al3(ha))
        # ha = F.relu(self.al4(ha))
        q = self.al1(state)

        return pfrl.action_value.DiscreteActionValue(q)

class Q_Func(nn.Module):
    def __init__(self, n_actions, n_input_channels, n_added_input=0, img_width=48, img_height=27):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Q_Func, self).__init__()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)

        # 畳み込み
        #convolution2D(in_channels,out_channels,K_size,stride,pad)
        self.conv1_1 = nn.Conv2d(n_input_channels, 8, 5)#27*48*6=>44*23*8
        nn.init.kaiming_normal_(self.conv1_1.weight)
        
        self.conv1_2 = nn.Conv2d(8, 16, 5)#40*19*16
        nn.init.kaiming_normal_(self.conv1_2.weight)
        #ここでpooling2*2
        self.conv2_1 = nn.Conv2d(16, 32*2, 5)#20*10*16=>16*6*64
        nn.init.kaiming_normal_(self.conv2_1.weight)

        self.conv2_2 = nn.Conv2d(32*2, 64*2, 5)#2*12*128
        nn.init.kaiming_normal_(self.conv2_2.weight)
        #pooling2*2:1*6*128

        # Advantage
        self.al1 = nn.Linear(128*6+n_added_input, 512)
        nn.init.kaiming_normal_(self.al1.weight)

        self.al2 = nn.Linear(512, 512)
        nn.init.kaiming_normal_(self.al2.weight)

        self.al3 = nn.Linear(512, n_actions)
        nn.init.kaiming_normal_(self.al3.weight)
    
    def forward(self, state):
        if self.n_added_input:
            img = state[:,:-self.n_added_input]
            sen = state[:,-self.n_added_input:]
            # img = F.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))
        else:
            img = state
        
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))
        #convolution
        h = F.relu(self.conv1_1(img))
        h = self.pool(F.relu(self.conv1_2(h)))
        h = F.relu(self.conv2_1(h))
        h = self.pool(F.relu(self.conv2_2(h)))
        
        h = h.view(-1,6*128)#reshape 
        
        if self.n_added_input:
            h = torch.cat((h,sen), axis=1)

        ha = F.relu(self.al1(h))
        ha = F.relu(self.al2(ha))
        q = self.al3(ha)

        return pfrl.action_value.DiscreteActionValue(q)

class Dueling_Q_Func(nn.Module):
    def __init__(self, n_actions, n_input_channels, n_added_input=0, img_width=48, img_height=27):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Dueling_Q_Func, self).__init__()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)

        # 畳み込み
        #convolution2D(in_channels,out_channels,K_size==tuple(N,M),stride,pad)
        self.conv1_1 = nn.Conv2d(n_input_channels, 8, 5)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv1_2 = nn.Conv2d(8, 16, 5)
        nn.init.kaiming_normal_(self.conv1_2.weight)

        self.conv2_1 = nn.Conv2d(16, 64, 5)
        nn.init.kaiming_normal_(self.conv2_1.weight)
        self.conv2_2 = nn.Conv2d(64, 128, 5)
        nn.init.kaiming_normal_(self.conv2_2.weight)

        self.conv3_1 = nn.Conv2d(128, 256, 3)
        nn.init.kaiming_normal_(self.conv3_1.weight)
        self.conv3_2 = nn.Conv2d(256, 1200, 3)
        nn.init.kaiming_normal_(self.conv3_2.weight)
        """
        self.conv4_1 = nn.Conv2d(32, 64, 3)#48*100=>44*96
        nn.init.kaiming_normal_(self.conv1_1.weight)
        
        self.conv4_2 = nn.Conv2d(64, 64, 3)#40*92
        nn.init.kaiming_normal_(self.conv1_2.weight)
        """
        # Advantage
        self.al1 = nn.Linear(128*6+self.n_added_input, 512)
        nn.init.kaiming_normal_(self.al1.weight)

        self.al2 = nn.Linear(512, 512)
        nn.init.kaiming_normal_(self.al2.weight)

        self.al3 = nn.Linear(2000, 2000)
        nn.init.kaiming_normal_(self.al3.weight)

        self.al4 = nn.Linear(512, n_actions)
        nn.init.kaiming_normal_(self.al4.weight)
        
        # State Value
        self.vl1 = nn.Linear(128*6+self.n_added_input, 512)
        nn.init.kaiming_normal_(self.vl1.weight)

        self.vl2 = nn.Linear(512,512)
        nn.init.kaiming_normal_(self.vl2.weight)

        self.vl3 = nn.Linear(1200,1200)
        nn.init.kaiming_normal_(self.vl3.weight)

        self.vl4 = nn.Linear(512, 1)
        nn.init.kaiming_normal_(self.vl4.weight)

    def forward(self, state):
        if self.n_added_input:
            img = state[:,:-self.n_added_input]
            sen = state[:,-self.n_added_input:]
            # img = F.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))
        else:
            img = state
        
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))
        #convolution
        h = F.relu(self.conv1_1(img))
        h = self.pool(F.relu(self.conv1_2(h)))
        h = F.relu(self.conv2_1(h))
        h = self.pool(F.relu(self.conv2_2(h)))
        #h = F.relu(self.conv3_1(h))
        #h = self.pool(F.relu(self.conv3_2(h)))
        #h = F.relu(self.conv4_1(h))
        #h = self.pool(F.relu(self.conv4_2(h)))
        h = h.view(-1, 6*128)#reshape 
        
        
        # Advantage
        batch_size = img.shape[0]#=1
        h= torch.cat((torch.reshape(h,(h.shape[0], -1)), sen),dim=1)
        ha = F.relu(self.al1(h))
        ha = F.relu(self.al2(ha))
        ya = self.al4(ha)
        mean = torch.reshape(
            torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State Value
        hs = F.relu(self.vl1(h))
        hs = F.relu(self.vl2(hs))
        ys = self.vl4(hs)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys

        return pfrl.action_value.DiscreteActionValue(q)

#これがRainbow
class DistributionalDueling_old(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(
        self,
        n_actions,
        n_input_channels,
        n_added_input,
        n_atoms = 100,
        v_min = -10,
        v_max = 10,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms
        self.img_width = 48
        self.img_height = 27
        self.n_added_input = n_added_input

        super().__init__()

        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)
        self.drop1d_B = nn.Dropout(0.5)
        self.drop1d_S = nn.Dropout(0.5)
        self.drop2d_B = nn.Dropout2d(0.4)
        self.drop2d_S = nn.Dropout2d(0.2)
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        # 畳み込み

        self.conv1_1 = nn.Conv2d(n_input_channels, 16, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv1_2 = nn.Conv2d(16, 32, 3)
        nn.init.kaiming_normal_(self.conv1_2.weight)

        self.conv2_1 = nn.Conv2d(32, 64, 3)
        nn.init.kaiming_normal_(self.conv2_1.weight)
        self.conv2_2 = nn.Conv2d(64, 128, 3)
        nn.init.kaiming_normal_(self.conv2_2.weight)

        self.conv3_1 = nn.Conv2d(128, 256, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv3_2 = nn.Conv2d(128, 128*2,3)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        self.conv3_3 = nn.Conv2d(256, 256*2,3)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        
        self.conv4_1 = nn.Conv2d(64, 128, 5)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv4_2 = nn.Conv2d(128, 128, 5)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        
        """
        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))
        """

        # Advantage
        self.al1 = nn.Linear(6144+self.n_added_input, 1200)
        nn.init.kaiming_normal_(self.al1.weight)

        self.al2 = nn.Linear(1200, 1200)
        nn.init.kaiming_normal_(self.al2.weight)

        self.al3 = nn.Linear(2000, 2000)
        nn.init.kaiming_normal_(self.al3.weight)

        self.al4 = nn.Linear(1200, n_actions*n_atoms)
        nn.init.kaiming_normal_(self.al4.weight)
        
        # State Value
        self.vl1 = nn.Linear(6144+self.n_added_input, 1200)
        nn.init.kaiming_normal_(self.vl1.weight)

        self.vl2 = nn.Linear(1200,1200)
        nn.init.kaiming_normal_(self.vl2.weight)

        self.vl3 = nn.Linear(2000,2000)
        nn.init.kaiming_normal_(self.vl3.weight)

        self.vl4 = nn.Linear(1200, n_atoms)
        nn.init.kaiming_normal_(self.vl4.weight)

    def softmax(self,a):
        a_max = max(a)
        x = np.exp(a-a_max)
        u = np.sum(x)
        return x/u


    def forward(self, state):
        img = state[:,:-self.n_added_input]
        sen = state[:,-self.n_added_input:]
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))

        """
        for layer in self.conv_layers:
            h = self.activation(layer(h))
        """

        h = F.relu(self.conv1_1(img))
        h = self.pool(F.relu(self.conv1_2(h)))
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = self.pool(F.relu(self.conv3_1(h)))
        
        """
        h = F.relu(self.conv3_2(h))
        h = self.pool(F.relu(self.conv3_3(h)))
        
        
        h = self.pool(F.relu(self.conv4_2(h)))
        """
        h = h.view(-1,  6144)#reshape 

        # Advantage
        batch_size = img.shape[0]

        h= torch.cat((torch.reshape(h,(h.shape[0], -1)), sen),dim=1)

        ha = F.relu(self.al1(h))
        #ha = F.relu(self.al2(ha))
        ya = self.al4(ha).reshape((batch_size, self.n_actions, self.n_atoms))

        #分布が見たいとき以外は死ぬほど重くなるのでコメントアウトしといてください
        ####################################################
        """
        x = ya.to('cpu').detach().numpy().copy()
        
        height1 = self.softmax(x[0][0])
        height2 = self.softmax(x[0][2])#なぜか出力画像が左右反転しているのでそれに合わせています
        height3 = self.softmax(x[0][1])


        left = np.arange(len(height1))
        
        width = 1
        
        fig = plt.figure()

        ax1 = fig.add_subplot(311)
        ax1.bar(left+1, height1, color='r', width=width, align='center')
        ax1.axis([0.5,35.5,0,0.1])
        ax1.set_title('straight')
        ax1.set_ylabel('plobability')

        ax2 = fig.add_subplot(312)
        ax2.bar(left+1, height2, color='b', width=width, align='center')
        ax2.axis([0.5,35.5,0,0.1])
        ax2.set_title('right')
        ax2.set_ylabel('plobability')

        ax3 = fig.add_subplot(313)
        ax3.bar(left+1, height3, color='g', width=width, align='center')
        ax3.axis([0.5,35.5,0,0.1])
        ax3.set_title('left')
        ax3.set_ylabel('plobability')

        fig.canvas.draw()

        im = np.array(fig.canvas.renderer._renderer)
        im = cv2.cvtColor(im,cv2.COLOR_RGBA2BGR)
        
        cv2.imshow('distribution',im)

        cv2.waitKey(1)

        plt.close('all')        
        """
        ############################################################
        mean = torch.sum(ya, dim=1,keepdim=True) / self.n_actions
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        """
        h = self.activation(self.main_stream(h.view(batch_size, -1)))
        h_a, h_v = torch.chunk(h, 2, dim=1)
        ya = self.a_stream(h_a).reshape((batch_size, self.n_actions, self.n_atoms))
        
        
        mean = ya.sum(dim=1, keepdim=True) / self.n_actions
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean
        """

        # State value
        """
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)
        """
        hs = F.relu(self.vl1(h))
        #hs = F.relu(self.vl2(hs))
        ys = self.vl4(hs).reshape((batch_size, 1, self.n_atoms))

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(state.device)
        return pfrl.action_value.DistributionalDiscreteActionValue(q, self.z_values)

#実機実験を想定した簡易版Rainbow
class DistributionalDueling(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(
        self,
        n_actions,
        n_input_channels,
        n_added_input,
        n_atoms = 35,
        v_min = -10,
        v_max = 10,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms
        self.img_width = 48
        self.img_height = 27
        self.n_added_input = n_added_input

        super().__init__()

        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)
        self.drop1d_B = nn.Dropout(0.5)
        self.drop1d_S = nn.Dropout(0.5)
        self.drop2d_B = nn.Dropout2d(0.4)
        self.drop2d_S = nn.Dropout2d(0.2)
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        # 畳み込み

        self.conv1_1 = nn.Conv2d(n_input_channels, 16, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv1_2 = nn.Conv2d(16, 32, 3)
        nn.init.kaiming_normal_(self.conv1_2.weight)

        self.conv2_1 = nn.Conv2d(32, 64, 3)
        nn.init.kaiming_normal_(self.conv2_1.weight)
        self.conv2_2 = nn.Conv2d(64, 128, 3)
        nn.init.kaiming_normal_(self.conv2_2.weight)

        self.conv3_1 = nn.Conv2d(128, 256, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv3_2 = nn.Conv2d(128, 128*2,3)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        self.conv3_3 = nn.Conv2d(256, 256*2,3)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        
        self.conv4_1 = nn.Conv2d(64, 128, 5)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv4_2 = nn.Conv2d(128, 128, 5)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        
        """
        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))
        """

        # Advantage
        self.al1 = nn.Linear(6144+self.n_added_input, 1200)
        nn.init.kaiming_normal_(self.al1.weight)

        self.al2 = nn.Linear(1200, 1200)
        nn.init.kaiming_normal_(self.al2.weight)

        self.al3 = nn.Linear(2000, 2000)
        nn.init.kaiming_normal_(self.al3.weight)

        self.al4 = nn.Linear(1200, n_actions*n_atoms)
        nn.init.kaiming_normal_(self.al4.weight)
        
        # State Value
        self.vl1 = nn.Linear(6144+self.n_added_input, 1200)
        nn.init.kaiming_normal_(self.vl1.weight)

        self.vl2 = nn.Linear(1200,1200)
        nn.init.kaiming_normal_(self.vl2.weight)

        self.vl3 = nn.Linear(2000,2000)
        nn.init.kaiming_normal_(self.vl3.weight)

        self.vl4 = nn.Linear(1200, n_atoms)
        nn.init.kaiming_normal_(self.vl4.weight)

    def softmax(self,a):
        a_max = max(a)
        x = np.exp(a-a_max)
        u = np.sum(x)
        return x/u


    def forward(self, state):
        img = state[:,:-self.n_added_input]
        sen = state[:,-self.n_added_input:]
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))

        """
        for layer in self.conv_layers:
            h = self.activation(layer(h))
        """

        h = F.relu(self.conv1_1(img))
        h = self.pool(F.relu(self.conv1_2(h)))
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = self.pool(F.relu(self.conv3_1(h)))
        
        """
        h = F.relu(self.conv3_2(h))
        h = self.pool(F.relu(self.conv3_3(h)))
        
        
        h = self.pool(F.relu(self.conv4_2(h)))
        """
        h = h.view(-1,  6144)#reshape 

        # Advantage
        batch_size = img.shape[0]

        h= torch.cat((torch.reshape(h,(h.shape[0], -1)), sen),dim=1)

        ha = F.relu(self.al1(h))
        ha = F.relu(self.al2(ha))
        ya = self.al4(ha).reshape((batch_size, self.n_actions, self.n_atoms))

        #分布が見たいとき以外は死ぬほど重くなるのでコメントアウトしといてください
        ####################################################
        """
        x = ya.to('cpu').detach().numpy().copy()
        
        height1 = self.softmax(x[0][0])
        height2 = self.softmax(x[0][2])#なぜか出力画像が左右反転しているのでそれに合わせています
        height3 = self.softmax(x[0][1])


        left = np.arange(len(height1))
        
        width = 1
        
        fig = plt.figure()

        ax1 = fig.add_subplot(311)
        ax1.bar(left+1, height1, color='r', width=width, align='center')
        ax1.axis([0.5,35.5,0,0.1])
        ax1.set_title('straight')
        ax1.set_ylabel('plobability')

        ax2 = fig.add_subplot(312)
        ax2.bar(left+1, height2, color='b', width=width, align='center')
        ax2.axis([0.5,35.5,0,0.1])
        ax2.set_title('right')
        ax2.set_ylabel('plobability')

        ax3 = fig.add_subplot(313)
        ax3.bar(left+1, height3, color='g', width=width, align='center')
        ax3.axis([0.5,35.5,0,0.1])
        ax3.set_title('left')
        ax3.set_ylabel('plobability')

        fig.canvas.draw()

        im = np.array(fig.canvas.renderer._renderer)
        im = cv2.cvtColor(im,cv2.COLOR_RGBA2BGR)
        
        cv2.imshow('distribution',im)

        cv2.waitKey(1)

        plt.close('all')        
        """
        ############################################################
        mean = torch.sum(ya, dim=1,keepdim=True) / self.n_actions
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        """
        h = self.activation(self.main_stream(h.view(batch_size, -1)))
        h_a, h_v = torch.chunk(h, 2, dim=1)
        ya = self.a_stream(h_a).reshape((batch_size, self.n_actions, self.n_atoms))
        
        
        mean = ya.sum(dim=1, keepdim=True) / self.n_actions
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean
        """

        # State value
        """
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)
        """
        hs = F.relu(self.vl1(h))
        hs = F.relu(self.vl2(hs))
        ys = self.vl4(hs).reshape((batch_size, 1, self.n_atoms))

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(state.device)
        return pfrl.action_value.DistributionalDiscreteActionValue(q, self.z_values)

class Distributional(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(
        self,
        n_actions,
        n_input_channels,
        n_added_input,
        n_atoms = 35,
        v_min = -10,
        v_max = 10,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms
        self.img_width = 48
        self.img_height = 27
        self.n_added_input = n_added_input

        super().__init__()

        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)
        
        # 同じ環境だから過学習しなさそうだけどしてる
        self.drop1d_B = nn.Dropout(0.5)
        self.drop1d_S = nn.Dropout(0.5)
        self.drop2d_B = nn.Dropout2d(0.4)
        self.drop2d_S = nn.Dropout2d(0.2)
        
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        # 畳み込み

        self.conv1_1 = nn.Conv2d(n_input_channels, 16, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv1_2 = nn.Conv2d(16, 32, 3)
        nn.init.kaiming_normal_(self.conv1_2.weight)

        self.conv2_1 = nn.Conv2d(32, 64, 3)
        nn.init.kaiming_normal_(self.conv2_1.weight)
        self.conv2_2 = nn.Conv2d(64, 128, 3)
        nn.init.kaiming_normal_(self.conv2_2.weight)

        self.conv3_1 = nn.Conv2d(128, 256, 3)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv3_2 = nn.Conv2d(128, 128*2,3)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        self.conv3_3 = nn.Conv2d(256, 256*2,3)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        
        self.conv4_1 = nn.Conv2d(64, 128, 5)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        self.conv4_2 = nn.Conv2d(128, 128, 5)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        
        """
        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))
        """

        # Advantage
        self.al1 = nn.Linear(6144+self.n_added_input, 1200)
        nn.init.kaiming_normal_(self.al1.weight)

        self.al2 = nn.Linear(1200, 1200)
        nn.init.kaiming_normal_(self.al2.weight)

        self.al3 = nn.Linear(2000, 2000)
        nn.init.kaiming_normal_(self.al3.weight)

        self.al4 = nn.Linear(1200, n_actions*n_atoms)
        nn.init.kaiming_normal_(self.al4.weight)


    def forward(self, state):
        img = state[:,:-self.n_added_input]
        sen = state[:,-self.n_added_input:]

        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))

        """
        for layer in self.conv_layers:
            h = self.activation(layer(h))
        """

        h = F.relu(self.conv1_1(img))
        h = self.pool(F.relu(self.conv1_2(h)))
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = self.pool(F.relu(self.conv3_1(h)))
        h = h.view(-1,  6144)#reshape 

        # Advantage
        batch_size = img.shape[0]

        h= torch.cat((torch.reshape(h,(h.shape[0], -1)), sen),dim=1)

        ha = F.relu(self.al1(h))
        ha = F.relu(self.al2(ha))
        ya = self.al4(ha).reshape((batch_size, self.n_actions, self.n_atoms))

        q = F.softmax(ya, dim=2)

        self.z_values = self.z_values.to(state.device)
        return pfrl.action_value.DistributionalDiscreteActionValue(q, self.z_values)

class LSTM_Q_Func(nn.Module):    

    def __init__(self, n_actions, n_input_channels):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        super(LSTM_Q_Func, self).__init__()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)

        # 畳み込み
        #convolution2D(in_channels,out_channels,K_size,stride,pad)
        self.conv1_1 = nn.Conv2d(n_input_channels, 8, 5)#27*48*6=>44*23*8
        nn.init.kaiming_normal_(self.conv1_1.weight)
        
        self.conv1_2 = nn.Conv2d(8, 16, 5)#40*19*16
        nn.init.kaiming_normal_(self.conv1_2.weight)
        #ここでpooling2*2
        self.conv2_1 = nn.Conv2d(16, 32*2, 5)#20*10*16=>16*6*64
        nn.init.kaiming_normal_(self.conv2_1.weight)

        self.conv2_2 = nn.Conv2d(32*2, 64*2, 5)#2*12*128
        nn.init.kaiming_normal_(self.conv2_2.weight)
        #pooling2*2:1*6*128

        # Advantage
        self.al1 = nn.LSTM(128*6, 1200,batch_first=True)
        #nn.init.kaiming_normal_(self.al1.weight)

        self.al2 = nn.Linear(1200, 1200)
        nn.init.kaiming_normal_(self.al2.weight)

        self.al3 = nn.Linear(1200, n_actions)
        nn.init.kaiming_normal_(self.al3.weight)
        
        # State Value
        self.vl1 = nn.LSTM(128*6, 1200,batch_first=True)
        #nn.init.kaiming_normal_(self.vl1.weight)

        self.vl2 = nn.Linear(1200, 1200)
        nn.init.kaiming_normal_(self.vl2.weight)

        self.vl3 = nn.Linear(1200, 1)
        nn.init.kaiming_normal_(self.vl3.weight)

    def forward(self, state):
        #convolution
        h = F.relu(self.conv1_1(state))
        h = self.pool(F.relu(self.conv1_2(h)))
        h = F.relu(self.conv2_1(h))
        h = self.pool(F.relu(self.conv2_2(h)))
        
        h = h.view(-1,6*128)#reshape 
        
        # Advantage
        batch_size = state.shape[0]
        ha = F.relu(self.al1(h))
        ha = F.relu(self.al2(ha))
        ya = self.al3(ha)
        mean = torch.reshape(
            torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State Value
        hs = F.relu(self.vl1(h))
        hs = F.relu(self.vl2(hs))
        ys = self.vl3(hs)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys

        return pfrl.action_value.DiscreteActionValue(q)
