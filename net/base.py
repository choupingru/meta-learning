import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
	def __init__(self, way, shot, query):
		super().__init__()
		self.way= way
		self.shot = shot
		self.query = query
		self.conv1 = ConvBlock(3, 64)
		self.conv2 = ConvBlock(64, 64)
		self.conv3 = ConvBlock(64, 64)
		self.conv4 = ConvBlock(64, 64)	
		self.mlp = nn.Sequential(
			nn.Linear(1600, 3200),
			nn.ReLU(True),
			nn.Linear(3200, 1024)
		)

	def forward(self, input, mode='support'):
		input = self.conv1(input)
		input = self.conv2(input)
		input = self.conv3(input)
		input = self.conv4(input)
		input = input.view(input.size(0), -1)
		input = self.mlp(input)
		if mode == 'support':
			input = input.view(self.way, self.shot, -1).mean(1)
		else:
			input = input.view(self.query * self.way, -1)
		return input 

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.encode = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, 1, 1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(True),
			nn.MaxPool2d(2)
		)
	def forward(self, input):
		return self.encode(input)

def get_model(way, shot, query):
	net = ConvNet(way, shot, query)
	return net
