import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.encode = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, 1, 1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(True),
			nn.MaxPool2d(2)
		)
	def forward(self, input, mode='support'):
		return self.encode(input)

class Hallucination(nn.Module):

	def __init__(self, m, way, shot, query):
		super().__init__()
		self.m = m
		self.way = way
		self.shot = shot
		self.query = query
		self.conv1 = ConvBlock(3, 64)
		self.conv2 = ConvBlock(64, 64)
		self.conv3 = ConvBlock(64, 64)
		self.conv4 = ConvBlock(64, 64)	
		self.mlp = nn.Sequential(
			nn.Linear(1600, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 512)
		)

		self.augment = nn.Sequential(
			nn.Linear(1024, 2048),
			nn.ReLU(True),
			nn.Linear(2048, 512)
		)

	def forward(self, input, mode='support'):
		input = self.conv1(input)
		input = self.conv2(input)
		input = self.conv3(input)
		input = self.conv4(input)
		input = input.view(input.size(0), -1)
		input = self.mlp(input)
		if mode == 'support':
			sfeat = input.size(-1)
			noise = torch.rand(self.way, self.m, sfeat).cuda()
			input = input.view(self.way, self.shot, sfeat)
			input_repeat = input.repeat(1, int(self.m / self.shot), 1)
			combine = torch.cat((input_repeat, noise), -1)
			combine = combine.view(self.way * self.m, sfeat)
			aug = self.augment(combine)
			aug = aug.view(self.way, self.m, sfeat)
			input = input.view(self.way, self.shot, sfeat)
			support_mean = torch.cat((aug, input), 1).mean(1)
			return support_mean
		input = input.view(self.query * self.way, -1)
		return input 

def get_model(way, shot, query):
	net = Hallucination(50, way, shot, query)
	return net
