import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = ConvBlock(3, 64)
		self.conv2 = ConvBlock(64, 64)
		self.conv3 = ConvBlock(64, 64)
		self.conv4 = ConvBlock(64, 64)

		self.mlp = nn.Sequential(
			nn.Linear(1600, 800),
			nn.ReLU(True),
			nn.Linear(800, 400)
		)

	def forward(self, input):
		input = self.conv1(input)
		input = self.conv2(input)
		input = self.conv3(input)
		input = self.conv4(input)
		input = input.view(input.size(0), -1)
		input = self.mlp(input)
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

class Loss(nn.Module):

	def __init__(self, way, shot, query):
		super().__init__()
		self.way = way
		self.shot = shot
		self.query = query
		self.ce = nn.CrossEntropyLoss()

	def forward(self, supports, queries):

		supports = supports.view(self.shot, self.way, -1).mean(0)
		supports = supports.view(self.way, -1)
		total_loss = 0
		for index, query in enumerate(queries):
			query = query.view(1, -1)
			dist = -torch.cdist(supports, query, p=2).view(1, -1)
	
			label = index // 15
			label = torch.LongTensor([label]).cuda()
			total_loss += self.ce(dist, label)

		return total_loss / queries.size(0)


def get_model(way, shot, query):
	net = ConvNet()
	criterion = Loss(way, shot, query)
	return net, criterion











