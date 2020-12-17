import torch
import torch.nn as nn

class EuclideanLoss(nn.Module):

	def __init__(self, way, shot, query):
		super().__init__()
		self.way = way
		self.shot = shot
		self.query = query
	def forward(self, supports, queries):
		dist = -torch.cdist(supports, queries, p=2).T
		label = torch.LongTensor([i // self.query for i in range(queries.size(0))]).cuda()
		return dist, label

class CosineSimilarityLoss(nn.Module):

	def __init__(self, way, shot, query):
		super().__init__()
		self.way = way
		self.shot = shot
		self.query = query
		self.sim = nn.CosineSimilarity(dim=1)

	def forward(self, supports, queries):
		fsize = supports.size(-1)
		supports = supports.view(self.way, 1, -1).expand(self.way, self.query * self.way, fsize).contiguous().view(-1, fsize)
		queries = queries.view(1, self.query * self.way, -1).expand(self.way, self.query * self.way, fsize).contiguous().view(-1, fsize)
		cosine_similarity = self.sim(supports, queries)
		cosine_similarity = cosine_similarity.view(self.way, -1).T * self.way
		label = torch.LongTensor([i // self.query for i in range(self.query * self.way)]).cuda()
		return cosine_similarity, label

class DistanceModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.linear = nn.Sequential(
			nn.Linear(2, 16),
			nn.ReLU(True),
			nn.Linear(16, 32),
			nn.ReLU(True),
			nn.Linear(32, 1)
		)

	def forward(self, score):
		return self.linear(score)



