import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)
filenameToPILImage = lambda x: Image.open(x)
	
class MetaDataset(Dataset):

	def __init__(self, data_dir, csv_path, way, shot, query, val):

		self.data_dir = data_dir
		self.way = way
		self.shot = shot
		self.query = query
		self.val = val
		
		train_data = pd.read_csv(csv_path)
		self.train_image_names = train_data['filename'].values.flatten()
		train_labels_id = train_data['label'].values.flatten()
		labels_set = set(train_labels_id)
		# self.train_labels_id_to_num = {id : index for index, id in enumerate(labels_set)}
		self.train_cls_datas = defaultdict(list)
		self.train_cls_datas_len = defaultdict(int)
		# train
		for name, label in zip(self.train_image_names, train_labels_id):
			self.train_cls_datas[label].append(name)
		for key in self.train_cls_datas:
			self.train_cls_datas_len[key] = len(self.train_cls_datas[key])

		self.class_combination = list(combinations(list(labels_set), self.way))
		self.size = len(self.class_combination)

		self.transform = transforms.Compose([
			filenameToPILImage,
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

	def __len__(self):
		return int(len(self.train_image_names) / self.val)

	def __getitem__(self, index):
		# select 5 class
		classes = self.class_combination[np.random.randint(0, self.size)]
		datas, target = [], []
		support, query = [], []
		for index, clss in enumerate(classes):
			select_index = np.random.choice(self.train_cls_datas_len[clss], self.shot+self.query)
			cnt = 0
			for _ in range(self.shot):
				datas.append(self.train_cls_datas[clss][select_index[cnt]])
				cnt += 1
			for _ in range(self.query):
				support.append(self.train_cls_datas[clss][select_index[cnt]])
				cnt += 1
		datas = [self.transform(os.path.join(self.data_dir, name)) for name in datas]
		datas = torch.stack(datas, 0)
		support = [self.transform(os.path.join(self.data_dir, name)) for name in support]
		support = torch.stack(support, 0)
		return datas, support 



# train_df, val_df = csv_split_train_val('./hw4_data/train.csv', 0.8)

if __name__ == '__main__':
	train_loader = MetaDataset('./hw4_data/train', './hw4_data/train.csv', way=5, shot=1, query=15)
	train_loader = DataLoader(train_loader, batch_size=1)


	for index, batch in enumerate(train_loader):
		timg, vimg = batch
		# print('Start')
		# print(timg.size(), " / ", tlab.size())
		# print(vimg.size(), " / ", vlab.size())
		# print()
		print('Train :')
		print(timg.size())
		print('Val : ')
		print(vimg.size())
		
		if index == 5:
			break



