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
	def __init__(self, data_dir, csv_path, way, shot, query, val=1):
		self.data_dir = data_dir
		self.way = way
		self.shot = shot
		self.query = query
		self.val = val
		train_data = pd.read_csv(csv_path)
		self.train_image_names = train_data['filename'].values.flatten()
		train_labels_id = train_data['label'].values.flatten()
		self.labels_list = list(set(train_labels_id))
		self.train_cls_datas = defaultdict(list)
		for name, label in zip(self.train_image_names, train_labels_id):
			self.train_cls_datas[label].append(name)
		self.transform = transforms.Compose([
			filenameToPILImage,
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	def __len__(self):
		return int(len(self.train_image_names) / self.val)
	def __getitem__(self, index):
		classes = np.random.choice(self.labels_list, self.way, replace=False)
		datas, support = [], []
		for index, clss in enumerate(classes):
			select = np.random.choice(self.train_cls_datas[clss], self.shot+self.query)
			datas.extend(select[:self.shot])
			support.extend(select[self.shot:])
		datas = torch.stack([self.transform(os.path.join(self.data_dir, name)) for name in datas], 0)
		support = torch.stack([self.transform(os.path.join(self.data_dir, name)) for name in support], 0)
		return datas, support 

