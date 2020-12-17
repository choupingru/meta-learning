import os 
import argparse
from os.path import join
from pathlib import Path
from importlib import import_module
from dataloader import MetaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn 
from loss import *
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model', default='base', type=str)
parser.add_argument('-save-dir', '--save-dir', default='./results', type=str)
parser.add_argument('-resume', '--resume', default=None, type=str)
parser.add_argument('-test', '--test', action='store_true')
parser.add_argument('-lr', '--lr', default=1e-3, type=float)
parser.add_argument('-val', '--val', default=20, type=int)
parser.add_argument('-way', '--way', default=5, type=int)
parser.add_argument('-shot', '--shot', default=1, type=int)
parser.add_argument('-query', '--query', default=15, type=int)
parser.add_argument('-loss', '--loss', default='l2', type=str)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss = 0

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def train(net, dataloader, optimizer, criterion, episode):
	net.train()
	total_loss = 0
	ce_loss = nn.CrossEntropyLoss()
	for epoch, batch in enumerate(tqdm(dataloader, ncols=50)):
		optimizer.zero_grad()
		supports, queries = batch
		supports = supports.squeeze()
		queries = queries.squeeze()
		supports, queries = supports.to(device), queries.to(device)
		supports_feat, queries_feat = net(supports), net(queries, mode='query')
		dist, label = criterion(supports_feat, queries_feat)
		loss = ce_loss(dist, label)
		total_loss += loss.item()
		loss.backward()
		optimizer.step()

	print('Train Episode : {}, Loss : {}'.format(episode, total_loss / len(dataloader)))

def validation(net, dataloader, way, shot, query, criterion):
	net.eval()
	total_correct = 0
	total_number = 0
	for epoch, batch in enumerate(tqdm(dataloader, ncols=50)):
		with torch.no_grad():
			supports, queries = batch
			supports = supports.squeeze()
			queries = queries.squeeze()
			supports, queries = supports.to(device), queries.to(device)
			supports_feat, queries_feat = net(supports), net(queries, mode='query')
			dist, label = criterion(supports_feat, queries_feat)
			pred = dist.argmax(dim=1)
			total_number += queries_feat.size(0)
		
			total_correct += (pred == label).sum().item()	
	acc =  total_correct / total_number
	print('Validation Acc : %3.4f' % (acc))
	return acc

def main():
	global best_loss
	root = Path('./results')
	if not os.path.isdir(root):
		os.mkdir(root)
	if not os.path.isdir(root / args.model):
		os.mkdir(root / args.model)
	model_root = 'net'
	model = import_module('{}.{}'.format(model_root, args.model))

	net = model.get_model(args.way, args.shot, args.query)
	if args.resume:
		print('Loading weight from' , args.resume)
		checkpoint = torch.load(args.resume)
		net.load_state_dict(checkpoint['state_dict'])

	train_loader = MetaDataset('./hw4_data/train/', './hw4_data/train.csv', way=args.way, shot=args.shot, query=args.query, val=64)
	train_loader = DataLoader(train_loader, batch_size=1, num_workers=8, worker_init_fn=worker_init_fn)
	val_loader = MetaDataset('./hw4_data/val', './hw4_data/val.csv', way=args.way, shot=args.shot, query=args.query, val=16)
	val_loader = DataLoader(val_loader, batch_size=1, num_workers=8, worker_init_fn=worker_init_fn)

	pytorch_total_params = 0
	pytorch_total_params += sum(p.numel() for p in net.parameters())
	print("Total number of params = ", pytorch_total_params)

	# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
	if args.loss == 'l2':
		loss = 'EuclideanLoss'
	elif args.loss == 'cos':
		loss = 'CosineSimilarityLoss'
	else:
		loss = 'ParameterDistance'
	criterion = eval(loss+'(args.way, args.shot, args.query)')
	val_criterion = eval(loss+'(5, 1, 15)')
	
	optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
	net.to(device)
	criterion.to(device)
	val_criterion.to(device)
	for episode in range(args.val * 5):
		train(net, train_loader, optimizer, criterion, episode)
		acc = validation(net, val_loader, 5, 1, 15, val_criterion)
		if acc > best_loss:
			best_loss = acc
			state_dicts = net.state_dict()
			state_dict = {k:v.cpu() for k, v in state_dicts.items()}
			state = {'epoch': episode,
					 'state_dict': state_dict}
			torch.save(state, root / args.model / ('{:>03d}.ckpt'.format(episode)))

if __name__ == '__main__':
	main()
