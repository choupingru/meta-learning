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
parser = argparse.ArgumentParser()
parser.add_argument('-model', '--model', default='base', type=str)
parser.add_argument('-save-dir', '--save-dir', default='./results', type=str)
parser.add_argument('-resume', '--resume', default=None, type=str)
parser.add_argument('-test', '--test', action='store_true')
parser.add_argument('-lr', '--lr', default=1e-4, type=float)
parser.add_argument('-val', '--val', default=100, type=int)
parser.add_argument('-way', '--way', default=5, type=int)
parser.add_argument('-shot', '--shot', default=1, type=int)
parser.add_argument('-query', '--query', default=15, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss = 0



def train(net, dataloader, optimizer, criterion, episode):

	net.train()
	total_loss = 0
	for epoch, batch in enumerate(tqdm(dataloader, ncols=50)):
		
		optimizer.zero_grad()

		supports, queries = batch
		supports = supports.squeeze()
		queries = queries.squeeze()

		supports, queries = supports.to(device), queries.to(device)

		supports_feat, queries_feat = net(supports), net(queries)
		loss = criterion(supports_feat, queries_feat)
		total_loss += loss.item()
		loss.backward()
		optimizer.step()
		break
	
	print('Train Episode : {}, Loss : {}'.format(episode, total_loss))

def validation(net, dataloader, way, shot, query):

	net.eval()
	total_correct = 0
	total_number = 0
	for epoch, batch in enumerate(tqdm(dataloader, ncols=50)):
		
		supports, queries = batch
		supports = supports.squeeze()
		queries = queries.squeeze()
		supports, queries = supports.to(device), queries.to(device)
		supports_feat, queries_feat = net(supports), net(queries)
		supports_feat = supports_feat.view(shot, way, -1).mean(0)

		for index in range(queries_feat.size(0)):
			query_feat = queries_feat[index]
			query_feat = query_feat.view(1, -1)
			dist = torch.cdist(supports_feat, query_feat, p=2).view(-1)
			pred = dist.argmin().item()
			total_number += 1
			if pred == index // query:
				total_correct += 1	



	acc =  total_correct / total_number
	print('Validation Episode : {}, Acc : {%.4f}'.format(episode, acc))
	return acc



def main():
	global best_loss

	root = Path('./results')
	if not os.path.isdir(root):
		os.mkdir(root)
	if not os.path.isdir(root / args.save_dir):
		os.mkdir(root / args.save_dir)

	model_root = 'net'
	model = import_module('{}.{}'.format(model_root, args.model))
	net, criterion = model.get_model(args.way, args.shot, args.query)

	if args.resume:
		print('Loading weight from' , args.resume)
		checkpoint = torch.load(args.resume)
		net.load_state_dict(checkpoint['state_dict'])
		best_loss = checkpoint['best_loss']

	train_loader = MetaDataset('./hw4_data/train/', './hw4_data/train.csv', way=5, shot=1, query=15, val=args.val)
	train_loader = DataLoader(train_loader, batch_size=1)

	val_loader = MetaDataset('./hw4_data/val', './hw4_data/val.csv', way=5, shot=1, query=15, val=args.val)
	val_loader = DataLoader(val_loader, batch_size=1)

	pytorch_total_params = 0
	pytorch_total_params += sum(p.numel() for p in net.parameters())
	print("Total number of params = ", pytorch_total_params)

	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

	net.to(device)
	criterion.to(device)

	for i in range(args.val):
		train(net, train_loader, optimizer, criterion, i)
		acc = validation(net, val_loader, args.way, args.shot, args.query)

		if acc > best_loss:
			best_loss = acc
			state_dicts = net.state_dicts()
			state_dict = {k:v.cpu() for k, v in state_dicts.items()}
			state = {'epoch': episode,
					 'state_dict': state_dict}
			torch.save(state, root / args.model / ('{:>03d}.ckpt'.format(episode)))



if __name__ == '__main__':
	main()
