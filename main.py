import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from importlib import import_module
from PIL import Image
from os.path import join
from dataloader import *
from torch.utils.data import DataLoader
import time
from pathlib import Path
from config import config

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=100, type=int, metavar='N')
parser.add_argument('--b', '--batch-size', default=4, type=int,metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,metavar='LR')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--save-dir', default="./results", type=str, metavar='SAVE')
parser.add_argument('--test', default=0, type=int, metavar='TEST')
parser.add_argument('--save-freq', default='1', type=int, metavar='S')
parser.add_argument('--task', default='VAE', type=str)
parser.add_argument('--src', default='undefined', type=str)
parser.add_argument('--tar', default='undefined', type=str)
parser.add_argument('--ep_src', default=-1, type=int)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_loss = 10000000

def main():

	global best_loss
	root = Path(args.save_dir)
	if args.src != 'undefined':
		if not os.path.isdir(root / (args.src+'2'+args.tar)):
			os.mkdir(root / (args.src+'2'+args.tar))
		if not os.path.isdir(root / (args.src+'2'+args.tar) / args.model):
			os.mkdir(root / (args.src+'2'+args.tar) / args.model)
	else:
		if not os.path.isdir(root / args.model):
			os.mkdir(root / args.model)
	model_root = 'net'
	model = import_module('{}.{}'.format(model_root, args.model))

	functions = model.get_model()

	if len(functions) == 3:
		net, criterion, train = functions
		test = None
	else:
		net, criterion, train, test = functions
	

	if args.resume:
		for key in net:
			name = args.resume.replace('.ckpt', '_'+key+'.ckpt')
			print('Loading weight from' , name)
			checkpoint = torch.load(name)
			net[key].load_state_dict(checkpoint['state_dict'])
			best_loss = checkpoint['best_loss']
			start_epoch = checkpoint['epoch'] + 1
	else:
		start_epoch = 1

	loader_table = {}
	if args.src != 'undefined' and args.tar != 'undefined':
		src_loader = get_dataset(dataset_name=args.task, domain=args.src)
		tar_loader = get_dataset(dataset_name=args.task, domain=args.tar)
		test_loader = get_dataset(dataset_name=args.task, domain=args.tar, mode='test')
		loader_table['src_loader'] = DataLoader(src_loader, batch_size=args.b , shuffle=True)
		loader_table['tar_loader'] = DataLoader(tar_loader, batch_size=args.b, shuffle=True)
		test_loader = DataLoader(tar_loader, batch_size=args.b, shuffle=True)
	else:
		train_loader = get_dataset(dataset_name=args.task)
		loader_table['train_loader'] = DataLoader(train_loader, batch_size=args.b, shuffle=True)
	optimizer_table = {}
	for key in net:
		optimizer_table[key] = torch.optim.Adam(net[key].parameters(), lr=config[args.model][key], weight_decay=1e-4)
	assert isinstance(net, dict), 'Net must be a dictionary'
	pytorch_total_params = 0
	for key in net:
		pytorch_total_params += sum(p.numel() for p in net[key].parameters())
	print("Total number of params = ", pytorch_total_params)
	criterion = criterion.to(device)
	### setting the training parameters ###
	training_param = {}
	training_param['criterion'] = criterion
	for key in loader_table:
		training_param[key] = loader_table[key]
	for key in optimizer_table:
		training_param['optimizer_' + key] =optimizer_table[key]
	training_param['net'] = net
	training_param['device'] = device
	if args.ep_src != -1:
		training_param['ep_src'] = args.ep_src
	### end of setting ###
	for epoch in range(start_epoch, args.epochs + 1):
		training_param['epoch'] = epoch
		if epoch == args.ep_src:
			print('\nTarget Encoder Loading Weight from Source Encoder\n')
			net['src'].eval()
			net['cls'].eval()
			net['tar'].load_state_dict(net['src'].state_dict())
		### training
		train(**training_param)
		### if have testing function, then test
		if hasattr(test, '__call__') and args.ep_src <= epoch:
			test(test_loader, net, epoch, device)
		### save checkpoint
		state_dicts = {}
		for key in net:
			state_dicts[key] = net[key].state_dict()
		for key in state_dicts:
			state_dict = {k:v.cpu() for k, v in state_dicts[key].items()}
			state = {'epoch': epoch,
					 'state_dict': state_dict}
			if args.src == 'undefined' or args.tar == 'undefined':
				torch.save(state, root / args.model / ('{:>03d}_'.format(epoch)+key+'.ckpt'))
			else:
				torch.save(state, root / (args.src + '2' + args.tar) / args.model / ('{:>03d}_'.format(epoch)+key+'.ckpt'))
		
if __name__ == '__main__':
	main()
