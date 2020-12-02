
import numpy as np
import torch
import torch.nn as nn
import os 
from tqdm import tqdm
from importlib import import_module
from torch.utils.data import DataLoader
import argparse
import time
from os.path import join
from PIL import Image
from sklearn.manifold import TSNE
import net.ADDA as adda
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataloader import *

src = 'mnistm'
tar = 'svhn'
ep = '150'
parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='base')
parser.add_argument('--src', '-src', metavar='SRC', default='undefined')
parser.add_argument('--tar', '-tar', metavar='TAR', default='undefined')
parser.add_argument('--ep', '-ep', metavar='EP', default='undefined')

if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_root = 'net'
	model = import_module('{}.{}'.format(model_root, args.model))
	function = model.get_model()
	net = function[0]

	state = torch.load('./results/'+args.src+'2'+args.tar+'/ADDA/'+str(ep)+'_tar.ckpt')

	encoder = net['tar']

	encoder.load_state_dict(state['state_dict'])
	encoder.to(device)

	src_loader = get_dataset(dataset_name='DANN', domain=args.src, mode='test')
	tar_loader = get_dataset(dataset_name='DANN', domain=args.tar, mode='test')
	src_dataloader = DataLoader(src_loader, batch_size=1)
	tar_dataloader = DataLoader(tar_loader, batch_size=1)

	start_time = time.time()
	encoder.eval()
	preds, labels_cls, labels_dom = [], [], []

	for index, dataloader in enumerate([src_dataloader, tar_dataloader]):
		pbar = tqdm(dataloader, ncols=50)
		for i, datas in enumerate(pbar):
			image = datas['image'].to(device)
			label = datas['label']
			label = label.view(-1)
			image = image.view(1, 3, 28, 28)
			pred = encoder(image)
			preds.append(pred.cpu().detach().numpy())
			labels_cls.append(label)
			labels_dom.append(np.array([index]))

	end_time = time.time()
	preds = np.concatenate([p for p in preds], 0)
	labels_cls = np.concatenate([l for l in labels_cls], 0)
	labels_dom = np.concatenate([l for l in labels_dom], 0)
	if not os.path.isdir('./tsne'):
		os.mkdir('./tsne')
	if not os.path.isdir('./tsne/'+args.model):
		os.mkdir('./tsne/'+args.model)

	np.save('./tsne/'+args.model+'/features'+args.src+'2'+args.tar+'.npy', preds)
	np.save('./tsne/'+args.model+'/labels'+args.src+'2'+args.tar+'_cls.npy', labels_cls)
	np.save('./tsne/'+args.model+'/labels'+args.src+'2'+args.tar+'_dom.npy', labels_dom)
	features = np.load('./tsne/'+args.model+'/features'+args.src+'2'+args.tar+'.npy', allow_pickle=True)
	labels_cls = np.load('./tsne/'+args.model+'/labels'+args.src+'2'+args.tar+'_cls.npy', allow_pickle=True)
	labels_dom = np.load('./tsne/'+args.model+'/labels'+args.src+'2'+args.tar+'_dom.npy', allow_pickle=True)
	features = PCA(n_components=100).fit_transform(features)
	tsne_features_fit = TSNE(n_components=2).fit_transform(features)
	np.save('./tsne/'+args.model+'/tsne_features_fit'+args.src+'2'+args.tar+'.npy', tsne_features_fit)
	tsne_features_fit = np.load('./tsne/'+args.model+'/tsne_features_fit'+args.src+'2'+args.tar+'.npy', allow_pickle=True)

	x, y = zip(*tsne_features_fit)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x, y, c=labels_cls, cmap='jet', s=1)
	plt.show()
	x, y = zip(*tsne_features_fit)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x, y, c=labels_dom, cmap='jet', s=1)
	plt.show()