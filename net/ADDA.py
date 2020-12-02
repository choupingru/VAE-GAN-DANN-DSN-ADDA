import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.animation as animation
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Function
from loss import DSNLoss
import torch.nn.functional as F


class Extractor(nn.Module):

	def __init__(self):
		super(Extractor, self).__init__()
	
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 64, 5, 1, 0),
			nn.ReLU(),
			nn.Conv2d(64, 128, 5, 1, 0),
			nn.ReLU(),
			nn.Conv2d(128, 256, 5, 1, 0),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(256, 512, 5, 1, 0),
			nn.ReLU(), 
			nn.Dropout()
		)


	def forward(self, x):	
		out = self.encoder(x)
		out = out.view(out.size(0), -1)

		#out = self.linear(out)
		return out
	
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.fc = nn.Sequential(
			nn.ReLU(),
			nn.Linear(512*4*4, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 10)

		)
				
	def forward(self, feature):
		return self.fc(feature)

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Sequential(
			nn.ReLU(),
			nn.Linear(512*4*4, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, 2)
		)
	def forward(self, input):
		out = self.fc(input)
		return out


def train(src_loader, tar_loader, net, criterion, epoch, optimizer_src, optimizer_tar, optimizer_cls, optimizer_dis, device='cpu', ep_src=4):
	for key in net:
		net[key].to(device)
		net[key].train()

	start_time = time.time()
	if epoch < ep_src:
		pbar = tqdm(src_loader, ncols=50)
	else:
		pbar = tqdm(zip(src_loader, tar_loader), ncols=50)

	total_loss = 0
	src_domain_total_loss, tar_domain_total_loss = 0, 0
	len_dataloader = len(src_loader)
	correct, src_class_correct, tar_class_correct = 0, 0, 0
	src_total, total = 0, 0
	
	domain_total_loss = 0
	criterion.to(device)
	for i, loaders in enumerate(pbar):

		if epoch < ep_src:
			data_src = loaders
			image_src, label_src = data_src['image'].to(device), data_src['label'].to(device)
			label_src = label_src.view(-1)
			b_src  = image_src.size(0)
			feature = net['src'](image_src)
			src_cls_output = net['cls'](feature)
			cls_loss = criterion(src_cls_output, label_src)
			src_class_correct += (src_cls_output.cpu().detach().numpy().argmax(1) == label_src.cpu().detach().numpy()).sum()
			optimizer_cls.zero_grad()
			optimizer_src.zero_grad()
			cls_loss.backward()
			optimizer_cls.step()
			optimizer_src.step()
		else:
			data_src, data_tar = loaders
			image_tar, label_tar = data_tar['image'].to(device), data_tar['label']
			image_src, label_src = data_src['image'].to(device), data_src['label'].to(device)
			label_src, label_tar = label_src.view(-1), label_tar.view(-1)

			b_src, b_tar = image_src.size(0), image_tar.size(0)
			min_loader = min(len(src_loader), len(tar_loader))
			# train discriminator
			
			optimizer_dis.zero_grad()
			domain_src_feature = net['src'](image_src)
			domain_tar_feature = net['tar'](image_tar)
			feature_combine = torch.cat((domain_src_feature, domain_tar_feature), 0)
			pred_combine = net['dis'](feature_combine.detach())

			domain_src_label, domain_tar_label = torch.ones(b_src).long().to(device), torch.zeros(b_tar).long().to(device)
			label_combine = torch.cat((domain_src_label, domain_tar_label), 0)
			domain_loss = criterion(pred_combine, label_combine)
			
			domain_loss.backward()
			optimizer_dis.step()

			domain_total_loss += domain_loss.item()
			correct += (pred_combine.argmax(1).view(-1) == label_combine.view(-1)).sum()
			total += pred_combine.size(0)
			# train target encoder
			optimizer_tar.zero_grad()

			reverse_tar_label = torch.ones(b_tar).long().to(device)
			domain_tar_feature = net['tar'](image_tar)
			domain_tar_output = net['dis'](domain_tar_feature)

			tar_domain_loss = criterion(domain_tar_output, reverse_tar_label)

			tar_domain_loss.backward()
			optimizer_tar.step()
			

		src_total += b_src
		
	end_time = time.time()
	if epoch < ep_src:
		print('Train Epoch : %d, Accuracy Class Src : %3.5f' % (epoch, src_class_correct / src_total))
	else:
		print('Train Epoch : %d, Domain Loss : %3.5f' % (epoch, domain_total_loss / min_loader))
		print('Accuracy Domain : %d / %d' % (correct, total))


def test(tar_loader, net, epoch, device='cpu', encoder='tar'):
	for key in net:
		net[key].to(device)
		net[key].eval()
	start_time = time.time()
	pbar = tqdm(tar_loader, ncols=50)
	total_loss = 0
	
	tar_class_correct = 0
	tar_total = 0
	with torch.no_grad():
		for i, data_tar in enumerate(pbar):
			
			image_tar, label_tar = data_tar['image'].to(device), data_tar['label']
			label_tar = label_tar.view(-1)
			b_tar = image_tar.size(0)
			tar_cls_output = net[encoder](image_tar)
			tar_cls_output = net['cls'](tar_cls_output)

			tar_class_correct += (tar_cls_output.cpu().detach().numpy().argmax(1) == label_tar.cpu().detach().numpy()).sum()
			tar_total += b_tar

	end_time = time.time()
	print('Testing Accuracy Class %s : %3.5f' % (encoder.upper(), tar_class_correct / tar_total))


def get_model():
	criterion = torch.nn.CrossEntropyLoss(reduction='mean')	
	net = {'src' : Extractor(),
		   'tar' : Extractor(),
		   'cls' : Classifier(),
		   'dis':Discriminator()}
	return net, criterion, train, test

