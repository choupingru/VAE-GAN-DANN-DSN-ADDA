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

class ReverseLayerF(Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)
	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha
		return output, None

class conv5x5(nn.Module):

	def __init__(self, in_ch, out_ch, stride=1, act='mish'):
		super().__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 5, stride, padding=2)
		self.norm = nn.BatchNorm2d(out_ch)
		self.act = nn.ReLU(inplace=True)
	def forward(self, input):
		return self.act(self.norm(self.conv(input)))

class conv3x3(nn.Module):
	def __init__(self, in_ch, out_ch, stride=1, act='mish', dilation=1):
		super().__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 5, stride, dilation=dilation, padding=2)
		self.norm = nn.BatchNorm2d(out_ch)
		self.act = nn.ReLU(inplace=True)
	def forward(self, input):
		return self.act(self.norm(self.conv(input)))


class conv1x1(nn.Module):

	def __init__(self, in_ch, out_ch, stride=1, act='mish'):
		super().__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1, stride, padding=0)
		self.norm = nn.BatchNorm2d(out_ch)
		self.act = nn.ReLU(inplace=True)
	def forward(self, input):
		return self.act(self.norm(self.conv(input)))

class ResidualBlock(nn.Module):

	def __init__(self, in_ch, residual=True):
		super().__init__()
		self.residual = residual
		self.conv1 = conv1x1(in_ch, in_ch // 2)
		self.conv2 = conv3x3(in_ch // 2, in_ch)
		self.conv3 = nn.Sequential(
			nn.Conv2d(in_ch, in_ch * 2, 1, 1, 0),
			nn.BatchNorm2d(in_ch * 2)
		)
		self.downsample = nn.Sequential(
			nn.Conv2d(in_ch, in_ch * 2, 1, 1, 0),
			nn.BatchNorm2d(in_ch * 2)
		)
		self.relu = nn.ReLU()

	def forward(self, input):
		identity = self.downsample(input)
		out = self.conv1(input)
		out = self.conv2(out)
		out = self.conv3(out)
		out = out + identity
		out = self.relu(out)
		return out


class SharedExtractor(nn.Module):

	def __init__(self, in_ch):
		super(SharedExtractor, self).__init__()
	
		self.conv1 = conv5x5(3, 16)
		self.conv2 = conv3x3(16, 32)
		self.maxpool1 = nn.MaxPool2d(2, 2)
		self.conv3 = conv5x5(32, 48)
		self.conv4 = conv3x3(48, 64)
		
		self.linear = nn.Sequential(
			nn.Linear(64 * 14 * 14, 512),
			nn.BatchNorm1d(512),
			nn.ReLU()
		)
	

	def forward(self, x):	
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.maxpool1(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out
		
class PrivateExtractor(nn.Module):

	def __init__(self, in_ch):
		super(PrivateExtractor, self).__init__()
	
		self.conv1 = conv5x5(3, 16)
		self.conv2 = conv3x3(16, 32)
		self.maxpool1 = nn.MaxPool2d(2, 2)
		self.conv3 = conv5x5(32, 48)
		self.conv4 = conv3x3(48, 64)
		
		self.linear = nn.Sequential(
			nn.Linear(64 * 14 * 14, 512),
			nn.BatchNorm1d(512),
			nn.ReLU()
		)
	
		
	def forward(self, x):	
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.maxpool1(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

class Classifier(nn.Module):

	def __init__(self):
		super(Classifier, self).__init__()
		self.fc = nn.Sequential(
			
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 10)
		)
				
	def forward(self, feature):
		return self.fc(feature)

class DomainClassifier(nn.Module):

	def __init__(self):
		super(DomainClassifier, self).__init__()
		self.fc = nn.Sequential(
			
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 2),
		)
				
	def forward(self, feature):
		return self.fc(feature)

class Decoder(nn.Module):

	def __init__(self):
		super().__init__()
		self.linear = nn.Sequential(
			
			nn.Linear(512, 16 * 7 * 7),
			nn.ReLU()
		)
		self.conv1 = conv3x3(16, 32)
		self.upsample = nn.Upsample(scale_factor=2)
		self.conv2 = conv3x3(32, 64)
		self.conv3 = nn.Conv2d(64, 3, 1, 1, 0)

	def forward(self, input):
		bs = input.size(0)
		input = self.linear(input)
		input = input.view(bs, -1, 7, 7)
		input = self.conv1(input)
		input = self.upsample(input)
		input = self.conv2(input)
		input = self.upsample(input)
		input = self.conv3(input)
		return input


class DSN(nn.Module):

	def __init__(self):
		super().__init__()
		self.shared_encoder = SharedExtractor(3)
		self.src_encoder = PrivateExtractor(3)
		self.tar_encoder = PrivateExtractor(3)
		self.cls_clf = Classifier()
		self.dom_clf = DomainClassifier()
		self.decoder = Decoder()

	def forward(self, input, alpha, mode='src'):	
		shared_input = self.shared_encoder(input)
		input_private = self.src_encoder(input) if mode == 'src' else self.tar_encoder(input)
		class_output = self.cls_clf(shared_input)
		reverse_input = ReverseLayerF.apply(shared_input, alpha)
		dom_output = self.dom_clf(reverse_input)
		combine = shared_input + input_private
		recon = self.decoder(combine)
		return shared_input, input_private, recon, dom_output, class_output
		


def train(src_loader, tar_loader, net, criterion, epoch, optimizer_DSN, device='cpu'):
	net['DSN'] = net['DSN'].to(device)
	start_time = time.time()
	net['DSN'].train()
	pbar = tqdm(zip(src_loader, tar_loader), ncols=50)
	total_loss = 0
	task_total_loss, recon_total_loss, difference_total_loss, domain_total_loss = 0, 0, 0, 0
	min_loader = len(src_loader)
	src_domain_correct, tar_domain_correct, src_class_correct, tar_class_correct = 0, 0, 0, 0
	src_total, tar_total = 0, 0
	min_loader = min([len(src_loader), len(tar_loader)])
	for i, (data_src, data_tar) in enumerate(pbar):
		
		p = float(i + 100 * min_loader) / epoch / min_loader
		alpha = 2. / (1. + np.exp(-10 * p)) - 1

		image_src, label_src = data_src['image'].to(device), data_src['label'].to(device)
		image_tar, label_tar = data_tar['image'].to(device), data_tar['label']
		label_src = label_src.view(-1)
		label_tar = label_tar.view(-1)

		b_src, b_tar = image_src.size(0), image_tar.size(0)
		domain_src_label, domain_tar_label = torch.ones(b_src).long().to(device), torch.zeros(b_tar).long().to(device)

		src_shared, src_private, src_recon, domain_src_output, src_cls_output = net['DSN'](image_src, alpha, mode='src')
		src_task_loss, src_recon_loss, src_difference_loss, src_domain_loss = criterion(domain_src_output, src_private, src_shared, src_recon, image_src, label_src, src_cls_output)
		src_loss = src_task_loss + src_recon_loss + src_difference_loss + src_domain_loss
		
		tar_shared, tar_private, tar_recon, domain_tar_output, tar_cls_output = net['DSN'](image_tar, alpha, mode='tar')
		tar_recon_loss, tar_difference_loss, tar_domain_loss = criterion(domain_tar_output, tar_private, tar_shared, tar_recon, image_tar, mode='tar')
		tar_loss = tar_recon_loss + tar_difference_loss + tar_domain_loss
		
		optimizer_DSN.zero_grad()
		combine_loss = src_loss + tar_loss
		combine_loss.backward()
		optimizer_DSN.step()

		total_loss += (src_loss + tar_loss).item()
		domain_total_loss += (src_domain_loss + tar_domain_loss).item()
		task_total_loss += src_task_loss.item()
		recon_total_loss += (src_recon_loss + tar_recon_loss).item()
		difference_total_loss += (src_difference_loss + tar_difference_loss).item()
		
		src_domain_correct += (domain_src_output.cpu().detach().numpy().argmax(1) == domain_src_label.cpu().detach().numpy()).sum()
		tar_domain_correct += (domain_tar_output.cpu().detach().numpy().argmax(1) == domain_tar_label.cpu().detach().numpy()).sum()
		src_class_correct += (src_cls_output.cpu().detach().numpy().argmax(1) == label_src.cpu().detach().numpy()).sum()
		tar_class_correct += (tar_cls_output.cpu().detach().numpy().argmax(1) == label_tar.cpu().detach().numpy()).sum()
		src_total += b_src
		tar_total += b_tar
		

	end_time = time.time()
	print('Train Epoch : %d, Task Loss : %3.5f, Recon Loss : %3.5f, Diff Loss : %3.5f, Domain Loss : %3.5f, Total Loss : %3.5f' % (epoch, task_total_loss / min_loader, recon_total_loss / min_loader, difference_total_loss / min_loader, domain_total_loss / min_loader, total_loss / min_loader))
	print('Accuracy Domain Src : %3.5f, Accuracy Domain Tar : %3.5f' % (src_domain_correct / src_total, tar_domain_correct / tar_total))
	print('Accuracy Class Src : %3.5f, Accuracy Class Tar : %3.5f' % (src_class_correct / src_total, tar_class_correct / tar_total))

def test(test_loader, net, epoch, device='cpu'):
	net['DSN'] = net['DSN'].to(device)
	start_time = time.time()
	net['DSN'].eval()
	pbar = tqdm(test_loader, ncols=50)
	total_loss = 0
	
	tar_class_correct = 0
	tar_total = 0
	with torch.no_grad():
		for i, data_tar in enumerate(pbar):
			
			p = float(i + 100 * len(test_loader)) / epoch / len(test_loader)
			alpha = 2. / (1. + np.exp(-10 * p)) - 1
			image_tar, label_tar = data_tar['image'].to(device), data_tar['label']
			label_tar = label_tar.view(-1)
			b_tar = image_tar.size(0)
			tar_shared, tar_private, tar_recon, domain_tar_output, tar_cls_output = net['DSN'](image_tar, alpha, mode='tar')
			tar_class_correct += (tar_cls_output.cpu().detach().numpy().argmax(1) == label_tar.cpu().detach().numpy()).sum()
			tar_total += b_tar

	end_time = time.time()
	print('Testing Accuracy Class Tar : %3.5f' % (tar_class_correct / tar_total))

def get_model():
	criterion = DSNLoss()
	net = {}
	net['DSN'] = DSN()
	return net, criterion, train, test

