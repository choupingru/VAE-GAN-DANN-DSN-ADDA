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


class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

class conv3x3(nn.Module):

	def __init__(self, in_ch, out_ch, stride=1, act='mish', dilation=1):
		super().__init__()

		self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, dilation=dilation, padding=dilation, bias=False)
		self.norm = nn.BatchNorm2d(out_ch)
		self.act = nn.ReLU(inplace=True)

	def forward(self, input):
		return self.act(self.norm(self.conv(input)))

class conv1x1(nn.Module):

	def __init__(self, in_ch, out_ch, stride=1, act='mish'):
		super().__init__()

		self.conv = nn.Conv2d(in_ch, out_ch, 1, stride, padding=0, bias=False)
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
		self.relu = nn.ReLU(True)

	def forward(self, input):
		identity = self.downsample(input)
		out = self.conv1(input)
		out = self.conv2(out)
		out = self.conv3(out)
		out = out + identity
		out = self.relu(out)
		return out

class Extractor(nn.Module):

	def __init__(self, in_ch, mode='easy'):
		super(Extractor, self).__init__()
		self.basci_conv = conv3x3(3, 32)
		self.maxpool = nn.MaxPool2d(2, 2)
		self.conv2 = ResidualBlock(32)
		self.conv3 = ResidualBlock(64)
		self.linear = nn.Sequential(
			nn.Linear(128 * 7 * 7, 2048),
			nn.ReLU(True)
		)
	def forward(self, x):
		out = self.basci_conv(x)
		out = self.maxpool(out)
		out = self.conv2(out)
		out = self.maxpool(out)
		out = self.conv3(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out
		

class ClassClassifier(nn.Module):

	def __init__(self):
		super(ClassClassifier, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(2048, 512),
			nn.ReLU(True),
			nn.Linear(512, 10)
		)

	def forward(self, feature):
		return F.log_softmax(self.fc(feature), 1)

class DomainClassifier(nn.Module):

	def __init__(self):
		super(DomainClassifier, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(2048, 1024),
			nn.ReLU(True),
			nn.Linear(1024, 512),
			nn.ReLU(True),
			nn.Linear(512, 2)
		)	

	def forward(self, feature):
		return F.log_softmax(self.fc(feature), 1)

class DANN(nn.Module):

	def __init__(self):
		super().__init__()
		self.backbone = Extractor(3)
		self.cls_clf = ClassClassifier()
		self.dom_clf = DomainClassifier()


	def forward(self, input, alpha):
		input = self.backbone(input)
		reverse_input = ReverseLayerF.apply(input, alpha)
		cls_output = self.cls_clf(input)
		dom_output = self.dom_clf(reverse_input)
		return cls_output, dom_output



def train(src_loader, tar_loader, net, criterion, epoch, optimizer_DANN, device='cpu'):
	net['DANN'] = net['DANN'].to(device)
	start_time = time.time()
	net['DANN'].train()
	pbar = tqdm(zip(src_loader, tar_loader), ncols=50)
	total_loss = 0
	src_domain_total_loss, tar_domain_total_loss = 0, 0
	len_dataloader = len(src_loader)
	src_domain_correct, tar_domain_correct, src_class_correct, tar_class_correct = 0, 0, 0, 0
	src_total, tar_total = 0, 0
	min_loader = min([len(src_loader), len(tar_loader)])
	for i, (data_src, data_tar) in enumerate(pbar):
		
		p = float(i + 100 * len_dataloader) / epoch / len_dataloader
		alpha = 2. / (1. + np.exp(-10 * p)) - 1

		image_src, label_src = data_src['image'].to(device), data_src['label'].to(device)
		image_tar, label_tar = data_tar['image'].to(device), data_tar['label']
		label_src = label_src.view(-1)
		label_tar = label_tar.view(-1)

		b_src, b_tar = image_src.size(0), image_tar.size(0)
		domain_src_label, domain_tar_label = torch.ones(b_src).long().to(device), torch.zeros(b_tar).long().to(device)

		src_cls_output, domain_src_output = net['DANN'](image_src, alpha)
		tar_cls_output, domain_tar_output = net['DANN'](image_tar, alpha)

		cls_loss = criterion(src_cls_output, label_src)
		src_domain_loss = criterion(domain_src_output, domain_src_label)
		tar_domain_loss = criterion(domain_tar_output, domain_tar_label)
		domain_loss = src_domain_loss + tar_domain_loss
		loss = cls_loss + domain_loss 

		optimizer_DANN.zero_grad()
		loss.backward()
		optimizer_DANN.step()
		total_loss += loss.item()
		src_domain_total_loss += src_domain_loss.item()
		tar_domain_total_loss += tar_domain_loss.item()
		
		src_domain_correct += (domain_src_output.cpu().detach().numpy().argmax(1) == domain_src_label.cpu().detach().numpy()).sum()
		tar_domain_correct += (domain_tar_output.cpu().detach().numpy().argmax(1) == domain_tar_label.cpu().detach().numpy()).sum()
		src_class_correct += (src_cls_output.cpu().detach().numpy().argmax(1) == label_src.cpu().detach().numpy()).sum()
		tar_class_correct += (tar_cls_output.cpu().detach().numpy().argmax(1) == label_tar.cpu().detach().numpy()).sum()
		src_total += b_src
		tar_total += b_tar
		

	end_time = time.time()
	print('Train Epoch : %d, Domain Src Loss : %3.5f, Domain Tar Loss : %3.5f, Total Loss : %3.5f ' % (epoch, src_domain_total_loss / min_loader, tar_domain_total_loss / min_loader, total_loss / min_loader))
	print('Accuracy Domain Src : %3.5f, Accuracy Domain Tar : %3.5f' % (src_domain_correct / src_total, tar_domain_correct / tar_total))
	print('Accuracy Class Src : %3.5f, Accuracy Class Tar : %3.5f' % (src_class_correct / src_total, tar_class_correct / tar_total))

def get_model():
	criterion = torch.nn.NLLLoss(reduction='mean')	
	net = {}
	net['DANN'] = DANN()
	return net, criterion, train

