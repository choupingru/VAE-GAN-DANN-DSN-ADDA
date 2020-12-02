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
from PIL import Image
class DenseBlock(nn.Module):

	def __init__(self, in_ch, num_layer, growth_rate, out_ch, isDown=False):
		super().__init__()
		inplane = in_ch
		for i in range(num_layer):
			setattr(self, 'dense{}'.format(i), nn.Sequential(
				nn.Conv2d(inplane, growth_rate, 3, 1, 1, bias=False),
				nn.BatchNorm2d(growth_rate),
				nn.ReLU(True)
			))
			inplane += growth_rate
		self.num_layer = num_layer
		self.trans = nn.Sequential(
			nn.Conv2d(in_ch + num_layer * growth_rate, out_ch, 1, 1, 0, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(True)
		)
		self.isDown = isDown
		if self.isDown:
			self.downsample = nn.MaxPool2d(2, 2)

	def forward(self, input):
		for i in range(self.num_layer):
			new_feature = getattr(self, 'dense{}'.format(i))(input)
			input = torch.cat((input, new_feature), 1)
		output = self.trans(input)
		if self.isDown:
			output = self.downsample(output)

		return output

class Encoder(nn.Module):

	def __init__(self, in_ch, hidden_size):
		super().__init__()

		self.basic = nn.Sequential(
			nn.Conv2d(in_ch, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			nn.MaxPool2d(2, 2)
		)

		self.dense1 = DenseBlock(32, 4, 8, 64, isDown=True)
		self.dense2 = DenseBlock(64, 4, 16, 128, isDown=True)
		self.dense3 = DenseBlock(128, 4, 32, 256, isDown=True)
		self.final_conv = nn.Sequential(
			nn.Conv2d(256, 128, 1, 1, 0, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True)
		)
		self.linear = nn.Sequential(
			nn.Linear(128 * 4 * 4, 1024),
			nn.ReLU(True),
			nn.Linear(1024, hidden_size),
			nn.ReLU(True)
		)

	def forward(self, input):

		input = self.basic(input)
		input = self.dense1(input)
		input = self.dense2(input)
		input = self.dense3(input)
		input = self.final_conv(input)
		b, c, w, h = input.size()
		input = input.view(b, -1)
		input = self.linear(input)

		return input

class LatentZ(nn.Module):

	def __init__(self, hidden_size, latent_size):
		super().__init__()

		self.mu = nn.Linear(hidden_size, latent_size)
		self.logvar = nn.Linear(hidden_size, latent_size)
		
	def forward(self, input):
		mu = self.mu(input)
		logvar = self.logvar(input)
		std = logvar.mul(0.5).exp_()
		if torch.cuda.is_available():
			eps = torch.cuda.FloatTensor(std.size()).normal_()
		else:
			eps = torch.FloatTensor(std.size()).normal_()
		
		return eps.mul(std) + mu, mu, logvar

class Decoder(nn.Module):

	def __init__(self, latent_size):
		super().__init__()
		self.dense1 = DenseBlock(16, 2, 8, 32)
		self.dense2 = DenseBlock(32, 2, 16, 32)
		self.dense3 = DenseBlock(32, 2, 16, 3)
	
	def forward(self, input):
		b, c = input.size()
		input = input.view(b, -1, 4, 4)
		input = F.interpolate(input, scale_factor=2)
		input = self.dense1(input)
		input = F.interpolate(input, scale_factor=2)
		input = self.dense2(input)
		input = F.interpolate(input, scale_factor=2)
		input = self.dense3(input)
		input = F.interpolate(input, scale_factor=2)

		input = torch.sigmoid(input)
		return input

class VAE(nn.Module):

	def __init__(self, in_ch):
		super().__init__()

		self.encoder = Encoder(3, 512)
		self.latent_layer = LatentZ(512, 256)
		self.decoder = Decoder(128)

	def forward(self, input):

		hidden_space = self.encoder(input)
		latent_space, mu, logvar = self.latent_layer(hidden_space)
		recon = self.decoder(latent_space)
		b, c, w, h = recon.size()
		
		recon = recon.view(b, -1)
		return recon, mu, logvar


def train(train_loader, net, criterion, epoch, optimizer_VAE, device='cpu'):
	net['VAE'] = net['VAE'].to(device)
	net['VAE'].train()

	pbar = tqdm(train_loader, ncols=50)

	preds, labels = [], []
	total_loss = 0

	for i, datas in enumerate(pbar):

		image = datas['image'].to(device)
		b = image.size(0)
		recon, mu, logvar = net['VAE'](image)
		image = image.view(b, -1)
		recon_loss = criterion(recon, image)
		kl_loss =  (-0.5 * (1 + logvar - mu**2 - logvar.exp())).mean()
		loss = recon_loss + kl_loss * 1e-5
		optimizer_VAE.zero_grad()
		loss.backward()
		optimizer_VAE.step()
		
		total_loss += loss.item()

		if i == 0:
			for index, img in enumerate(recon):
				
				img = img.view(3, 64, 64).permute(1, 2, 0)
				img = img.cpu().detach().numpy()
				img = img * 255
				img = Image.fromarray(img.astype(np.uint8))
				img.save('./recon/{}.jpg'.format(index))


	end_time = time.time()
	print('Train Epoch : %d, Recon Loss : %3.5f, KL Loss : %3.5f, Total Loss : %3.f ' % (epoch, recon_loss, kl_loss, recon_loss + kl_loss))
	

def get_model():
	criterion = torch.nn.MSELoss(reduction='mean')	
	net = {}
	net['VAE'] = VAE(3)
	return net, criterion, train

