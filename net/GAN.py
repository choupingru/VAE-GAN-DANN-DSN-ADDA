import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.animation as animation
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class Generator(nn.Module):

	def __init__(self, latent_size):

		super(Generator, self).__init__()
		self.latent_dim = latent_size
		
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, self.latent_dim * 2, 4, 1, 0),
			nn.BatchNorm2d(self.latent_dim * 2),
			nn.ReLU(True),
			nn.ConvTranspose2d(self.latent_dim * 2, self.latent_dim * 4, 4, 2, 1),
			nn.BatchNorm2d(self.latent_dim * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(self.latent_dim * 4, self.latent_dim * 2, 4, 2, 1),
			nn.BatchNorm2d(self.latent_dim * 2),
			nn.ReLU(True),
			nn.ConvTranspose2d(self.latent_dim * 2, self.latent_dim, 4, 2, 1),
			nn.BatchNorm2d(self.latent_dim),
			nn.ReLU(True),
			nn.ConvTranspose2d(self.latent_dim, 3, 4, 2, 1),
			nn.Tanh(),
			)

	def forward(self, x):
		x = x.view(x.size(0), self.latent_dim, 1, 1)
		img = self.deconv(x)
		img = img.view(x.size(0), 3, 64, 64)
		return img

class Discriminator(nn.Module):

	def __init__(self):
		super(Discriminator, self).__init__()

		self.cnn = nn.Sequential(
			# 3 * 64 * 64
			nn.Conv2d(3, 32,3, 2, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2),
			# 32 * 64 * 64
			nn.Conv2d(32, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			)
		self.fc = nn.Sequential(
			nn.Linear(64 * 16 * 16, 512),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
			nn.Linear(512, 1),
			nn.Sigmoid()
			)

	def forward(self, img):
		img = self.cnn(img)
		img = img.view(img.size(0), -1)
		res = self.fc(img)

		return res

def train(train_loader, net, criterion, epoch, optimizer_generator, optimizer_discriminator, fixed_noise=None, device='cpu'):
	net['generator'] = net['generator'].to(device)
	net['discriminator'] = net['discriminator'].to(device)
	if not fixed_noise:
		torch.manual_seed(1)
		fixed_noise = torch.randn(32, net['generator'].latent_dim, 1, 1, device=device)
	start_time = time.time()

	net['generator'].train()
	net['discriminator'].train()

	pbar = tqdm(train_loader, ncols=50)
	img_list = []
	D_real_probability, D_fake_probability = 0, 0
	D_total_loss, G_total_loss = 0, 0
	for i, datas in enumerate(pbar):

		image = datas['image'].to(device)
		b = image.size(0)
		
		### Discriminator
		optimizer_discriminator.zero_grad()
			### real
		pred_real = net['discriminator'](image)		
		label_real = torch.ones(b).to(device)
		loss_real = criterion(pred_real, label_real)
		loss_real.backward()
		D_real_probability += pred_real.mean().item()

		###fake
		noise = torch.randn(b, net['generator'].latent_dim).to(device)
		fake_image = net['generator'](noise)
		label_fake = torch.zeros(b).to(device)
		pred_fake = net['discriminator'](fake_image.detach()).view(-1)
		loss_fake = criterion(pred_fake, label_fake)
		loss_fake.backward()
		D_fake_probability += pred_fake.mean().item()

		D_total_loss = D_total_loss + loss_real.item() + loss_fake.item()
		optimizer_discriminator.step()


		### Generator
		optimizer_generator.zero_grad()
		pred_fake = net['discriminator'](fake_image).view(-1)
		G_loss = criterion(pred_fake, label_real)
		G_loss.backward()
		G_total_loss += G_loss.item()
		optimizer_generator.step()


	with torch.no_grad():
		fake = net['generator'](fixed_noise).detach().cpu()
	img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
	fig = plt.figure(figsize=(8,8))
	plt.axis("off")
	ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
	ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
	plt.savefig('./results/GAN/{}'.format(epoch))
	end_time = time.time()
	print('Train Epoch : %d, D Loss : %3.5f, G Loss : %3.5f' % (epoch, D_total_loss / len(train_loader), G_total_loss / len(train_loader)))
	print('Discriminator on Real Image Probability : %3.3f' % (D_real_probability / len(train_loader)))
	print('Discriminator on Fake Image Probability : %3.3f' % (D_fake_probability / len(train_loader)))
	


def get_model():
	generator, discriminator = Generator(100), Discriminator()
	net = {'generator':generator, 'discriminator':discriminator}
	criterion = nn.BCELoss()
	return net, criterion, train
