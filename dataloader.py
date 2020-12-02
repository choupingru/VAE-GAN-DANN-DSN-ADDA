import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
from os.path import join
from torchvision import transforms
from PIL import Image
from pathlib import Path

class FaceDataloader(Dataset):
	def __init__(self, path='./hw3_data/face/train'):
		self.path = Path(path)
		self.image_name = os.listdir(path)
		self.image_name = [name for name in self.image_name if name.endswith('.png')]
		self.transform = transforms.Compose([
			transforms.ToTensor()
		])
	def __len__(self):
		return len(self.image_name)
	def __getitem__(self, idx):
		name = self.image_name[idx]
		image = Image.open(self.path / name)
		image = self.transform(image)
		return {
			'image' : image,
			'name' : name
		}

class DigitDataloader(Dataset):

	def __init__(self, domain, path='./hw3_data/digits/', mode='train'):
		
		if mode != 'gg':
			df = pd.read_csv(Path(path) / domain / (mode + '.csv'))
			self.name, self.label = df['image_name'], df['label']
			self.path = Path(path) / domain / mode
		else:
			self.name = os.listdir(path)
			self.path = path
		self.transform = transforms.Compose([
			transforms.ToTensor(),
		])
		self.domain =domain
	def __len__(self):
		return len(self.name)

	def __getitem__(self, index):
		image_name = self.name[index]
		label = self.label[index]
		image = Image.open(join(self.path, image_name))
		image = np.array(image)
		if len(image.shape) != 3:
			image = image[:, :, np.newaxis]
			image = image.repeat(3, 2)
		image = self.transform(image)
		label = torch.LongTensor([label])
		return {
			'image':image,
			'label':label
		}


def get_dataset(dataset_name, domain=None, mode='train'):
	if dataset_name == 'VAE':
		return FaceDataloader()
	elif dataset_name == 'GAN':
		return FaceDataloader()
	else:
		return DigitDataloader(domain, mode=mode)