import torch.nn as nn
import torch
import torch.nn.functional as F

class SiMSE(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, shared, private):

		n = torch.numel(shared)
		loss = torch.cdist(shared, private, p=2).mean()
		loss = loss / n 
		return loss

class diffLoss(nn.Module):

	def __init__(self):

		super().__init__()

	def forward(self, shared, private):
		batch_size = shared.size(0)
		shared = shared.view(batch_size, -1)
		private = private.view(batch_size, -1)

		shared_l2_norm = torch.norm(shared, p=2, dim=1, keepdim=True).detach()
		shared_l2 = shared.div(shared_l2_norm.expand_as(shared) + 1e-6)

		private_l2_norm = torch.norm(private, p=2, dim=1, keepdim=True).detach()
		private_l2 = private.div(private_l2_norm.expand_as(private) + 1e-6)

		diff_loss = torch.mean((private_l2.t().mm(shared_l2)).pow(2))

		# bs = shared.size(0)
		# shared = shared - shared.mean(dim=0)
		# private = private - private.mean(dim=0)
		# shared = F.normalize(shared, dim=1, p=2)
		# private = F.normalize(private, dim=1, p=2)
		# loss = private.t().mm(shared).pow(2).sum().pow(0.5)
		return diff_loss 

class DSNLoss(nn.Module):

	def __init__(self, alpha=0.01, beta=0.05, gamma=0.3):

		super().__init__()
		self.diff_loss = diffLoss()
		self.mseloss = nn.MSELoss(reduction='mean')
		self.ceLoss = nn.CrossEntropyLoss(reduction='mean')
		self.simse = SiMSE()
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

	def forward(self, pred_dom,  private, share, recon, label, class_label=None, pred_cls=None, mode='src'):
		bs = share.size(0)
		differnce = self.diff_loss(share, private)
		recon_loss = self.mseloss(recon, label) - self.simse(recon, label)
		domain_label = torch.ones(bs).long().cuda() if mode=='src' else torch.zeros(bs).long().cuda()
		domain_loss = self.ceLoss(pred_dom, domain_label)

		recon_loss = recon_loss * 0 if torch.isnan(recon_loss) else recon_loss
		differnce = differnce * 0 if torch.isnan(differnce) else differnce
		domain_loss = domain_loss * 0 if torch.isnan(domain_loss) else domain_loss
		if mode == 'src':
			task_loss = self.ceLoss(pred_cls, class_label)
			task_loss = task_loss * 0 if torch.isnan(task_loss) else task_loss
			return task_loss, self.alpha * recon_loss, self.beta * differnce, self.gamma * domain_loss
		return self.alpha * recon_loss, self.beta * differnce, self.gamma * domain_loss
		











