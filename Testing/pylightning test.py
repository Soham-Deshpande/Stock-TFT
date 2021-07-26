import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD


class model(pl.LightningModule):
	def __init__(self):
		super(model, self).__init__()
		self.fc1 = nn.Linear(28*28, 256)
		self.fc2 = nn.Linear(256, 128)
		self.out = nn.Linear(128, 10)
		self.lr = 0.01
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x):
		batch_size, _, _, _ = x.size()
		x = x.view(batch_size, -1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.out(x)

	def configure_optimizers(self):
		return torch.optim.SGD(self.parameters(), lr=self.lr)

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		logits = self.forward(x)
		loss = self.loss(logits, y)
		return loss

	def validation_step(self, valid_batch, batch_idx):
		x, y = valid_batch
		logits = self.forward(x)
		loss = self.loss(logits, y)


class Data(pl.LightningDataModule):
	def prepare_data(self):
		transform = transforms.Compose([
			transforms.ToTensor()
		])

		self.train_data = datasets.MNIST(
			'', train=True, download=True, transform=transform)
		self.test_data = datasets.MNIST(
			'', train=False, download=True, transform=transform)

	def train_dataloader(self):
		return DataLoader(self.train_data, batch_size=32, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.test_data, batch_size=32, shuffle=True)


clf = model()
mnist = Data()
trainer = pl.Trainer(max_epochs=20)
trainer.fit(clf, mnist)
