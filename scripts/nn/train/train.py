# Plotting Includes
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
# External Includes
import numpy as np
from pprint import pprint
import torch as th
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from scripts.utils.dataset import Dataset
from scripts.utils.encoder import Encoder


from scripts.nn.model import Model


class TrainLoop(object):

	def __init__(self, lr: float = 10e-4, n_epochs: int = 3, gpu: bool = True):
		self.lr = lr
		self.n_epochs = n_epochs
		self.gpu = gpu

	def __repr__(self):
		ret = self.__class__.__name__
		ret += "(lr={}, n_epochs={}, gpu={})".format(self.lr, self.n_epochs, self.gpu)
		return ret

	def __call__(
			self, model: nn.Module, training: Dataset, validation: Dataset, le: Encoder
	):
		criterion = nn.CrossEntropyLoss()

		if self.gpu:
			model.to("mps")
			criterion.to("mps")

		optimizer = Adam(model.parameters(), lr=self.lr)

		train_data = DataLoader(
			training.as_torch(le=le), shuffle=True, batch_size=1
		)
		val_data = DataLoader(
			validation.as_torch(le=le), shuffle=True, batch_size=1
		)

		# Save two lists for plotting a convergence graph at the end
		ret_train_loss = list()
		ret_val_loss = list()

		for epoch in range(self.n_epochs):
			train_loss = self._train_one_epoch(
				model=model, data=train_data, loss_fn=criterion, optimizer=optimizer
			)
			print("On Epoch {} the training loss was {}".format(epoch, train_loss))
			ret_train_loss.append(train_loss)

			val_loss = self._validate_once(
				model=model, data=val_data, loss_fn=criterion
			)
			print("---- validation loss was {}".format(val_loss))
			ret_val_loss.append(val_loss)

		return ret_train_loss, ret_val_loss

	def _train_one_epoch(
			self, model: nn.Module, data: DataLoader, loss_fn: nn.CrossEntropyLoss, optimizer: Adam
	) -> float:
		total_loss = 0.0
		# Switch the model mode so it remembers gradients, induces dropout, etc.
		model.train()

		for i, batch in enumerate(data):
			x, y = batch

			# Push datasets to GPU if necessary
			if self.gpu:
				x = Variable(x.to("mps"))
				y = Variable(y.to("mps"))
			else:
				x = Variable(x)
				y = Variable(y)

			# Forward pass of prediction
			outputs = model(x)

			# Zero out the parameter gradients, because they are cumulative,
			# compute loss, compute gradients (backward), update weights
			loss = loss_fn(outputs, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		mean_loss = total_loss / (i + 1.0)
		return mean_loss

	def _validate_once(
			self, model: nn.Module, data: DataLoader, loss_fn: nn.CrossEntropyLoss
	) -> float:
		total_loss = 0.0
		# Switch the model back to test mode (so that batch norm/dropout doesn't
		# take effect)
		model.eval()
		for i, batch in enumerate(data):
			x, y = batch

			if self.gpu:
				x = x.to("mps")
				y = y.to("mps")

			outputs = model(x)
			loss = loss_fn(outputs, y)
			total_loss += loss.item()

		mean_loss = total_loss / (i + 1.0)
		return mean_loss