import torch.nn as nn
from scripts.nn.model import Model
try:
	from collections import OrderedDict
except ImportError:
	OrderedDict = dict
from scripts.preprocess import emdExtract

import torch
from torch import nn
from torch.nn import functional as F
from scripts.preprocess import iqEMD

class OShea16(Model):
	"""Convolutional Neural Network based on the "VT_CNN2" Architecture

	This network is based off of a network for modulation classification first
	introduced in O'Shea et al and later updated by West/Oshea and Hauser et al
	to have larger filter sizes.

	Modifying the first convolutional layer to not use a bias term is a
	modification made by Bryse Flowers due to the observation of vanishing
	gradients during training when ported to PyTorch (other authors used Keras).

	Including the PowerNormalization inside this network is a simplification
	made by Bryse Flowers so that utilization of DSP blocks in real time for
	datasets generation does not require knowledge of the normalization used during
	training as that is encapsulated in the network and not in a pre-processing
	stage that must be matched up.

	References
	    T. J. O'Shea, J. Corgan, and T. C. Clancy, “Convolutional radio modulation
	    recognition net,” in International Conference on Engineering Applications
	    of Neural Networks, pp. 213–226, Springer,2016.

	    N. E. West and T. O’Shea, “Deep architectures for modulation recognition,” in
	    IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN), pp.
	    1–6, IEEE, 2017.

	    S. C. Hauser, W. C. Headley, and A. J.  Michaels, “Signal detection effects on
	    deep neural net utilizing raw iq for modulation classification,” in
	    Military Communications Conference, pp. 121–127, IEEE, 2017.
	"""

	def __init__(self, input_samples: int, n_classes: int):
		super().__init__(input_samples, n_classes)
		# Batch x 1-channel x IQ x input_samples
		self.conv1 = nn.Conv2d(
			in_channels=1,
			out_channels=256,
			kernel_size=(1, 7),
			padding=(0, 3),
			bias=False,
		)
		self.a1 = nn.ReLU()
		self.n1 = nn.BatchNorm2d(256)

		self.conv2 = nn.Conv2d(
			in_channels=256,
			out_channels=80,
			kernel_size=(2, 7),
			padding=(0, 3),
			bias=True,
		)
		self.a2 = nn.ReLU()
		self.n2 = nn.BatchNorm2d(80)

		# Batch x Features
		self.dense1 = nn.Linear(80 * 1 * input_samples, 256)
		self.a3 = nn.ReLU()
		self.n3 = nn.BatchNorm1d(256)

		self.dense2 = nn.Linear(256, n_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.a1(x)
		x = self.n1(x)

		x = self.conv2(x)
		x = self.a2(x)
		x = self.n2(x)

		# Flatten the input layer down to 1-d by using Tensor operations
		x = x.contiguous()
		x = x.view(x.size()[0], -1)

		x = self.dense1(x)
		x = self.a3(x)
		x = self.n3(x)

		x = self.dense2(x)

		return x


class VGG16(Model):
	def __init__(self, input_samples: int, n_classes: int):
		super().__init__(input_samples, n_classes)
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU())
		self.layer4 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer5 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.layer6 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU())
		self.layer7 = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer8 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU())
		self.layer9 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU())
		self.layer10 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer11 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU())
		self.layer12 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU())
		self.layer13 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(7 * 7 * 512, 4096),
			nn.ReLU())
		self.fc1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU())
		self.fc2 = nn.Sequential(
			nn.Linear(4096, n_classes))

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.layer5(out)
		out = self.layer6(out)
		out = self.layer7(out)
		out = self.layer8(out)
		out = self.layer9(out)
		out = self.layer10(out)
		out = self.layer11(out)
		out = self.layer12(out)
		out = self.layer13(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		out = self.fc1(out)
		out = self.fc2(out)
		return out