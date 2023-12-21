import torch.nn as nn


# Creating a CNN class
class Snoap_CNN(nn.Module):
	#  Determine what layers and their order in CNN object
	def __init__(self, num_classes):
		super(Snoap_CNN, self).__init__()
		filter_size = 23

		self.conv_layer1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=filter_size, padding=11)
		self.batch_norm1 = nn.BatchNorm2d(16)
		self.ReLU1 = nn.ReLU()
		self.max_pool1 = nn.MaxPool2d(kernel_size=filter_size, stride=2)

		self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=filter_size,padding = 11)
		self.batch_norm2 = nn.BatchNorm2d(24)
		self.ReLU2 = nn.ReLU()
		self.max_pool2 = nn.MaxPool2d(kernel_size=filter_size, stride=2)

		self.conv_layer3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=filter_size, padding=11)
		self.batch_norm3 = nn.BatchNorm2d(32)
		self.ReLU3 = nn.ReLU()
		self.max_pool3 = nn.MaxPool2d(kernel_size=filter_size, stride=2)

		self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=filter_size, padding=11)
		self.batch_norm4 = nn.BatchNorm2d(48)
		self.ReLU4 = nn.ReLU()
		self.max_pool4 = nn.MaxPool2d(kernel_size=filter_size, stride=2)

		self.conv_layer5 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=filter_size, padding=11)
		self.batch_norm5 = nn.BatchNorm2d(64)
		self.ReLU5 = nn.ReLU()
		self.max_pool5 = nn.MaxPool2d(kernel_size=filter_size, stride=2)

		self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=filter_size, padding=11)
		self.batch_norm6 = nn.BatchNorm2d(96)
		self.ReLU6 = nn.ReLU()

		self.avgpool = nn.AdaptiveMaxPool1d(1)
		self.fc1 = nn.Linear(96, num_classes)
		self.softmax = nn.Softmax(dim=0)

	# Progresses data across layers
	def forward(self, x):
		out = self.conv_layer1(x)
		out = self.batch_norm1(out)
		out = self.ReLU1(out)
		out = self.max_pool1(out)

		out = self.conv_layer2(out)
		out = self.batch_norm2(out)
		out = self.ReLU2(out)
		out = self.max_pool2(out)

		out = self.conv_layer3(out)
		out = self.batch_norm3(out)
		out = self.ReLU3(out)
		out = self.max_pool3(out)

		out = self.conv_layer4(out)
		out = self.batch_norm4(out)
		out = self.ReLU4(out)
		out = self.max_pool4(out)

		out = self.conv_layer5(out)
		out = self.batch_norm5(out)
		out = self.ReLU5(out)
		out = self.max_pool5(out)

		out = self.conv_layer6(out)
		out = self.batch_norm6(out)
		out = self.ReLU6(out)

		out = self.avgpool(out)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = self.softmax(out, dim=1)
		return out