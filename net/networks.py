import torch.nn as nn


# Creating a CNN class
class Snoap_CNN(nn.Module):
	#  Determine what layers and their order in CNN object
	def __init__(self, num_classes):
		super(Snoap_CNN, self).__init__()
		filter_size = 23

		self.conv_layer1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=filter_size, padding=11)
		self.batch_norm1 = nn.BatchNorm1d(16)
		self.ReLU1 = nn.ReLU()
		self.max_pool1 = nn.MaxPool1d(kernel_size=filter_size, stride = 2, padding = 11)

		self.conv_layer2 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=filter_size, padding = 11)
		self.batch_norm2 = nn.BatchNorm1d(24)
		self.ReLU2 = nn.ReLU()
		self.max_pool2 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding = 11)

		self.conv_layer3 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=filter_size, padding = 11)
		self.batch_norm3 = nn.BatchNorm1d(32)
		self.ReLU3 = nn.ReLU()
		self.max_pool3 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding=11)

		self.conv_layer4 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=filter_size, padding = 11)
		self.batch_norm4 = nn.BatchNorm1d(48)
		self.ReLU4 = nn.ReLU()
		self.max_pool4 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding = 11)

		self.conv_layer5 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=filter_size, padding = 11)
		self.batch_norm5 = nn.BatchNorm1d(64)
		self.ReLU5 = nn.ReLU()
		self.max_pool5 = nn.MaxPool1d(kernel_size=filter_size, stride=2, padding = 11)

		self.conv_layer6 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=filter_size, padding = 11)
		self.batch_norm6 = nn.BatchNorm1d(96)
		self.ReLU6 = nn.ReLU()

		self.avgpool = nn.AvgPool1d(kernel_size=32,stride=2,padding = 0)
		self.drop = nn.Dropout(0.5)
		self.fc1 = nn.Linear(96, num_classes)
		self.softmax = nn.Softmax(dim=1)

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
		out = self.drop(out)
		out = self.fc1(out)
		out = self.softmax(out)

		return out

class ResidualUnit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResidualUnit, self).__init__()
		filter_size = 23
		self.conv1 = nn.Conv1d(32, 32, kernel_size=filter_size,
		                             padding=11)
		self.bn1 = nn.BatchNorm1d(32)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv1d(32, 32, kernel_size=filter_size,
		                       padding =11)
		self.bn2 = nn.BatchNorm1d(32)
		self.relu2 = nn.ReLU()

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out += residual  # Skip connection
		return out


class ResidualStack(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResidualStack, self).__init__()
		self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
		self.bn = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU()
		self.residual_units = nn.Sequential(
			ResidualUnit(in_channels, out_channels),
			ResidualUnit(in_channels, out_channels)
		)
		self.max_pool = nn.MaxPool1d(kernel_size=2)

	def forward(self, x):
		print("x {}".format(x.size()))
		out = self.conv1x1(x)
		print("conv1x1 {}".format(out.size()))
		out = self.bn(out)
		out = self.relu(out)
		out = self.residual_units(out)
		out = self.max_pool(out)
		print("output {}".format(out.size()))
		return out


class Snoap_ResidualNN(nn.Module):
	def __init__(self, num_classes=10):
		super(Snoap_ResidualNN, self).__init__()
		self.residual_stacks = nn.Sequential(
			ResidualStack(2, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32),
			ResidualStack(32, 32)
		)
		self.final_layers = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(32 * 16, 128),
			nn.SELU(),
			nn.Dropout(0.5),
			nn.Linear(128, 128),
			nn.SELU(),
			nn.Dropout(0.5),
			nn.Linear(128, num_classes),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		out = self.residual_stacks(x)
		out = out.view(out.size(0), -1)
		print("res_stack out {}".format(out.size()))
		out = self.final_layers(out)
		return out

