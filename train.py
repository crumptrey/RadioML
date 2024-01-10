import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import utils.load_datasets
from utils import *
from net import networks
from sklearn.model_selection import train_test_split


device = torch.device("mps")
def evaluate(model, data_loader, criterion, phase):
	model.eval()
	total_loss = 0.0
	correct_predictions = 0
	total_samples = 0
	snr_loss_dict = {}
	snr_correct_dict = {}

	with torch.no_grad():
		for batch in data_loader:
			data, mod_types, snrs = batch
			data = torch.from_numpy(data).float().to(device)
			labels = torch.tensor(mod_types).long().to(device)

			outputs = model(data)

			loss = criterion(outputs, labels)
			total_loss += loss.item()

			_, predicted = torch.max(outputs, 1)
			correct_predictions += (predicted == labels).sum().item()
			total_samples += labels.size(0)

			for i, snr in enumerate(snrs):
				if snr not in snr_loss_dict:
					snr_loss_dict[snr] = 0.0
					snr_correct_dict[snr] = 0

				snr_loss_dict[snr] += loss.item()
				snr_correct_dict[snr] += (predicted[i] == labels[i]).item()

	avg_loss = total_loss / len(data_loader)
	accuracy = correct_predictions / total_samples

	print(f'{phase} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

	if phase == 'Test':
		# Plot Probability of Correct Classification vs. SNR
		snr_accuracy_dict = {snr: snr_correct_dict[snr] / total_samples for snr in snr_loss_dict}
		snr_values = list(snr_accuracy_dict.keys())
		accuracy_values = list(snr_accuracy_dict.values())

		plt.plot(snr_values, accuracy_values, marker='o')
		plt.xlabel('SNR (dB)')
		plt.ylabel('Probability of Correct Classification')
		plt.title(f'Probability of Correct Classification vs. SNR - {phase}')
		plt.grid(True)
		plt.show()

model = networks.Snoap_CNN(num_classes=11)
model = model.to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the split ratios
train_ratio, val_ratio, test_ratio = 0.75, 0.125, 0.125

train_dataset = utils.load_datasets.DeepSig2018Dataset("datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5", 'train')
val_dataset = utils.load_datasets.DeepSig2018Dataset("datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5", 'valid')
test_dataset = utils.load_datasets.DeepSig2018Dataset("datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5", 'test')
# Split the dataset
#total_size = len(dataset)
#train_size = int(train_ratio * total_size)
#val_size = int(val_ratio * total_size)
#test_size = total_size - train_size - val_size

#train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
epochs = 12

for epoch in range(epochs):
	model.train()
	total_train_loss = 0.0

	for train_batch in train_loader:
		data, mod_types, snrs = train_batch
		data = data.to(device)
		labels = mod_types.to(device)

		print(data.size())

		optimizer.zero_grad()
		outputs = model(data)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		total_train_loss += loss.item()
		print(total_train_loss)

	avg_train_loss = total_train_loss / len(train_loader)
	print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}')

	# Validation
	evaluate(model, val_loader, criterion, 'Validation')

# Testing
evaluate(model, test_loader, criterion, 'Test')