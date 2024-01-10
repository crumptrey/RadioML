import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
import pickle
import numpy as np
import h5py
import json

MOD_TYPES = {'AM-SSB', 'CPFSK', 'QPSK', 'GFSK', 'PAM4', 'QAM16', 'WBFM', '8PSK', 'QAM64', 'AM-DSB',
             'BPSK'}


def dataset_split(data,
                  modulations_classes,
                  modulations, snrs,
                  target_modulations,
                  mode,
                  target_snrs, train_proportion=0.7,
                  valid_proportion=0.2,
                  test_proportion=0.1,
                  seed=48):
	np.random.seed(seed)
	train_split_index = int(train_proportion * 4096)
	valid_split_index = int((valid_proportion + train_proportion) * 4096)
	test_split_index = int((test_proportion + valid_proportion + train_proportion) * 4096)
	X_output = []
	Y_output = []
	Z_output = []

	target_modulation_indices = [modulations_classes.index(modu) for modu in target_modulations]

	for modu in target_modulation_indices:
		for snr in target_snrs:
			snr_modu_indices = np.where((modulations == modu) & (snrs == snr))[0]

			np.random.shuffle(snr_modu_indices)
			train, valid, test, remaining = np.split(snr_modu_indices,
			                                         [train_split_index, valid_split_index,
			                                          test_split_index])
			if mode == 'train':
				X_output.append(data[np.sort(train)])
				Y_output.append(modulations[np.sort(train)])
				Z_output.append(snrs[np.sort(train)])
			elif mode == 'valid':
				X_output.append(data[np.sort(valid)])
				Y_output.append(modulations[np.sort(valid)])
				Z_output.append(snrs[np.sort(valid)])
			elif mode == 'test':
				X_output.append(data[np.sort(test)])
				Y_output.append(modulations[np.sort(test)])
				Z_output.append(snrs[np.sort(test)])
			else:
				raise ValueError(f'unknown mode: {mode}. Valid modes are train, valid and test')
	X_array = np.vstack(X_output)
	Y_array = np.concatenate(Y_output)
	Z_array = np.concatenate(Z_output)
	for index, value in enumerate(np.unique(np.copy(Y_array))):
		Y_array[Y_array == value] = index
	return X_array, Y_array, Z_array

class DeepSig2016Dataset(Dataset):
	def __init__(self, file_dir, transform=None, target_transform=None):
		self.file_dir = file_dir
		self.transform = transform
		self.target_transform = target_transform
		with open(self.file_dir, 'rb') as f:
			u = pickle._Unpickler(f)
			u.encoding = 'latin1'
			self.pickle = u.load()
		snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.pickle.keys())))), [1, 0])

		X = []
		self.lbl = []
		for mod in mods:
			for snr in snrs:
				X.append(self.pickle[(mod, snr)])
				for i in range(self.pickle[(mod, snr)].shape[0]):
					self.lbl.append((mod, snr))
		self.X = np.vstack(X)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		x = self.X[idx]
		mod = self.lbl[idx][0]
		mod_type = list(MOD_TYPES).index(mod)
		snr = self.lbl[idx][1]
		return x, mod_type, snr


class DeepSig2018Dataset(Dataset):
	def __init__(self, file_dir, mode:str, seed=48):
		nf_train = 1024
		nf_valid = 1024
		nf_test = 1024
		self.file_dir = file_dir
		self.modulation_classes = ['32PSK',
                                 '16APSK',
                                 '32QAM',
                                 'FM',
                                 'GMSK',
                                 '32APSK',
                                 'OQPSK',
                                 '8ASK',
                                 'BPSK',
                                 '8PSK',
                                 'AM-SSB-SC',
                                 '4ASK',
                                 '16PSK',
                                 '64APSK',
                                 '128QAM',
                                 '128APSK',
                                 'AM-DSB-SC',
                                 'AM-SSB-WC',
                                 '64QAM',
                                 'QPSK',
                                 '256QAM',
                                 'AM-DSB-WC',
                                 'OOK',
                                 '16QAM']
		# load data
		hdf5_file = h5py.File(self.file_dir, 'r')
		self.X = hdf5_file['X']
		self.Y = np.argmax(hdf5_file['Y'], axis=1)
		print(self.X.shape[0])
		self.Z = hdf5_file['Z'][:, 0]
		train_proportion = (24 * 26 * nf_train) / self.X.shape[0]
		valid_proportion = (24 * 26 * nf_valid) / self.X.shape[0]
		test_proportion = (24 * 26 * nf_test) / self.X.shape[0]
		self.target_modulations = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', '256QAM']
		self.target_snrs = np.unique(self.Z)

		self.X_data, self.Y_data, self.Z_data = dataset_split(
			data=self.X,
			modulations_classes=self.modulation_classes,
			modulations=self.Y,
			snrs=self.Z,
			mode=mode,
			train_proportion=0.75,
			valid_proportion=0.125,
			test_proportion=0.125,
			target_modulations=self.target_modulations,
			target_snrs=self.target_snrs,
			seed=seed
		)


	def __len__(self):
		return len(self.X_data)

	def __getitem__(self, idx):
		x, y, z = self.X_data[idx], self.Y_data[idx], self.Z_data[idx]
		x, y, z = torch.Tensor(x).transpose(0, 1), y, z
		return x, y, z