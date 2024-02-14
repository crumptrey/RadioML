import h5py
import numpy as np
from scipy.io import savemat

file_dir = "datasets/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"
modulation_classes = ['OOK',
                           '4ASK',
                           '8ASK',
                           'BPSK',
                           'QPSK',
                           '8PSK',
                           '16PSK',
                           '32PSK',
                           '16APSK',
                           '32APSK',
                           '64APSK',
                           '128APSK',
                           '16QAM',
                           '32QAM',
                           '64QAM',
                           '128QAM',
                           '256QAM',
                           'AM-SSB-WC',
                           'AM-SSB-SC',
                           'AM-DSB-WC',
                           'AM-DSB-SC',
                           'FM',
                           'GMSK',
                           'OQPSK']
hdf5_file = h5py.File(file_dir, 'r')
X = hdf5_file['X']
data = X
modulations = np.argmax(hdf5_file['Y'], axis=1)
print(X.shape[0])
Z = hdf5_file['Z'][:, 0]
snrs = Z
target_modulations = ["BPSK", "QPSK", "8PSK",
                      "16QAM", "64QAM", "256QAM"]
target_snrs = np.arange(-20, 21, 2)
print(target_snrs)
target_modulation_indices = [modulation_classes.index(modu) for modu in target_modulations]
X_array = []
Y_array = []
Z_array = []
# Create a mapping between indices and modulation values
index_to_modulation = {index: modulation for index, modulation in enumerate(modulation_classes)}
index = 0
for modu in target_modulation_indices:
    for snr in target_snrs:
        snr_modu_indices = np.where((modulations == modu) & (snrs == snr))[0]
        for ind in snr_modu_indices:
            X_output = data[ind]
            Y_output = index_to_modulation[modulations[ind]]
            Z_output = snrs[ind]
            X_array.append(data[ind])
            Y_array.append(index_to_modulation[modulations[ind]])
            Z_array.append(snrs[ind])
            output_mat_file = 'data/frame_{0}_{1}_{2}.mat'.format(Y_output, Z_output, index)
            savemat(output_mat_file, {'frame': X_output, 'label': Y_output, 'SNR': Z_output}, do_compression=True,
            format='5')
            index += 1

#X_array = np.vstack(X_output)
#Y_array = np.concatenate(Y_output)
#Z_array = np.concatenate(Z_output)
#print(len(X_array))
print(np.unique(Y_array))

print(np.unique(Z_array))
print("saved")