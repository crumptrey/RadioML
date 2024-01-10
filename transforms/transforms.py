import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def EMD(signal, maxIMFs, samplingFreq):
	tAxis = np.linspace(0, len(signal) / samplingFreq, len(signal))
	resultEMD = []
	s = signal
	IMFs = 0
	while IMFs < maxIMFs:
		IMFs += 1
		upper_peaks, _ = find_peaks(s)
		lower_peaks, _ = find_peaks(-s)
		f1 = interp1d(upper_peaks / samplingFreq, s[upper_peaks], kind='cubic', fill_value='extrapolate')
		f2 = interp1d(lower_peaks / samplingFreq, s[lower_peaks], kind='cubic', fill_value='extrapolate')
		y1 = f1(tAxis)
		y2 = f2(tAxis)
		# Zero-padding
		y1[0:5] = 0
		y1[-5:] = 0
		y2[0:5] = 0
		y2[-5:] = 0
		avg_envelope = (y1 + y2) / 2
		intrinsicModeFunction = s - avg_envelope
		resultEMD.append(intrinsicModeFunction)
		s = avg_envelope
	resultEMD = np.array(resultEMD)
	return resultEMD

def plotIMF(IMFs, samplingFreq, spectralWindow):
	for IMF in range(0, len(IMFs)):
		# Calculate Nyquist
		nyquistFreq = round(samplingFreq / 2)
		# Calculate Fast Fourier Transform
		xfft = np.abs(fft(IMFs[IMF], spectralWindow))
		# Take one side of the FFT, i.e., N/2 where N = spectralWindow
		xfft = xfft[0:round(spectralWindow / 2)]
		fAxis = np.linspace(0, nyquistFreq, round(spectralWindow / 2))  # in Hz
		tAxis = np.linspace(0, len(IMFs[IMF]) / samplingFreq, len(IMFs[IMF]))
		plt.rcParams.update({'font.size': 16})
		plt.figure(figsize=(20, 8))
		plt.subplot(1, 2, 1)
		plt.plot(tAxis, IMFs[IMF])
		plt.xlabel('Time [s]')
		timeStr = str('IMF {0} in time domain').format(IMF)
		plt.title(timeStr)
		plt.subplot(1, 2, 2)
		plt.plot(fAxis, xfft)
		freqStr = str('IMF {0} in frequency domain').format(IMF)
		plt.title(freqStr)
		plt.xlabel('Frequency [Hz]')
		plt.show()
	return plt
