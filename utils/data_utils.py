import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def plotSignal(signal, samplingFreq):
	tAxis = np.linspace(0, len(signal) / samplingFreq, len(signal))
	plt.rcParams.update({'font.size': 16})
	plt.figure(figsize=(20, 8))
	plt.plot(tAxis, signal)
	plt.xlabel('Time [s]')
	plt.title('Signal in time domain')
	plt.show()
def plotSignalAndSpectrum(signal, samplingFreq, spectralWindow):
	# Calculate Nyquist
	nyquistFreq = round(samplingFreq / 2)
	# Calculate Fast Fourier Transform
	xfft = np.abs(fft(signal,spectralWindow))
	# Take one side of the FFT, i.e., N/2 where N = spectralWindow
	xfft = xfft[0:round(spectralWindow/2)]
	fAxis = np.linspace(0, nyquistFreq , round(spectralWindow/2))  # in Hz
	tAxis = np.linspace(0, len(signal)/samplingFreq, len(signal))
	plt.rcParams.update({'font.size': 16})
	plt.figure(figsize=(20, 8))
	plt.subplot(1, 2, 1)
	plt.plot(tAxis, signal)
	plt.xlabel('Time [s]')
	plt.title('Signal in time domain')
	plt.subplot(1, 2, 2)
	plt.plot(fAxis, xfft)
	plt.title('Signal in frequency domain')
	plt.xlabel('Frequency [Hz]')
	plt.show()