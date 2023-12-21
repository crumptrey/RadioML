import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
def rectBPSK(bitPeriod, carrierFreq, signalLen):
	# Create the bit sequence
	bitSequence = np.random.randint(2, size=signalLen)
	# Create the symbol sequence from bit sequence
	symbolSequence = 2 * bitSequence - 1
	zeroMatrix = np.zeros((bitPeriod - 1, signalLen))
	symbolSequence = np.vstack([symbolSequence, zeroMatrix])
	symbolSequence = np.reshape(symbolSequence, [bitPeriod*signalLen])
	# Create pulse function
	p_of_t = np.ones([bitPeriod])
	# Convolve bit sequence with pulse function
	s_of_t = signal.lfilter(p_of_t, [1], symbolSequence)

	e_vec = np.exp(1j*2*np.pi*carrierFreq*np.linspace(1,len(s_of_t),1))
	x_of_t = s_of_t * e_vec

	n_of_t = np.random.random_integers(len(x_of_t)) + 1j*np.random.random_integers(len(x_of_t))
	return s_of_t

