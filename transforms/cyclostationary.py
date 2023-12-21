import numpy as np



def autoCorrelation(data,lags):
	# Pre-allocate autocorrelation table
	acorr = len(lags) * [0]
	# Mean
	mean = sum(data) / len(data)
	# Variance
	var = sum([(x - mean) ** 2 for x in data]) / len(data)
	# Normalized data
	ndata = [x - mean for x in data]
	# Go through lag components one-by-one
	for l in lags:
		c = 1  # Self correlation
		if (l > 0):
			tmp = [ndata[l:][i] * ndata[:-l][i]
			       for i in range(len(data) - l)]
			c = sum(tmp) / len(data) / var
		acorr[l] = c
	return acorr

