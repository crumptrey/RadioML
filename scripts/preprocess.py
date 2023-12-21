import emd
import torch as th
import numpy as np
import io
import matplotlib
from scipy import ndimage
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import matlab.engine
def iqEMD(x):
	x= x.cpu()
	x1 = x[0,0,:]
	x2 = x1[1,:]
	x1 = x1[0,:]
	hht1 = emdExtract(x1.numpy())
	hht2 = emdExtract(x2.numpy())
	hht = np.column_stack((hht1, hht2))
	hht = resize(hht, (224, 224))
	hht = th.from_numpy(hht).float()
	hht = hht[None, :].to('mps')
	hht = hht.repeat(3,1,1)
	hht = hht[None,:]
	return hht
def emdExtract(x):
	eng = matlab.engine.start_matlab()
	t = np.linspace(0,10,128)
	x1 = np.sin(np.pi**t)
	plt.plot(t,x1)
	plt.show()
	Fs = 200e3
	efs = eng.emd(x1)
	hht = eng.hht(efs, Fs)
	hht = hht.toarray()
	eng.quit()
	return hht
