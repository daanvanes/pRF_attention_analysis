from IPython import embed as shell
from matplotlib import pyplot as plt
import seaborn as sn

import numpy as np
import scipy as sp
from GroupLevelPlotting import General_functions 
GF = General_functions()


image_folder = '/home/vanes/gaussian_interactions/'

AF_sizes = [1,3,5]
SD_size = 1
SD_x = 1
AF_xs = [2,4,8]


"""
First, let's understand simple Gaussian interactions

functions work as follows:

GF.one_d_gauss(xo,sigma,amplitude=1,res=101,max_eccentricity=1)
GF.two_d_gauss(xo, yo, sigma_x, sigma_y=None, amplitude = 1, theta=0,res=101,max_eccentricity=1)
"""

## gaussian interaction with medium AF
f = plt.figure(figsize=(5,5))
k=0
for AF_size in AF_sizes:
	for AF_x in AF_xs:
		k+=1
		s = f.add_subplot(len(AF_sizes),len(AF_xs),k)
		SD = GF.one_d_gauss(xo = SD_x,sigma=SD_size,amplitude=1,res=101,max_eccentricity=10)
		AF = GF.one_d_gauss(xo = AF_x,sigma=AF_size,amplitude=1,res=101,max_eccentricity=10)
		PRF = SD*AF
		plt.plot(SD,'r',lw=2,label='SD')
		plt.plot(AF,'b',lw=2,label='AF')
		plt.plot(PRF,'g',lw=2,label='PRF')
		if k > ((len(AF_xs)-1)*len(AF_sizes)):
			plt.xlabel('eccentricity')
		if np.mod(k,len(AF_sizes))==0:
			plt.ylabel('amplitude')
		sn.despine(offset=2)
		# pl.title('AF ')
		# plt.legend(loc='best')
plt.savefig(image_folder+'basic.pdf')


