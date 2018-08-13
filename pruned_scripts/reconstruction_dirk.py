from __future__ import division
import numpy as np
import scipy as sp
import pylab as pl
from skimage.morphology import disk
import sys, os
sys.path.append( os.environ['ANALYSIS_HOME'] )
from Tools.Operators.CommandLineOperator import FlirtOperator, BETOperator,InvertFlirtOperator
import glob
from IPython import embed

flirt_results = True
create_reconstruction = False

def two_d_gauss(xo, yo, sigma_x, sigma_y=None, amplitude = 1, theta=0,res=101,max_eccentricity=1):
	
	"""
	Input:
	- xo: float
	- yo: float
	- sigma_x: float
	- sigma_y: float
	- amplitude: float
	- theta: float in degrees
	- res: int 

	Output:
	- 2D array

	This function positions a 2D-Gaussian at [xo,yo] (where the units are defined
	by max_eccentricity), with defined amplitude and sigma_x. 
	If sigma_y is not given, it will produce an isotropic Gaussian.
	Res sets the resolution i.e. the amount of pixels in 
	the 2D output array. 
	"""

	X = np.linspace(-max_eccentricity, max_eccentricity, res)
	Y = np.linspace(-max_eccentricity, max_eccentricity, res)
	MG = np.meshgrid(X, Y)

	(x,y) = MG	
	if sigma_y == None:
		sigma_y = sigma_x
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	gauss = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
	# gauss[disk((res-1)/2)==0] = 0
	if np.isnan(gauss).sum()>0:
		gauss = np.zeros((res,res))
	return gauss

def reconstruction(xo,yo,sizes,weights,res,timecourses):
	"""
	Input:
	- xo: array of pRF xs
	- yo: array of pRF ys
	- sizes: array of pRF sizes
	- weights: array of pRF weights
	- res: resolution (int)
	- timecourses: reconstruction is created for each timepoint

	Output:
	- 2D array

	This function creates a reconstruction based on input pRF parameters and 
	a series of either BOLD timecourses or beta-weights
	"""	

	reconstruction = np.sum([functions.two_d_gauss(xo[voxno],yo[voxno],sigma_x = sizes[voxno], amplitude = timecourses[voxno]*weights[voxno],max_eccentricity=max_eccentricity,res=res) for voxno in range(len(xo))],axis=0)
	distribution = np.sum([functions.two_d_gauss(xo[voxno],yo[voxno],sigma_x = sizes[voxno], amplitude =  weights[voxno],max_eccentricity=max_eccentricity,res=res)  for voxno in range(len(xo))],axis=0)
	result = reconstruction/distribution
	result[(disk((res-1)/2)==False)] = 0

	return result








