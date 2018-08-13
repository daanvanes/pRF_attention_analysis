# !/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

# import python packages:
from __future__ import division
from sklearn.linear_model import Ridge
from IPython import embed as shell
import numpy as np
import pylab as pl
from scipy.stats import pearsonr,spearmanr,kendalltau
from scipy.signal import fftconvolve
from scipy import ndimage
from scipy import interpolate
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, report_errors
from skimage.morphology import disk
from matplotlib import cm 
import seaborn as sn
from sympy.solvers import solve
from sympy import Symbol, exp
import hrf_estimation as he
import time as t
import os
sn.set(style="ticks")

class gpf(object):
	def __init__(self, design_matrix, max_eccentricity, n_pixel_elements, ssr, rtime,slice_no):
		self.design_matrix = design_matrix
		self.max_eccentricity = max_eccentricity
		self.n_pixel_elements = n_pixel_elements
		self.ssr = ssr
		self.rtime = rtime	
		self.slice = slice_no

		X = np.linspace(-max_eccentricity, max_eccentricity, n_pixel_elements)
		Y = np.linspace(-max_eccentricity, max_eccentricity, n_pixel_elements)
		self.MG = np.meshgrid(X, Y)

	#define model function and pass independent variables x and y as a list
	def twoD_Gaussian(self,  xo, yo, sigma):
		(x,y) = self.MG
		theta=0
		a = (np.cos(theta)**2)/(2*sigma**2) + (np.sin(theta)**2)/(2*sigma**2)
		b = -(np.sin(2*theta))/(4*sigma**2) + (np.sin(2*theta))/(4*sigma**2)
		c = (np.sin(theta)**2)/(2*sigma**2) + (np.cos(theta)**2)/(2*sigma**2)
		gauss = np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
		gauss[disk((self.n_pixel_elements-1)/2)==0] = 0 
		return gauss

	def raw_model_prediction(self,  xo, yo, sigma):
		g = self.twoD_Gaussian(xo, yo, sigma).reshape(self.n_pixel_elements**2)
		return np.dot(self.design_matrix.reshape(-1,self.n_pixel_elements**2), g)

	def hrf_model_prediction(self, xo, yo, sigma,hrf_params):

		rmp = self.raw_model_prediction( xo, yo, sigma)
		original_size = len(rmp)
		rmp = np.repeat(rmp, self.ssr, axis=0)
		xx = np.arange(0,32,self.rtime/float(self.ssr))
		self.hrf_kernel = hrf_params[0] * he.hrf.spmt(xx) +hrf_params[1]* he.hrf.dspmt(xx) +hrf_params[2] * he.hrf.ddspmt(xx)
		if self.hrf_kernel.shape[0] % 2 == 1:
			self.hrf_kernel = np.r_[self.hrf_kernel, 0]
		self.hrf_kernel /= np.abs(self.hrf_kernel).sum()

		# add slice timing correction to fit
		convolved_mp = fftconvolve( rmp, self.hrf_kernel, 'full' )[int(self.slice)::int(self.ssr)][:int(original_size)]

		return convolved_mp, self.hrf_kernel

def fitRidge_for_Dumoulin(design_matrix, timeseries, n_iter = 50, compute_score = False, verbose = True,valid_regressors=[],n_pixel_elements=[], alpha = 1.0):
	"""fitRidge fits a design matrix to a given timeseries.
	It computes the coefficients and returns these coefficients
	plus the correlation between the model fit and timeseries.
	"""
	br = Ridge(alpha = alpha)
	br.fit(design_matrix, timeseries)
	predicted_signal = br.coef_ * design_matrix
	srp = list(spearmanr(timeseries, predicted_signal.sum(axis = 1)))
	srp = [srp[0], -np.log10(srp[1])]

	PRF = np.zeros(n_pixel_elements**2)
	PRF[valid_regressors] = br.coef_
	PRF = np.reshape(PRF,(n_pixel_elements,n_pixel_elements))
	maximum = ndimage.measurements.maximum_position(PRF)

	start_params = {}
	start_params['xo'], start_params['yo'] = maximum[1]/float(n_pixel_elements)*2-1, maximum[0]/float(n_pixel_elements)*2-1

	return start_params, PRF, predicted_signal.sum(axis = 1)

def fit_PRF_on_concatenated_data(data_shared,voxels_in_this_slice,n_TRs,n_slices,fit_on_all_data,plotbool,raw_design_matrices, dm_for_BR,
	valid_regressors, n_pixel_elements_convolved, n_pixel_elements_raw,plotdir,voxno,slice_no,randint,roi,TR,model,hrf_params_shared,all_results_shared,conditions,
	results_frames,	postFix=[],max_eccentricity=1,max_xy = 5,orientations=['0','45','90','135','180','225','270','315','X'],stim_radius = 7.5,
	nuisance_regressors = []):
	"""
	"""

	# grab data for this fit procedure from shared memory
	time_course = np.array(data_shared[:,voxels_in_this_slice][:,voxno])
	hrf_params = np.array(hrf_params_shared[:,voxels_in_this_slice][:,voxno])

	n_orientations = len(orientations)

	# already initialize the final PRF dict
	PRFs = {}

	if fit_on_all_data:

		#########################################################################################################################################################################################################################
		#### Instantiate parameters 
		#########################################################################################################################################################################################################################

		## initiate search space with Ridge prefit
		Ridge_start_params, PRFs['Ridge'], BR_predicted = fitRidge_for_Dumoulin(dm_for_BR, time_course, valid_regressors=valid_regressors, n_pixel_elements=n_pixel_elements_convolved, alpha=1e14)

		## initiate parameters:
		params = Parameters()
		
		# one baseline parameter
		params.add('baseline',value=0.0)

		# two location parameters
		params.add('xo_%s'%conditions[0], value = Ridge_start_params['xo'])
		params.add('yo_%s'%conditions[0], value = Ridge_start_params['yo'])

		params.add('sigma_center_%s'%conditions[0],value=0.1,min=1e-20)#min=1e-201 # this means initialization at 0.1 * 7.5 = 0.75 degrees, with minimum of 0.075 degrees
		params.add('amp_center_%s'%conditions[0],value=0.05,min=1e-20)#min=1e-201 # this is initialized at 0.001

		# surround parameters
		params.add('sigma_surround_%s'%conditions[0],value=0.3,expr='sigma_center_%s+delta_sigma_%s'%(conditions[0],conditions[0])) # surround size should roughly be 5 times that of the center
		params.add('delta_sigma_%s'%conditions[0],value=0.4,min=1e-20) # this difference parameter ensures that the surround is always larger than the center#,min=1e-20000000001
		params.add('amp_surround_%s'%conditions[0],value=-0.005,max=1e-20,expr='-amp_center_%s+delta_amplitude_%s'%(conditions[0],conditions[0])) # initialized at 10% of center amplitude #max=-0.0000000001,
		params.add('delta_amplitude_%s'%conditions[0],value=0.045,min=1e-20) # this difference parameter ensures that the surround is never deeper than the center is high,min=0.0000000001

		# when fitting an OG model, set all surround and delta parameters to 0 and to not vary and set the expression to None, otherwise it will start to vary anyway
		if model == 'OG':	
			params['amp_surround_%s'%conditions[0]].value,params['amp_surround_%s'%conditions[0]].vary,params['amp_surround_%s'%conditions[0]].expr = 0, False, None
			params['delta_sigma_%s'%conditions[0]].vary,params['sigma_surround_%s'%conditions[0]].vary =  False, False
			params['delta_amplitude_%s'%conditions[0]].vary = False
	else:

		#########################################################################################################################################################################################################################
		#### INITIATING PARAMETERS with all results
		#########################################################################################################################################################################################################################

		# grab data for this fit procedure from shared memory
		all_results = np.array(all_results_shared[:,voxels_in_this_slice][:,voxno])

		## initiate parameters:
		params = Parameters()

		# shared baseline param:
		params.add('baseline', value = all_results[results_frames['baseline']])

		# location parameters
		for condition in conditions:
			params.add('xo_%s'%condition, value = all_results[results_frames['xo']])
			params.add('yo_%s'%condition, value = all_results[results_frames['yo']])

			# center parameters:
			params.add('sigma_center_%s'%condition,value=all_results[results_frames['sigma_center']]/stim_radius,min=1e-20) # this means initialization at 0.05/2 * 15 = 1.5 degrees, ,min=0.0084
			params.add('amp_center_%s'%condition,value=all_results[results_frames['amp_center']],min=1e-20) # this is initialized at 0.001 ,min=0.0000000001

			# surround parameters
			params.add('sigma_surround_%s'%condition,value=all_results[results_frames['sigma_surround']]/stim_radius,expr='sigma_center_%s+delta_sigma_%s'%(condition,condition)) # surround size should roughly be 5 times that of the center
			params.add('amp_surround_%s'%condition,value=all_results[results_frames['amp_surround']],max=-1e-20,expr='-amp_center_%s+delta_amplitude_%s'%(condition,condition)) # initialized at 10% of center amplitudemax=-0.0000000001
			params.add('delta_sigma_%s'%condition,value=all_results[results_frames['delta_sigma']],min=1e-20) # this difference parameter ensures that the surround is always larger than the centermin=0.0000000001
			params.add('delta_amplitude_%s'%condition,value=all_results[results_frames['delta_amplitude']],min=1e-20) # this difference parameter ensures that the surround is never deeper than the center is highmin=0.0000000001

			# when fitting an OG model, set all surround and delta parameters to 0 and to not vary and set the expression to None, otherwise it will start to vary anyway
			if model == 'OG':	
				params['amp_surround_%s'%condition].value,params['amp_surround_%s'%condition].vary,params['amp_surround_%s'%condition].expr = 0, False, None
				params['delta_sigma_%s'%condition].vary,params['sigma_surround_%s'%condition].vary = False, False
				params['delta_amplitude_%s'%condition].vary=False

		g = gpf(design_matrix = raw_design_matrices[conditions[0]], max_eccentricity = max_eccentricity, n_pixel_elements = n_pixel_elements_raw, rtime = TR, ssr = 1,slice_no=slice_no)
		
		# recreate PRFs
		this_surround_PRF = g.twoD_Gaussian(all_results[results_frames['xo']],all_results[results_frames['yo']],
			all_results[results_frames['sigma_surround']]/stim_radius) * all_results[results_frames['amp_surround']]
		this_center_PRF = g.twoD_Gaussian(all_results[results_frames['xo']], all_results[results_frames['yo']],
			all_results[results_frames['sigma_center']]/stim_radius) * all_results[results_frames['amp_center']]
		PRFs['All_fit'] = this_center_PRF + this_surround_PRF

	#########################################################################################################################################################################################################################
	#### Prepare fit object and function
	#########################################################################################################################################################################################################################

	# initiate model prediction object
	ssr = np.round(1/(TR/float(n_slices)))
	
	gpfs = {}
	for condition in conditions:
		gpfs[condition] = gpf(design_matrix = raw_design_matrices[condition], max_eccentricity = max_eccentricity, n_pixel_elements = n_pixel_elements_raw, rtime = TR, ssr = ssr,slice_no=slice_no)

	def residual(params,recreate=False):

		# combine all stimulus regressors
		combined_model_prediction = np.ones_like(time_course)*params['baseline'].value
		for ci,condition in enumerate(conditions):
			combined_model_prediction += gpfs[condition].hrf_model_prediction(params['xo_%s'%condition].value, params['yo_%s'%condition].value, 
				params['sigma_center_%s'%condition].value,hrf_params)[0] * params['amp_center_%s'%condition].value
			combined_model_prediction += gpfs[condition].hrf_model_prediction(params['xo_%s'%condition].value, params['yo_%s'%condition].value, 
				params['sigma_surround_%s'%condition].value, hrf_params)[0] * params['amp_surround_%s'%condition].value

		return time_course - combined_model_prediction

	#########################################################################################################################################################################################################################
	#### evalute fit
	#########################################################################################################################################################################################################################

	# optimize parameters
	minimize(residual, params, args=(), kws={},method='powell')

	#########################################################################################################################################################################################################################
	#### Recreate resulting predictions and PRFs with optimized parameters
	#########################################################################################################################################################################################################################

	# initiate model prediction at baseline value
	combined_model_prediction = np.ones_like(time_course) * params['baseline'].value

	# now loop over conditions, create prediction and add to total prediction
	model_predictions = {}
	for ci,condition in enumerate(conditions):
		this_center_model_prediction = gpfs[condition].hrf_model_prediction(params['xo_%s'%condition].value, params['yo_%s'%condition].value, 
			params['sigma_center_%s'%condition].value,hrf_params)[0] * params['amp_center_%s'%condition].value
		this_surround_model_prediction = gpfs[condition].hrf_model_prediction(params['xo_%s'%condition].value, params['yo_%s'%condition].value, 
			params['sigma_surround_%s'%condition].value, hrf_params)[0] * params['amp_surround_%s'%condition].value
		model_predictions[condition] = this_center_model_prediction + this_surround_model_prediction
		combined_model_prediction += model_predictions[condition]

		# recreate PRFs
		this_center_PRF = gpfs[condition].twoD_Gaussian(params['xo_%s'%condition].value, params['yo_%s'%condition].value,
			params['sigma_center_%s'%condition].value) * params['amp_center_%s'%condition].value
		this_surround_PRF = gpfs[condition].twoD_Gaussian(params['xo_%s'%condition].value, params['yo_%s'%condition].value,
			params['sigma_surround_%s'%condition].value) * params['amp_surround_%s'%condition].value
		PRFs[condition] = this_center_PRF + this_surround_PRF


	#########################################################################################################################################################################################################################
	#### Get fit diagnostics
	#########################################################################################################################################################################################################################

	# add fwhm, necessary when fitting DoG
	reconstruction_radius = 10
	this_ssr = 1000 
	t = np.linspace(-reconstruction_radius,reconstruction_radius,this_ssr*reconstruction_radius)
	
	fwhms = {}
	surround_sizes = {}
	for condition in conditions:
		PRF_2D = params['amp_center_%s'%condition].value * np.exp(-t**2/(2*params['sigma_center_%s'%condition].value**2)) + params['amp_surround_%s'%condition].value * np.exp(-t**2/(2*(params['sigma_surround_%s'%condition].value)**2))
		## then, we fit a spline through this line, and get the roots (the fwhm points) of the spline:
		spline=interpolate.UnivariateSpline(range(len(PRF_2D)),PRF_2D-np.max(PRF_2D)/2,s=0)
		## and compute the distance between them
		try:
			fwhms[condition] = ((np.diff(spline.roots())/len(t)*reconstruction_radius) * stim_radius)[0]
		except:
			## when this procedure fails, set fwhm to 0:
			fwhms[condition] = 0
		
		## now find the surround size in the same way
		if (model == 'OG') + (params['amp_surround_%s'%condition].value == 0):
			surround_sizes[condition] = 0
		else:
			spline = interpolate.UnivariateSpline(range(len(PRF_2D)),PRF_2D+np.min(PRF_2D),s=0)
			surround_sizes[condition] = ((np.diff(spline.roots())/len(t)*reconstruction_radius) * stim_radius)[0]

	## EVALUATE OVERALL MODEL FIT QUALITY
	stats = {}
	stats['spearman'] = spearmanr(time_course,combined_model_prediction)[0]
	stats['pearson'] = pearsonr(time_course,combined_model_prediction)[0]
	stats['RSS'] = np.sum((time_course - combined_model_prediction)**2)
	stats['r_squared'] = 1 - stats['RSS']/np.sum((time_course - np.mean(time_course)) ** 2) 
	stats['kendalls_tau'] = kendalltau(time_course,combined_model_prediction)[0]

	## CREATE SEPERATE RESULTS DICT PER CONDITION
	results = {}
	for condition in conditions:
		results[condition] = {}
		results[condition]['baseline'] = params['baseline'].value
		# params from fit
		for key in params.keys():
			if condition in key:
				if condition in key:
					# leave out the condition in the keys (as the results frames are identical across conditions)
					new_key = key[:-len(condition)-1]
				else:
					new_key = key
				results[condition][new_key] = params[key].value

		results[condition]['ecc'] = np.linalg.norm([params['xo_%s'%condition].value,params['yo_%s'%condition].value]) * stim_radius
		results[condition]['sigma_center'] *= stim_radius
		results[condition]['sigma_surround'] *= stim_radius

		# derived params
		results[condition]['polar'] = np.arctan2(params['yo_%s'%condition].value,params['xo_%s'%condition].value)
		results[condition]['fwhm'] = fwhms[condition]
		results[condition]['surround_size'] = surround_sizes[condition]
		results[condition]['SI'] = ((params['amp_surround_%s'%condition].value * (params['sigma_surround_%s'%condition].value**2) ) 
			/ (params['amp_center_%s'%condition].value * (params['sigma_center_%s'%condition].value**2) ))
		
		# if the resulting PRF falls outside of the stimulus radius,
		# set the multiplier here to 0 so that it falls off the retmaps
		if results[condition]['ecc'] < (stim_radius):
			multiplier = stats['r_squared']
		else:
			multiplier = 0.001

		# here for only voxels within stim region:
		results[condition]['real_polar_stim_region'] = np.cos(results[condition]['polar'])*np.arctanh(multiplier)
		results[condition]['imag_polar_stim_region'] = np.sin(results[condition]['polar'])*np.arctanh(multiplier)
		
		# and for all voxels:
		results[condition]['real_polar'] = np.cos(results[condition]['polar'])*np.arctanh(stats['r_squared'])
		results[condition]['imag_polar'] = np.sin(results[condition]['polar'])*np.arctanh(stats['r_squared'])


	#########################################################################################################################################################################################################################
	#### Plot results
	#########################################################################################################################################################################################################################

	if plotbool * (stats['r_squared']>0.0):# (np.random.randint(10)<10):#* (stats['r_squared']>0.1):#(stats['r_squared']>0.1):# * :# :#* (results['ecc'] < 3) :#:# * * randint ) #* :#* )

		n_TRs = n_TRs[0]
		n_runs = int(len(time_course) / n_TRs)
		if fit_on_all_data:
			plot_conditions = ['Ridge','All']
		else:
			plot_conditions = conditions + ['All_fit']
		plot_dir = os.path.join(plotdir, '%s'%roi)
		if not os.path.isdir(plot_dir): os.mkdir(plot_dir)

		f=pl.figure(figsize=(20,8)); rowi = (n_runs+4)

		import colorsys
		colors = np.array([colorsys.hsv_to_rgb(c,0.6,0.9) for c in np.linspace(0,1,3+1)])[:-1]

		for runi in range(n_runs):
			s = f.add_subplot(rowi,1,runi+1)
			pl.plot(time_course[n_TRs*runi:n_TRs*(runi+1)],'-ok',linewidth=0.75,markersize=2.5)#,label='data'
			if not fit_on_all_data:
				for ci, condition in enumerate(conditions):
					pl.plot(model_predictions[condition][n_TRs*runi:n_TRs*(runi+1)]+params['baseline'].value,color=colors[ci],label='%s model'%condition,linewidth=2)				
				pl.plot([0,n_TRs],[params['baseline'].value,params['baseline'].value],color=colors[0],linewidth=1)	
			else:
				pl.plot(combined_model_prediction[n_TRs*runi:n_TRs*(runi+1)],color=colors[0],label='model',linewidth=2)	
			sn.despine(offset=10)
			pl.xlim(0,850)
			if runi == (n_runs-1):
				pl.xlabel('TRs')
			else:
				pl.xticks([])
			if runi == (n_runs/2):
				pl.legend(loc='best',fontsize=8)
				if 'psc' in postFix:
					pl.ylabel('% signal change')
				else:
					pl.ylabel('unkown unit')	
			pl.yticks([int(np.min(time_course)),0,int(np.max(time_course))])	
			pl.ylim([int(np.min(time_course)),int(np.max(time_course))])


		rowi = (n_runs+2)/2
		k = 0
		for ci, condition in enumerate(plot_conditions):
			k+= 1
			s = f.add_subplot(rowi,len(plot_conditions)*2,(rowi-1)*len(plot_conditions)*2+k,aspect='equal')
			pl.imshow(PRFs[condition],origin='lowerleft',interpolation='nearest',cmap=cm.coolwarm)

			pl.axis('off')
			s.set_title('%s PRF'%condition)
			
			k+= 1
			if not (condition == 'Ridge') + (condition == 'All_fit'):
				s = f.add_subplot(rowi,len(plot_conditions)*2,(rowi-1)*len(plot_conditions)*2+k)
				pl.imshow(np.ones((n_pixel_elements_raw,n_pixel_elements_raw)),cmap='gray')
				pl.clim(0,1)
				if model == 'OG':
					s.text(n_pixel_elements_raw/2,n_pixel_elements_raw/2, "\n%s PARAMETERS: \n\nbaseline: %.2f\nsize: %.2f\namplitude: %.6f\n\n\nDERIVED QUANTIFICATIONS: \n\nr-squared: %.2f\necc: %.2f\nFWHM: %.2f"%
						(condition,results[condition]['baseline'],results[condition]['sigma_center'],results[condition]['amp_center'],
							stats['r_squared'],results[condition]['ecc'],results[condition]['fwhm']),
						horizontalalignment='center',verticalalignment='center',fontsize=10,bbox={'facecolor':'white', 'alpha':1, 'pad':10})
				elif model == 'DoG':
					s.text(n_pixel_elements_raw/2,n_pixel_elements_raw/2, "\n%s PARAMETERS: \n\nbaseline: %.2f\nsd center: %.2f\nsd surround: %.2f\namp center: %.6f\namp surround: %.6f\n\nDERIVED QUANTIFICATIONS: \n\nr squared: %.2f\necc: %.2f\nFWHM: %.2f\nsurround size: %.2f\nsupression index: %.2f"
						%(condition,results[condition]['baseline'],results[condition]['sigma_center'],results[condition]['sigma_surround'],results[condition]['amp_center'],
						results[condition]['amp_surround'],stats['r_squared'],results[condition]['ecc'],results[condition]['fwhm'],results[condition]['surround_size'],
						results[condition]['SI']),horizontalalignment='center',verticalalignment='center',fontsize=10,bbox={'facecolor':'white', 'alpha':1, 'pad':10})
				pl.axis('off')

		# pl.tight_layout()
		pl.savefig(os.path.join(plot_dir, 'vox_%d_%d_%d.pdf'%(slice_no,voxno,n_pixel_elements_raw)))
		pl.close()

	return results, stats


	
