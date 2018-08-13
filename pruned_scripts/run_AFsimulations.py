# !/usr/bin/env python
# encoding: utf-8

from datetime import date
import os, sys, datetime
import subprocess, logging
from IPython import embed as shell
import numpy as np

sys.path.append( os.environ['ANALYSIS_HOME'] )
from AFSimulations import *

#########################################
# SETUP OPTIONS
#########################################
import socket
if socket.gethostname() == 'aeneas.psy.vu.nl':
	plot_dir = os.path.join('/home','shared','PRF_2','AF_simulations')
	data_dir = os.path.join('/home','shared','PRF_2','data','_group_level','plots','AF_modeling')
else:
	plot_dir = os.path.join('/projects','0','pqsh283','PRF_2','AF_simulations')
	data_dir = os.path.join('/projects','0','pqsh283','PRF_2','data','_group_level','plots','AF_modeling')


res = 131
model_area_increase_factor = 1.5

stim_radius = 3.6
ecc_thresholds=[0,3.3]
# TR = 1.6
# bar_pass_dur_TR = 24

# Initiate AF simulation object
AFO = AFSimulations(plot_dir,data_dir,res,stim_radius,ecc_thresholds)

# AF_timecourse(plot_dir)

#########################################
# RUN FUNCTIONS
#########################################

AFO.simulate_single_RF(
	
	plot=True,						# create shift plot?
	animate=True,					# create model animation?

	SDxy = [3,0],  					# SD x and y
	model_area_increase_factor=model_area_increase_factor,
	use_fit_params = False,
	roi='V1',

	coordinate_system = 'max_scaling',
	logtype='combined',
	ecc_logliness=0.001,

	circle_mask = True,
	center_method = 'maxval',
	upsample=10,

	# if not use_fit_params, define here:	
	eccen_size_intercept=0.8,#0.48276512036023028,		# This determines SD size. Also, when eccen_dir=True, this determines size of RF after central position has been projected onto eccentricity axis
	eccen_size_slope=0.2,#0.1989041747236992,				# 
	bar_AF_length=500,

	AF_ecc_size_intercept=0.5,#0.50003698485237047,		# this is only for when the bar should be pulled through filter bank
	AF_ecc_size_slope=0.07,#0.067309184787422982,			# this is only for when the bar should be pulled through filter bank

	AF_shape='bar_convolved',			# 'isotropic','anisotropic','bar_convolved',bar_convolved_bank'

	# old
	distance_slope=1,
	ecc_polar_ratio=0.5,
	restriction = 'none',#'none','only_ecc','only_angle'
	AF_slopes='positive',#'positive','zero'
	fix_AF_size=2,#8.8154201805688253,				# determines strength of attend fixation pull
	bar_AF_width=2,									# this determines the strength of the attend bar pull
	distance_effect=False, 			# use this option to get reduced effect of AF with distance
	distance_factor=5,				# use this option to get reduced effect of AF with distance
	restrict_x=False,				# When centre of AF crosses x-boundary: cancel AF effect
	restrict_y=False,				# When centre of AF crosses y-boundary: cancel AF effect
	eccen_dir=False,					# this projects the found RF to the ecc axis

	)

# for bar_AF_intercept in [0.5]:#[0.5,1,2,4]:
# 	for bar_AF_slope in [0.07]:#[0.05,0.1,0.5,1,2]:#0.5,0.75,1.0]:#,1.25,1.5,1.75]:
# 		for fix_AF_size in [2]:#[0.5,1,1.5,2,4,8]:#,11,12,13,14]:

# 			AFO.simulate_population_of_RFs(
				
# 				circle_mask = False,
# 				n_pRFs = 500,					# amount of PRFs to simulate per 'visual area'
# 				fix_AF_size = fix_AF_size,		# determines strength of attend fixation pull
# 				AF_ecc_size_intercept=bar_AF_intercept,#2,#0.50003698485237047,		# 
# 				AF_ecc_size_slope=bar_AF_slope,#0.5,#0.067309184787422982,			# 
								
# 				AF_shape='bar_convolved',		# 'isotropic','anisotropic','bar_convolved',bar_convolved_with_filterbank'
# 				roi_names = ['V3AB'],#,'V4','LO','IPS'],
# 				use_fit_params = False,
# 				model_name = 'bar_convolved_positive_AF_slope_coord_sys_max_scaling_combined_proj_False_maf_4_res_131_circle_mask_True_meanval_hV4',

# 				# if use_fit_params is False, use:
# 				logtype='combined', #'separate'
# 				center_method = 'maxval',
# 				upsample=10,

# 				ecc_size_slopes= [[0.8,0.2]],#[0.48276512036023028,0.1989041747236992]],#,[0.8,0.4],[0.9,0.5]], # [slope,intercept] for different 'visual areas'[0.5,0.1],[0.6,0.2],

				
# 				coordinate_system = 'cartesian',
# 				ecc_logliness=0.001,#0.001,	
# 				model_area_increase_factor=4,

# 				n_jobs=1,

# 				# old
# 				bar_AF_length = 1e6,			# this only takes effect when AF_shape == 'anisotropic'
# 				bar_AF_width = 2,	#bar_AF_width		,					# this determines the strength of the attend bar pull
# 				ecc_polar_ratio=0.5,
# 				distance_slope=1,
# 				restrict_x=False,				# When centre of AF crosses x-boundary: cancel AF effect
# 				restrict_y=False,				# When centre of AF crosses y-boundary: cancel AF effect
# 				eccen_dir=False,					# this projects the found RF to the ecc axis

# 				distance_effect=False,			# use this option to get reduced effect of AF with distance
# 				distance_factor=5,				# use this option to get reduced effect of AF with distance
# 				)


