
from __future__ import division
from IPython import embed as shell

# import python moduls
import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import numpy as np
import colorsys
from skimage.morphology import disk
import copy
import seaborn as sn
sn.set(style="ticks")

from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, report_errors
import time as t
# import toolbox funcitonality
import os, sys
sys.path.append( os.environ['ANALYSIS_HOME'] )
from Tools.Sessions.Session import *


class modelAFinResults(object):


	def residual(params,these_fix_results,these_stim_results,ecc_size_intercept,ecc_size_slope):

		# voxels = np.arange(10)
		voxels = np.arange(len(these_fix_results))

		res = Parallel(n_jobs = -1, verbose = 0)(delayed(model_attention)
		(these_fix_results[voxno,results_frames['xo']]*7.5,these_fix_results[voxno,results_frames['yo']]*7.5,these_fix_results[voxno,results_frames['sigma_center']]*7.5,
			params['AF_bar_width'].value,params['AF_bar_length'].value,params['AF_fix_size'].value,ecc_size_intercept,ecc_size_slope) 
		for voxno in voxels)#range(len(these_fix_results)))	

		shell()
		residuals = (np.array([these_stim_results[:,results_frames['xo']],these_stim_results[:,results_frames['yo']]]) - np.array(res).T)

		# residuals = np.linalg.norm([these_stim_results[voxels,results_frames['xo']],these_stim_results[voxels,results_frames['yo']]],axis=0) - np.linalg.norm(res,axis=1)


		RSS = np.sum(residuals)

		global iteration
		global start_time
		fit_time = (t.time()-start_time)/60
		iteration += 1
		sys.stdout.write('\rfinished iteration %d, elapsed time: %.2f minutes with RSS of %.2f, '%(iteration,fit_time,RSS)+
			'AF_bar_width = %.2f, AF_bar_length = %.2f, AF_fix_size = %.2f'%(params['AF_bar_width'].value,params['AF_bar_length'].value,params['AF_fix_size'].value))
		sys.stdout.flush()

		return residuals

	def model_attention(fix_PRF_x,fix_PRF_y,fix_PRF_sigma,AF_bar_width,AF_bar_length,AF_fix_size,ecc_size_intercept,ecc_size_slope,res=100):

		"""
		res should be an even number
		"""

		fix_PRF = twodgauss(fix_PRF_x,fix_PRF_y,fix_PRF_sigma,res=res)
		# create the fix AF
		fix_AF = twodgauss(0,0,AF_fix_size,res=res)

		# then, we'll want to invert the effect of the attention on fixation by dividing the fix PRF by the fix AF
		SD = fix_PRF / fix_AF
		# scale SD back to 1
		SD /= np.max(SD)

		# now let's run the bar simulation on this SD
		stim_PRF = []
		all_AFs=[]
		orientations = [0,45,90,135,180,225,270,315]
		n_timepoints = 25 # 24 timepoints + 1 for the start point (which is why we move from 8.25 and not 7.5)
		for oi,orient in enumerate(orientations):
			# the 0 direction is from top to bottom

			y = np.linspace(8.25,-8.25,n_timepoints)
			x = np.zeros(len(y))
			rotation_matrix = np.matrix([[np.cos(np.radians(orient)), -np.sin(np.radians(orient))],[np.sin(np.radians(orient)), np.cos(np.radians(orient))]])
			xys = np.array(np.mat([x,y]).T * rotation_matrix)

			try:
				AFs = np.array([twodgauss(xy[0],xy[1],sigma_y=AF_bar_width,sigma_x=AF_bar_length,res=res,theta=np.radians(orient)) for xy in xys])
				if sp.ndimage.maximum_position(SD)[1] < (res*0.5):
					if sp.ndimage.maximum_position(SD)[0] > (res*0.5):
						for tp in np.arange(n_timepoints):
							if np.max(AFs[tp,int(res/2):,:int(res/2)]) < 0.9: 
								AFs[tp] = np.zeros((res,res))
					else:
						for tp in np.arange(n_timepoints):
							if np.max(AFs[tp,:int(res/2),:int(res/2)]) < 0.9:
								AFs[tp] = np.zeros((res,res))
				else:
					if sp.ndimage.maximum_position(SD)[0] > (res*0.5):
						for tp in np.arange(n_timepoints):
							if np.max(AFs[tp,int(res/2):,int(res/2):]) < 0.9:
								AFs[tp] = np.zeros((res,res))
					else:
						for tp in np.arange(n_timepoints):
							if np.max(AFs[tp,:int(res/2),int(res/2):]) < 0.9:
								AFs[tp] = np.zeros((res,res))

				# restrict the movement of the SD to it's eccentricity axis
				these_RFs = [ (SD*AF)/np.max(SD*AF) for AF in AFs if np.sum(AF) > 0 ]
				RFs_x = np.round((np.argmax(np.mean(these_RFs,axis=1),axis=1)+0.5)/res,2) * 15 - 7.5
				RFs_y = np.round((np.argmax(np.mean(these_RFs,axis=2),axis=1)+0.5)/res,2) * 15 - 7.5
				SD_x = np.round((np.argmax(np.mean(SD,axis=0))+0.5)/res,2) * 15 - 7.5
				SD_y = np.round((np.argmax(np.mean(SD,axis=1))+0.5)/res,2) * 15 - 7.5	

				projected_RFs_xy = [np.dot(np.dot([SD_x,SD_y],[RFs_x[i],RFs_y[i]])/(np.linalg.norm([SD_x,SD_y])**2),[SD_x,SD_y]) for i in range(len(these_RFs))]
				projected_RFs_eccen = np.linalg.norm(projected_RFs_xy,axis=1)
				projected_RFs_size = ecc_size_intercept + projected_RFs_eccen * ecc_size_slope

				projected_RFs = [twodgauss(projected_RFs_xy[i][0],projected_RFs_xy[i][1],projected_RFs_size[i],res=res) for i in range(len(these_RFs))]
				projected_RFs_normalized = [RF/np.max(RF) for RF in projected_RFs]

				stim_PRF.append(np.mean(projected_RFs_normalized,axis=0))
			except:
				stim_PRF.append(np.zeros((res,res)))
			all_AFs.append(np.mean(AFs,axis=0))

		overall_stim_PRF = np.mean(stim_PRF,axis=0)
		if np.sum(stim_PRF) > 0:
			stim_PRF_x = np.round((np.argmax(np.mean(overall_stim_PRF,axis=0))+0.5)/res,2) * 15 - 7.5
			stim_PRF_y = np.round((np.argmax(np.mean(overall_stim_PRF,axis=1))+0.5)/res,2) * 15 - 7.5
		else:
			stim_PRF_x = 100
			stim_PRF_y = 100
		overall_stim_PRF_size = ecc_size_intercept + np.linalg.norm([stim_PRF_x,stim_PRF_y]) * ecc_size_slope
		overall_AF = np.mean(all_AFs,axis=0)

		# stim_PRF_ecc = np.linalg.norm([stim_PRF_x,stim_PRF_y])

		return stim_PRF_x, stim_PRF_y#, overall_stim_PRF_size, overall_AF
		# return overall_stim_PRF


	def model_data_per_roi():

		load_data(PRF=True)
		# these_subjects = subjects + ['super_subject']
		these_subjects = ['super_subject']

		for si, subject in enumerate(these_subjects):
			all_params = {}
			for ri,roi in enumerate(rois_for_plot):

				all_params[roi] = {}
				sys.stdout.write('\n\n##############################################\n'+
				'now fitting AF parameters %s for subject %s\n'%(roi,subject) +
				'##############################################\n\n')

				iteration = 0
				start_time = t.time()
				global start_time
				global iteration

				params = Parameters()

				params.add('AF_bar_width', value= 4, min = 0.01, max = 50)
				params.add('AF_bar_length', value= 100, min = 0.01 , max = 100,vary=False)
				params.add('AF_fix_size', value= 12.5, min = 0.01, max = 100)

				mask = np.ones_like(np.squeeze(all_stats[subject]['Fix'][roi])).astype(bool)
				for sub_condition in ['Fix','Stim']:
					mask *= (np.squeeze(all_stats[subject][sub_condition][roi]) > r_squared_threshold)
					mask *= (np.array(all_results[subject][sub_condition][roi])[:,results_frames['ecc']] > ecc_thresholds[0])
					mask *= (np.array(all_results[subject][sub_condition][roi])[:,results_frames['ecc']] < ecc_thresholds[1])

				these_fix_results = np.array(all_results[subject]['Fix'][roi])[mask,:]
				these_stim_results = np.array(all_results[subject]['Stim'][roi])[mask,:]

				# do a weighted linear regression on the eccen-surf relation to find intercept and slope of this area
				x = np.array(all_results[subject]['Fix'][roi])[mask,results_frames['ecc']]
				data = np.array(all_results[subject]['Fix'][roi])[mask,results_frames['sigma_center']]*7.5
				weights = np.ravel(all_stats[subject]['Fix'][roi])[mask]
				ecc_size_slope, ecc_size_intercept = sp.polyfit(x, data, 1, w= weights)

				minimize(residual, params, args=(), 
					kws={'these_fix_results':these_fix_results,'these_stim_results':these_stim_results,
					'ecc_size_slope':ecc_size_slope,'ecc_size_intercept':ecc_size_intercept,},
					method='powell')
				
				for param in params.keys():
					all_params[roi][param] = params[param].value

			# save all params to disk
			with open(os.path.join((group_plot_dir),'AF_params_%s'%subject), 'w') as f:
				pickle.dump(all_params, f)

	def evaluate_AF_fit():

		load_data(PRF=True)
		# these_subjects = subjects + ['super_subject']
		these_subjects = ['super_subject']

		for si, subject in enumerate(these_subjects):
		
			# load parameters
			with open(os.path.join((group_plot_dir),'AF_params_%s'%subject)) as f:
				picklefile = pickle.load(f)

			for ri,roi in enumerate(rois_for_plot):

				sys.stdout.write('\n\n##############################################\n'+
				'now fitting AF parameters %s for subject %s\n'%(roi,subject) +
				'##############################################\n\n')

				iteration = 0
				start_time = t.time()
				global start_time
				global iteration

				params = Parameters()

				params.add('AF_bar_width', value= 4, min = 0.01, max = 50)
				params.add('AF_bar_length', value= 100, min = 0.01 , max = 100,vary=False)
				params.add('AF_fix_size', value= 12.5, min = 0.01, max = 100)

				mask = np.ones_like(np.squeeze(all_stats[subject]['Fix'][roi])).astype(bool)
				for sub_condition in ['Fix','Stim']:
					mask *= (np.squeeze(all_stats[subject][sub_condition][roi]) > r_squared_threshold)
					mask *= (np.array(all_results[subject][sub_condition][roi])[:,results_frames['ecc']] > ecc_thresholds[0])
					mask *= (np.array(all_results[subject][sub_condition][roi])[:,results_frames['ecc']] < ecc_thresholds[1])

				these_fix_results = np.array(all_results[subject]['Fix'][roi])[mask,:]
				these_stim_results = np.array(all_results[subject]['Stim'][roi])[mask,:]

				# do a weighted linear regression on the eccen-surf relation to find intercept and slope of this area
				x = np.array(all_results[subject]['Fix'][roi])[mask,results_frames['ecc']]
				data = np.array(all_results[subject]['Fix'][roi])[mask,results_frames['sigma_center']]*7.5
				weights = np.ravel(all_stats[subject]['Fix'][roi])[mask]
				ecc_size_slope, ecc_size_intercept = sp.polyfit(x, data, 1, w= weights)

				# minimize(residual, params, args=(), 
				# 	kws={'these_fix_results':these_fix_results,'these_stim_results':these_stim_results,
				# 	'ecc_size_slope':ecc_size_slope,'ecc_size_intercept':ecc_size_intercept,},
				# 	method='powell')
				
				for param in params.keys():
					all_params[roi][param] = params[param].value

			# save all params to disk
			with open(os.path.join((group_plot_dir),'AF_params_%s'%subject), 'w') as f:
				pickle.dump(all_params, f)

				# try some stuff out
				# fix_PRF_xy = [7,0]
				# fix_PRF_size = ecc_size_intercept + np.linalg.norm(fix_PRF_xy) * ecc_size_slope
				# fix_PRF = twodgauss(fix_PRF_xy[0],fix_PRF_xy[1],fix_PRF_size,res=100)
				# modelled_PRFx,modelled_PRFy,modelled_PRFsize,stim_AF = model_attention(fix_PRF_xy[0],fix_PRF_xy[1],fix_PRF_size,
				# 	params['AF_bar_width'].value,params['AF_bar_length'].value,params['AF_fix_size'].value,ecc_size_intercept,ecc_size_slope)
				# modelled_PRF = twodgauss(modelled_PRFx,modelled_PRFy,modelled_PRFsize,res=100)

				# voxno = 30
				# modelled_PRFx,modelled_PRFy,modelled_PRFsize,stim_AF = model_attention(these_fix_results[voxno,results_frames['xo']]*7.5,these_fix_results[voxno,results_frames['yo']]*7.5,these_fix_results[voxno,results_frames['sigma_center']]*7.5,
				# 	params['AF_bar_width'].value,params['AF_bar_length'].value,params['AF_fix_size'].value,ecc_size_intercept,ecc_size_slope)
				# modelled_PRF = twodgauss(modelled_PRFx,modelled_PRFy,modelled_PRFsize)
				# fix_PRF = twodgauss(these_fix_results[voxno,results_frames['xo']]*7.5,these_fix_results[voxno,results_frames['yo']]*7.5,these_fix_results[voxno,results_frames['sigma_center']]*7.5)
				# stim_PRF = twodgauss(these_stim_results[voxno,results_frames['xo']]*7.5,these_stim_results[voxno,results_frames['yo']]*7.5,these_stim_results[voxno,results_frames['sigma_center']]*7.5)
				# fix_AF = twodgauss(0,0,params['AF_fix_size'].value)
				# SD = fix_PRF / fix_AF

				# with sn.axes_style("dark"):
				# 	f=pl.figure(figsize=(8,8))
				# 	s = f.add_subplot(321)
				# 	pl.title('actual fix PRF')
				# 	imshow(fix_PRF,interpolation='nearest',aspect=1,cmap=cm.coolwarm)
				# 	max_pos = sp.ndimage.maximum_position(fix_PRF)
				# 	max_pos_deg = np.round((np.array(max_pos)+0.5)/100,2)*15.-7.5
				# 	pl.axhline(max_pos[0],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(max_pos[1],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(50,color='white',linestyle='-')
				# 	pl.axhline(50,color='white',linestyle='-')
				# 	pl.xticks([max_pos[1]],['%.2f'%max_pos_deg[1]])
				# 	pl.yticks([max_pos[0]],['%.2f'%max_pos_deg[0]])
				# 	ecc = np.linalg.norm([max_pos_deg[1],max_pos_deg[0]])
				# 	pl.text(0.1*100,0.9*100,'ecc: %.2f'%ecc,color='white',fontsize=10)

				# 	# s = f.add_subplot(122)
				# 	# pl.title('modelled stim PRF')
				# 	# imshow(modelled_PRF,interpolation='nearest',aspect=1,cmap=cm.coolwarm)
				# 	# max_pos = sp.ndimage.maximum_position(modelled_PRF)
				# 	# max_pos_deg = np.round((np.array(max_pos)+0.5)/100,2)*15.-7.5
				# 	# pl.axhline(max_pos[0],color='w',linestyle='--',linewidth=1)
				# 	# pl.axvline(max_pos[1],color='w',linestyle='--',linewidth=1)
				# 	# pl.axvline(50,color='white',linestyle='-')
				# 	# pl.axhline(50,color='white',linestyle='-')
				# 	# pl.xticks([max_pos[1]],['%.2f'%max_pos_deg[1]])
				# 	# pl.yticks([max_pos[0]],['%.2f'%max_pos_deg[0]])
				# 	# ecc = np.linalg.norm([max_pos_deg[1],max_pos_deg[0]])
				# 	# pl.text(0.1*100,0.9*100,'ecc: %.2f'%ecc,color='white',fontsize=10)

				# 	s = f.add_subplot(322)
				# 	pl.title('actual stim PRF')
				# 	imshow(stim_PRF,interpolation='nearest',aspect=1,cmap=cm.coolwarm)
				# 	max_pos = sp.ndimage.maximum_position(stim_PRF)
				# 	max_pos_deg = np.round((np.array(max_pos)+0.5)/100,2)*15.-7.5
				# 	pl.axhline(max_pos[0],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(max_pos[1],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(50,color='white',linestyle='-')
				# 	pl.axhline(50,color='white',linestyle='-')
				# 	pl.xticks([max_pos[1]],['%.2f'%max_pos_deg[1]])
				# 	pl.yticks([max_pos[0]],['%.2f'%max_pos_deg[0]])
				# 	ecc = np.linalg.norm([max_pos_deg[1],max_pos_deg[0]])
				# 	pl.text(0.1*100,0.9*100,'ecc: %.2f'%ecc,color='white',fontsize=10)

				# 	s = f.add_subplot(323)
				# 	pl.title('fix AF')
				# 	imshow(fix_AF,interpolation='nearest',aspect=1,cmap=cm.coolwarm)
				# 	max_pos = sp.ndimage.maximum_position(fix_AF)
				# 	max_pos_deg = np.round((np.array(max_pos)+0.5)/100,2)*15.-7.5
				# 	pl.axhline(max_pos[0],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(max_pos[1],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(50,color='white',linestyle='-')
				# 	pl.axhline(50,color='white',linestyle='-')
				# 	pl.xticks([max_pos[1]],['%.2f'%max_pos_deg[1]])
				# 	pl.yticks([max_pos[0]],['%.2f'%max_pos_deg[0]])
				# 	ecc = np.linalg.norm([max_pos_deg[1],max_pos_deg[0]])
				# 	pl.text(0.1*100,0.9*100,'ecc: %.2f'%ecc,color='white',fontsize=10)	

				# 	s = f.add_subplot(324)
				# 	pl.title('stim AF')
				# 	imshow(stim_AF,interpolation='nearest',aspect=1,cmap=cm.coolwarm)
				# 	max_pos = sp.ndimage.maximum_position(stim_AF)
				# 	max_pos_deg = np.round((np.array(max_pos)+0.5)/100,2)*15.-7.5
				# 	pl.axhline(max_pos[0],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(max_pos[1],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(50,color='white',linestyle='-')
				# 	pl.axhline(50,color='white',linestyle='-')
				# 	pl.xticks([max_pos[1]],['%.2f'%max_pos_deg[1]])
				# 	pl.yticks([max_pos[0]],['%.2f'%max_pos_deg[0]])
				# 	ecc = np.linalg.norm([max_pos_deg[1],max_pos_deg[0]])
				# 	pl.text(0.1*100,0.9*100,'ecc: %.2f'%ecc,color='white',fontsize=10)

				# 	s = f.add_subplot(325)
				# 	pl.title('inferred SD')
				# 	imshow(SD,interpolation='nearest',aspect=1,cmap=cm.coolwarm)
				# 	max_pos = sp.ndimage.maximum_position(SD)
				# 	max_pos_deg = np.round((np.array(max_pos)+0.5)/100,2)*15.-7.5
				# 	pl.axhline(max_pos[0],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(max_pos[1],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(50,color='white',linestyle='-')
				# 	pl.axhline(50,color='white',linestyle='-')
				# 	pl.xticks([max_pos[1]],['%.2f'%max_pos_deg[1]])
				# 	pl.yticks([max_pos[0]],['%.2f'%max_pos_deg[0]])
				# 	ecc = np.linalg.norm([max_pos_deg[1],max_pos_deg[0]])
				# 	pl.text(0.1*100,0.9*100,'ecc: %.2f'%ecc,color='white',fontsize=10)		

				# 	s = f.add_subplot(326)
				# 	pl.title('modelled stim PRF')
				# 	imshow(modelled_PRF,interpolation='nearest',aspect=1,cmap=cm.coolwarm)			
				# 	max_pos = sp.ndimage.maximum_position(modelled_PRF)
				# 	max_pos_deg = np.round((np.array(max_pos)+0.5)/100,2)*15.-7.5
				# 	pl.axhline(max_pos[0],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(max_pos[1],color='w',linestyle='--',linewidth=1)
				# 	pl.axvline(50,color='white',linestyle='-')
				# 	pl.axhline(50,color='white',linestyle='-')
				# 	pl.xticks([max_pos[1]],['%.2f'%max_pos_deg[1]])
				# 	pl.yticks([max_pos[0]],['%.2f'%max_pos_deg[0]])
				# 	ecc = np.linalg.norm([max_pos_deg[1],max_pos_deg[0]])
				# 	pl.text(0.1*100,0.9*100,'ecc: %.2f'%ecc,color='white',fontsize=10)