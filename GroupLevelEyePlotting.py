from __future__ import division
import numpy as np
import os
from IPython import embed as shell
import tables as tb
import pandas as pd
from math import *
from matplotlib import pyplot as pl
import seaborn as sn
sn.set_style('ticks')
import copy
import matplotlib as mpl
mpl.rc_file_defaults()
from utilities import CustomStatUtilities

class GroupLevelEyePlotting(object):
	
	def __init__(self,group_dir,subjects,task_names,screen_size_pixs,screen_distance_cm,screen_size_cm,
		stim_radius_ratio,bar_width_ratio,num_steps,ci_factor,TR,sample_rate,comparison_colors):

		self.task_names = task_names
		self.subjects = subjects
		self.group_dir = group_dir
		self.group_h5 = os.path.join(group_dir,'group_level.hdf5')

		self.screen_size_pixs =screen_size_pixs
		self.screen_distance_cm = screen_distance_cm
		self.screen_size_cm = screen_size_cm
		self.stim_radius_ratio =stim_radius_ratio
		self.bar_width_ratio =bar_width_ratio
		self.num_steps =num_steps
		self.comparison_colors=comparison_colors

		self.screen_size_degrees = [np.degrees(np.arctan((self.screen_size_cm[0]/2.)/self.screen_distance_cm))*2,np.degrees(np.arctan((self.screen_size_cm[1]/2.)/self.screen_distance_cm))*2]
		self.stim_radius_cm = self.screen_size_cm[1]*0.25
		self.stim_radius_pixs = self.screen_size_pixs[1]*0.25
		self.stim_radius_degrees = np.degrees(np.arctan((self.stim_radius_cm)/self.screen_distance_cm))
		self.pixs_per_degree = self.stim_radius_pixs/self.stim_radius_degrees
		# self.pixs_per_degree = np.mean(np.array(self.screen_size_pixs)/np.array(self.screen_size_degrees))
		# self.stim_radius_degrees = self.screen_size_degrees[1]*self.stim_radius_ratio*0.5
		self.bar_width_cm = self.stim_radius_cm*2*self.bar_width_ratio
		self.bar_width_degrees = np.degrees(np.arctan((self.bar_width_cm/2.)/self.screen_distance_cm))*2
		self.step_size = (self.stim_radius_degrees*2+self.bar_width_degrees)/self.num_steps
		self.ci_factor=ci_factor
		self.TR = TR
		self.sample_rate = sample_rate

		self.TRs_per_trial = 24

	########################
	#### CREATE PLOT DIR
	########################

	def create_plot_dir(self,plot_type):
		"""Creates an empty dir at self.group_dir/plot_type"""
		self.group_plot_dir = os.path.join(self.group_dir,'plots')
		if not os.path.isdir(self.group_plot_dir): 
			os.mkdir(self.group_plot_dir)
		this_plot_dir = os.path.join(self.group_plot_dir,plot_type)
		if not os.path.isdir(this_plot_dir):
			os.mkdir(this_plot_dir)


	def rotate_eye_pos(self,xy,angle):

		ccw_rotation_matrix = np.matrix([[cos(angle), -sin(angle)],[sin(angle), cos(angle)]])
		rotated_xy = ccw_rotation_matrix * np.matrix(xy).T

		return rotated_xy

	########################
	#### PLOTS
	########################

	def test_eye_rotation(self,):

		n_datapoints = 20
		# let's create some data that runs from left to right
		x = np.linspace(-10,10,n_datapoints)
		y = np.zeros(n_datapoints)
		order = np.linspace(0.1,0.5,n_datapoints)
		rgba_colors = np.zeros((n_datapoints,4))
		# for red the first column needs to be one
		rgba_colors[:,0] = 1.0
		# the fourth column needs to be your alphas
		rgba_colors[:, 3] = order

		# first let's plot the data
		pl.figure(figsize=(3,3))
		pl.subplot(111)
		pl.scatter(x,y,color=rgba_colors)
		pl.xlim(-10,10)
		pl.ylim(-10,10)

		# and now how the positions are rotated
		pl.figure(figsize=(9,9))
		orients = [0,45,90,135,180,225,270,315]
		for i,orient in enumerate(orients):
			rotated_pos = self.rotate_eye_pos(np.array([x,y]).T,np.radians(orient))
			pl.subplot(3,3,i+1)
			pl.title(orient)
			pl.scatter(rotated_pos[0],rotated_pos[1],color=rgba_colors)
			pl.xlim(-10,10)
			pl.ylim(-10,10)
		pl.show()


	def plot_rotated_eye_pos(self,average_per_TR=True):

		self.create_plot_dir('plot_rotated_eye_pos')
		self.CUO = CustomStatUtilities()

		## first rotate the data to bar position per subject and add to subject array
		all_trial_median = {}
		all_trial_std = {}
		all_total_std = {}

		for subject in self.subjects:
			
			print 'computing rotated eye position for subject %s'%subject

			h5 = tb.open_file(self.group_h5,'r')
			aliases = ['PRF_run_%d'%(ri+1) for ri in range(len(h5.list_nodes('/%s'%subject)))]
			h5.close()
			dict_keys = np.ravel([['%s_%s'%(task,direction) for task in self.task_names] for direction in ['orth_dir','bar_dir','norm']])
			eye_data_dict = {key: [] for key in dict_keys}
			# let's define one trial duration for all:
			num_samples = int(self.num_steps*self.TR*self.sample_rate)

			with pd.get_store(self.group_h5) as h5_file:

				for alias in aliases:
					exp_start_time = np.array(h5_file['/%s/%s/trial_phases'%(subject,alias)]['trial_phase_EL_timestamp'])[2]
					phases = np.array(h5_file['%s/%s/trial_phases'%(subject,alias)]['trial_phase_index'])
					stim_on_times = (np.array(h5_file['%s/%s/trial_phases'%(subject,alias)]['trial_phase_EL_timestamp'])[phases==3] - exp_start_time)/self.sample_rate
					stim_off_times = (np.array(h5_file['%s/%s/trial_phases'%(subject,alias)]['trial_phase_EL_timestamp'])[phases==4] - exp_start_time)/self.sample_rate
					task_indices = np.array(h5_file['%s/%s/parameters'%(subject,alias)]['task_index']).astype('int32')
					measured_eye = h5_file['%s/%s/block_0'%(subject,alias)].keys()[1][0]
					xy_data = np.array([np.array(h5_file['%s/%s/block_0'%(subject,alias)]['%s_gaze_x_int_cleaned'%measured_eye]),np.array(h5_file['%s/%s/block_0'%(subject,alias)]['%s_gaze_y_int_cleaned'%measured_eye])]).T
					time = (np.array(h5_file['%s/%s/block_0'%(subject,alias)]['time']) - exp_start_time)/self.sample_rate
					orientations = np.round(np.degrees(np.array(h5_file['%s/%s/parameters'%(subject,alias)]['orientation']))).astype('int32')

				for ti in range(len(stim_on_times)):
					
					start_index = np.argmin(np.abs(time-stim_on_times[ti]))
					end_index = start_index + num_samples
					# get the xy positions for this trial
					these_xy_data = xy_data[start_index:end_index]
					# get the median xy position for this trial
					median_gaze = np.nanmedian(these_xy_data,axis=0)
					# median-center the xy data
					these_xy_data -= median_gaze

					# now, we'll want to rotate these xy positions to a top-to-bottom pass
					# we'll need a counter clock wise rotation matrix, to reverse the orientation
					# of the bar back to the 0 degree pass. Only rotate in bar-present trials however.

					if self.task_names[task_indices[ti]] != 'Fix_no_stim':
						orient = np.radians(orientations[ti])
					else:
						orient = 0
					
					rotated_xy= self.rotate_eye_pos(these_xy_data,orient)

					# ccw_rotation_matrix = np.matrix([[cos(orient), -sin(orient)],[sin(orient), cos(orient)]])
					# rotated_xy = ccw_rotation_matrix * np.matrix(these_xy_data).T

					# and convert to visual degrees
					rotated_xy /= self.pixs_per_degree

					# if averaging over TRs:
					if average_per_TR:
						eye_samples_per_TR = int(self.sample_rate * self.TR)
						rotated_xy = np.squeeze(np.swapaxes(np.array([np.nanmean(rotated_xy[:,TRi*eye_samples_per_TR:(TRi+1)*eye_samples_per_TR],axis=1) for TRi in range(self.TRs_per_trial)]),0,1))

					# as we rotated all data to the top-to-bottom pass, the x data has become
					# irrelevant, and all information is contained in the y data
					# now let's append this to the correct dict
					eye_data_dict[self.task_names[task_indices[ti]]+'_orth_dir'].append(np.squeeze(rotated_xy[0,:]))
					eye_data_dict[self.task_names[task_indices[ti]]+'_bar_dir'].append(np.squeeze(rotated_xy[1,:])) # the y dimension now refers to bar-direction
					eye_data_dict[self.task_names[task_indices[ti]]+'_norm'].append(np.linalg.norm(rotated_xy,axis=0))
				
			# median over trials for this subject
			trial_median_eye_position_dict = {}
			trial_std_eye_position_dict = {}
			total_std_eye_position_dict = {}
			for task in dict_keys:
				# median across trials
				trial_median_eye_position_dict[task] = np.nanmedian(eye_data_dict[task],axis=0)
				# std for all data together for certain trialtype
				trial_std_eye_position_dict[task] = np.nanstd(eye_data_dict[task],axis=0) 
				# sd for all data taken together
				total_std_eye_position_dict[task] = np.nanstd(eye_data_dict[task]) 

			# now also add a variable to totay_std_eye_position_dict for std over all tasks together
			total_std_eye_position_dict['Stim_norm'] = np.nanstd([np.ravel(eye_data_dict[task]) for task in dict_keys if (not 'Fix_no_stim' in task)*('norm' in task)])

			all_trial_median[subject] = trial_median_eye_position_dict
			all_trial_std[subject] = trial_std_eye_position_dict
			all_total_std[subject] = total_std_eye_position_dict

		print 'creating plot'
		import colorsys
		# condition_colors = np.array([colorsys.hsv_to_rgb(c,0.6,0.9) for c in np.linspace(0.0,1,len(self.task_names)+1)])[:-1]


		# for the plot, let's recreate y position of the bar
		if average_per_TR:
			num_samples = self.TRs_per_trial
		sample_time = np.linspace(0,self.num_steps*self.TR,num_samples)
		# let's recreate y position of bar
		bar_mid = np.repeat(np.linspace(self.stim_radius_degrees+self.bar_width_degrees*0.5,-self.stim_radius_degrees-self.bar_width_degrees*0.5,self.num_steps,endpoint=True),2)
		bar_times = np.repeat(np.linspace(0,np.max(sample_time),(len(bar_mid)+2)/2),2)[1:-1]
		bar_upper = bar_mid + self.bar_width_degrees*0.5
		bar_lower = bar_mid - self.bar_width_degrees*0.5
		

		# and create a plot with 4 subplots for each task
		f = pl.figure(figsize=(2,1.5))
		spc = 0

		direction_names = {
		'orth_dir': 'orthogonal direction',
		'bar_dir': 'colinear direction'
		}

		condition_combinations = {
		'Color':['Color'],
		'Speed':['Speed'],
		'Fix':['Fix']
		}


		for ci, comparison in enumerate(['position','variability']):
			s = f.add_subplot(2,1,ci+1)
			sn.despine(offset=2)

			for taski, combo in enumerate(['Fix','TF','Color']):

				linestyle='-'
				pl.title('%s'%(comparison))

				for direction in ['bar_dir']:#'orth_dir',
					
					median_data = [];ci_data=[]

					if comparison == 'position':
						these_data = np.array([all_trial_median[subject][combo+'_'+direction] for subject in self.subjects])
					else:
						these_data = np.array([all_trial_std[subject][combo+'_'+direction] for subject in self.subjects])
					mean_data, se_data, p,N = self.CUO.bootstrap(data=these_data.T,test_value=0,ci_factor=self.ci_factor,detect_inliers=False)

					# the data
					pl.plot(sample_time,mean_data,ls=linestyle,linewidth=2,label=combo,color=self.comparison_colors[combo])#,alpha=0.5)
					pl.fill_between(sample_time,se_data[:,0],se_data[:,1],alpha=0.1,color=self.comparison_colors[combo])

				if ci == 0:
					pl.ylim(-0.25,0.25)
					pl.yticks([-0.25,0,0.25])
				else:
					pl.ylim(0.1,0.75)
					pl.yticks([0.1,0.75])
					pl.xlabel('time (s)')
				pl.ylabel('dva')
				pl.xticks([0,np.round(np.max(sample_time),1)])

		pl.tight_layout(pad=0)
		pl.savefig(os.path.join(self.group_plot_dir,'plot_rotated_eye_pos', 'Eye_position_relative_to_bar.pdf'))
		pl.close()

		# and write out a file for the per TR data
		direction = 'bar_dir'
		# now combine conditions

		# add attend stimulus condition
		for subject in self.subjects:
			# for median data
			col_data = all_trial_median[subject]['Color_'+direction]
			speed_data = all_trial_median[subject]['TF_'+direction]
			stim_data = np.mean([col_data,speed_data],axis=0)
			all_trial_median[subject]['Stim_'+direction] = stim_data

			# for std data
			col_data = all_trial_std[subject]['Color_'+direction]
			speed_data = all_trial_std[subject]['TF_'+direction]
			stim_data = np.mean([col_data,speed_data],axis=0)
			all_trial_std[subject]['Stim_'+direction] = stim_data

		tasks_to_write = ['Stim','Fix','Color','TF']
		column_titles = np.ravel([[task+'_'+str(TRi) for TRi in range(self.TRs_per_trial)] for task in tasks_to_write])

		# for median data
		data = np.reshape([[all_trial_median[subject][task+'_'+direction] for task in tasks_to_write] for subject in self.subjects],(len(self.subjects),-1))
		df = pd.DataFrame(data ,columns=column_titles,index=self.subjects)
		df.to_csv(os.path.join(self.group_plot_dir,'plot_rotated_eye_pos','eye_position_median_per_TR.csv'))	
		
		# for std data
		data = np.reshape([[all_trial_std[subject][task+'_'+direction] for task in tasks_to_write] for subject in self.subjects],(len(self.subjects),-1))
		df = pd.DataFrame(data ,columns=column_titles,index=self.subjects)
		df.to_csv(os.path.join(self.group_plot_dir,'plot_rotated_eye_pos','eye_position_std_per_TR.csv'))	





