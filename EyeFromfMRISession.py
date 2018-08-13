# !/usr/bin/env python
# encoding: utf-8

from __future__ import division
import datetime, os, sys
from Tools.Operators.HDFEyeOperator import *
from Tools.Operators.CommandLineOperator import *
from joblib import Parallel, delayed
from Tools.Operators.EyeSignalOperator import detect_saccade_from_data
from itertools import chain
import seaborn as sn
import numpy.linalg as LA
import colorsys
from Tools.Sessions.Session import *
from Tools.Operators.EyeSignalOperator import detect_saccade_from_data

from pylab import *
import numpy as np

class EyeFromfMRISession(object):
	"""eyePreprocessing"""
	def __init__(self, PRFsession, subject, experiment_name,  project_directory, loggingLevel = logging.DEBUG, sample_rate = 1000.0):

		# set these values!!!
		self.screen_size_pixs = [1920,1080]
		self.screen_distance_cm = 156
		self.screen_size_cm = [69.84,39.29]
		self.stim_radius_ratio = 0.5
		self.bar_width_ratio = 0.125
		self.num_steps = 24

		self.ci_factor=1.96

		# derived measures
		self.screen_size_degrees = [np.degrees(arctan((self.screen_size_cm[0]/2)/self.screen_distance_cm))*2,np.degrees(arctan((self.screen_size_cm[1]/2)/self.screen_distance_cm))*2]
		self.pixs_per_degree = np.mean(np.array(self.screen_size_pixs)/np.array(self.screen_size_degrees))
		self.stim_radius_degrees = self.screen_size_degrees[1]*self.stim_radius_ratio*0.5
		self.bar_width_degrees = self.stim_radius_degrees*self.bar_width_ratio
		self.step_size = self.stim_radius_degrees*2/self.num_steps

		self.subject = subject
		self.experiment_name = experiment_name
		# self.experiment = experiment_nr
		# self.version = version
		try:
			os.mkdir(os.path.join(project_directory, experiment_name))
			os.mkdir(os.path.join(project_directory, experiment_name, self.subject.initials))
		except OSError:
			pass
		self.project_directory = project_directory
		self.base_directory = os.path.join(self.project_directory, self.experiment_name, self.subject.initials)
		self.plot_dir = os.path.join(self.base_directory,'figs')
		self.create_folder_hierarchy()
		self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
		self.group_lvl_hdf5_filename = os.path.join(self.project_directory, self.experiment_name, '_group_level', 'group_level.hdf5')
		self.ho = HDFEyeOperator(self.hdf5_filename)
		self.velocity_profile_duration = self.signal_profile_duration = 100
		self.loggingLevel = loggingLevel
		self.logger = logging.getLogger(self.__class__.__name__)
		self.logger.setLevel(self.loggingLevel)
		addLoggingHandler(logging.handlers.TimedRotatingFileHandler(os.path.join(self.base_directory, 'log', 'sessionLogFile.log'), when='H', delay=2, backupCount=10), loggingLevel=self.loggingLevel)
		loggingLevelSetup()
		for handler in logging_handlers:
			self.logger.addHandler(handler)
		
		self.sample_rate = sample_rate
		self.PRFsession = PRFsession
		
	def create_folder_hierarchy(self):
		"""createFolderHierarchy does... guess what."""
		this_dir = self.project_directory
		for d in [self.experiment_name, self.subject.initials]:
			try:
				this_dir = os.path.join(this_dir, d)
				os.mkdir(this_dir)
			except OSError:
				pass

		for p in ['raw','processed','figs','log']:
			try:
				os.mkdir(os.path.join(self.base_directory, p))
			except OSError:
				pass
	
	def delete_hdf5(self):
		os.system('rm {}'.format(os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')))
	
	def import_raw_data(self, edf_files, aliases):
		"""import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
		for (edf_file, alias,) in zip(edf_files, aliases):
			self.logger.info('importing file ' + edf_file + ' as ' + alias)
			ExecCommandLine('cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"'))
	
	def import_all_data(self, aliases):
		"""import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
		for alias in aliases:
			#first check if there's already a node with this name

			if os.path.isfile(self.hdf5_filename):
				h5file = open_file(self.hdf5_filename, mode = "r+")
				try:
					h5file.remove_node(where='/',name=alias,recursive=True)
				except:
					pass

			self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))
			self.ho.edf_message_data_to_hdf(alias=alias)
			self.ho.edf_gaze_data_to_hdf(alias=alias)
	
	def write_trial_timing_text_files(self,aliases,condition):

		for alias, run in zip(aliases,[self.PRFsession.runList[i] for i in self.PRFsession.conditionDict[condition]]):

			with pd.get_store(self.hdf5_filename) as h5_file:

				# timing
				exp_start_time = np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[0]
				phases = np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_index'])

				if condition == 'Mapper':
					task_indices = np.array(h5_file['%s/parameters'%(alias)]['task']).astype('int32')
					orientations = np.zeros_like(task_indices)
					stim_on_times = (np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[phases==2] - exp_start_time)/self.sample_rate
					stim_off_times = (np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[phases==3] - exp_start_time)/self.sample_rate

				elif condition == 'PRF':
					task_indices = np.array(h5_file['%s/parameters'%(alias)]['task_index']).astype('int32')
					orientations = np.round(np.degrees(np.array(h5_file['%s/parameters'%(alias)]['orientation']))).astype('int32')
					stim_on_times = (np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[phases==3] - exp_start_time)/self.sample_rate
					stim_off_times = (np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[phases==4] - exp_start_time)/self.sample_rate
			
			# write trial timings
			valid_trials = np.ones(len(task_indices)).astype(bool)
			trial_times = np.array([task_indices,stim_on_times,stim_off_times,valid_trials,orientations]).T

			np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'trial_times.txt'), trial_times, delimiter="\t",fmt='%.3f')

			# also create fsl readable format for mapper
			if condition == 'Mapper':

				for task_condition in self.PRFsession.task_names['Mapper']:

					task_index = np.where(np.array(self.PRFsession.task_names['Mapper'])==task_condition)[0][0]
					mask = (task_indices==task_index) 
					exec("%s = np.array([stim_on_times[mask],stim_off_times[mask]-stim_on_times[mask],np.ones_like(stim_on_times)[mask]]).T"%(task_condition))
					np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'%s.txt'%task_condition), eval(task_condition), delimiter="\t",fmt='%.3f')

	def write_blink_text_files(self,aliases,condition):

		for alias, run in zip(aliases,[self.PRFsession.runList[i] for i in self.PRFsession.conditionDict[condition]]):

			with pd.get_store(self.hdf5_filename) as h5_file:

				# timing
				exp_start_time = np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[0]
				blink_start_times = (np.array(h5_file['%s/blinks_from_message_file'%(alias)]['start_timestamp'])- exp_start_time)/self.sample_rate
				valid_blinks = (blink_start_times>0)
				blink_start_times = blink_start_times[valid_blinks]
				blink_durations = (np.array(h5_file['%s/blinks_from_message_file'%(alias)]['duration']))/self.sample_rate
				blink_durations = blink_durations[valid_blinks]

			blink_times = np.array([blink_start_times,blink_durations,np.ones(len(blink_durations))]).T

			np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'blink_times.txt'), blink_times, delimiter="\t",fmt='%.3f')

	def write_button_text_files(self,aliases,condition):
		
		for alias, run in zip(aliases,[self.PRFsession.runList[i] for i in self.PRFsession.conditionDict[condition]]):

			with pd.get_store(self.hdf5_filename) as h5_file:
				
				exp_start_time = np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[0]
				phases = np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_index'])
				buttons_pressed = np.array(h5_file['%s/buttons'%alias]['button_pressed'])
				button_press_times_L = (np.array(h5_file['%s/buttons'%alias]['EL_timestamp'])[buttons_pressed==4] - exp_start_time) / self.sample_rate # the 5th letter in the alphabet is e
				button_press_times_R = (np.array(h5_file['%s/buttons'%alias]['EL_timestamp'])[buttons_pressed==1] - exp_start_time) / self.sample_rate # the 2nd letter in the alphabet is b

			# and add button press text files
			button_presses_L = np.array([button_press_times_L,np.ones_like(button_press_times_L)*0.25,np.ones_like(button_press_times_L)]).T
			button_presses_R = np.array([button_press_times_R,np.ones_like(button_press_times_R)*0.25,np.ones_like(button_press_times_R)]).T
			np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'button_presses_L.txt'), button_presses_L, delimiter="\t",fmt='%.3f')
			np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'button_presses_R.txt'), button_presses_R, delimiter="\t",fmt='%.3f')

	def write_transient_text_files(self,aliases,condition):
		
		for alias, run in zip(aliases,[self.PRFsession.runList[i] for i in self.PRFsession.conditionDict[condition]]):

			with pd.get_store(self.hdf5_filename) as h5_file:
			
				exp_start_time = np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[0]
				phases = np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_index'])

				if condition == 'Mapper':
					transient_times = (np.array(h5_file['%s/mapper_transients'%(alias)]['EL_timestamp']) - exp_start_time) / self.sample_rate
				elif condition == 'PRF':
					task_indices = np.array(h5_file['%s/parameters'%(alias)]['task_index']).astype('int32')
					fns_trial_start =((np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[phases==1] - exp_start_time)/self.sample_rate)[task_indices==3]
					fns_trial_end = ((np.array(h5_file['%s/trial_phases'%(alias)]['trial_phase_EL_timestamp'])[phases==4] - exp_start_time)/self.sample_rate)[task_indices==3]
			
					transient_type = np.array(h5_file['%s/transients'%(alias)]['transient_type'])
					transient_times_0 = (np.array(h5_file['%s/transients'%(alias)]['EL_timestamp'])[(transient_type==0)+(transient_type==3)] - exp_start_time) / self.sample_rate
					transient_times_1 = (np.array(h5_file['%s/transients'%(alias)]['EL_timestamp'])[transient_type==1] - exp_start_time) / self.sample_rate
					transient_times_2 = (np.array(h5_file['%s/transients'%(alias)]['EL_timestamp'])[transient_type==2] - exp_start_time) / self.sample_rate

			# and add button press text files
			if condition == 'Mapper':
				transient_times = np.array([transient_times,np.ones_like(transient_times)*self.PRFsession.TR/3.,np.ones_like(transient_times)]).T
				np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'transient_times.txt'), transient_times, delimiter="\t",fmt='%.3f')
			elif condition == 'PRF':
				# get rid of Color and Speed transient times that fall within fns trials
				bad_transients_1 = np.zeros_like(transient_times_1).astype(bool)
				bad_transients_2 = np.zeros_like(transient_times_2).astype(bool)
				for start, end in zip(fns_trial_start,fns_trial_end):
					for t1i, t1 in enumerate(transient_times_1):
						if (t1 > start) * (t1 < end):
							bad_transients_1[t1i] = True
					for t2i, t2 in enumerate(transient_times_2):
						if (t2 > start) * (t2 < end):
							bad_transients_2[t2i] = True
				transient_times_1 = transient_times_1[bad_transients_1==False]
				transient_times_2 = transient_times_2[bad_transients_2==False]

				print 'there were %d fix events, %d color transients and %d speed transients in %s'%(len(transient_times_0),len(transient_times_1),len(transient_times_2),alias)

				transient_times_0 = np.array([transient_times_0,np.ones_like(transient_times_0)*self.PRFsession.TR/3.,np.ones_like(transient_times_0)]).T
				transient_times_1 = np.array([transient_times_1,np.ones_like(transient_times_1)*self.PRFsession.TR/3.,np.ones_like(transient_times_1)]).T
				transient_times_2 = np.array([transient_times_2,np.ones_like(transient_times_2)*self.PRFsession.TR/3.,np.ones_like(transient_times_2)]).T
				np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'transient_times_0.txt'), transient_times_0, delimiter="\t",fmt='%.3f')
				np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'transient_times_1.txt'), transient_times_1, delimiter="\t",fmt='%.3f')
				np.savetxt(os.path.join(self.PRFsession.runFolder(stage = 'processed/mri', run = run),'transient_times_2.txt'), transient_times_2, delimiter="\t",fmt='%.3f')

	def add_to_group_level_hdf5(self,aliases,condition):


		# define files
		import h5py
		try:
			group_lvl_h5 = h5py.File(self.group_lvl_hdf5_filename, 'r+')
		except:
			group_lvl_h5 = h5py.File(self.group_lvl_hdf5_filename, 'w')
		subject_h5_file = h5py.File(self.hdf5_filename, 'r')

		# delete this subject in the group file if it already exists:
		try:
			del group_lvl_h5[self.subject.initials]
		except:
			pass

		# and copy subject to group hdf5
		group_lvl_h5.create_group(self.subject.initials)
		for alias in aliases:
			subject_h5_file.copy(alias, group_lvl_h5[self.subject.initials])
