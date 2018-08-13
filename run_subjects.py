# !/usr/bin/env python
# encoding: utf-8

# import python packages
from IPython import embed as shell
from datetime import date
import os, sys, datetime
import subprocess, logging
import scipy as sp
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as pl
import socket

# import toolbox functionality (available on github.com/xxx)
sys.path.append( os.environ['ANALYSIS_HOME'] )
from Tools.Subjects.Subject import *
from Tools.Projects.Project import *
from Tools.Sessions.Session import *

# import rest of analysis functionality
from PopulationReceptiveFieldMappingSession import *

# determine current host
if socket.gethostname() == 'aeneas':
	this_raw_folder = '/home/raw_data/2015/visual/PRF_2/'
	this_project_folder = '/home/shared/2015/visual/PRF_2/'
else: # if on cartesius:
	this_raw_folder = '/projects/0/pqsh283/raw_data/'
	this_project_folder = '/projects/0/pqsh283/PRF_2'

################################################
# parse command line input arguments
################################################
# 1. the first additional input refers to which subjects to fit and can be 
# * 'all', meaning fit all subjects,
# * a multipel subjects, e.g. 'DE JS'
# * a single subject e.g. DE
# 2. the second input argument is the number of jobs
# 3. slice number to fit prfs on
# 4. condition, 'PRF' or 'Mapper' for which files to use for preprocessing

# if wanting to compute analyses for all subjects, specify who they are:
if sys.argv[1] == 'all':
	which_subjects = ['TK','DE','JW','JS','NA']
# if more than one subject is entered, split them
elif len(sys.argv[1]) > 2:
	which_subjects = sys.argv[1].split(' ')
# otherwise take over single subject initials:
else:
	which_subjects = [sys.argv[1]]

# n jobs
if np.size(sys.argv) > 2:
	n_jobs = int(sys.argv[2])
else:
	n_jobs = 20

# slice no
if np.size(sys.argv) == 4: 
	slice_no = int(sys.argv[3])
	this_condition = sys.argv[4]
else:
	slice_no = None
	this_condition = 'PRF' # this determines which files the are in the 'runarray'

################################################
# some analysis parameters:
################################################

# anatomical mask
anat_mask = 'bet_mask_dilated'
all_mask = 'bet_mask_dilated'

# this determines which data is fitted on
# mcf = motion corrected file
# sgtf = savitzky golay temporally filtered
# psc = percent signal changed
postFix = ['mcf','fnirted','sgtf','psc']

# which pRF model (OG or DoG)
model = 'OG'

# how to combine hrfs across voxels:
hrf_type = 'median'

################################################
# determine which part of analysis is run:
################################################

# eye / behavior preprocessing
add_behavior_to_hdf5 = True
eye_analysis = False

# fMRI preprocessing
spatial_preprocessing = False
create_masks = False
temporal_preprocessing = False
regressor_preparation = False

# mapper
mapper_analyses = False

# pRF fitting
fit_single_pRF = False
combine_all_slice_niftis = False
fit_multiple_pRFs = False
combine_multiple_pRF_slice_niftis = False
convert_to_surf = False

# output data
add_data_to_hdf5 = True

### START SUBJECT LOOP
for which_subject in which_subjects:

	def runWholeSession( rA, session ):
		conditions = np.array([r['condition'] for r in rA])
		run_idx = np.where(conditions==this_condition)[0][0]
		for ri,r in enumerate(rA):
			thisRun = Run( **r )
			session.addRun(thisRun)
		session.parcelateConditions()
		session.parallelize = True

		########################
		#### ANALYZE BEHAVIOR
		########################

		if add_behavior_to_hdf5:
			session.add_behavior_to_hdf5()

		#######################################################################
		##### RUN PREPROCESSING ON AENEAS:
		#######################################################################

		if spatial_preprocessing:
			# SETUP FILES: 
			session.setupFiles(rawBase = presentSubject.initials, process_eyelink_file = False)

			# WE'LL FIRST MOTION CORRECT THE EPIS TO THEMSELVES
			session.motionCorrectFunctionals(use_ref_file=False)
			session.create_moco_check_gifs()

			## NOW, LET'S FLIRT ALL MEAN MOTION CORRECTED VOLUMES TO THE SESSIONS T2 
			# SO THAT WE CAN VISUALLY CHECK WHICH ONE HAS THE LEAST B0 DISTORTION
			session.flirt_mean_moco_to_session_T2()
			session.create_B0_distortion_check_gifs()

			## WITH A TARGET EPI VOLUME SELECTED AND MARKED IN THE SUBJECT DEFINITION
			# WE CAN NOW REGISTER TO THE FREESURFER T1
			session.registerSession(input_type='target_meanvol_moco_epi')

			## WITH A TARGET EPI VOLUME SELECTED AND MARKED IN THE SUBJECT DEFINITION
			# WE CAN NOW FLIRT ALL MEAN MOTION CORRECTED EPIS TO THAT EPI
			# AND CREATE VISUAL SANITY CHECKS
			session.flirt_mean_moco_to_mean_target_EPI()
			session.check_EPI_alignment(postFix=['mcf','meanvol','NB','flirted2targetEPI'])

			# ## FOR THE FINAL TOUCH, WE'LL NOW FNIRT THE MEAN MOTION CORRECTED AND FLIRTED
			# EPI TO THE TARGET MEAN MOTION CORRECTED EPI
			session.fnirt_mean_moco_to_mean_target_EPI()
			session.check_EPI_alignment(postFix=['mcf','meanvol','fnirted2targetEPI'])

			# NOW COMBINE MOCO / FLIRT / FNIRT AND APPLY TO ALL DATA
			session.applywarp_to_moco_data()
			session.create_mean_vol(postFix=['mcf','fnirted'])
			session.check_EPI_alignment(postFix=['mcf','fnirted','meanvol'])

		if eye_analysis:
			# EYE ANALYSIS
			session.eye_analysis(conditions=['PRF'],
				delete_hdf5=True,
				import_raw_data=True,
				import_all_data=True,
				write_trial_timing_text_files=True,
				add_to_group_level_hdf5 = True)

		if create_masks:
			# MASKS
			session.dilate_and_move_func_bet_mask()
			session.createMasksFromFreeSurferLabels(annot = False, annotFile = 'aparc.a2009s', labelFolders = ['retmap_PRF'], cortex = False)
		 	session.create_dilated_cortical_mask(dilation_sd = 0.5, label = 'cortex')
			session.create_WM_GM_CSF_masks()

		if temporal_preprocessing:
		 	# TEMPORAL FILTERING AND PERCENT SIGNAL CHANGE
		 	for condition in ['Mapper','PRF']:
				session.rescaleFunctionals(condition=condition,operations = ['sgtf'],filterFreqs={'highpass':120}, funcPostFix = ['mcf','fnirted'], mask_file = os.path.join(session.stageFolder('processed/mri/masks/anat'), 'bet_mask_dilated.nii.gz'))
				session.rescaleFunctionals(condition=condition,operations = ['percentsignalchange'], funcPostFix = ['mcf','fnirted','sgtf'])

		if regressor_preparation:
			# REGRESSOR PREPARATION
			session.retroicorFSL(conditions=['Mapper'], postFix=['mcf','fnirted','sgtf'], shim_slice=True, prepare=True, run=False)
			session.dt_ddt_moco_pars(conditions=['Mapper'])	

		# MAPPER ANALYSIS	
		if mapper_analyses:	
			session.Mapper_GLM(mask = 'bet_mask_dilated',postFix = ['mcf','fnirted','sgtf','psc'])
			session.hrf_from_mapper()

		#######################################################################
		##### PRF fitting
		#######################################################################

		if fit_single_pRF:
			session.design_matrices_for_concatenated_data(n_pixel_elements_raw = 101,n_pixel_elements_convolved=31,task_conditions=['All'])
			session.setup_fit_PRF_on_concatenated_data(
				anat_mask_file_name = anat_mask, 
				all_mask_file_name = all_mask, 
				n_jobs = n_jobs, 
				postFix = postFix, 
				plotbool = True,
				model = 'OG',
				hrf_type = hrf_type,
				fit_on_all_data = True,
				slice_no = slice_no
				)

		if combine_all_slice_niftis:
			session.combine_seperate_slice_niftis(mask,postFix,'OG',task_conditions=['All'],hrf_type=hrf_type)
		
		if convert_to_surf:
			session.convert_to_surf(mask_file = mask,postFix=postFix,model='OG',hrf_type=hrf_type,depth_min=-1.0,depth_max=2.0,depth_step=0.25,task_conditions=['Fix'],sms=[0])
			session.combine_surfaces(mask_file = mask,postFix=postFix,model='OG',hrf_type=hrf_type,depth_min=-1.0,depth_max=2.0,depth_step=0.25,task_conditions=['Fix'],sms=[0])

		if fit_multiple_pRFs:
			# fit separate pRF per attention condition
			# run_num is only for fitting cross-validated, which is not included in this version of the analyses
			task_conditions = ['Fix','Color','Speed']
			session.design_matrices_for_concatenated_data(n_pixel_elements_raw = 101,n_pixel_elements_convolved=31,
				change_type=change_type,run_num=run_num,task_conditions=task_conditions)
			r_squared_threshold = 0.0005 # threshold for which voxels to fit in the non-ALL condition
			session.setup_fit_PRF_on_concatenated_data(
				anat_mask_file_name = anat_mask, 
				all_mask_file_name =all_mask, 
				n_jobs = n_jobs, 
				postFix = postFix, 
				plotbool = True,
				model = 'OG',
				hrf_type = 'median',
				fit_on_all_data = False,
				r_squared_threshold = r_squared_threshold,
				slice_no = slice_no,
				change_type = change_type,
				run_num = run_num,
				)	
		if combine_multiple_pRF_slice_niftis:
			session.combine_seperate_slice_niftis(anat_mask,postFix,'OG',task_conditions = ['Fix','Color','Speed'],hrf_type=hrf_type)
		
		if add_data_to_hdf5:
			# adding data to the subject specific and group level hdf5 files:
			task_conditions = ['Fix','Color','Speed']
			session.mask_stats_to_hdf(mask_file = anat_mask , postFix = postFix, task_conditions = task_conditions,model='OG',hrf_type=hrf_type,
				add_regular_fit_results=True,add_mapper_data=True,add_hrf_params=True)

	if __name__ == '__main__':	

		#########################################################################
		# subject information
		#########################################################################

		if which_subject == 'TK':
			initials = 'TK'
			firstName = 'anonymous'
			standardFSID = 'TK_290615'
			birthdate = date(1950,01,01)
			labelFolderOfPreference = ''
			presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
			presentProject = Project( 'Population Receptive Field Mapping', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data'))
		
			sessionDate = date(2015, 6, 1)
			sessionID = 'PRF_' + presentSubject.initials
			sj_init_data_code = 'TK_010615'
		
			subject_session = PopulationReceptiveFieldMappingSession(sessionID, sessionDate, presentProject, presentSubject, this_project_folder=this_project_folder,targetEPIID=11 )
		
			try:
				os.mkdir(os.path.join(this_project_folder, 'data', initials))
				os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
			except OSError:
				subject_session.logger.debug('output folders already exist')
		
			subject_run_array = [

				# SESSION 1 in chronological scan order
				{'ID' : 1, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','TK_WIP_RetMap_2.5_1.6_20.32_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','tk_1_2015-06-01_13.16.47.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','tk_1_2015-06-01_13.16.47_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150601131622.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','TK_010615_3_to_12_NB.mat'),					
					'thisSessionT2ID':3,
					},
				{'ID' : 2, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','TK_WIP_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','tk_2_2015-06-01_13.42.40.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','tk_2_2015-06-01_13.42.40_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150601133716.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','TK_010615_3_to_12_NB.mat'),
					'thisSessionT2ID':3,
					},	
				{'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw','mri','TK_WIP_T2W_RetMap_1.25_CLEAR_6_1.nii.gz' ), 
					'targetSessionT2anatID':12
					},	
				{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','TK_WIP_Mapper_2.5_1.6_13.45_SENSE_7_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','tk_1_2015-06-01_14.11.33.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','tk_1_2015-06-01_14.11.33_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150601140915.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','TK_010615_3_to_12_NB.mat'),
					'thisSessionT2ID':3,	
					},

				# SESSION 2 in chronological order
				{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','TK_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','tk_4_2015-06-02_13.07.35.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','tk_4_2015-06-02_13.07.35_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150602130732.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','TK_010615_7_to_12_NB.mat'),
					'thisSessionT2ID':7,
					},
				{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','TK_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','tk_2_2015-06-02_13.30.07.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','tk_2_2015-06-02_13.30.07_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150602132834.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','TK_010615_7_to_12_NB.mat'),
					'thisSessionT2ID':7,
					},
				{'ID' : 7, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_2','mri','TK_T2W_RetMap_1.25_CLEAR_4_1.nii.gz' ), 
					'targetSessionT2anatID':12
					},	
				{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','TK_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','tk_5_2015-06-02_13.51.21.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','tk_5_2015-06-02_13.51.21_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150602135006.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','TK_010615_7_to_12_NB.mat'),
					'thisSessionT2ID':7,
					},	

				# SESSION 3 in chronological order
				{'ID' : 9, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','TK_3_WIP_RetMap_2.5_1.6_20.32_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','tk_6_2015-06-03_12.30.05.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','tk_6_2015-06-03_12.30.05_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150603123010.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','12','TK_010615_12_to_12_NB.mat'),	
					'thisSessionT2ID':12,
					},
				{'ID' : 10, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','TK_3_WIP_Mapper_2.5_1.6_13.45_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','tk_3_2015-06-03_12.52.14.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','tk_3_2015-06-03_12.52.14_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150603125106.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','12','TK_010615_12_to_12_NB.mat'),
					'thisSessionT2ID':12,
					},
				{'ID' : 11, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','TK_3_WIP_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','tk_7_2015-06-03_13.07.58.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','tk_7_2015-06-03_13.07.58_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150603130654.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','12','TK_010615_12_to_12_NB.mat'),	
					'thisSessionT2ID':12,
					},	
				{'ID' : 12, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_3','mri','TK_3_WIP_T2W_RetMap_1.25_CLEAR_6_1.nii.gz' ), 
					'targetSessionT2anatID':12
					},	



			]
		
			runWholeSession(subject_run_array, subject_session)

		elif which_subject == 'DE':
			# first subject; WK
			#########################################################################
			# subject information
			initials = 'DE'
			firstName = 'anonymous'
			standardFSID = 'DE_110412'
			birthdate = date(1950,01,01)
			labelFolderOfPreference = ''
			presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
			presentProject = Project( 'Population Receptive Field Mapping', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data'))
		
			sessionDate = date(2015, 6, 1)
			sessionID = 'PRF_' + presentSubject.initials
			sj_init_data_code = 'DE_010615'
		
			subject_session = PopulationReceptiveFieldMappingSession(sessionID, sessionDate, presentProject, presentSubject, this_project_folder=this_project_folder,targetEPIID=9)
		
			try:
				os.mkdir(os.path.join(this_project_folder, 'data', initials))
				os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
			except OSError:
				subject_session.logger.debug('output folders already exist')
		
		
			subject_run_array = [

				# SESSION 1 in chronological order
				{'ID' : 1, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','DE_WIP_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','de_1_2015-06-01_15.10.36.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','de_1_2015-06-01_15.10.36_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150601150932.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','4','DE_010615_4_to_11_NB.mat'),
					'thisSessionT2ID':4,
					'mocoT2anatIDtarget':11,
					},
				{'ID' : 2, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','DE_WIP_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','de_1_2015-06-01_15.33.48.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','de_1_2015-06-01_15.33.48_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150601153103.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','4','DE_010615_4_to_11_NB.mat'),
					'thisSessionT2ID':4,
					'mocoT2anatIDtarget':11,
					},
				{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','DE_WIP_RetMap_2.5_1.6_20.32_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','de_2_2015-06-01_15.51.06.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','de_2_2015-06-01_15.51.06_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150601154841.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','4','DE_010615_4_to_11_NB.mat'),
					'thisSessionT2ID':4,
					'mocoT2anatIDtarget':11,
					},	
				{'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw','mri','DE_WIP_T2W_RetMap_1.25_CLEAR_5_1.nii.gz' ), 
					'targetSessionT2anatID':11,
					},
					
				# SESSION 2 in chronological order
				{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','DE_2_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','de_3_2015-06-02_14.29.29.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','de_3_2015-06-02_14.29.29_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150602142744.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','DE_010615_7_to_11_NB.mat'),
					'thisSessionT2ID':7,
					'mocoT2anatIDtarget':11,
					},
				{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','DE_2_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','de_2_2015-06-02_14.51.32.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','de_2_2015-06-02_14.51.32_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150602145005.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','DE_010615_7_to_11_NB.mat'),
					'thisSessionT2ID':7,
					'mocoT2anatIDtarget':11,
					},
				{'ID' : 7, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_2','mri','DE_2_T2W_RetMap_1.25_CLEAR_4_1.nii.gz' ), 
					'targetSessionT2anatID':11
					},	
				{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','DE_2_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','de_4_2015-06-02_15.13.12.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','de_4_2015-06-02_15.13.12_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150602151054.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','DE_010615_7_to_11_NB.mat'),
					'thisSessionT2ID':7,
					'mocoT2anatIDtarget':11,
					},	

				# # SESSION 3 in chronological order
				{'ID' : 9, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','DE_3_WIP_RetMap_2.5_1.6_20.32_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','de_5_2015-06-03_14.16.01.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','de_5_2015-06-03_14.16.01_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150603142105.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','DE_010615_11_to_11_NB.mat'),
					'thisSessionT2ID':11,
					'mocoT2anatIDtarget':11,
					},
				{'ID' : 10, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','DE_3_WIP_Mapper_2.5_1.6_13.45_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','de_3_2015-06-03_14.43.06.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','de_3_2015-06-03_14.43.06_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150603144203.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','DE_010615_11_to_11_NB.mat'),
					'mocoT2anatIDtarget':11,
					'thisSessionT2ID':11,
					},
				{'ID' : 11, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_3','mri','DE_3_WIP_T2W_RetMap_1.25_CLEAR_5_1.nii.gz' ), 
					'targetSessionT2anatID':11
					},	
				{'ID' : 12, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','DE_3_WIP_RetMap_2.5_1.6_20.32_SENSE_6_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','de_6_2015-06-03_15.05.52.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','de_6_2015-06-03_15.05.52_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150603150409.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','DE_010615_11_to_11_NB.mat'),
					'thisSessionT2ID':11,
					'mocoT2anatIDtarget':11,
					},	
				{'ID' : 13, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','DE_3_WIP_RetMap_2.5_1.6_20.32_SENSE_7_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','de_7_2015-06-03_15.28.00.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','de_7_2015-06-03_15.28.00_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150603152604.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','DE_010615_11_to_11_NB.mat'),
					'thisSessionT2ID':11,
					'mocoT2anatIDtarget':11,
					},	
			]
		
			runWholeSession(subject_run_array, subject_session)
		
		elif which_subject == 'JW':
			# first subject; WK
			#########################################################################
			# subject information
			initials = 'JW'
			firstName = 'anonymous'
			standardFSID = 'JW_310312'
			birthdate = date(1950,01,01)
			labelFolderOfPreference = ''
			presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
			presentProject = Project( 'Population Receptive Field Mapping', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data'))
		
			sessionDate = date(2015, 6, 2)
			sessionID = 'PRF_' + presentSubject.initials
			sj_init_data_code = 'JW_020615'
		
			subject_session = PopulationReceptiveFieldMappingSession(sessionID, sessionDate, presentProject, presentSubject, this_project_folder=this_project_folder,targetEPIID=4 )
		
			try:
				os.mkdir(os.path.join(this_project_folder, 'data', initials))
				os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
			except OSError:
				subject_session.logger.debug('output folders already exist')
		
		
			subject_run_array = [

			# 	# SESSION 1 in chronological order
				{'ID' : 1, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','JW_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','jw_1_2015-06-02_15.44.19.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','jw_1_2015-06-02_15.44.19_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150602154242.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','5','JW_020615_5_to_9_NB.mat'),
					'thisSessionT2ID':5,
					},
				# # NO BEHAVIOR FILE WRITTEN DURING SCAN. Try and retrieve trial info from edf...
				{'ID' : 2, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','JW_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','jw_1_2015-06-02_16.05.36.edf' ),
					# 'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150602160434.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','5','JW_020615_5_to_9_NB.mat'),
					'thisSessionT2ID':5,
					},
				{'ID' : 3, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','JW_RetMap_2.5_1.6_20.32_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','jw_2_2015-06-02_16.22.27.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','jw_2_2015-06-02_16.22.27_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150602161945.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','5','JW_020615_5_to_9_NB.mat'),
					'thisSessionT2ID':5,
					},	
				# scanid 5 was failed T2, because JW moved
				{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','JW_RetMap_2.5_1.6_20.32_SENSE_6_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','jw_3_2015-06-02_16.49.10.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','jw_3_2015-06-02_16.49.10_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150602164849.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','5','JW_020615_5_to_9_NB.mat'),
					'thisSessionT2ID':5,
					},	
				{'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw','mri','JW_T2W_RetMap_1.25_CLEAR_7_1.nii.gz' ), 
					'targetSessionT2anatID':9
					},
				
				# SESSION 2 in chronological order
				{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','JW_2_WIP_RetMap_2.5_1.6_20.32_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','jw_4_2015-06-03_16.20.22.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','jw_4_2015-06-03_16.20.22_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150603161920.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','9','JW_020615_9_to_9_NB.mat'),
					'thisSessionT2ID':9,
					},
				{'ID' : 7, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','JW_2_WIP_Mapper_2.5_1.6_13.45_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','jw_2_2015-06-03_16.41.35.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','jw_2_2015-06-03_16.41.35_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150603164022.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','9','JW_020615_9_to_9_NB.mat'),
					'thisSessionT2ID':9,
					},
				{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','JW_2_WIP_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','jw_5_2015-06-03_16.57.43.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','jw_5_2015-06-03_16.57.43_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150603165538.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','9','JW_020615_9_to_9_NB.mat'),
					'thisSessionT2ID':9,
					},	
				{'ID' : 9, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_2','mri','JW_2_WIP_T2W_RetMap_1.25_CLEAR_6_1.nii.gz' ), 
					'targetSessionT2anatID':9
					},	
				{'ID' : 10, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','JW_2_WIP_RetMap_2.5_1.6_20.32_SENSE_7_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','jw_6_2015-06-03_17.19.59.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','jw_6_2015-06-03_17.19.59_outputDict.pickle' ), 
					# timing of physlogfile is again later than the edf, probably turned on the experiment during the T2
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150603172438.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','9','JW_020615_9_to_9_NB.mat'),
					'thisSessionT2ID':9,
					},	
			]
		
			runWholeSession(subject_run_array, subject_session)

		elif which_subject == 'NA':
			# first subject; WK
			#########################################################################
			# subject information
			initials = 'NA'
			firstName = 'anonymous'
			standardFSID = 'NA_220813_12'
			birthdate = date(1950,01,01)
			labelFolderOfPreference = ''
			presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
			presentProject = Project( 'Population Receptive Field Mapping', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data'))
		
			sessionDate = date(2015, 6, 23) 
			sessionID = 'PRF_' + presentSubject.initials
			sj_init_data_code = 'NA_230615'
		
			subject_session = PopulationReceptiveFieldMappingSession(sessionID, sessionDate, presentProject, presentSubject, this_project_folder=this_project_folder,targetEPIID=12 )
		
			try:
				os.mkdir(os.path.join(this_project_folder, 'data', initials))
				os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
			except OSError:
				subject_session.logger.debug('output folders already exist')
		
		
			subject_run_array = [

				# SESSION 1 in chronological order
				{'ID' : 1, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','NA_RetMap_2.5_1.6_20.32_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','na_1_2015-06-22_15.09.36.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','na_1_2015-06-22_15.09.36_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150622150912.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','NA_230615_3_to_11_NB.mat'),
					'thisSessionT2ID':3,
					},
				{'ID' : 2, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','NA_Mapper_2.5_1.6_13.45_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','na_1_2015-06-22_15.31.24.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','na_1_2015-06-22_15.31.24_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150622153050.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','NA_230615_3_to_11_NB.mat'),
					'thisSessionT2ID':3,
					},
				{'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw','mri','NA_T2W_RetMap_1.25_CLEAR_6_1.nii.gz' ), 
					'targetSessionT2anatID':11
					},
				{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','NA_RetMap_2.5_1.6_20.32_SENSE_7_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','na_2_2015-06-22_15.52.07.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','na_2_2015-06-22_15.52.07_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150622155140.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','NA_230615_3_to_11_NB.mat'),
					'thisSessionT2ID':3,
					},	

				
				# SESSION 2 in chronological order
				{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','NA_WIP_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','na_3_2015-06-23_14.05.04.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','na_3_2015-06-23_14.05.04_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150623140509.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','NA_230615_7_to_11_NB.mat'),
					'thisSessionT2ID':7,
					},
				{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','NA_WIP_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','na_2_2015-06-23_14.27.52.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','na_2_2015-06-23_14.27.52_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150623142751.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','NA_230615_7_to_11_NB.mat'),
					'thisSessionT2ID':7,
					},
				{'ID' : 7, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_2','mri','NA_WIP_T2W_RetMap_1.25_CLEAR_4_1.nii.gz' ), 
					'targetSessionT2anatID':11
					},
				{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','NA_WIP_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','na_4_2015-06-23_14.48.42.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','na_4_2015-06-23_14.48.42_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150623144752.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','NA_230615_7_to_11_NB.mat'),
					'thisSessionT2ID':7,
					},	


				# SESSION 3 in chronological order
				{'ID' : 9, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','NA_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','na_5_2015-06-25_12.13.06.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','na_5_2015-06-25_12.13.06_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150625121250.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','NA_230615_11_to_11_NB.mat'),
					'thisSessionT2ID':11,
					},
				{'ID' : 10, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','NA_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','na_3_2015-06-25_12.35.15.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','na_3_2015-06-25_12.35.15_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150625123349.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','NA_230615_11_to_11_NB.mat'),
					'thisSessionT2ID':11,
					},
				{'ID' : 11, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_3','mri','NA_T2W_RetMap_1.25_CLEAR_4_1.nii.gz' ), 
					'targetSessionT2anatID':11
					},
				{'ID' : 12, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','NA_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','na_6_2015-06-25_12.55.46.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','na_6_2015-06-25_12.55.46_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150625125444.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','NA_230615_11_to_11_NB.mat'),
					'thisSessionT2ID':11,
					},				
			]
			runWholeSession(subject_run_array, subject_session)

		elif which_subject == 'JS':
			# first subject; WK
			#########################################################################
			# subject information
			initials = 'JS'
			firstName = 'anonymous'
			standardFSID = 'JVS_091014'
			birthdate = date(1950,01,01)
			labelFolderOfPreference = ''
			presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
		
			presentProject = Project( 'Population Receptive Field Mapping', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data'))
		
			sessionDate = date(2015, 6, 23) 
			sessionID = 'PRF_' + presentSubject.initials
			sj_init_data_code = 'JS_230615'
		
			subject_session = PopulationReceptiveFieldMappingSession(sessionID, sessionDate, presentProject, presentSubject, this_project_folder=this_project_folder,targetEPIID=16 )
		
			try:
				os.mkdir(os.path.join(this_project_folder, 'data', initials))
				os.mkdir(os.path.join(this_project_folder, 'data', initials, sj_init_data_code))
			except OSError:
				subject_session.logger.debug('output folders already exist')
		
		
			subject_run_array = [


				# SESSION 1 in chronological order
				{'ID' : 1, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','JS_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','js_1_2015-06-22_16.33.23.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','js_1_2015-06-22_16.33.23_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150622163455.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','JS_230615_3_to_15_NB.mat'),
					'thisSessionT2ID':3,
					},
				{'ID' : 2, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','JS_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','js_1_2015-06-22_16.57.02.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','js_1_2015-06-22_16.57.02_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150622165555.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','JS_230615_3_to_15_NB.mat'),
					'thisSessionT2ID':3,
					},
				{'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw','mri','JS_T2W_RetMap_1.25_CLEAR_4_1.nii.gz' ), 
					'targetSessionT2anatID':15
					},
				{'ID' : 4, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw','mri','JS_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw','edf','js_2_2015-06-22_17.18.07.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw','behavior','js_2_2015-06-22_17.18.07_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw','hr','SCANPHYSLOG20150622171703.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','3','JS_230615_3_to_15_NB.mat'),
					'thisSessionT2ID':3,
					},	

				
				# SESSION 2 in chronological order
				{'ID' : 5, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','JS_WIP_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','js_3_2015-06-23_12.27.39.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','js_3_2015-06-23_12.27.39_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150623122720.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','JS_230615_7_to_15_NB.mat'),
					'thisSessionT2ID':7,
					},
				{'ID' : 6, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','JS_WIP_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','js_2_2015-06-23_12.51.27.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','js_2_2015-06-23_12.51.27_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150623125055.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','JS_230615_7_to_15_NB.mat'),
					'thisSessionT2ID':7,
					},
				{'ID' : 7, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_2','mri','JS_WIP_T2W_RetMap_1.25_CLEAR_4_1.nii.gz' ), 
					'targetSessionT2anatID':15
					},
				{'ID' : 8, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_2','mri','JS_WIP_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_2','edf','js_4_2015-06-23_13.11.30.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_2','behavior','js_4_2015-06-23_13.11.30_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_2','hr','SCANPHYSLOG20150623131129.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','7','JS_230615_7_to_15_NB.mat'),
					'thisSessionT2ID':7,
					},	

				# SESSION 3 in chronological order
				{'ID' : 9, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','JS_RetMap_2.5_1.6_20.32_SENSE_2_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','js_5_2015-06-25_16.07.49.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','js_5_2015-06-25_16.07.49_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150625160719.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','JS_230615_11_to_15_NB.mat'),
					'thisSessionT2ID':11,
					},
				{'ID' : 10, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','JS_Mapper_2.5_1.6_13.45_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','js_3_2015-06-25_16.29.36.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','js_3_2015-06-25_16.29.36_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150625162835.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','JS_230615_11_to_15_NB.mat'),
					'thisSessionT2ID':11,
					},
				{'ID' : 11, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_3','mri','JS_T2W_RetMap_1.25_CLEAR_4_1.nii.gz' ), 
					'targetSessionT2anatID':15
					},
				{'ID' : 12, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_3','mri','JS_RetMap_2.5_1.6_20.32_SENSE_5_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_3','edf','js_6_2015-06-25_16.52.02.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_3','behavior','js_6_2015-06-25_16.52.02_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_3','hr','SCANPHYSLOG20150625164936.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','11','JS_230615_11_to_15_NB.mat'),
					'thisSessionT2ID':11,
					},		

				# SESSION 4 in chronological order
				{'ID' : 13, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_4','mri','JS_RetMap_2.5_1.6_20.32_SENSE_3_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_4','edf','js_7_2015-07-02_18.00.51.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_4','behavior','js_7_2015-07-02_18.00.51_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_4','hr','SCANPHYSLOG20150702180023.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','15','JS_230615_15_to_15_NB.mat'),
					'thisSessionT2ID':15,
					},
				{'ID' : 14, 'scanType': 'epi_bold', 'condition': 'Mapper', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_4','mri','JS_Mapper_2.5_1.6_13.45_SENSE_4_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_4','edf','js_4_2015-07-02_18.22.54.edf' ),
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_4','behavior','js_4_2015-07-02_18.22.54_outputDict.pickle' ),  
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_4','hr','SCANPHYSLOG20150702182153.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','15','JS_230615_15_to_15_NB.mat'),
					'thisSessionT2ID':15,
					},
				{'ID' : 15, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials, 'raw_4','mri','JS_T2W_RetMap_1.25_CLEAR_5_1.nii.gz' ), 
					'targetSessionT2anatID':15
					},
				{'ID' : 16, 'scanType': 'epi_bold', 'condition': 'PRF', 
					'rawDataFilePath': os.path.join(this_raw_folder, initials,  'raw_4','mri','JS_RetMap_2.5_1.6_20.32_SENSE_6_1.nii.gz' ), 
					'eyeLinkFilePath': os.path.join(this_raw_folder, initials,  'raw_4','edf','js_8_2015-07-02_18.44.11.edf' ), 
					'rawBehaviorFile': os.path.join(this_raw_folder, initials,  'raw_4','behavior','js_8_2015-07-02_18.44.11_outputDict.pickle' ), 
					'physiologyFile': os.path.join(this_raw_folder,  initials,  'raw_4','hr','SCANPHYSLOG20150702184345.log' ), 
					'transformationMatrixFile': os.path.join(this_project_folder, 'data', initials, sj_init_data_code,'processed','mri','T2_anat','15','JS_230615_15_to_15_NB.mat'),
					'thisSessionT2ID':15,
					},			
			]
			runWholeSession(subject_run_array, subject_session)

