# !/usr/bin/env python
# encoding: utf-8

# import python packages
from __future__ import division
from datetime import date
import os, sys, datetime
import subprocess, logging
from IPython import embed as shell
import numpy as np

# import own functionality
from GroupLevelPlotting import *

#########################################
# Which plots to make:
#########################################

# roi selection figure
figure_2AB        = False 
figure_2C          = False
figure_2D       	  = False

# shift dir fig
figure_3B       = False 
figure_3C       = False 
figure_3D       = False 
rayleigh           = False 

# ecc / size change fig:
figure_4AB      		= False 
figure_4C         	= False 

# af model fig:
figure_5runmodel= False 
figure_5plot    		= False 

# feature ami fig
figure_6AB      = False
figure_6C        = False  
figure_6D        = False 

# feature pref vs ecc
figure_7scatter = False
figure_7corrs   = True 

# AF model feature
figure_8          = False 

# behavior figure:
figure_9        = False 

#########################################
# PLOT VARIABLES
#########################################

# 1. the group_dir should refer to the directory that contains the group_level
#     and the frames files (see readme)
# 2. the AF_dir should refer to the directory where you downloaded the design_matrix file
# note: it's fine to let this be the same dir!

###########################################################################################
# ---------CHANGE THESE DIR NAMES TO YOUR OWN DOWNLOADED FIGSHARE directories:------------#

import socket
if socket.gethostname() == 'aeneas':
    group_dir = os.path.join('/home','shared','2015','visual','PRF_2','data','_group_level')
    AF_dir = os.path.join('/home','shared','2015','visual','PRF_2','AF_simulations')
else:
    AF_dir = os.path.join('/projects','0','pqsh283','PRF_2','AF_simulations')
    group_dir = os.path.join('/projects/0/pqsh283/','PRF_2','data','_group_level')

############################################################################################

filename = os.path.join(group_dir, 'frames.pickle')
with open(filename) as f:
    picklefile = pickle.load(f)
stats_frames = picklefile['stats_frames']
results_frames = picklefile['results_frames']

# settings
subjects = ['NA','JS','JW','TK','DE']
stim_radius = 3.6 # in dva
mask_type = 'cond_1' # cond_1 refers to fixation data later on, base mask on this data
mask_ecc_thresholds = [0.0,3.3]
plot_ecc_thresholds = [0.0,3.6]
r_squared_threshold = 0.1
detect_outliers = True
outlier_num_stds = 5 # amount of absolute median deviations for outlier rejection 
rescale_factor = 7.5/stim_radius # this was a necessary step as stim radius was scaled to 7.5 in fit procedure
size_threshold = stim_radius*2
ci_factor = 1.96 # amounts to 95 % CIs
reps = int(1e5) # amount of bootstrap reps. Can be lowered if plots need to be created faster.

# get given options:
if len(sys.argv)>1:
    # when fitting the AF model fit:
    if fit_AF_model:
      rois_for_plot = [sys.argv[1]]
      comparison = sys.argv[2]
      subject_method = sys.argv[3]

    if plot_AF_params:
      # for plotting of af model: 
      subject_method = sys.argv[1]
      rois_for_plot = ['V1','V2','V3','hV4','VO','LO','V3AB','IPS0','MT+','combined']

else:
    rois_for_plot = ['V1','V2','V3','hV4','VO','LO','V3AB','IPS0','MT+','combined']


rois = {
        'V1':['V1'],
        'V2':['V2v','V2d'],
        'V3':['V3v','V3d'],
        'hV4':['V4'], 
        'VO':['VO1','VO2'],
        'LO':['LO1','LO2'],
        'V3AB':['V3A','V3B'],
        'MT+':['TO1','TO2'],
        'IPS0':['IPS0'],
        'combined':['V1','V2v','V2d','V3v','V3d','V4','VO1','VO2','V3A','V3B','LO1','LO2','TO1','TO2','IPS0'],
        }

roi_groups_for_plot = {
        '1_early':['V1','V2','V3',],
        '2_ventral':['hV4','VO','LO'],
        '3_dorsal':['V3AB','IPS0','MT+'],
        '4_combined':['combined']
}

roi_subplot_grid = [10,1]
roi_group_subplot_grid = [1,4]
total_n_rois = 10
roi_colors = np.array([colorsys.hsv_to_rgb(c,0.6,0.9) for c in np.linspace(0,1,total_n_rois+2)])[:-2]

comparison_colors = {
    'Color - Fix':colorsys.hsv_to_rgb(0,0.0,0.75), #
    'Speed - Fix':colorsys.hsv_to_rgb(0,0.0,0.5), 
    'Stim - Fix':colorsys.hsv_to_rgb(0.0,0.0,0.0,) #
                    }

condition_colors = {
    'Color':colorsys.hsv_to_rgb(0,0.0,0.75), #
    'Speed':colorsys.hsv_to_rgb(0,0.0,0.5), 
    'Stim':colorsys.hsv_to_rgb(0.0,0.0,0.25), #
    'Fix':colorsys.hsv_to_rgb(0.0,0.0,0.0) #
                    }

# setup plot object
pO = GroupLevelPlots(subjects,mask_ecc_thresholds,plot_ecc_thresholds,r_squared_threshold,stim_radius,outlier_num_stds,
        rois,ci_factor,rois_for_plot,results_frames,stats_frames,group_dir,roi_subplot_grid,roi_colors,
        roi_groups_for_plot,roi_group_subplot_grid,mask_type,rescale_factor,size_threshold,comparison_colors,condition_colors,detect_outliers,reps)


#########################################
# Figure 2 - ROI evaluation
#########################################
if figure_2AB:
    pO.color_wheel()
    pO.var_over_var(
          conditions=['Fix'],
          n_bins = 5,
          bin_type='fixed',
          plot_types = ['eccen_surf'],
          subjects=['super_subject','DE','JS','TK','JW','NA','over_subjects'])#,'super_subject','DE','JS','TK','JW','NA'])

if figure_2C:
    pO.R2_distributions(
        )

if figure_2D:
    pO.mask_visualization(
        )

#########################################
# Figure 3 - shift dir
#########################################

### Figure 3B
if figure_3B:
    pO.arrow_plot(
          comparisons = {'Stim - Fix': ['Stim','Fix']},
          subjects = ['over_subjects','super_subject','DE','JS','TK','JW','NA'],
          fields = ['quadrant'],
          bin_type = 'fixed',
          bin_data = 'cond_1_data',
          n_eccen_bins = 8,
          n_polar_bins = 8,
          location='cond_1',
          min_vox_per_arrow=1,
          )

# Figure 3C
if figure_3C:
    pO.shift_explained_by_ecc(
      comparisons = {'Stim - Fix': ['Stim','Fix']},
      measures = ['ecc_diff','x_diff','y_diff'],
      subjects=['over_subjects','DE','JS','TK','JW','NA','super_subject'])#['DE','JS','TK','JW','NA'])'super_subject',

# Figure 3D
if figure_3D:
    pO.shift_vector_explained_by_x_y_ecc_over_polar_angle(
      comparisons = {'Stim - Fix': ['Stim','Fix']},
      subjects=['DE','JS','TK','JW','NA','super_subject','over_subjects'],#'super_subject','over_subjects','DE','JS','TK','JW','NA'],#'super_subject''over_subjects','over_subjects',
      n_bins = 3,
      data_units=['voxels'],#'voxels'])#,'voxels''voxels',
      # if data_units == arrows:
      n_eccen_bins = 8,
      n_polar_bins = 8,
      location='cond_1')#'voxels'])#,'voxels'

if rayleigh:
    # rayleigh tests
    pO.pRF_distributions(
      condition='Fix',
      subjects=['super_subject','over_subjects'])

##############################
###  Figure 4:
##############################

# Figure 4A + B
if figure_4AB:
    pO.diff_over_ecc(
      comparisons = {'Stim - Fix': ['Stim','Fix']},#'Color - Fix': ['Color','Fix'],'Speed - Fix': ['Speed','Fix']},#,'Color - Speed':['Color','Speed']},,
      measures=['ecc','size'],#'ecc','size'],#'ecc','size','combined'],#'amp_center'],#'amp_center'],#'ecc','size'],#'r_squared','ecc','size','amp_center'],#'ecc','size'],#,'amp_center'],#,'size'],
      n_bins=4,
      bin_type='fixed',#'fixed',
      diff_type = 'abs',
      over = 'ecc',
      sig_test=True,
      # stat_methods=['super_subject'])#'over_subjects','super_subject'])#,'super_subject'])#,'super_subject'])#,'super_subject','over_super_subjects'])'over_subjects',
      subjects = ['super_subject'])#'DE','JS','TK','JW','NA','super_subject','over_subjects'])#,'over_subjects','DE','JS','NA','TK','JW',])#'super_subject'])#,'DE','JS','NA','TK','JW',

# Figure 5C
if figure_4C:
    pO.correlate_diffs(
      comparisons = {'Stim - Fix': ['Stim','Fix']},#,'Color - Fix': ['Color','Fix'],'Speed - Fix': ['Speed','Fix']},#,'Color - Speed':['Color','Speed']},#,'Speed - Fix':['Speed','Fix'],'Speed - Color':['Speed','Color']},
      ecc_size_relation_condition = 'Fix',
      n_bins=20,
      target_regions = rois_for_plot,#,'IPS0-2']
      # stat_methods=['super_subject'],#'over_subjects',
      subjects = ['over_subjects','DE','JS','TK','JW','NA','super_subject'],#'DE','JS','NA','TK','JW','super_subject','over_subjects',
      correlation_types = ['pearson'],
      measure_comparisons = {'ecc-size': ['ecc','sigma_center']},
      corr_overs=['bins'])#,'ecc-amp': ['ecc','amp_center'],'size-amp': ['sigma_center','amp_center']})#'ecc-size': ['ecc','sigma_center'],

##############################
### Figure 5 AF model 
##############################

if figure_5runmodel:
    # comparison = 'Color'
    # Runs model
    subject_method = 'over_subjects'
    if subject_method == 'over_subjects':
      jacknife = False
      per_subject = True
    elif subject_method == 'super_subjects':
      jacknife = True
      per_subject = False      
    if comparison == 'Stim':
      init_fix_from_stim = False
      AF_intercepts = np.linspace(0.6,1.6,50)
      model_tag = 'OG_1.5-2.5_50steps'
    else:
      init_fix_from_stim = False
      AF_intercepts = np.linspace(0.6,1.6,50)
      # model_tag = 'OG_fix_AF_from_Stim'
      model_tag = 'OG_1.5-2.5_50steps'

    pO.AF_fitting(
      # stuff that varies:
      model_tag = model_tag,
      # bar and fix AF sizes:

      AF_intercepts = AF_intercepts,
      AF_fix_sizes = np.linspace(1.5,2.5,50),
      init_fix_from_stim = init_fix_from_stim,
      fix_init_model_name = 'bar_convolved_OG_1.5-2.5_50steps_zero_AFslope',
      jacknife = jacknife,
      per_subject = per_subject,

      ### do not change:
      # surround params:
      AF_surround_amps = [0],
      AF_surround_ratios = [2],

      # bar AF slope params:
      AF_slopes = 'zero',
      AF_slopelist = [0],

      # some model properties
      rectify = False,
      overlap = False,
      center_method = 'maxval', 
      circle_mask = True,
      upsample_factor = 10, #increasing position estimates
      model_area_increase_factor = 4, #model larger area than visual field
      res = 131, #resolution at which multiplication is performed
      AF_shape='bar_convolved',

      # fit search options
      initialize_on = 'custom',# 'grid' or 'custom'. If custom, it takes the first values from the lists above
      n_jobs = -1,
      # minimization options
      fit_method = 'grid',# either 'grid', or 'fit'
      # what to fit on
      these_subjects = ['DE','TK','NA','JS','JW'],
      conditions = [comparison,'Fix'],
      AF_plot_dir = AF_dir,  # this should be the directory where the dm is
      )

if figure_5plot:
    subject_method = 'over_subjects'

    if subject_method == 'over_subjects':
      jacknife = False
      per_subject = True
    elif subject_method == 'super_subjects':
      jacknife = True
      per_subject = False  
    # Creates figure
    pO.compare_AF_models(
          model_name = 'bar_convolved_OG_1.5-2.5_50steps_zero_AFslope',
          AF_fit = False,
          save_postfix = 'for_paper',
          conditions = ['Stim'],
          subjects = ['DE','TK','NA','JS','JW'],
          ecc_diff_n_bins = 4,
          ecc_plot_rois = rois_for_plot,#['V1','V2','V3','hV4','VO','LO','V3AB','IPS0','MT+','combined'],
          markers = ['o','s','^','v','*','o','s','^','v','*'],
          ecc_plot = True ,
          AF_bar_param_plot = True,
          arrow_plot = True,
          AF_mechanics_plot = True,

          jacknife = jacknife,
          per_subject = per_subject,  
          )

##############################
### Figure 6: feature AMI fig
##############################

if figure_6AB:
  # Figure 6A:
  pO.diff_over_ecc_col_tf(
    comparisons = {'Speed - Fix': ['Speed','Fix'],'Color - Fix': ['Color','Fix']},
    measures=['ecc','size','ami','fami'],
    n_bins=4,
    bin_type='fixed',
    diff_type = 'abs',
    subjects = ['super_subject','over_subjects','DE','JS','TK','JW','NA'],#'DE','JS','NA','TK','JW'],#'over_subjects','DE','JS','NA','TK','JW','super_subject'],
    plot_types = ['per_roi'])#,'across_rois'])#,'combined'])#['combined'])

if figure_6C:
  pO.mapper_fba_ecc(
      create_per_roi_fami_bar_plot = True,
      measure_types = ['combined'],#,'ecc','size'],      
      subjects = ['over_subjects','super_subject','NA','JS','TK','JW','DE']
      )

if figure_6D:
  pO.mapper_fba_ecc(
      create_over_roi_scatter_plot = True,
      # subjects = ['super_subject'],
      measure_types = ['combined','ecc','size'],      
      )

##############################
### Figure 7: feature pref vs ecc
##############################

if figure_7scatter:
  pO.diff_over_ecc(
    comparisons = {'Stim - Fix': ['Stim','Fix']},#'Color - Fix': ['Color','Fix'],'Speed - Fix': ['Speed','Fix']},#,'Color - Speed':['Color','Speed']},,
    measures=['feature_pref'],#'ecc','size'],#'ecc','size','combined'],#'amp_center'],#'amp_center'],#'ecc','size'],#'r_squared','ecc','size','amp_center'],#'ecc','size'],#,'amp_center'],#,'size'],
    n_bins=6,
    bin_type='percentile',#'fixed',
    diff_type = 'abs',
    over = 'ecc',
    sig_test = False,
    subjects = ['super_subject'])#,'over_subjects','DE','JS','NA','TK','JW',])#'super_subject'])#,'DE','JS','NA','TK','JW',

if figure_7corrs:
  pO.mapper_fba_ecc(
      create_over_ecc_corrs = True
      )  

##############################
### Figure 8: AF model feature
##############################

if figure_8:
  subject_method = 'over_subjects'
  if subject_method == 'over_subjects':
    jacknife = False
    per_subject = True
  elif subject_method == 'super_subjects':
    jacknife = True
    per_subject = False      

  pO.compare_AF_params_between_conditions(
      
      jacknife = jacknife,
      per_subject = per_subject,

      model_name = 
      'bar_convolved_OG_1.5-2.5_50steps_zero_AFslope',
      AF_fit = False,
      save_postfix = 'for_revisions',
      conditions = ['Color','Speed'],
      ecc_plot_rois = rois_for_plot,#['V1','V2','V3','hV4','VO','LO','V3AB','IPS0','MT+','combined'],
      subjects = ['DE','TK','NA','JS','JW'],
      bar_param_plot = True,
      )


##############################
### Figure 9: behavior
##############################
if figure_9:
    pO.analyze_behavior(
        conditions=['Speed','Color','Fix','Fix_no_stim'],
        these_subjects=subjects,
        with_first_block=True)




