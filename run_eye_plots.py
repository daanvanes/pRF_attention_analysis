 # !/usr/bin/env python
# encoding: utf-8

from datetime import date
import os, sys, datetime
import subprocess, logging
from IPython import embed as shell
import numpy as np

sys.path.append( os.environ['ANALYSIS_HOME'] )
from GroupLevelEyePlotting import *

#########################################
# SETUP PLOTTING
#########################################

# set these values!!!
screen_size_pixs = [1920,1080]
screen_distance_cm = 156
screen_size_cm = [69.84,39.29]
stim_radius_ratio = 0.5
bar_width_ratio = 0.125
num_steps = 24

ci_factor=1.96
TR = 1.6
sample_rate = 1000
		
subjects = ['TK','DE','JW','JS','NA']
task_names = ['TF','Color','Fix','Fix_no_stim']

import colorsys
comparison_colors = {
	'Color':colorsys.hsv_to_rgb(0,0.0,0.75), #
	'TF':colorsys.hsv_to_rgb(0,0.0,0.5), 
	'Fix':colorsys.hsv_to_rgb(0.0,0.0,0.0) #
					}

### change this dir to the directory where you downloaded the file (see readme)
group_dir = os.path.join('/home','shared','2015','visual','PRF_2','PRF_eye_analysis','_group_level')

# setup plot object
pO = GroupLevelEyePlotting(group_dir = group_dir,subjects=subjects,task_names=task_names,screen_size_pixs=screen_size_pixs,
	screen_distance_cm=screen_distance_cm,screen_size_cm=screen_size_cm,stim_radius_ratio=stim_radius_ratio,bar_width_ratio=bar_width_ratio,
	num_steps=num_steps,ci_factor=ci_factor,TR=TR,sample_rate=sample_rate,comparison_colors=comparison_colors)

#########################################
# CREATE DESIRED PLOTS
# #########################################

pO.plot_rotated_eye_pos()











