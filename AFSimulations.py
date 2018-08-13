# !/usr/bin/env python
# encoding: utf-8

# import python packages
from __future__ import division

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import animation

import scipy as sp
from skimage.morphology import disk
import skimage
import sys, os
from IPython import embed as shell
from joblib import Parallel, delayed
import copy
import pickle
import colorsys
import hrf_estimation as he
import shutil
sys.path.append( os.environ['ANALYSIS_HOME'] )


def find_profile_maxval(profile,radius,use_circle_mask=True,upsample_factor=10):

    res = np.shape(profile)[0]*upsample_factor
    x_view = sp.signal.resample(np.max(profile,axis=0),res)
    y_view = sp.signal.resample(np.max(profile,axis=1),res)
    x,y = np.argmax(x_view), np.argmax(y_view)
    x_deg = x/res*(radius*2)-(radius)
    y_deg = y/res*(radius*2)-(radius)

    return [x_deg,y_deg]

def convolve_bar_with_AF(AF_size_intercept,AF_size_slope,model_area_increase_factor,res,plot_dir,stim_radius,circle_mask,AF_surround_amp,AF_surround_ratio):#plot_dir,res,n_jobs,):
    orientations = [0,45,90,135,180,225,270,315]

    # get dm
    dm_filename = os.path.join(plot_dir,'design_matrix_for_AF_model.pickle')
    with open(dm_filename) as f:
        picklefile = pickle.load(f)
    dm = picklefile['raw_design_matrix']
    dm_res = dm.shape[-1]

    high_res = dm_res*model_area_increase_factor
    if high_res%2 == 0:
        high_res-=1
    edge = int((high_res-dm_res)/2)
    # now pad dm with zeros to match larger area that is modelled
    enlarged_dm = np.zeros((np.shape(dm)[0],high_res,high_res))
    if edge != 0:
        enlarged_dm[:,edge:-edge,edge:-edge] = dm
    else:
        enlarged_dm = dm
    high_res = np.shape(enlarged_dm)[-1]
    downsample_factor = res/high_res
    downsampled_dm = np.array([sp.ndimage.interpolation.zoom(this_dm,(downsample_factor,downsample_factor)) for this_dm in enlarged_dm])
    downsampled_dm[downsampled_dm>0.5] = 1
    downsampled_dm[downsampled_dm<=0.5] = 0

    TRs_per_pass = int(len(dm)/(len(orientations)+1))
    bars = downsampled_dm
    bars_1_pass = np.reshape(downsampled_dm,(len(orientations)+1,TRs_per_pass,res,res))[0]
    barsums = np.sum(np.reshape(bars,(bars.shape[0],-1)),axis=-1)
    barsums_1_pass = np.sum(np.reshape(bars_1_pass,(bars_1_pass.shape[0],-1)),axis=-1)
    valid_bar_timepoints = (barsums>1) # larger than 1 to get rid of this single pixel in horizontal bar pass
    valid_bar_timepoints_1_pass = (barsums_1_pass>1) # larger than 1 to get rid of this single pixel in horizontal bar pass
    AF_eccs = np.abs(np.linspace(-stim_radius,stim_radius,TRs_per_pass))
    AF_eccs_1_pass = np.hstack([np.zeros(1),np.abs(np.linspace(-stim_radius,stim_radius,np.sum(valid_bar_timepoints_1_pass))),np.zeros(3)])
    AF_eccs = np.hstack([np.tile(AF_eccs_1_pass,len(orientations)),np.zeros(TRs_per_pass)])

    dms = []
    AFs = []
    for t in np.arange(len(downsampled_dm)):
        if valid_bar_timepoints[t]:
            dms.append(downsampled_dm[t])

            center = twodgauss(0,0,AF_eccs[t]*AF_size_slope+AF_size_intercept,amplitude=1,res=res,max_eccentricity=stim_radius*model_area_increase_factor)
            surround = twodgauss(0,0,AF_eccs[t]*AF_size_slope+AF_size_intercept*AF_surround_ratio,amplitude = AF_surround_amp,res=res,max_eccentricity=stim_radius*model_area_increase_factor)
            kernel = center + surround
            AF = sp.signal.fftconvolve(downsampled_dm[t],kernel,mode='same')
            if circle_mask:
                AF[disk(int((res-1)/2))==0] = 0

            if AF.max() > 0:
                AF = AF/AF.max()

            AFs.append(AF)

    return AFs,dms

def twodgauss(xo, yo, sigma_x, sigma_y=None, amplitude = 1, theta=0,res=100,max_eccentricity=3.6,circle_mask=False,baseline=0,):
    
    X = np.linspace(-max_eccentricity, max_eccentricity, res)
    Y = np.linspace(-max_eccentricity, max_eccentricity, res)
    MG = np.meshgrid(X, Y)

    (x,y) = MG  
    if sigma_y == None:
        sigma_y = sigma_x
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    gauss = baseline+amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    if circle_mask:
        gauss[disk(int((res-1)/2))==0] = 0
    return gauss

def run_AF_bar_model(x,y,size,fix_AF_size,AFs,
    stim_radius,res,model_area_increase_factor,
    circle_mask,center_method,upsample_factor,rectify,dms,overlap,AF_surround_amp,AF_surround_ratio):

    # we want to model the effect of attention in an area that is larger than the original
    # stimulus aperture. This is because cancellation of attention at fixation
    # can result in SDs that fall off the stim area.
    high_stim_radius = stim_radius * model_area_increase_factor

    fix_AF = twodgauss(0,0,fix_AF_size,res=res,max_eccentricity=high_stim_radius,circle_mask=circle_mask)
    fix_RF_center = twodgauss(x,y,size,res=res,max_eccentricity=high_stim_radius,circle_mask=circle_mask)
    fix_RF_surround = twodgauss(x,y,size*AF_surround_ratio,res=res,max_eccentricity=high_stim_radius,circle_mask=circle_mask,amplitude=AF_surround_amp)
    fix_RF = fix_RF_center+fix_RF_surround
    # invert effect of attention at fixation 
    SD = fix_RF/fix_AF  
    # make sure there are no nans
    if np.isnan(SD).sum()>0:
        SD[np.isnan(SD)] = 0
    # normalize back to max 1
    SD /= np.max(SD)
    # circlemask SD
    if circle_mask:
        SD[disk(int((res-1)/2))==0] = 0

    # now simulate attention to bar effect on this SD:
    all_RFs = []
    SD_dm_overlaps = []
    for t, AF in enumerate(AFs):
        this_RF = (SD*AF)/np.max(SD*AF)
        if rectify:
            this_RF[this_RF<0] = 0

        if overlap:
            SD_dm_overlaps.append(np.dot(np.ravel(dms[t]),np.ravel(this_RF)))

        all_RFs.append(this_RF)

    # average over all bar positions
    if overlap:
        bar_RF = np.average(all_RFs,weights=SD_dm_overlaps,axis=0)
    else:
        bar_RF = np.mean(all_RFs,axis=0) 

    if np.isnan(bar_RF).sum()>0:
        bar_RF[np.isnan(bar_RF)] = 0
    if circle_mask:
        bar_RF[disk(int((res-1)/2))==0] = 0    

    # find peaks
    if center_method == 'maxval':
        SD_RFx,SD_RFy = find_profile_maxval(SD,high_stim_radius,circle_mask,upsample_factor)
        bar_RFx,bar_RFy = find_profile_maxval(bar_RF,high_stim_radius,circle_mask,upsample_factor)
    elif center_method == 'meanval':
        SD_RFx,SD_RFy = find_profile_means(SD,high_stim_radius,circle_mask,upsample_factor)     
        bar_RFx,bar_RFy = find_profile_means(bar_RF,high_stim_radius,circle_mask,upsample_factor)
    elif center_method == 'com':
        bar_RFx,bar_RFy = find_profile_center_of_mass(bar_RF,high_stim_radius,circle_mask,upsample_factor)
        SD_RFx,SD_RFy = find_profile_center_of_mass(SD,high_stim_radius,circle_mask,upsample_factor)    

    return x, y, bar_RFx, bar_RFy, SD_RFx, SD_RFy

