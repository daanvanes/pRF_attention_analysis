# !/usr/bin/env python
# -*- coding: utf-8 -*-

# import python packages:
from __future__ import division
import os, sys
from tempfile import mkdtemp
import gc
from sklearn.externals import joblib
from random import sample as randomsample
from pylab import *
import numpy as np
import scipy as sp
from scipy.stats import spearmanr,spearmanr
from scipy import ndimage
import seaborn as sns
import statsmodels
import socket
from matplotlib import animation
import hrf_estimation as he
import random as random
import multiprocessing as mp
import time as time_module
from nifti import *
from math import *
import shutil
from joblib import Parallel, delayed
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from skimage.morphology import disk
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, report_errors
from IPython import embed as shell

# import toolbox functionality (available at )
from Tools.Operators.ArrayOperator import *
from Tools.Operators.EyeOperator import *
from Tools.Operators.PhysioOperator import *
from Tools.Operators.CommandLineOperator import *
from Tools.Sessions.Session import *
from ModelExperiment import *
from FitPRFModel import *
from EyeFromfMRISession import *
from Tools.other_scripts.plotting_tools import *

class PopulationReceptiveFieldMappingSession(Session):
    """
    Class for population receptive field mapping sessions analysis.
    """
    
    def __init__(self, ID, date, project, subject, this_project_folder, parallelize = True, loggingLevel = logging.DEBUG,**kwargs):
        super(PopulationReceptiveFieldMappingSession, self).__init__(ID, date, project, subject, parallelize = parallelize, 
            loggingLevel = loggingLevel,this_project_folder=this_project_folder)

        self.n_pixel_elements_raw = 101
        self.n_pixel_elements_convolved = 31
        self.this_project_folder = this_project_folder
        self.TR = 1.594
        for k,v in kwargs.items():
            setattr(self, k, v)

        self.task_names = {'PRF':['Color', 'Speed', 'Fix', 'Fix_no_stim'],'Mapper':['fix_no_stim','no_color_no_speed','yes_color_no_speed','no_color_yes_speed','yes_color_yes_speed']}
        self.n_TR = {'PRF':765,'Mapper':510}
        self.run_types = ['PRF','Mapper']

    #########################################################################################################
    # %%% GENERAL
    #########################################################################################################

    def create_mean_vol(self,postFix=['mcf','warped']):

        for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

            filename = self.runFile(stage = 'processed/mri', run = r, postFix=postFix )
            whole_data = NiftiImage(filename).data
            mean_data = np.mean(whole_data,axis=0)

            mean_nifti = NiftiImage(mean_data)
            mean_nifti.header = NiftiImage(filename).header
            save_postFix = postFix + ['meanvol']
            save_filename = self.runFile(stage = 'processed/mri', run = r, postFix=save_postFix )
            mean_nifti.save(save_filename)

    #########################################################################################################
    # %%% MASKS
    #########################################################################################################


    def dilate_and_move_func_bet_mask(self,dilation_sd = 0.5):

        target_run_idx = np.where(np.array([self.runList[ri].ID for ri in range(len(self.runList))]) == self.targetEPIID)[0][0]
        bet_mask_fn = self.runFile( stage = 'processed/mri', run = self.runList[target_run_idx],postFix = ['mcf','meanvol','NB','mask'])
        bet_mask_ofn =  os.path.join(self.stageFolder(stage='processed/mri/masks/anat'),'bet_mask.nii.gz')
        
        shutil.copyfile(bet_mask_fn,bet_mask_ofn)

        smoothed_bet_mask_ofn =  os.path.join(self.stageFolder(stage='processed/mri/masks/anat'),'bet_mask_dilated.nii.gz')
        fmO = FSLMathsOperator(bet_mask_ofn)
        fmO.configureSmooth(smoothing_sd = dilation_sd)
        fmO.execute()
            
        fmO = FSLMathsOperator(fmO.outputFileName)
        fmO.configure(outputFileName = smoothed_bet_mask_ofn, **{'-bin': ''})
        fmO.execute()

    def create_dilated_cortical_mask(self, dilation_sd = 0.5, label = 'cortex'):
        """create_dilated_cortical_mask takes the rh and lh cortex files and joins them to one cortex.nii.gz file.
        it then smoothes this mask with fslmaths, using a gaussian kernel. 
        This is then thresholded at > 0.0, in order to create an enlarged cortex mask in binary format.
        """
        self.logger.info('creating dilated %s mask with sd %f'%(label, dilation_sd))
        # take rh and lh files and join them.
        fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.' + label + '.nii.gz'))
        fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'), 
            **{'-add': os.path.join(self.stageFolder('processed/mri/masks/anat'), 'lh.' + label + '.nii.gz')})
        fmO.execute()
        
        fmO = FSLMathsOperator(os.path.join(self.stageFolder('processed/mri/masks/anat'), '' + label + '.nii.gz'))
        fmO.configureSmooth(smoothing_sd = dilation_sd)
        fmO.execute()
        
        fmO = FSLMathsOperator(fmO.outputFileName)
        fmO.configure(outputFileName = os.path.join(self.stageFolder('processed/mri/masks/anat'), label + '_dilated_mask.nii.gz'), **{'-bin': ''})
        fmO.execute()


    def combine_rois(self,rois=[],output_roi=''):
        """
        combine rois combines labels of input rois into one output_roi nifti mask
        """

        mask_path = os.path.join(self.stageFolder('processed/mri/masks/anat'))
        temp_combined = zeros((30,96,96))
        for roi in rois:
            roi_mask = np.array(NiftiImage(os.path.join(mask_path,roi)).data,dtype=bool)
            temp_combined += roi_mask

        combined_rois = np.zeros((30,96,96))
        combined_rois[temp_combined!=0] =1 

        new_nifti = NiftiImage(combined_rois.astype(int))
        new_nifti.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), rois[0])).header
        new_nifti.save(os.path.join(self.stageFolder('processed/mri/masks/anat'), output_roi + '.nii.gz'))

    def combine_lh_rh(self):

        rois = ['V1','V2','V3','V4','VO','MT','LO','PHC','V3ab','IPS0','IPS1','IPS2','IPS3','IPS4','IPS']
        mask_path = os.path.join(self.stageFolder('processed/mri/masks/anat'))

        for roi in rois:

            lh_roi_mask = np.array(NiftiImage(os.path.join(mask_path,'lh.'+roi)).data,dtype=bool)
            rh_roi_mask = np.array(NiftiImage(os.path.join(mask_path,'rh.'+roi)).data,dtype=bool)
            bh_roi_mask = lh_roi_mask + rh_roi_mask

            new_nifti = NiftiImage(bh_roi_mask.astype(int))
            new_nifti.header = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'rh.' + roi)).header
            new_nifti.save(os.path.join(self.stageFolder('processed/mri/masks/anat'), roi + '.nii.gz'))

    def create_combined_label_mask(self):
        """
        create combined label mask combines all labels that contain bh, lh or rh in it's name
        into one large 'combined label mask'.
        """

        anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, 
            shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
        anatRoiFileNames = [anRF for anRF in anatRoiFileNames if np.all([np.any(['bh' in anRF,'lh' in anRF,'rh' in anRF]),'cortex' not in anRF])]
        rois_combined = zeros((30,96,96)).astype('bool')
        for this_roi in anatRoiFileNames:
            rois_combined += NiftiImage(this_roi).data.astype('bool')
        
        new_nifti = NiftiImage(rois_combined.astype('int32'))
        new_nifti.header = NiftiImage(this_roi).header
        new_nifti.save(os.path.join(self.stageFolder('processed/mri/masks/anat'), 'combined_labels.nii.gz'))

    #########################################################################################################
    # %%% SPATIAL NORMALISATION
    #########################################################################################################

    def flirt_mean_moco_to_session_T2(self):

        flirts = []

        for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

            # first flirt mean mcf volume to target epi
            target_T2_run_idx = r.mocoT2anatIDtarget
            # target_T2_run_idx = np.where(np.array([self.runList[ri].ID for ri in range(len(self.runList))]) == r.thisSessionT2ID)[0][0]
            flirt_target_fn = self.runFile( stage = 'processed/mri', run = self.runList[target_T2_run_idx])
            flirt_target_fn_betted = self.runFile( stage = 'processed/mri', run = self.runList[target_T2_run_idx],postFix=['NB'])
            in_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol'])
            in_fn_betted = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','NB'])
            out_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','NB','flirted2sessionT2'])

            # as flirt works best with BETTED brains, let's first BET the flirt target
            if not os.path.isfile(flirt_target_fn_betted):
                better = BETOperator( inputObject = flirt_target_fn)
                better.configure( outputFileName = flirt_target_fn_betted)
                better.execute()
            # and also the mean motion corrected volume 
            if not os.path.isfile(in_fn_betted):
                better = BETOperator( inputObject = in_fn)
                better.configure( outputFileName = in_fn_betted)
                better.execute()

            # now create and configure the flirt object with the betted brains
            flO = FlirtOperator(inputObject = in_fn_betted, referenceFileName = flirt_target_fn_betted)
            flO.configureRun(outputFileName = out_fn)
            # add it to the flirt command list for later parallel execution
            flirts.append(flO)

        # now execute in parallel
        ppservers = ()
        job_server = pp.Server(ppservers=ppservers, secret='mc')
        self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
        ppResults = [job_server.submit(ExecCommandLine,(flirt.runcmd,),(),('subprocess','tempfile',)) for flirt in flirts]
        for fMcf in ppResults:
            fMcf()
        job_server.print_stats()

    def combine_moco_with_FLIRT(self,conditions=['PRF','Mapper']):

        applyxfms = []
        for condition in conditions:

            # creat single run animation with shuffled TRs
            for r in [self.runList[i] for i in self.conditionDict[condition]]:

                in_fn = self.runFile(stage = 'processed/mri', run = r, postFix = [])    
                ofn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf_with_flirt'])  
                transm_folder = in_fn[:-7]+'.mat/'
                transm_fns = np.sort(glob.glob(transm_folder+'*'))
                flirt_fn = in_fn[:-7]+'_mcf_meanvol_NB_trans.mat'
                flirt_m = np.loadtxt(flirt_fn)
                new_folder = in_fn[:-7]+'_moco_flirt.mat/'
                if os.path.isdir(new_folder): shutil.rmtree(new_folder)
                os.mkdir(new_folder)
                # loop over timepoints
                for transm in transm_fns:
                    this_transm = np.loadtxt(transm)    
                    # transformation calculation, reverse order:
                    new_trans = np.mat(flirt_m) * np.mat(this_transm)
                    # save
                    np.savetxt(os.path.join(new_folder,transm.split('/')[-1]),new_trans)

                # now create applyxfm command
                xO = XFM4DOperator(inputObject=in_fn)
                xO.configure(transformMatrixDir=new_folder,outputFileName=ofn,regFile=in_fn)
                applyxfms.append(xO)
        
        # now execute in parallel
        ppservers = ()
        job_server = pp.Server(ppservers=ppservers, secret='mc')
        self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
        ppResults = [job_server.submit(ExecCommandLine,(xfm.runcmd,),(),('subprocess','tempfile',)) for xfm in applyxfms]
        for fMcf in ppResults:
            fMcf()
        job_server.print_stats()

    def create_moco_check_gifs(self,conditions=['PRF','Mapper'],postFix=[],fps=10):

        pl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        general_plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/'))
        if not os.path.isdir(general_plotdir): os.mkdir(general_plotdir)
        plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/moco/'))
        if not os.path.isdir(plotdir): os.mkdir(plotdir)    

        for condition in conditions:

            # creat single run animation with shuffled TRs
            for r in [self.runList[i] for i in self.conditionDict[condition]]:
                
                self.logger.info('loading nifti file for condition %s, run %s'%(condition,r.ID))

                func_fn = self.runFile(stage = 'processed/mri', run = r, postFix=postFix )
                func_data = NiftiImage(func_fn).data
                
                # slice_dim = np.argmin(np.shape(func_data))
                timepoints = np.arange(len(func_data))
                # timepoints = randomsample(timepoints,100)
                np.random.shuffle(timepoints)
                for direction in range(3):
                    this_mid = int(np.shape(func_data)[direction+1]/2)

                    if direction == 0:
                        f=pl.figure(figsize=(3,3))
                    else:
                        f = pl.figure(figsize=(3,1))
                    ims = []
                    for t in timepoints:
                        s=f.add_subplot(111)
                        if direction == 0:
                            im=pl.imshow(func_data[t,this_mid,:,:],origin='lowerleft',interpolation='nearest',cmap='gray')
                        elif direction == 1:
                            im=pl.imshow(func_data[t,:,this_mid,:],origin='lowerleft',interpolation='nearest',cmap='gray')
                        elif direction == 2:
                            im=pl.imshow(func_data[t,:,:,this_mid],origin='lowerleft',interpolation='nearest',cmap='gray')
                        pl.axis('off')
                        ims.append([im])
                    ani = animation.ArtistAnimation(f, ims)#, interval=5, blit = True, repeat_delay = 1000)
                    mywriter = animation.FFMpegWriter(fps = fps)
                    # pl.tight_layout()
                    self.logger.info('saving to %s_%s_run_%s_dir_%d.mp4'%('_'.join(postFix),condition,r.ID,direction))
                    ani.save(os.path.join(plotdir,'%s_%s_run_%s_dir_%d.mp4'%('_'.join(postFix),condition,r.ID,direction)),writer=mywriter,dpi=200,bitrate=200)#,fps=2)#,dpi=100,bitrate=50)
                    pl.close()

    def create_B0_distortion_check_gifs(self,conditions=['PRF','Mapper'],fps=2,n_repetitions=50,func_data_postFix=['mcf','meanvol','NB','flirted2sessionT2']):

        pl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        general_plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/'))
        if not os.path.isdir(general_plotdir): os.mkdir(general_plotdir)
        plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/B0distortion/'))
        if not os.path.isdir(plotdir): os.mkdir(plotdir)    
        sub_plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/B0distortion/%s'%'_'.join(func_data_postFix)))
        if not os.path.isdir(sub_plotdir): os.mkdir(sub_plotdir)    

        for condition in conditions:

            for r in [self.runList[i] for i in self.conditionDict[condition]]:

                target_T2_run_idx = np.where(np.array([self.runList[ri].ID for ri in range(len(self.runList))]) == r.thisSessionT2ID)[0][0]
                func_data = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = func_data_postFix )).data
                T2_data = NiftiImage(self.runFile( stage = 'processed/mri', run = self.runList[target_T2_run_idx],postFix=['NB'])).data

                for direction in range(3):

                    for depth in [1,2]:

                        f=pl.figure(figsize=(4,4))
                        ims = []
                        s=f.add_subplot(111)
                        for repetition in range(n_repetitions):
                            if direction ==0:
                                im=pl.imshow(func_data[int(np.shape(func_data)[0]/3*depth),:,:],origin='lowerleft',interpolation='nearest',cmap='gray')
                            elif direction ==1:
                                im=pl.imshow(func_data[:,int(np.shape(func_data)[1]/3*depth),:],origin='lowerleft',interpolation='nearest',cmap='gray')
                            elif direction ==2:
                                im=pl.imshow(func_data[:,:,int(np.shape(func_data)[2]/3*depth)],origin='lowerleft',interpolation='nearest',cmap='gray')
                            pl.axis('off')
                            ims.append([im])

                            if direction ==0:
                                im=pl.imshow(T2_data[int(np.shape(T2_data)[0]/3*depth),:,:],origin='lowerleft',interpolation='nearest',cmap='gray')
                            elif direction ==1:
                                im=pl.imshow(T2_data[:,int(np.shape(T2_data)[1]/3*depth),:],origin='lowerleft',interpolation='nearest',cmap='gray')
                            elif direction ==2:
                                im=pl.imshow(T2_data[:,:,int(np.shape(T2_data)[2]/3*depth)],origin='lowerleft',interpolation='nearest',cmap='gray')
                            pl.axis('off')
                            ims.append([im])

                        ani = animation.ArtistAnimation(f, ims)#, interval=5, blit = True, repeat_delay = 1000)
                        mywriter = animation.FFMpegWriter(fps = 3)
                        self.logger.info('saving to %s_run_%s_direction_%s_depth_%s.mp4'%(condition,r.ID,direction,depth))
                        ani.save(os.path.join(sub_plotdir,'%s_run_%s_direction_%s_depth_%s.mp4'%(condition,r.ID,direction,depth)),writer=mywriter,dpi=200,bitrate=200)#,fps=2)#,dpi=100,bitrate=50)
                        pl.close()

    def flirt_mean_moco_to_mean_target_EPI(self):

        flirts = []

        for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

            # flirt mean mcf volume to target epi
            target_run_idx = np.where(np.array([self.runList[ri].ID for ri in range(len(self.runList))]) == self.targetEPIID)[0][0]
            flirt_target_fn = self.runFile( stage = 'processed/mri', run = self.runList[target_run_idx],postFix = ['mcf','meanvol'])
            flirt_target_fn_betted = self.runFile(stage = 'processed/mri', run = self.runList[target_run_idx],postFix=['mcf','meanvol','NB'])
            in_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol'])
            in_fn_betted = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','NB'])
            out_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','NB','flirted2targetEPI'])
            output_mat_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','NB','flirted2targetEPI'], extension='.mat')
            # as flirt works best with BETTED brains, let's first BET the flirt target
            if not os.path.isfile(flirt_target_fn_betted):
                better = BETOperator( inputObject = flirt_target_fn)
                better.configure( outputFileName = flirt_target_fn_betted)
                better.execute()
            # and also the mean motion corrected volume 
            if not os.path.isfile(in_fn_betted):
                better = BETOperator( inputObject = in_fn)
                better.configure( outputFileName = in_fn_betted)
                better.execute()

            # now create and configure the flirt object with the betted brains
            flO = FlirtOperator(inputObject = in_fn_betted, referenceFileName = flirt_target_fn_betted)
            flO.configureRun(outputFileName = out_fn,transformMatrixFileName = output_mat_fn)
            # add it to the flirt command list for later parallel execution
            flirts.append(flO)

        # now execute in parallel
        ppservers = ()
        job_server = pp.Server(ppservers=ppservers, secret='mc')
        self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
        ppResults = [job_server.submit(ExecCommandLine,(flirt.runcmd,),(),('subprocess','tempfile',)) for flirt in flirts]
        for fMcf in ppResults:
            fMcf()
        job_server.print_stats()

        # now convert the hexidecimal transformation matrices to decimal format
        for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:
            mat_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','NB','flirted2targetEPI'], extension='.mat')
            flirt_out_mat = np.loadtxt(mat_fn)
            np.savetxt(mat_fn,flirt_out_mat)

    def fnirt_mean_moco_to_mean_target_EPI(self):

        fnirts = []

        for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

            # fnirt mean mcf volume to target epi
            target_run_idx = np.where(np.array([self.runList[ri].ID for ri in range(len(self.runList))]) == self.targetEPIID)[0][0]
            fnirt_target_fn = self.runFile( stage = 'processed/mri', run = self.runList[target_run_idx],postFix = ['mcf','meanvol'])
            input_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol'])
            output_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','fnirted2targetEPI'])
            initial_flirt_mat_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','NB','flirted2targetEPI'], extension='.mat')
            coefs_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','fnirted2targetEPI','coefs'])

            flO = FnirtOperator(inputObject=input_fn, referenceFileName = fnirt_target_fn)
            flO.configure(AffineTransMatrixFileName=initial_flirt_mat_fn,outputFileName=output_fn,coefsFileName=coefs_fn)
            fnirts.append(flO)

        # now execute in parallel
        ppservers = ()
        job_server = pp.Server(ppservers=ppservers, secret='mc')
        self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
        ppResults = [job_server.submit(ExecCommandLine,(fnirt.runcmd,),(),('subprocess','tempfile',)) for fnirt in fnirts]
        for fMcf in ppResults:
            fMcf()
        job_server.print_stats()


    def check_EPI_alignment(self,conditions=['PRF','Mapper'],fps=10,n_repetitions=10,postFix=['mcf','meanvol','NB','flirted2targetEPI']):

        pl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        general_plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/'))
        if not os.path.isdir(general_plotdir): os.mkdir(general_plotdir)
        plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/between_session_registration/'))
        if not os.path.isdir(plotdir): os.mkdir(plotdir)    
        sub_plotdir = os.path.join(self.stageFolder('processed/mri/figs/check_registration/between_session_registration/%s'%'_'.join(postFix)))
        if not os.path.isdir(sub_plotdir): os.mkdir(sub_plotdir)    

        for condition in self.run_types:

            # load epis
            all_epis = []
            self.logger.info('loading all %s niftis'%condition)
            for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:
            # for r in [self.runList[i] for i in self.conditionDict[condition]]:

                filename = self.runFile(stage = 'processed/mri', run = r, postFix=postFix )
                all_epis.append(NiftiImage(filename).data)

            all_epis = np.array(all_epis)

            self.logger.info('creating %s animation'%condition)
            pl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
            ims = []
            for direction in range(3):
                for depth in [1,2]:
                    f=pl.figure(figsize=(4,4))
                    ims = []
                    s=f.add_subplot(111)
                    for repetition in range(n_repetitions):
                        for epir in range(len([self.runList[i] for i in self.conditionDict[condition]])):
                            if direction ==0:
                                im=pl.imshow(all_epis[epir,int(np.shape(all_epis)[0]/3*depth),:,:],origin='lowerleft',interpolation='nearest',cmap='gray')
                            elif direction ==1:
                                im=pl.imshow(all_epis[epir,:,int(np.shape(all_epis)[1]/3*depth),:],origin='lowerleft',interpolation='nearest',cmap='gray')
                            elif direction ==2:
                                im=pl.imshow(all_epis[epir,:,:,int(np.shape(all_epis)[2]/3*depth)],origin='lowerleft',interpolation='nearest',cmap='gray')
                            pl.axis('off')
                            ims.append([im])

                    ani = animation.ArtistAnimation(f, ims)#, interval=5, blit = True, repeat_delay = 1000)
                    mywriter = animation.FFMpegWriter(fps = fps)
                    self.logger.info('saving to %s_direction_%s_depth_%s.mp4'%(condition,direction,depth))
                    ani.save(os.path.join(sub_plotdir,'%s_direction_%s_depth_%s.mp4'%(condition,direction,depth)),writer=mywriter,dpi=200,bitrate=300)#,fps=2)#,dpi=100,bitrate=50)
                    pl.close()


    def fnirt_mean_moco_to_mean_target_EPI(self):

        AWOs = []

        # the fnirt includes the flirt matrix, so we can simply apply the warp
        # to the motion corrected data
        for r in [self.runList[i] for i in self.scanTypeDict['epi_bold']]:

            # get the rawest func file to apply to
            func_fn = self.runFile(stage = 'processed/mri', run = r,postFix = ['mcf'])

            # the fnirt warpfields:
            warpfield_fn = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','meanvol','fnirted2targetEPI','coefs'])

            # then, we'll create the applywarp command
            ofn = self.runFile(stage = 'processed/mri', run = r, postFix=['mcf','fnirted'])
            AWO = ApplyWarpOperator(inputObject = func_fn, referenceFileName = func_fn )
            AWO.configure(outputFileName = ofn,warpfieldFileName=warpfield_fn   )
            AWOs.append(AWO)

        # execute the applywarp operations in parallel
        ppservers = ()
        job_server = pp.Server(ppservers=ppservers, secret='mc')
        self.logger.info("starting pp with", job_server.get_ncpus(), "workers for " + sys._getframe().f_code.co_name)
        ppResults = [job_server.submit(ExecCommandLine,(AWO.runcmd,),(),('subprocess','tempfile',)) for AWO in AWOs]
        for fMcf in ppResults:
            fMcf()
        job_server.print_stats()

    #########################################################################################################
    # %%% MAPPER FUNCTIONALITY
    #########################################################################################################

    def Mapper_GLM(self, mask = 'bet_mask_dilated', postFix = ['mcf', 'sgtf']):
        """
        This function executes a GLM including task variables (e.g. motion vs color presence)
        and a lot of nuissance variables (e.g. eye blinks, physiology regressors)
        """

        n_TRs = np.array([NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix )).timepoints for r in [self.runList[i] for i in self.conditionDict['Mapper']]])
        cortex_mask = np.array(NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), mask + '.nii.gz')).data, dtype = bool)

        # the GLM function requires:
        # dm of shape [timepoints,reg]
        # data of shape [timepoints, voxels]
        # c of shape [n_contrasts,n_regressors]

        # # now let's setup some interesting contrasts
        # # for memory:
        # # regressor 1 = bw_static
        # # regressor 2 = bw_moving
        # # regressor 3 = col_static
        # # regressor 4 = col_moving

        n_retroicor_regressors = 34
        n_nuissance_regressors = 4
        num_n_r = n_retroicor_regressors + n_nuissance_regressors
        contrasts = [
        [0,1,0,0,0]+[0]*num_n_r,    # 0 bw_static > baseline
        [0,0,1,0,0]+[0]*num_n_r,    # 1 bw_moving > baseline
        [0,0,0,1,0]+[0]*num_n_r,    # 2 col_static > baseline
        [0,0,0,0,1]+[0]*num_n_r,    # 3 col_moving > baseline
        [0,0,0,-1,1]+[0]*num_n_r,       # 4 col_moving > col_static:                                                    
        [0,-1,0,0,1]+[0]*num_n_r,       # 5 col_moving > bw_static:                             
        [0,-1,1,-1,1]+[0]*num_n_r,  # 6 (bw_moving+col_moving) > (bw_static+col_static)     (TF > static, irrespective of color)    
        [0,0,-1,0,1]+[0]*num_n_r,       # 7 col_moving > bw_moving                                                  
        [0,-1,0,1,0]+[0]*num_n_r,       # 8 col_static > bw_static:                             (color > bw)
        [0,-1,-1,1,1]+[0]*num_n_r,      # 9 (col_moving+col_static) > (bw_moving+bw_static)     (color > BW, irrespective of TF)
        [0,0,-1,1,0]+[0]*num_n_r,       # 10 col_static > bw_moving                             (col > TF )
        [0,-1,1,0,0]+[0]*num_n_r        # 11 bw_moving > bw_static                              (TF > static)       
        ]

        from utilities import CustomStatUtilities
        CUO = CustomStatUtilities()

        nruns = len(self.conditionDict['Mapper'])
        # preallocate variables to save:
        betas_3d = np.zeros([nruns]+[43]+list(cortex_mask.shape))
        # residuals_3d = np.reshape(residuals,(np.sum(n_TRs),-1))
        copes_3d = np.zeros(([nruns]+[len(contrasts)]+list(cortex_mask.shape)))
        varcopes_3d = np.zeros(([nruns]+[len(contrasts)]+list(cortex_mask.shape)))
        t_stat_3d = np.zeros(([nruns]+[len(contrasts)]+list(cortex_mask.shape)))
        z_stat_3d = np.zeros(([nruns]+[len(contrasts)]+list(cortex_mask.shape)))
        residuals = np.zeros(([nruns]+[len(contrasts)]+list(cortex_mask.shape)))

        for ri,r in enumerate([self.runList[i] for i in self.conditionDict['Mapper']]):

            self.logger.info('loading %s'%self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
            ## load nifti data
            nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix ))
            data = nii_file.data
            if nii_file.rtime > 1000:
                rtime = nii_file.rtime/1000
            elif nii_file.rtime < 0.01:
                rtime = nii_file.rtime * 1000
            else:
                rtime = nii_file.rtime

            n_slices = np.min(nii_file.volextent)
            n_timepoints = nii_file.timepoints

            ## for mapper, load the stimulus times 
            no_color_no_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'no_color_no_speed.txt'))
            no_color_no_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in no_color_no_speed]
            no_color_yes_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'no_color_yes_speed.txt'))
            no_color_yes_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in no_color_yes_speed]
            yes_color_no_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'yes_color_no_speed.txt'))
            yes_color_no_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in yes_color_no_speed]
            yes_color_yes_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'yes_color_yes_speed.txt'))
            yes_color_yes_speed_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in yes_color_yes_speed]

            run_design = NewDesign(nii_file.timepoints, rtime, sample_duration = rtime/float(n_slices))
            run_design.configure([no_color_no_speed_list,no_color_yes_speed_list,yes_color_no_speed_list,yes_color_yes_speed_list], 
                hrf_parameters=[1,0,0])
            interest_design = run_design.convolved_design_matrix

            ## baseline per run
            baseline_regressor = np.ones(n_timepoints)[:,np.newaxis]

            ## transients
            self.logger.info('loading transient times')
            transients = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'transient_times.txt'))
            transient_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in transients]

            ## button presses
            self.logger.info('loading button press times')
            button_L = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'button_presses_L.txt'))
            button_L_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in button_L]
            button_R = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'button_presses_R.txt'))
            button_R_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in button_R]

            ## blinks
            self.logger.info('loading blink times')
            this_blink_events = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'blink_times.txt'))
            blink_times_list = [[float(tt[0]), float(tt[1]),tt[2]] for tt in this_blink_events]
            
            ## create nuissance design matrix
            self.logger.info('creating design matrices')
            nuissance_design = NewDesign(nii_file.timepoints, rtime, sample_duration = rtime/float(n_slices))
            nuissance_design.configure([transient_list,blink_times_list,button_L_list,button_R_list], 
                hrf_parameters=[1,0,0])
            nuissance_design = nuissance_design.convolved_design_matrix

            ## motion correction parameters
            self.logger.info('loading motion correction parameters')
            mcf = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '.par' ))
            mcf_dt = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '_dt.par' ))
            mcf_ddt = np.loadtxt(self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf'], extension = '_ddt.par' ))
            # sgtf these regressors, because they are estimated from the non-sgtf data
            mcf_sgtf = mcf - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf.T]).T
            mcf_dt_sgtf = mcf_dt - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf_dt.T]).T
            mcf_ddt_sgtf = mcf_ddt - np.array([sp.signal.savgol_filter(par,window_length=int(120/rtime),polyorder=3) for par in mcf_ddt.T]).T
            all_moco_regressors = np.hstack([mcf_sgtf,mcf_dt_sgtf,mcf_ddt_sgtf]).T

            ## retroicor regressors
            self.logger.info('loading retroicor regressors')
            retroicor_dir = os.path.join(self.runFolder('processed/mri/',run=r),'retroicor')
            retroicor_regressors = np.squeeze(np.array([NiftiImage(os.path.join(retroicor_dir,'retroicorev00%d.nii.gz'%(reg+1))).data if reg < 9 else NiftiImage(os.path.join(retroicor_dir,'retroicorev0%d.nii.gz'%(reg+1))).data for reg in np.arange(34)]))

            # now loop over slices and perform GLM per slice so we won't have to perform slice timing correction
            for sl in np.arange(n_slices):
                self.logger.info('executing Mapper GLM on slice %d/%d'%(sl+1,n_slices))     

                # first, we can simply hstack the regressors of interest and select relevant slice
                this_interest_dm = interest_design[:,sl::n_slices].T
                # select the behavioral nuisances
                this_nuissance_dm = nuissance_design[:,sl::n_slices].T
                # now also take the retroicor regressors
                this_retroicor_dm = retroicor_regressors[:,:,sl].T
                # combine all regressors
                this_dm = np.hstack([baseline_regressor,this_interest_dm,this_nuissance_dm,this_retroicor_dm])

                # select relevant voxels:
                slice_mask = np.zeros_like(cortex_mask)
                slice_mask[sl,:,:] = cortex_mask[sl,:,:]
                these_voxels = data[:,slice_mask]

                # now perform GLM:
                prediction,betas,t,cope,varcope,p_own,z = CUO.GLM(this_dm,these_voxels,contrasts)

                # and store output:
                betas_3d[ri,:,slice_mask] = betas.T
                copes_3d[ri,:,slice_mask] = cope.T
                varcopes_3d[ri,:,slice_mask] = varcope.T
                t_stat_3d[ri,:,slice_mask] = t.T
                z_stat_3d[ri,:,slice_mask] = z.T

        self.logger.info('combining results from different runs based on inverse of varcope')       

        # combine results from different runs based on inverse of varcope
        copes = np.average(copes_3d,weights=1/varcopes_3d,axis=0)
        t_stat = np.average(t_stat_3d,weights=1/varcopes_3d,axis=0)
        z_stat = np.average(z_stat_3d,weights=1/varcopes_3d,axis=0)
        varcopes = np.average(varcopes_3d,axis=0)
        betas = np.average(betas_3d,axis=0)

        # and save results
        output_folder = self.stageFolder('processed/mri/Mapper/GLM/')
        if not os.path.isdir(output_folder): os.mkdir(output_folder)
        
        self.logger.info('saving betas')            
        betas_nii_file = NiftiImage(betas)
        betas_nii_file.header = nii_file.header
        betas_nii_file.save(os.path.join(output_folder,'betas.nii.gz'))

        self.logger.info('saving copes')    
        copes_nii_file = NiftiImage(copes)
        copes_nii_file.header = nii_file.header
        copes_nii_file.save(os.path.join(output_folder,'copes.nii.gz'))

        self.logger.info('saving varcopes') 
        varcopes_nii_file = NiftiImage(varcopes)
        varcopes_nii_file.header = nii_file.header
        varcopes_nii_file.save(os.path.join(output_folder,'varcopes.nii.gz'))

        self.logger.info('saving t_stat')   
        t_stat_nii_file = NiftiImage(t_stat)
        t_stat_nii_file.header = nii_file.header
        t_stat_nii_file.save(os.path.join(output_folder,'t_stat.nii.gz'))

        self.logger.info('saving z_stat')   
        z_stat_nii_file = NiftiImage(z_stat)
        z_stat_nii_file.header = nii_file.header
        z_stat_nii_file.save(os.path.join(output_folder,'z_stat.nii.gz'))

        self.logger.info('projecting to surface')       
        surf_folder = self.stageFolder('processed/mri/Mapper/GLM/surf/')
        if not os.path.isdir(surf_folder): os.mkdir(surf_folder)
        # and turn into surfaces
        for sm in [0,1,2,3,5]:
            for measure in ['copes','z_stat']:
                # and project to surface
                for copenr in [6,9,10]:
                    vsO = VolToSurfOperator(inputObject =os.path.join(output_folder,'%s.nii.gz'%measure))
                    ofn = os.path.join(os.path.join(surf_folder,'%s_cope_%d_sm_%d'%(measure,copenr,sm)))
                    vsO.configure(outputFileName = ofn, threshold = 0.5, surfSmoothingFWHM = sm,frames = {'_f':copenr}, surfType = 'paint' ,register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ) )
                    vsO.execute()

    def hrf_from_mapper(self,n_slices=30,mask='bet_mask_dilated',plot=True,target_session=-1,n_jobs=-1):
        """
        This function determines the optimal hrf for each voxel from the feature mapper data.
        It does so using the hrf_estimation package (https://github.com/fabianp/hrf_estimation)

        note: this function was run to fit on the 'residuals' of the mapper glm procedure
        This 'residual' data contained the original data only minus the nuissance variables.
        In essence this is thus cleaned data.
        Although the current version of the mapper glm does not output such residuals, the 
        previous version did.
        """

        # load data from the best fitting voxels in the mapper glm
        GLM_folder = self.stageFolder('processed/mri/Mapper/GLM/')
        self.logger.info('loading z-stats')
        # fit on voxels that respond to both color and motion (cope 3)
        z_stats = NiftiImage(os.path.join(GLM_folder,'z_stat.nii.gz')).data[3]
        #  and take the 1000 most responding voxels
        cutoff_z_score = np.sort(np.ravel(z_stats))[-1001]
        cortex_mask = np.zeros_like(z_stats).astype(bool)
        cortex_mask[z_stats>cutoff_z_score] = 1

        # first, lets concatenate all the data and all the predictors across runs
        all_no_color_no_speed_list = []
        all_no_color_yes_speed_list = []
        all_yes_color_no_speed_list = []
        all_yes_color_yes_speed_list = []
        run_delay = 0
        total_TRs=0
        for r in [self.runList[i] for i in self.conditionDict['Mapper']]: 

            TR = NiftiImage(self.runFile(stage = 'processed/mri', run = r)).rtime
            if TR > 1000:
                TR /= 1000.
            if TR < 0.01:
                TR *= 1000
            n_TRs = NiftiImage(self.runFile(stage = 'processed/mri', run = r)).timepoints
            header = NiftiImage(self.runFile(stage = 'processed/mri', run = r)).header

            no_color_no_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'no_color_no_speed.txt'))
            all_no_color_no_speed_list.extend([[float(tt[0]+run_delay), float(tt[1]),tt[2]] for tt in no_color_no_speed])
            no_color_yes_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'no_color_yes_speed.txt'))
            all_no_color_yes_speed_list.extend([[float(tt[0]+run_delay), float(tt[1]),tt[2]] for tt in no_color_yes_speed])
            yes_color_no_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'yes_color_no_speed.txt'))
            all_yes_color_no_speed_list.extend([[float(tt[0]+run_delay), float(tt[1]),tt[2]] for tt in yes_color_no_speed])
            yes_color_yes_speed = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'yes_color_yes_speed.txt'))
            all_yes_color_yes_speed_list.extend([[float(tt[0]+run_delay), float(tt[1]),tt[2]] for tt in yes_color_yes_speed])

            run_delay += n_TRs*TR
            total_TRs += n_TRs

        regressor_list = [all_no_color_no_speed_list,all_no_color_yes_speed_list,all_yes_color_no_speed_list,all_yes_color_yes_speed_list]

        # create design matrix. 
        self.logger.info('creating design matrices')
        run_design = NewDesign(total_TRs, TR, sample_duration = TR/float(n_slices))
        run_design.configure(regressor_list, hrf_parameters = [1,0,0])
        hrf_dm = run_design.convolved_design_matrix
        run_design = NewDesign(total_TRs, TR, sample_duration = TR/float(n_slices))
        run_design.configure(regressor_list, hrf_parameters = [0,1,0])
        hrf_dt_dm = run_design.convolved_design_matrix          
        run_design = NewDesign(total_TRs, TR, sample_duration = TR/float(n_slices))
        run_design.configure(regressor_list, hrf_parameters = [0,0,1])
        hrf_ddt_dm = run_design.convolved_design_matrix

        self.logger.info('stacking different event regressors')
        combined_design_matrix = []
        for reg in np.arange(4):
            combined_design_matrix.append(hrf_dm[reg])
            combined_design_matrix.append(hrf_dt_dm[reg])
            combined_design_matrix.append(hrf_ddt_dm[reg])

        # load data (see note in function description)
        GLM_folder = self.stageFolder('processed/mri/Mapper/GLM/')
        all_data = NiftiImage(os.path.join(GLM_folder,'residuals.nii.gz')).data

        # and fit the hrfs
        hrfs = []
        for sl in range(n_slices):

            slice_mask = np.zeros_like(cortex_mask)
            slice_mask[sl,:,:] = cortex_mask[sl,:,:]

            if slice_mask.sum() > 0:
                
                this_dm = np.array(combined_design_matrix)[:,sl:total_TRs*n_slices:n_slices].T
                these_data = all_data[:,slice_mask]

                these_hrfs, betas = he.rank_one(this_dm,these_data,3)
                hrfs.append(these_hrfs)

            self.logger.info('now fitting HRF per slice - done %d/%d'%(sl+1,n_slices))

        hrfs = np.hstack(hrfs)

        # and plot
        mean_hrf_params = [np.median(hrfs[0]),np.median(hrfs[1]),np.median(hrfs[2])]
        all_hrfs = np.zeros([3]+list(cortex_mask.shape))
        all_hrfs[:,cortex_mask] = hrfs
        all_hrfs[:,cortex_mask == False] = np.tile(mean_hrf_params,(np.sum(cortex_mask == False),1)).T
        all_mean_hrfs = np.reshape(np.tile(mean_hrf_params,(np.size(cortex_mask),1)).T,([3] + list(cortex_mask.shape)))

        colors = np.array([colorsys.hsv_to_rgb(c,0.6,0.9) for c in np.linspace(0.0,1,3)])[:-1]
        f = pl.figure(figsize=(10,10))
        xx = np.arange(0,32,0.05)
        s = f.add_subplot(111)
        generated_hrfs = all_hrfs[0,cortex_mask] * he.hrf.spmt(xx)[:, None] + all_hrfs[1,cortex_mask]  * he.hrf.dspmt(xx)[:, None] + all_hrfs[2,cortex_mask]  * he.hrf.ddspmt(xx)[:, None]
        generated_hrfs /= np.sum(np.abs(generated_hrfs),axis=0)
        mean_hrf = mean_hrf_params[0] * he.hrf.spmt(xx) +mean_hrf_params[1]* he.hrf.dspmt(xx) +mean_hrf_params[2] * he.hrf.ddspmt(xx)
        mean_hrf /= np.sum(np.abs(mean_hrf))
        standard_hrf = he.hrf.spmt(xx) 
        standard_hrf /= np.sum(np.abs(standard_hrf))

        pl.plot(xx, generated_hrfs,linewidth=1,alpha=0.2,color='k')
        pl.plot(xx, mean_hrf,color=colors[0],linewidth=4,label = 'median hrf')
        pl.plot(xx, standard_hrf ,color=colors[1],linewidth=4,label='standard hrf')
        pl.legend(loc='best',fontsize=16)
        sn.despine(offset=10)
        pl.xlabel('time (s)')
        pl.ylabel('a.u.')

        pl.savefig(os.path.join(self.stageFolder('processed/mri/figs/'),'all_HRFs.pdf'))

        # and save
        self.logger.info('outputting hrf parameters')           
        all_hrf_nii_file = NiftiImage(all_hrfs)
        all_hrf_nii_file.header = header
        all_hrf_nii_file.save(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'hrf_parameters.nii.gz'))

        self.logger.info('outputting median hrf parameters')            
        mean_hrf_nii_file = NiftiImage(all_mean_hrfs)
        mean_hrf_nii_file.header = header
        mean_hrf_nii_file.save(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'mean_hrf_parameters.nii.gz'))

        np.save(os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),'mean_hrf_parameters.npy'),mean_hrf_params)

    #########################################################################################################
    # %%% PRF FUNCTIONALITY
    #########################################################################################################

    def stimulus_timings(self):
        """
        This function takes trial times as computed by the eye analysis function and stores it in run object 
        """

        for ri,r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]):            
            
            this_trial_time_file = np.loadtxt(os.path.join(self.runFolder(stage = 'processed/mri', run = r),'trial_times.txt'), delimiter="\t")

            r.trial_times = this_trial_time_file
            r.trial_duration = np.mean(r.trial_times[:,2]- r.trial_times[:,1])
            r.orientations = r.trial_times[:,4]

    def eye_analysis(self,conditions,delete_hdf5=False,import_raw_data=False,import_all_data=False,detect_saccades=False,
        write_trial_timing_text_files=False,add_to_group_level_hdf5=False,rotate_eye_position_to_bar=False):
        """
        This function takes info contained in the .edf files such as:
        * blink times
        * trial timings (important for design matrices)
        * button press times
        * gaze data
        """

        for condition in conditions:
            edfs=[]
            for r in [self.runList[i] for i in self.conditionDict[condition]]:
                all_eye_files = os.listdir(os.path.join(self.runFolder(stage = 'processed/eye', run = r)))
                edfs.append(os.path.join(self.runFolder(stage = 'processed/eye', run = r),[this_file  for this_file in all_eye_files if 'edf' in this_file][0]))
            aliases = ['%s_run_%s'%(condition,i+1) for i in range(len(edfs))]           

            eyesession = EyeFromfMRISession(PRFsession = self,subject = self.subject, experiment_name = 'PRF_eye_analysis', project_directory = self.this_project_folder)
            if import_raw_data:
                eyesession.import_raw_data(edf_files=edfs, aliases=aliases)
            if delete_hdf5:
                eyesession.delete_hdf5()
            if import_all_data: 
                eyesession.import_all_data(aliases=aliases)
            if write_trial_timing_text_files:
                eyesession.write_trial_timing_text_files(aliases=aliases,condition=condition)
                eyesession.write_blink_text_files(aliases=aliases,condition=condition)
                eyesession.write_button_text_files(aliases=aliases,condition=condition)
                eyesession.write_transient_text_files(aliases=aliases,condition=condition)
            if add_to_group_level_hdf5:
                eyesession.add_to_group_level_hdf5(aliases=aliases,condition=condition)

    def design_matrices_for_concatenated_data(self, gamma_hrfType = 'doubleGamma', hrf_params = [1,0,0],n_pixel_elements_raw = 101,
        n_pixel_elements_convolved=31,task_conditions=['All','Fix','Speed','Color'],change_type='all_data',run_num=0,animate_dm=False):
        """
        This function creates a design matrix per stimulus type.
        For this, it relies heavily on the ModelExperiment object.
        """

        dm_dir = os.path.join(self.stageFolder('processed/mri/PRF/design_matrices/'))
        try:
            os.mkdir(dm_dir)
        except:
            pass

        for this_condition in ['All','Stim','Fix','Color']:
            self.logger.info('Creating design matrix for condition %s'%this_condition)
            self.stimulus_timings()

            raw_dms = []
            convolved_dms = []
            for ri,r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]): 

                this_nii_file = NiftiImage(self.runFile(stage = 'processed/mri', run = r))
                TR = this_nii_file.rtime
                n_TRs = this_nii_file.timepoints
                n_slices = np.min(this_nii_file.extent)

                # recode TR to seconds if it was coded in milliseconds
                if TR > 10:
                    TR /= 1000
                elif TR < 0.01:
                    TR *= 1000      

                # select which trials to include based on attention condition
                if this_condition == 'All':
                    which_trials = np.ones(len(r.trial_times)).astype(bool)
                elif this_condition == 'Stim':
                    which_trials = (r.trial_times[:,0]==np.where(np.array(self.task_names['PRF'])=='Speed')[0][0])\
                    +(r.trial_times[:,0]==np.where(np.array(self.task_names['PRF'])=='Color')[0][0])
                else: 
                    which_trials = (r.trial_times[:,0]==np.where(np.array(self.task_names['PRF'])==this_condition)[0][0]) 
                r.orientations = np.radians(r.orientations[which_trials])
                r.trial_times = r.trial_times[which_trials]

                # first, let's create a 'raw' design matrix at TR time resolution with 101 pixel elements
                # the prediction will be slice timing corrected in the pRF fitting procedure
                sample_duration = TR
                bar_width = 0.25
                mr = PRFModelRun(r, n_samples = n_TRs, n_pixel_elements = n_pixel_elements_raw, sample_duration = sample_duration, bar_width = bar_width)
                self.logger.info('simulating model experiment run %d with %d pixel elements and %1.4f s sample_duration'%(ri,n_pixel_elements_raw, sample_duration))
                mr.simulate_run()
                raw_dms.append(mr.run_matrix)

                if this_condition == 'All':
                    # then, let's create a design matrix at slice time resolution with 31 pixel elements for convolution
                    sample_duration = TR/float(n_slices)
                    n_samples = n_TRs*n_slices
                    mr = PRFModelRun(r, n_samples = n_samples, n_pixel_elements = n_pixel_elements_convolved, sample_duration = sample_duration, bar_width = bar_width)
                    self.logger.info('simulating model experiment run %d with %d pixel elements and %1.4f s sample_duration'%(ri,n_pixel_elements_convolved, sample_duration))
                    mr.simulate_run()

                    self.logger.info('convolving design matrix with hrf')
                    run_design = Design(mr.run_matrix.shape[0], mr.sample_duration, subSamplingRatio = 1)
                    run_design.rawDesignMatrix = mr.run_matrix.reshape((n_TRs*n_slices,-1)).T
                    run_design.convolveWithHRF(hrfParameters = hrf_params)
                    convolved_dms.append(run_design.designMatrix.T.reshape(n_TRs*n_slices,n_pixel_elements_convolved,n_pixel_elements_convolved))

            # the convolved design matrices are only used when fitting on all data
            # (for indiviual attention conditions, initialization is from the 'all' direct pRF model fit)
            if this_condition == 'All':
                combined_convolved_dm = np.vstack(convolved_dms)
                del convolved_dms
                self.logger.info('saving convolved design matrix')
                np.save(os.path.join(dm_dir, 'convolved_design_matrix_for_concatenated_data_%s.npy'%(this_condition)),combined_convolved_dm)

            combined_dm = np.vstack(raw_dms)
            del raw_dms
            self.logger.info('saving raw design matrix')
            np.save(os.path.join(dm_dir, 'raw_design_matrix_for_concatenated_data_%s.npy'%(this_condition)),combined_dm)

            if animate_dm:
                n_timepoints = 300
                pl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
                ims = []
                f=pl.figure(figsize=(4,4))
                dm = combined_dm
                timepoints = np.arange(n_timepoints)
                for t in timepoints:
                    s=f.add_subplot(111)
                    im=pl.imshow(dm[t,:,:],origin='lowerleft',interpolation='nearest',cmap='Greys')
                    pl.clim(0,np.max(dm))
                    pl.axis('off')
                    ims.append([im])
                ani = animation.ArtistAnimation(f, ims)#, interval=5, blit = True, repeat_delay = 1000)
                mywriter = animation.FFMpegWriter(fps = 10)
                self.logger.info('saving animation')
                plotdir = os.path.join(self.stageFolder('processed/mri/figs/dm_animations'))

                ani.save(os.path.join(plotdir,'raw_design_matrix_conc_data_%s_%dx%d.mp4'%(this_condition,n_pixel_elements_raw, n_pixel_elements_raw)),writer=mywriter,dpi=300,bitrate=200)

    def setup_fit_PRF_on_concatenated_data(self, anat_mask_file_name = 'dilated_bet_mask',all_mask_file_name='dilated_bet_mask',postFix = [], n_jobs = 1, fit_on_all_data=True,n_slices = 30,
                n_vox_per_ROI=100,plotbool=False, model='OG',hrf_type='canonical',n_pixel_elements_raw = 101,n_pixel_elements_convolved=31,r_squared_threshold=0.1,
                use_shared_memory = False,slice_no=0,change_type='all_data',run_num=0):
        """
        This function fits a pRF model to the data.
        For this, it heavily relies on the FitPRFModel object.
        """

        # set tempdir 
        if socket.gethostname() == 'aeneas':
            tempdir = '/home/vanes/temp'
        else:
            tempdir = '/dev/shm'

        self.logger.info('starting PRF fit procedure')
        total_elapsed_time = 0

        # load anatomical mask - not large enough to store in shared data
        mask_file = NiftiImage(os.path.join(self.stageFolder('processed/mri/masks/anat'), anat_mask_file_name +  '.nii.gz'))
        cortex_mask = np.array(mask_file.data, dtype = bool)

        # in the first all-data fit, we must fit on all data
        # for we have no estimate for how 'visually' responsive
        # voxels are. After this, we can use an R2 threshold
        # on the all data fit to fit on smaller selection of voxels.
        if not fit_on_all_data:
            # and the results frames
            filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'frames.pickle')
            with open(filename) as f:
                picklefile = pickle.load(f)
            stats_frames = picklefile['stats_frames']
            results_frames = picklefile['results_frames']
            stats = NiftiImage(os.path.join(self.stageFolder('processed/mri/PRF/'), 'corrs_' + all_mask_file_name + '_' + '_'.join(postFix)  + '_' + 
                model + '_All_hrf_' + hrf_type+ '.nii.gz')).data[stats_frames['r_squared'],:]
            cortex_mask_size = cortex_mask.sum()
            cortex_mask *= (stats>r_squared_threshold)
            stat_mask_size = cortex_mask.sum()
            self.logger.info('%d/%d voxels selected(%.1f%s), R2 threshold = %.6f'%(stat_mask_size,cortex_mask_size,stat_mask_size/cortex_mask_size*100,'%',r_squared_threshold))
            del stats

        # load and concatenate the data
        all_data = []
        retroicor_regressors = []
        moco_regressors = []
        n_TRs = []
        import time as t
        t0=t.time()
        for ri,r in enumerate([self.runList[i] for i in self.conditionDict['PRF']]): 
            this_nii_file = self.runFile(stage = 'processed/mri', run = r, postFix=postFix)
            n_TRs.append(NiftiImage(this_nii_file).timepoints)
            rtime = NiftiImage(this_nii_file).rtime
            # recode to s if in ms:
            if rtime > 1000:
                rtime/=1000
            self.logger.info('loading %s'%this_nii_file)
            all_data.append(NiftiImage(this_nii_file).data[:,cortex_mask])

        # combine runs
        n_slices = np.min(NiftiImage(this_nii_file).extent)
        all_data = np.vstack(all_data)
        self.logger.info('Loading data lasted %.3f seconds'%(t.time()-t0))
    
        # convert it into a shared variable
        filename = os.path.join(tempdir, 'all_data.dat')
        fp = np.memmap(filename, dtype='float32', mode='write', shape=all_data.shape)
        fp[:] = all_data[:]
        all_data_shared = np.memmap(filename, dtype='float32', mode='readonly', shape=all_data.shape)
        del all_data, fp

        # get the hrf parameters
        if hrf_type != 'canonical':
            hrf_nifti_filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'mean_hrf_parameters.nii.gz') 
            all_hrf_parameters = NiftiImage(hrf_nifti_filename).data[:,cortex_mask]
            # convert it into a shared variable
            filename = os.path.join(tempdir, 'all_hrf_parameters.dat')
            fp = np.memmap(filename, dtype='float32', mode='write', shape=all_hrf_parameters.shape)
            fp[:] = all_hrf_parameters[:]
            all_hrf_parameters_shared = np.memmap(filename, dtype='float32', mode='readonly', shape=all_hrf_parameters.shape)
            del all_hrf_parameters, fp

        # number slices
        slices = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T[cortex_mask]
        slices_in_full = (np.ones(cortex_mask.shape).T * np.arange(cortex_mask.shape[0])).T

        # figure out roi label per voxel and how many there are 
        anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
        anatRoiFileNames = [anRF for anRF in anatRoiFileNames if np.all([np.any(['bh' in anRF,'lh' in anRF,'rh' in anRF]),'cortex' not in anRF])]
        roi_names = np.zeros_like(slices_in_full).astype('string')
        for this_roi in anatRoiFileNames:
            roi_nifti = NiftiImage(this_roi).data.astype('bool')
            roi_names[roi_nifti] = (this_roi.split('/')[-1]).split('.')[1]
        roi_names[roi_names=='0.0'] = 'unkown_roi'
        roi_names = roi_names[cortex_mask]
        roi_count = {}
        for roi in np.unique(roi_names):
            roi_count[roi] = np.size(roi_names[roi_names==roi]) 

        # load the all results for initialization when fitting on the invidual condition data
        if fit_on_all_data:
            task_conditions = ['All']
            all_params_shared = None
            results_frames = None
        else:
            task_conditions = ['Fix','Speed','Color']
            all_params = NiftiImage(os.path.join(self.stageFolder('processed/mri/PRF/'), 'results_' + all_mask_file_name + '_' + '_'.join(postFix)  + '_' + model + 
                '_All_hrf_' + hrf_type+ '.nii.gz')).data[:,cortex_mask]
            # put all_params in shared mem
            filename = os.path.join(tempdir, 'all_params.dat')
            fp = np.memmap(filename, dtype='float32', mode='write', shape=all_params.shape)
            fp[:] = all_params[:]    
            all_params_shared = np.memmap(filename, dtype='float32', mode='readonly', shape=all_params.shape)
            del all_params, fp

        # set up empty arrays for saving the data
        all_results = {}
        all_corrs = {}
        for condition in task_conditions:
            all_results[condition] = np.zeros([18] + list(cortex_mask.shape))
            all_corrs[condition] = np.zeros([5] + list(cortex_mask.shape))

        # create plot dir for individual voxel plots
        if plotbool: 
            plotbase = os.path.join(self.stageFolder('processed/mri/figs/PRF_fit_plots'))
            if not os.path.isdir(plotbase): os.mkdir(plotbase)
            if change_type == 'all_data':
                this_plot_dir = os.path.join(plotbase,'%s_%s_%s_%s_%s'%(anat_mask_file_name,model,'_'.join(postFix),'_'.join(task_conditions),time_module.strftime("%d.%m.%Y")))
            else:
                this_plot_dir = os.path.join(plotbase,'%s_%s_%s_%s_%s_%d_%s'%(anat_mask_file_name,model,'_'.join(postFix),'_'.join(task_conditions),change_type,run_num,time_module.strftime("%d.%m.%Y")))
            try:
                os.mkdir(this_plot_dir)
            except:
                if os.path.isdir(this_plot_dir):
                    self.logger.info('not making plot dir as it exists already')
                else:
                    self.logger.info('unkown error in plotdir generation')
        else:
            plotdir = []

        # load raw dms
        raw_dms = {}
        for this_condition in task_conditions:
            self.logger.info('loading raw design matrix for condition %s'%this_condition)
            raw_dms[this_condition] = np.load(os.path.join(self.stageFolder('processed/mri/PRF/design_matrices/'),'raw_design_matrix_for_concatenated_data_%s.npy'%(this_condition)))

        # load convolved dms
        if fit_on_all_data:
            self.logger.info('loading convolved design matrix')
            convolved_dm = np.load(os.path.join(self.stageFolder('processed/mri/PRF/design_matrices/'),'convolved_design_matrix_for_concatenated_data_All.npy')).reshape(-1,n_pixel_elements_convolved**2)
            valid_regressors = convolved_dm.sum(axis = 0) != 0
            convolved_dm = convolved_dm[:,valid_regressors]
            # put in shared mem
            filename = os.path.join(tempdir, 'convolved_dm.dat')
            fp = np.memmap(filename, dtype='float32', mode='write', shape=convolved_dm.shape)
            fp[:] = convolved_dm[:]
            convolved_dm_shared = np.memmap(filename, dtype='float32', mode='readonly', shape=convolved_dm.shape)
            del convolved_dm, fp
        else:
            valid_regressors = []
            convolved_dm_shared = []

        # determine slice iterable
        if slice_no != None:
            if (slices == slice_no).sum() > 0:
                slice_iterable = [slice_no]
            else: 
                slice_iterable = None
        else:
            slice_iterable = np.unique(slices)
    
        # estimate fit duration
        fit_start_time = time_module.time()
        if fit_on_all_data:
            minute_per_voxel = 0.07
        else:
            minute_per_voxel = 0.35

        if slice_iterable != None:
            num_voxels = np.sum(cortex_mask[np.array(slice_iterable).astype(int)])
        else:
            num_voxels = cortex_mask.sum()
        estimated_fit_duration = num_voxels * minute_per_voxel
        self.logger.info('starting PRF model fits on %d voxels total'%(int(num_voxels)))
        self.logger.info('estimated total duration: %dm (%.1fh)' % (estimated_fit_duration,estimated_fit_duration/60.0))
        
        # now loop over slices and fit PRF model for voxels parallel
        if slice_iterable != None:
            for sl in slice_iterable:

                plotdir = os.path.join(this_plot_dir,'slice_%d'%sl)
                if os.path.isdir(plotdir): shutil.rmtree(plotdir)
                os.mkdir(plotdir)

                # get voxels from this slice
                voxels_in_this_slice = (slices == sl)
                voxels_in_this_slice_in_full = (slices_in_full == sl)
                these_roi_names = roi_names[voxels_in_this_slice]

                # only get convolved dm if on all data
                if fit_on_all_data:
                    this_slice_convolved_dm = convolved_dm_shared[int(sl)::int(n_slices)]
                else:
                    this_slice_convolved_dm = []

                # this is just for plotting filenames
                randints_for_plot = [(np.random.randint(roi_count[these_roi_names[voxno]])<n_vox_per_ROI) for voxno in range(voxels_in_this_slice.sum())]
                
                self.logger.info('now fitting pRF models on slice %d, with %d voxels' % (sl, voxels_in_this_slice.sum()))

                # actually fit the pRF model
                res = Parallel(n_jobs = np.min([n_jobs,voxels_in_this_slice.sum()]), verbose = 9)(
                            delayed(fit_PRF_on_concatenated_data)(
                            data_shared                     = all_data_shared,
                            voxels_in_this_slice            = voxels_in_this_slice,
                            n_TRs                           = n_TRs,
                            n_slices                        = n_slices,
                            fit_on_all_data                 = fit_on_all_data,
                            plotbool                        = plotbool,
                            raw_design_matrices             = raw_dms, 
                            dm_for_BR                       = this_slice_convolved_dm, 
                            valid_regressors                = valid_regressors,
                            n_pixel_elements_convolved      = self.n_pixel_elements_convolved,
                            n_pixel_elements_raw            = self.n_pixel_elements_raw,
                            plotdir                         = plotdir,
                            voxno                           = voxno,
                            slice_no                        = sl,
                            randint                         = randints_for_plot[voxno],
                            roi                             = these_roi_names[voxno],
                            TR                              = self.TR,
                            model                           = model,
                            hrf_params_shared               = all_hrf_parameters_shared,
                            all_results_shared              = all_params_shared,
                            conditions                      = task_conditions,
                            results_frames                  = results_frames,
                            postFix                         = postFix,
                            )
                        for voxno in range(voxels_in_this_slice.sum())) 

                # insert this slice's results in the whole brain variables
                for condition in task_conditions:
                    all_corrs[condition][:, cortex_mask * voxels_in_this_slice_in_full] = np.array([rs[1].values() for rs in res]).T
                    all_results[condition][:, cortex_mask * voxels_in_this_slice_in_full] = np.array([rs[0][condition].values() for rs in res]).T

            results_frames = {}
            for ki,key in enumerate(res[0][0][task_conditions[0]].keys()):
                results_frames[key] = ki
            stats_frames = {}
            for ki,key in enumerate(res[0][1].keys()):
                stats_frames[key] = ki              
            
            # save frames
            filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'frames.pickle')
            with open(filename, 'w') as f:
                pickle.dump({'results_frames':results_frames,'stats_frames':stats_frames}, f)

            fit_end_time = time_module.time()
            fit_time_this_condition = (fit_end_time-fit_start_time)/60.
            self.logger.info('fitting this condition lasted: %dm'%(fit_time_this_condition))
            total_elapsed_time += fit_time_this_condition
            self.logger.info('total fit time: %dm'%(total_elapsed_time))
            
            # and save for every condition
            # note that the corrs file will be the same for the fix/color/speed condition
            for condition in task_conditions:

                self.logger.info('saving coefficients and correlations of PRF fits for condition %s'%condition)
                if change_type == 'all_data':
                    filename = '%s_%s_%s_%s_hrf_%s'%(anat_mask_file_name,'_'.join(postFix),model,condition,hrf_type)
                else:
                    filename = '%s_%s_%s_%s_hrf_%s_%s_%d'%(anat_mask_file_name,'_'.join(postFix),model,condition,hrf_type,change_type,run_num)

                if slice_no == None:
                    stage_dir = self.stageFolder('processed/mri/PRF/')
                    save_filename = filename
                else:
                    stage_dir = os.path.join(self.stageFolder('processed/mri/PRF/'),filename)
                    save_filename = filename + '_sl_%d'%slice_no

                if not os.path.isdir(stage_dir): os.mkdir(stage_dir)

                # replace infs in correlations with the maximal value of the rest of the array.
                all_corrs[condition][np.isinf(all_corrs[condition])] = all_corrs[condition][-np.isinf(all_corrs[condition])].max() + 1.0
                corr_nii_file = NiftiImage(all_corrs[condition])
                corr_nii_file.header = mask_file.header
                corr_nii_file.save(os.path.join(stage_dir,'corrs_'+save_filename+'.nii.gz'))

                results_nii_file = NiftiImage(all_results[condition])
                results_nii_file.header = mask_file.header
                results_nii_file.save(os.path.join(stage_dir,'results_'+save_filename+'.nii.gz'))
        else:
            self.logger.info('no voxels in slice %d'%slice_no)

    def combine_seperate_slice_niftis(self,mask_file_name='rh.V1',postFix=['mcf','sgtf','psc'],model='OG',task_conditions=['All'],hrf_type='median'):
        """
        As we fit each slice on a separate node on the supercomputer (cartesius),
        we now need to recombine slice niftis.
        """

        for condition in task_conditions:
            base_filenames = ['%s_%s_%s_%s_hrf_%s'%(mask_file_name,'_'.join(postFix),model,condition,hrf_type)]

            for this_base_filenames in base_filenames:
                slice_nifti_folder = os.path.join(self.stageFolder('processed/mri/PRF/'),this_base_filenames)
                slice_nifti_files = os.listdir(slice_nifti_folder)

                for data_type in ['corrs','results']:

                    self.logger.info('combining niftis for %s data'%data_type)

                    these_slice_nifti_files = [fn for fn in slice_nifti_files if data_type in fn] 
                    save_filename = os.path.join(self.stageFolder('processed/mri/PRF/'),data_type+'_'+this_base_filenames+'.nii.gz')

                    # load first slice to get dimensions
                    example_nifti = NiftiImage(os.path.join(slice_nifti_folder,these_slice_nifti_files[0]))
                    # pre allocate output nifti
                    all_data = np.zeros(example_nifti.data.shape)

                    # loop over filenames:
                    for fi,filename in enumerate(these_slice_nifti_files):
                        self.logger.info('slice%d/%d'%(fi+1,len(these_slice_nifti_files)))
                        this_slice_data = NiftiImage(os.path.join(slice_nifti_folder,filename)).data
                        slice_no = int(filename.split('.')[-3].split('_')[-1])
                        all_data[:,slice_no,:,:] = this_slice_data[:,slice_no,:,:]

                    self.logger.info('saving to %s'%save_filename)
                    all_data_nifti = NiftiImage(all_data)
                    all_data_nifti.header = example_nifti.header
                    all_data_nifti.save(save_filename)

    def convert_to_surf(self,mask_file='cortex',postFix = ['mcf','sgtf','res','psc'],all_data_fit=True,corr_threshold=0.3,retinotopy=True,attentotopy=False,
        model='OG',hrf_type='median',depth_min=-0.5,depth_max=1.5,depth_step=0.1,sms=[0],task_conditions=['Fix']):
        """
        This function converts the results to surface polar angles for 
        defining retinotopic maps by hand
        """

        depths = np.arange(depth_min,depth_max+depth_step,depth_step)

        filename = os.path.join(self.stageFolder('processed/mri/PRF/'), 'frames.pickle')
        with open(filename) as f:
            picklefile = pickle.load(f)
        stats_frames = picklefile['stats_frames']
        results_frames = picklefile['results_frames']

        for this_condition in task_conditions:

            for depth in depths:
                filename = '%s_%s_%s_%s_hrf_%s'%(mask_file,'_'.join(postFix),model,this_condition,hrf_type)
                # filename =  mask_file + '_' + '_'.join(postFix)+ '_' + model + '_' + str(this_condition) + '_' + hrf_type +'_hrf'

                # r_squared = np.ravel(stats[stats_frames['r_squared']])
                # hist(r_squared[r_squared>0.1])
                results = NiftiImage(self.stageFolder('processed/mri/PRF/results_'+filename+ '.nii.gz')).data
                stats = NiftiImage(self.stageFolder('processed/mri/PRF/corrs_'+filename+ '.nii.gz')).data

                for sm in sms:#,3,5]: # different smoothing values.
                    # reproject the original stats
                    self.results_to_surface(file_name = 'corrs_' + filename, output_file_name = 'corrs_' + filename + '_' + str(sm) + '_depth_%.2f'%depth, depth=depth, frames = {'_f':stats_frames['r_squared']}, smooth = sm)
                    # and the spatial values
                    self.results_to_surface(file_name = 'results_' + filename, output_file_name = 'results_' + filename + '_' + str(sm) + '_polar' + '_depth_%.2f'%depth, depth=depth,frames = {'_real':results_frames['real_polar'], '_imag':results_frames['imag_polar'] }, smooth = sm)


    def results_to_surface(self, file_name = 'corrs_cortex', output_file_name = 'polar', frames = {'_f':1}, smooth = 0.0,depth=0.5):
        """docstring for results_to_surface"""
        vsO = VolToSurfOperator(inputObject = os.path.join(self.stageFolder('processed/mri/PRF/'), file_name + '.nii.gz'))
        ofn = os.path.join(self.stageFolder('processed/mri/PRF/surf/'), output_file_name )
        vsO.configure(outputFileName = ofn, threshold = depth, surfSmoothingFWHM = smooth,frames = frames, surfType = 'paint' ,register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ) )
        vsO.execute()

    def combine_surfaces(self,mask_file,postFix,model,hrf_type,depth_min=-0.5,depth_max=1.5,depth_step=0.1,task_conditions=['All'],sms=[1,2,3]):

        depths = np.arange(depth_min,depth_max+depth_step,depth_step)

        import nibabel
        for this_condition in task_conditions:

            for sm in sms:
                for measure in ['polar']:
                    for hemi in ['rh','lh']:

                        all_real_data = []
                        all_imag_data = []
                        
                        base_filename =  os.path.join(self.stageFolder(stage='processed/mri/PRF/surf/'),'results_%s_%s_%s_%s_hrf_%s_%d_%s'%(mask_file,'_'.join(postFix),model,this_condition,hrf_type,sm,measure))

                        for depth in depths:

                            real_filename = '%s_depth_%.2f_real-%s.mgz'%(base_filename,depth,hemi)
                            imag_filename = '%s_depth_%.2f_imag-%s.mgz'%(base_filename,depth,hemi)

                            all_real_data.append(nibabel.load(real_filename).get_data())
                            mean_real_placeholder = nibabel.load(real_filename)
                            all_imag_data.append(nibabel.load(imag_filename).get_data())
                            mean_imag_placeholder = nibabel.load(imag_filename)

                        # now that we have all the depth files, let's average them together:
                        mean_real_data = np.mean(all_real_data,axis=0)
                        mean_imag_data = np.mean(all_imag_data,axis=0)
                        # and save
                        mean_real_mgh = nibabel.MGHImage(mean_real_data,mean_real_placeholder.affine,mean_real_placeholder.header)
                        real_avg_filename = '%s_depth_%.2f-%.2f_real-%s.mgz'%(base_filename,depths[0],depths[-1],hemi)
                        nibabel.save(mean_real_mgh,real_avg_filename)
                        mean_imag_mgh = nibabel.MGHImage(mean_imag_data,mean_imag_placeholder.affine,mean_imag_placeholder.header)
                        imag_avg_filename = '%s_depth_%.2f-%.2f_imag-%s.mgz'%(base_filename,depths[0],depths[-1],hemi)
                        nibabel.save(mean_imag_mgh,imag_avg_filename)

                        # and convert the complex numbers back to polar
                        common_filename = '%s_depth_%.2f-%.2f'%(base_filename,depth_min,depth_max) 
                        self.surface_to_polar(filename = common_filename)

    def surface_to_polar(self, filename,output_suffix = 'polar'):
        """surface_to_polar takes a (smoothed) surface file for both real and imaginary parts and re-converts it to polar and eccentricity angle."""
        self.logger.info('converting %s from (smoothed) surface to nii back to surface')
        for hemi in ['lh','rh']:
            for component in ['real', 'imag']:
                svO = SurfToVolOperator(inputObject = filename + '_' + component + '-' + hemi + '.mgz' )
                svO.configure(templateFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['PRF'][0]], postFix = ['mcf']), hemispheres = [hemi], register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), fsSubject = self.subject.standardFSID, outputFileName = filename + '_' + component + '.nii.gz', threshold = 0.5, surfType = 'paint')
                print svO.runcmd
                svO.execute()
                # 
                
            # now, there's a pair of imag and real nii files for this hemisphere. Let's open them and make polar and eccen phases before re-transforming to surface. 
            complex_values = NiftiImage(filename + '_real-' + hemi + '.nii.gz').data + 1j * NiftiImage(filename + '_imag-' + hemi + '.nii.gz').data
        
            comp = NiftiImage(np.array([np.angle(complex_values), np.abs(complex_values)]))
            comp.header = NiftiImage(filename + '_real-' + hemi + '.nii.gz').header
            comp.save(filename + '_polecc-' + hemi + '.nii.gz')
        
        # add the two polecc files together
        addO = FSLMathsOperator(filename + '_polecc-' + 'lh' + '.nii.gz')
        addO.configureAdd(add_file = filename + '_polecc-' + 'rh' + '.nii.gz', outputFileName = filename + '_polecc.nii.gz')
        addO.execute()
        
        # self.results_to_surface(file_name = filename + '_polecc.nii.gz', output_file_name = filename, frames = , smooth = 0)
        vsO = VolToSurfOperator(inputObject = os.path.join(self.stageFolder('processed/mri/PRF/'), filename + '_polecc.nii.gz'))
        # ofn = os.path.join(self.stageFolder('processed/mri/PRF/surf/'), output_file_name )
        vsO.configure(frames = {'_polar':0, '_ecc':1}, hemispheres = None, register = self.runFile(stage = 'processed/mri/reg', base = 'register', postFix = [self.ID], extension = '.dat' ), outputFileName = filename + '_sm', threshold = 0.5, surfSmoothingFWHM = 0.0, surfType = 'paint'  )
        vsO.execute()
    
    def mask_stats_to_hdf(self, mask_file = 'cortex_dilated_mask_all',CV_mask_file = 'cortex_dilated_mask_all', 
        postFix = ['mcf','sgtf','res','psc'], task_conditions = ['Fix','all','color','sf','orient','speed'],
        model='OG',hrf_type='canonical',add_regular_fit_results=False,add_mapper_data=False,add_hrf_params=False):

        """
        Create an hdf5 file to populate with the stats and parameter estimates of the Mapper and PRF results
        """
        
        # determine masks:
        anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
        anatRoiFileNames = [anRF for anRF in anatRoiFileNames if np.all([np.any(['bh' in anRF,'lh' in anRF,'rh' in anRF]),'cortex' not in anRF])]
        self.logger.info('Taking masks ' + str(anatRoiFileNames))
        rois, roinames = [], []
        for roi in anatRoiFileNames:
            rois.append(NiftiImage(roi))
            roinames.append(os.path.split(roi)[1][:-7])

        # open hdf5 filename
        self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/mri/PRF'),  'PRF.hdf5')
        h5file = open_file(self.hdf5_filename, mode = "r+", title = 'PRF')
        self.logger.info('starting table file ' + self.hdf5_filename)

        ############################################
        ###### first add to subject specific file
        ############################################

        this_run_group_name = 'prf'
        try:
            thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
            self.logger.info('data file already in ' + self.hdf5_filename)
        except NoSuchNodeError:
            # import actual data
            self.logger.info('Adding group ' + this_run_group_name + ' to this file')
            thisRunGroup = h5file.create_group("/", this_run_group_name, '')
    

        stat_files = {}
        if add_regular_fit_results:
            for c in task_conditions:
                filename = '_'.join(postFix)  + '_' + model + '_' + c + '_' + 'hrf_'+ hrf_type
                for res_type in ['results', 'corrs']:# 'coefs',
                    # this adds regular results
                        stat_files.update({c+'_'+res_type: os.path.join(self.stageFolder('processed/mri/PRF'), res_type+ '_' +mask_file + '_' + filename + '.nii.gz')})        
        if add_hrf_params:  
            # add hrf parameters
            stat_files.update({'HRF_params': os.path.join(self.stageFolder('processed/mri/PRF/mean_hrf_parameters.nii.gz'))})
        if add_mapper_data:
            # add cope data
            for mapper_results in ['copes','varcopes','t_stat','z_stat']:
                stat_files.update({'Mapper_%s'%mapper_results: os.path.join(self.stageFolder('processed/mri/Mapper/GLM/%s.nii.gz'%mapper_results))})

        # load all files as niftifiles and add to hdf5
        stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
        for (roi, roi_name) in zip(rois, roinames):
            try:
                thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
            except NoSuchNodeError:
                # import actual data
                self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
                thisRunGroup = h5file.create_group("/" + this_run_group_name, roi_name, 'ROI ' + roi_name +' imported' )
        
            for (i, sf) in enumerate(stat_files.keys()):
                # loop over stat_files and rois
                # to mask the stat_files with the rois:
                imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
                these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
                try:
                    h5file.get_node(where='/'+this_run_group_name+'/'+roi_name,name=sf,classname='Array')
                    h5file.remove_node('/'+this_run_group_name+'/'+roi_name+'/'+sf)
                    h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
                except NoSuchNodeError:
                    h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])

        h5file.close()

        ############################################
        ###### then add to group file
        ############################################
        
        # create group dir if it doesn't yet exist
        group_dir = os.path.join(self.this_project_folder,'data','_group_level')
        if not os.path.isdir(group_dir): os.mkdir(group_dir)
        self.hdf5_group_filename = os.path.join(group_dir,'group_level.hdf5')
        
        # create group lvl hdf5 if it does not yet exist
        if os.path.isfile(self.hdf5_group_filename):
            group_h5file = open_file(self.hdf5_group_filename, mode = "a", title = 'PRF')
        else:
            group_h5file = open_file(self.hdf5_group_filename, mode = "w", title = 'PRF')   

        this_run_group_name = self.subject.initials

        try:
            thisRunGroup = group_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
            self.logger.info('data file already in ' + self.hdf5_filename)
        except NoSuchNodeError:
            # import actual data
            self.logger.info('Adding group ' + this_run_group_name + ' to this file')
            thisRunGroup = group_h5file.create_group("/", this_run_group_name, '')

        stat_files = {}
        if add_regular_fit_results:
            for c in task_conditions:
                for res_type in ['results', 'corrs']:# 'coefs',
                    filename = '_'.join(postFix)  + '_' + model + '_' + c + '_' + 'hrf_'+ hrf_type
                    stat_files.update({c+'_'+res_type: os.path.join(self.stageFolder('processed/mri/PRF'), res_type+ '_' +mask_file + '_' + filename + '.nii.gz')})
        if add_mapper_data:
            for mapper_results in ['copes','varcopes','t_stat','z_stat']:
                stat_files.update({'Mapper_%s'%mapper_results: os.path.join(self.stageFolder('processed/mri/Mapper/GLM/%s.nii.gz'%mapper_results))})
        if add_hrf_params:
            stat_files.update({'HRF_params': os.path.join(self.stageFolder('processed/mri/PRF/mean_hrf_parameters.nii.gz'))})

        stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
        for (roi, roi_name) in zip(rois, roinames):
            try:
                thisRunGroup = group_h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
            except NoSuchNodeError:
                # import actual data
                self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
                thisRunGroup = group_h5file.create_group("/" + this_run_group_name, roi_name, 'ROI ' + roi_name +' imported' )
        
            for (i, sf) in enumerate(stat_files.keys()):
                # loop over stat_files and rois
                # to mask the stat_files with the rois:
                imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
                these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
                try:
                    group_h5file.get_node(where='/'+this_run_group_name+'/'+roi_name,name=sf,classname='Array')
                    group_h5file.remove_node('/'+this_run_group_name+'/'+roi_name+'/'+sf)
                    group_h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
                except NoSuchNodeError:
                    group_h5file.create_array(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])

        group_h5file.close()

    def add_behavior_to_hdf5(self,conditions=['Color','Speed','Fix','Fix_no_stim']):
        """
        This function reads behavior and adds it to the hdf5 file.
        """

        condition_task_values = {
        'Color':0,
        'Speed':1,
        'Fix':2,
        'Fix_no_stim':3
        }
        staircase_values = {}
        staircase_times = {}
        response_values = {}
        response_times = {}
        run_delay = 0
        for ri, runi in enumerate(self.conditionDict['PRF']):
            staircase_values[ri] = {}
            staircase_times[ri] = {}        
            response_values[ri] = {}
            response_times[ri] = {} 

            filename = self.runFile(stage = 'processed/behavior', run = self.runList[runi],extension = '.dat' )
            ipf = open(filename)
            picklefile = pickle.load(ipf)
            ipf.close()
            
            run_start_time = [float(e.split(' ')[-1]) for e in picklefile['eventArray'][0]  if ('trial 0 phase 1 started at' in e)][0]

            for condition in conditions:
                staircase_values[ri][condition] = {}
                staircase_times[ri][condition] = {}             
                response_values[ri][condition] = {}
                response_times[ri][condition] = {}
                if condition != 'Fix_no_stim':
                    for eccen_bin in np.arange(3):
                        staircase_values[ri][condition][eccen_bin] = np.concatenate([np.array([float(e.split(' ')[-4]) for e in picklefile['eventArray'][i]  if ('signal in feature: %s ecc bin: %d'%(condition,eccen_bin) in e) * (picklefile['parameterArray'][i]['task_index'] == condition_task_values[condition])]) for i in range(len(picklefile['eventArray']))])
                        staircase_times[ri][condition][eccen_bin] = np.concatenate([np.array([float(e.split(' ')[-2])-run_start_time+run_delay for e in picklefile['eventArray'][i] if ('signal in feature: %s ecc bin: %d'%(condition,eccen_bin) in e) * (picklefile['parameterArray'][i]['task_index'] == condition_task_values[condition])]) for i in range(len(picklefile['eventArray']))])
                        response_values[ri][condition][eccen_bin] = np.concatenate([np.array([float(e.split(' ')[-3]) for e in picklefile['eventArray'][i]  if 'staircase %s bin %d'%(condition,eccen_bin) in e]) for i in range(len(picklefile['eventArray']))])
                        response_times[ri][condition][eccen_bin] = np.concatenate([np.array([float(e.split(' ')[-1])-run_start_time+run_delay for e in picklefile['eventArray'][i]  if 'staircase %s bin %d'%(condition,eccen_bin) in e]) for i in range(len(picklefile['eventArray']))])
                else:
                    staircase_values[ri][condition][0] = np.concatenate([np.array([float(e.split(' ')[-4]) for e in picklefile['eventArray'][i]  if ('signal in feature: %s'%(condition) in e) * (picklefile['parameterArray'][i]['task_index'] == condition_task_values[condition])]) for i in range(len(picklefile['eventArray']))])
                    staircase_times[ri][condition][0] = np.concatenate([np.array([float(e.split(' ')[-2])-run_start_time+run_delay for e in picklefile['eventArray'][i] if ('signal in feature: %s'%(condition) in e) * (picklefile['parameterArray'][i]['task_index'] == condition_task_values[condition])]) for i in range(len(picklefile['eventArray']))])
                    response_values[ri][condition][0] = np.concatenate([np.array([float(e.split(' ')[-3]) for e in picklefile['eventArray'][i]  if 'staircase %s'%(condition) in e]) for i in range(len(picklefile['eventArray']))])
                    response_times[ri][condition][0] = np.concatenate([np.array([float(e.split(' ')[-1])-run_start_time+run_delay for e in picklefile['eventArray'][i]  if 'staircase %s'%(condition) in e]) for i in range(len(picklefile['eventArray']))])

            run_end_time = [float(e.split(' ')[-1]) for e in picklefile['eventArray'][-1]  if ('trial 27 phase 4 started at' in e)][0]
            run_duration = run_end_time - run_start_time
            run_delay += run_duration

        # first add to subject specific file
        self.hdf5_filename = os.path.join(self.stageFolder(stage = 'processed/behavior/PRF'),  'PRF.hdf5')

        if os.path.isfile(self.hdf5_filename):
            h5file = open_file(self.hdf5_filename, mode = "a", title = 'PRF')
            self.logger.info('adding behavior to hdf5 ' + self.hdf5_filename)
        else:
            h5file = open_file(self.hdf5_filename, mode = "w", title = 'PRF')   
            self.logger.info('creating new hdf5 ' + self.hdf5_filename)

        this_run_group_name = 'prf'
        try:
            thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
            self.logger.info('data file already in ' + self.hdf5_filename)
        except NoSuchNodeError:
            # import actual data
            self.logger.info('Adding group ' + this_run_group_name + ' to this file')
            thisRunGroup = h5file.create_group("/", this_run_group_name, '')
        
        for condition in conditions:
            try:
                thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = condition, classname='Group')
            except NoSuchNodeError:
                self.logger.info('Adding group ' + this_run_group_name + '_' + condition + ' to this file')
                thisRunGroup = h5file.create_group("/" + this_run_group_name, condition)
            
            if condition != 'Fix_no_stim':
                eccen_bins = range(3)
            else:
                eccen_bins = [0]

            for eccen_bin in eccen_bins:    

                for runi in range(len(self.conditionDict['PRF'])):
                    try:
                        # h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_responses'+sub_condition + '_run_' + str(runi),name=sub_condition,classname='Array')
                        h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_response_values_'+str(eccen_bin) + '_run_' + str(runi))
                        h5file.create_array(thisRunGroup, condition+'_response_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response values')
                    except NoSuchNodeError:
                        h5file.create_array(thisRunGroup, condition+'_response_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response values')

                    try:
                        # h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_responses'+eccen_bin + '_run_' + str(runi),name=eccen_bin,classname='Array')
                        h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_response_times_'+str(str(eccen_bin)) + '_run_' + str(runi))
                        h5file.create_array(thisRunGroup, condition+'_response_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response times')
                    except NoSuchNodeError:
                        h5file.create_array(thisRunGroup, condition+'_response_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response times')
                
                    try:
                        # h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_responses'+eccen_bin + '_run_' + str(runi),name=eccen_bin,classname='Array')
                        h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_staircase_values_'+str(eccen_bin) + '_run_' + str(runi))
                        h5file.create_array(thisRunGroup, condition+'_staircase_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase values')
                    except NoSuchNodeError:
                        h5file.create_array(thisRunGroup, condition+'_staircase_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase values')
                                        
                    try:
                        # h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_staircases'+eccen_bin + '_run_' + str(runi),name=eccen_bin,classname='Array')
                        h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_staircase_times_'+str(eccen_bin) + '_run_' + str(runi))
                        h5file.create_array(thisRunGroup, condition+'_staircase_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase times')
                    except NoSuchNodeError:
                        h5file.create_array(thisRunGroup, condition+'_staircase_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase times')
                                        
        h5file.close()

        # create group dir if it doesn't yet exist
        group_dir = os.path.join(self.this_project_folder,'data','_group_level')
        if not os.path.isdir(group_dir): os.mkdir(group_dir)
        self.hdf5_group_filename = os.path.join(group_dir,'group_level.hdf5')

        # create group lvl hdf5 if it does not yet exist
        if os.path.isfile(self.hdf5_group_filename):
            group_h5file = open_file(self.hdf5_group_filename, mode = "a", title = 'PRF')
        else:
            group_h5file = open_file(self.hdf5_group_filename, mode = "w", title = 'PRF')   

        this_run_group_name = self.subject.initials

        try:
            thisRunGroup = group_h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
            self.logger.info('data file already in ' + self.hdf5_filename)
        except NoSuchNodeError:
            # import actual data
            self.logger.info('Adding group ' + this_run_group_name + ' to this file')
            thisRunGroup = group_h5file.create_group("/", this_run_group_name, '')
        
        for condition in conditions:
            
            try:
                thisRunGroup = group_h5file.get_node(where = "/" + this_run_group_name, name = condition, classname='Group')
            except NoSuchNodeError:
                self.logger.info('Adding group ' + this_run_group_name + '_' + condition + ' to this file')
                thisRunGroup = group_h5file.create_group("/" + this_run_group_name, condition)
            
            if condition != 'Fix_no_stim':
                eccen_bins = range(3)
            else:
                eccen_bins = [0]

            for eccen_bin in eccen_bins:    

                for runi in range(len(self.conditionDict['PRF'])):
                    try:
                        # h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_responses'+sub_condition + '_run_' + str(runi),name=sub_condition,classname='Array')
                        group_h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_response_values_'+str(eccen_bin) + '_run_' + str(runi))
                        group_h5file.create_array(thisRunGroup, condition+'_response_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response values')
                    except NoSuchNodeError:
                        group_h5file.create_array(thisRunGroup, condition+'_response_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response values')

                    try:
                        # group_h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_responses'+eccen_bin + '_run_' + str(runi),name=eccen_bin,classname='Array')
                        group_h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_response_times_'+str(eccen_bin) + '_run_' + str(runi))
                        group_h5file.create_array(thisRunGroup, condition+'_response_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response times')
                    except NoSuchNodeError:
                        group_h5file.create_array(thisRunGroup, condition+'_response_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(response_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' response times')
                
                    try:
                        # group_h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_responses'+eccen_bin + '_run_' + str(runi),name=eccen_bin,classname='Array')
                        group_h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_staircase_values_'+str(eccen_bin) + '_run_' + str(runi))
                        group_h5file.create_array(thisRunGroup, condition+'_staircase_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase values')
                    except NoSuchNodeError:
                        group_h5file.create_array(thisRunGroup, condition+'_staircase_values_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_values[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase values')
                                        
                    try:
                        # group_h5file.get_node(where='/'+this_run_group_name+'/'+condition+'_staircases'+eccen_bin + '_run_' + str(runi),name=eccen_bin,classname='Array')
                        group_h5file.remove_node('/'+this_run_group_name+'/'+condition+'/'+condition+'_staircase_times_'+str(eccen_bin) + '_run_' + str(runi))
                        group_h5file.create_array(thisRunGroup, condition+'_staircase_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase times')
                    except NoSuchNodeError:
                        group_h5file.create_array(thisRunGroup, condition+'_staircase_times_'+str(eccen_bin) + '_run_' + str(runi), np.array(staircase_times[runi][condition][eccen_bin]).astype(np.float32), condition + ' ecc bin ' + str(eccen_bin) + ' run ' + str(runi)+ '_' + ' staircase times')
                                    
        group_h5file.close()


