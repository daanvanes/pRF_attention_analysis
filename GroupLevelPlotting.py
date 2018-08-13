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
sn.set(style="white")
import mne
import pickle
from utilities import CustomStatUtilities
from statsmodels.stats.weightstats import DescrStatsW
import scipy.stats as stats
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, report_errors
import time as t
from decimal import Decimal
import pandas as pd

# import toolbox funcitonality
import os, sys
sys.path.append( os.environ['ANALYSIS_HOME'] )
from Tools.Sessions.Session import *
from AFSimulations import *
mpl.rc_file_defaults()

#####################################
#### FUNCTIONS FOR PARALLELISATIONS
#####################################

def AF_fit_residual(params,AF_shape,fit_method,res,stim_radius,these_results,dm_dir,output_results,n_jobs,
    model_area_increase_factor,circle_mask,center_method,upsample_factor,mask_ecc_thresholds,
    detect_outliers,outlier_num_stds,n_eccen_bins,n_polar_bins,rectify,overlap):
    
    # initiate functions instance
    functions = General_functions()

    # devy up AF_size_combo
    if fit_method == 'fit':
        fix_AF_size, AF_ecc_size_intercept, AF_ecc_size_slope = params['fix_AF_size'].value, params['bar_AF_intercept'].value, params['bar_AF_slope'].value 
        AF_surround_amp,AF_surround_ratio = params['AF_surround_amp'].value, params['AF_surround_ratio'].value
    elif fit_method == 'grid':
        fix_AF_size, AF_ecc_size_intercept, AF_ecc_size_slope,AF_surround_ratio,AF_surround_amp = params

    # create the AFs:
    if AF_shape == 'bar_convolved':
        import socket
        if socket.gethostname() == 'dhcp-182-204.clients.vu.nl':
            dm_dir = os.path.join('/home','shared','PRF_2','AF_simulations')
        else:
            dm_dir = os.path.join('/projects','0','pqsh283','PRF_2','AF_simulations')

        all_AFs,dms = convolve_bar_with_AF(AF_ecc_size_intercept,AF_ecc_size_slope,model_area_increase_factor,res,dm_dir,stim_radius,circle_mask,AF_surround_amp,AF_surround_ratio) 
    elif AF_shape == 'anisotropic':
        
        all_AFs = anisotropic_AF(AF_ecc_size_intercept,AF_ecc_size_slope,model_area_increase_factor,res,dm_dir,stim_radius,circle_mask) 

    results = []
    for size,x,y in zip(these_results['cond_1_sizes'],these_results['cond_1_xo'],these_results['cond_1_yo']):
        results.append(run_AF_bar_model(x,y,size,fix_AF_size,all_AFs,stim_radius,res,model_area_increase_factor,
            circle_mask,upsample_factor,dms))

    results = np.array(results)

    these_results['fix_RFx'],these_results['fix_RFy'],these_results['bar_RFx'],these_results['bar_RFy'],\
    these_results['SD_RFx'],these_results['SD_RFy'], = results[:,0],results[:,1],results[:,2],results[:,3],results[:,4],results[:,5]

    return these_results

class General_functions(object):

    def __init__(self,reps=int(1e3)):
        self.CUO = CustomStatUtilities(reps=reps)    

    def two_d_gauss(self,xo, yo, sigma_x, sigma_y=None, amplitude = 1, theta=0,res=101,max_eccentricity=1):
        
        """
        Input:
        - xo: float
        - yo: float
        - sigma_x: float
        - sigma_y: float
        - amplitude: float
        - theta: float in degrees
        - res: int 

        Output:
        - 2D array

        This function positions a 2D-Gaussian at [xo,yo] (where the units are defined
        by max_eccentricity), with defined amplitude and sigma_x. 
        If sigma_y is not given, it will produce an isotropic Gaussian.
        Res sets the resolution i.e. the amount of pixels in 
        the 2D output array. 
        """

        X = np.linspace(-max_eccentricity, max_eccentricity, res)
        Y = np.linspace(-max_eccentricity, max_eccentricity, res)
        MG = np.meshgrid(X, Y)

        (x,y) = MG    
        if sigma_y == None:
            sigma_y = sigma_x
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        gauss = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        # gauss[disk((res-1)/2)==0] = 0
        if np.isnan(gauss).sum()>0:
            gauss = np.zeros((res,res))
        return gauss

    def one_d_gauss(self,xo,sigma,amplitude=1,res=101,max_eccentricity=1):
        """
        Input:
        - xo: float
        - sigma: float
        - amplitude: float
        - res: int
        - max_eccentricity: float

        Outpu:
        - 1D array of gaussian

        This function returns a 1-D gaussian function at desired location and width,
        amplitude and resolution. 
        xo and sigma are defined in terms of max_eccentricity 
        """

        x = np.linspace(0, max_eccentricity, res)
        gauss = amplitude*np.exp(-np.power(x - xo, 2.) / (2 * np.power(sigma, 2.)))

        return gauss

    def compute_FBA(self,these_data,weights,measures,outlier_num_stds):#,measure_type,diff_type = 'per_voxel_proj'):


        # first setup differences with attend fixation
        for condition1 in ['Stim','Color','Speed']:
            for condition2 in ['Fix']:
                these_data['%s-%s'%(condition1,condition2)] = {}
                for measure in measures:
                    these_data['%s-%s'%(condition1,condition2)][measure] = (these_data[condition1][measure] - these_data[condition2][measure])


        # now compute the standard deviation of ecc and size differences
        # based on the attend stimulus condition, so that ecc and size
        # differences are equalized (i.e. contribute equally to feature AMI),
        # but that differences between color and speed conditions are not 
        # filtered out
        for measure in measures:
            # compute z-scores based on Stim-Fix differences
            temp = these_data['Stim-Fix'][measure]
            temp_inliers = self.CUO.detect_inliers_mad(temp,outlier_num_stds)
            z_score = DescrStatsW(temp[temp_inliers], weights=weights[temp_inliers]).std
            # and apply to all condition differences
            for condition in ['Stim','Color','Speed']:
                    these_data['%s-Fix'%(condition)][measure] /= z_score
        
        # setup a vector with ecc and size differences
        diff_vectors = {}
        for condition in ['Stim','Color','Speed']:
                diff_vectors['%s-Fix'%(condition)] = []
                for measure in measures:
                    temp = these_data['%s-Fix'%(condition)][measure]
                    diff_vectors['%s-Fix'%(condition)].append(temp)

        # now compute the attentional modulation index as the 
        # length of these vectors:
        stim = np.linalg.norm(diff_vectors['Stim-Fix'],axis=0)
        col = np.linalg.norm(diff_vectors['Color-Fix'],axis=0)
        speed = np.linalg.norm(diff_vectors['Speed-Fix'],axis=0)

        # now compute fami as a contrast:
        fami = (col-speed)/(col+speed)

        return col,speed,fami


    def find_yticks(self,ylims,middle=0,ndec=None):
       
        # now see how many decs we have
        decs = np.array([len(str(this_tick).split('.')[-1]) for this_tick in ylims])
        if ndec == None:
            ndec = np.max(decs)

        # round the ticks:
        yticks = np.ceil(np.abs(ylims)*10**ndec)/10**ndec*np.sign(ylims)
        
        # now see how many decs we have after rounding
        decs = np.array([len(str(this_tick).split('.')[-1]) for this_tick in yticks])
        if ndec == None:
            ndec = np.max(decs)

        # add trailing 0s:
        new_ticks =[]
        for yi, ytick in enumerate(yticks):
            if ytick != 0:
                new_ticks.append(str(ytick)+'0'*(ndec-decs[yi]))
            else:
                new_ticks.append('0')
        yticks = new_ticks

        # remove leading 0:
        if np.all(np.abs(np.array(yticks).astype(float))<1):
            new_ticks = []
            for this_tick in yticks:
                if float(this_tick) == 0:
                    new_ticks.append('0')
                if (float(this_tick)>0)*(float(this_tick)<1):
                    new_ticks.append('.%s'%this_tick.split('.')[1])
                if (float(this_tick)<0)*(float(this_tick)>-1):
                    new_ticks.append('-.%s'%this_tick.split('.')[1])    
        else:
            new_ticks = yticks  

        # add middle 
        if (float(new_ticks[0]) < middle) * (float(new_ticks[1]) > middle):
            new_yticks = [new_ticks[0],str(middle),new_ticks[1]]
        else:
            new_yticks = new_ticks

        # now check how many numbers there are in each tick
        # max_nints = np.max([len(this_tick.strip('-').strip('.')) for this_tick in new_yticks])

        return np.array(new_yticks).astype(float), new_yticks

    def percentile_bins(self,y_data,x_data,weights=None,n_bins=10,outlier_num_stds=3,ci_factor=1.96,detect_inliers=True):

        """
        Input:
        - y_data: array
        - x_data: array
        - weights: array

        Output:
        - binned y means: array
        - binned cis: array
        - binned x means: array

        This function determines 100/n_bins percentile bins based on x_data.
        It then tries to calculate weighted averages and confidence intervals
        (where ci = weighted_std/np.sqrt(len(data))*outlier_num_stds).
        If this fails (because weights sum to zero for instance), a regular average
        and ci is calculated.
        """

        if weights is None:
            weights = np.ones_like(x_data)

        # first do outlier detection
        if detect_inliers:
            y_inliers = self.CUO.detect_inliers_mad(y_data,outlier_num_stds)
            x_inliers = self.CUO.detect_inliers_mad(x_data,outlier_num_stds)
            inliers = y_inliers*x_inliers
            # now apply
            y_data = y_data[inliers]
            x_data = x_data[inliers]
            weights = weights[inliers]

        # first, we'll sort the data based on the x_data
        x_order = np.argsort(x_data)
        y_data = y_data[x_order]
        x_data = x_data[x_order]
        weights = weights[x_order]

        mean_y_data = []
        se_y_data = []
        se_x_data = []
        mean_x_data = [] 
        p_y_data=[]
        ds = []
        Ns=[]

        bin_width = int(len(y_data)/n_bins)
        for bi in range(n_bins):    

            these_voxels = np.zeros(len(x_data)).astype(bool)
            these_voxels[bi * bin_width:(bi+1) * bin_width] = True
            
            mean_y, ci_y, p_y,Ny,dy = self.CUO.bootstrap(y_data[these_voxels],weights=weights[these_voxels],ci_factor=ci_factor,outlier_num_stds=outlier_num_stds,detect_inliers=False,return_d=True)
            mean_x, ci_x, p_x,Nx,dx = self.CUO.bootstrap(x_data[these_voxels],weights=weights[these_voxels],ci_factor=ci_factor,outlier_num_stds=outlier_num_stds,detect_inliers=False,return_d=True)

            mean_y_data.append(mean_y)
            mean_x_data.append(mean_x)
            p_y_data.append(p_y)
            se_y_data.append(ci_y)
            se_x_data.append(ci_x)
            ds.append(dy)
            Ns.append(Ny)

        return np.array(mean_x_data), np.array(se_x_data), np.array(mean_y_data), np.array(se_y_data), np.array(p_y_data),np.array(Ns),np.array(ds)

    def fixed_bins(self,y_data,x_data,weights=None,n_bins=10,ci_factor=1.96,ecc_thresholds=[0,3.6],outlier_num_stds=3,detect_inliers=True,return_d=False):


        """
        Input:
        - y_data: array
        - x_data: array
        - weights: array

        Output:
        - binned y means: array
        - binned cis: array
        - binned x means: array

        This function determines 100/n_bins percentile bins based on x_data.
        It then tries to calculate weighted averages and confidence intervals
        (where ci = weighted_std/np.sqrt(len(data))*outlier_num_stds).
        If this fails (because weights sum to zero for instance), a regular average
        and ci is calculated.
        """

        # 
        if weights is None:
            weights = np.ones_like(x_data)

        mean_y_data = []
        mean_x_data = []
        se_y_data = []
        se_x_data = []
        p_y_data = []
        Ns=[]
        ds=[]

        bin_edges = np.linspace(ecc_thresholds[0],ecc_thresholds[1],n_bins+1)
        for bi in range(n_bins):    

            these_voxels = (x_data > bin_edges[bi]) * (x_data < bin_edges[bi+1])

            if return_d:
                mean_y, se_y, p_y,N,d = self.CUO.bootstrap(y_data[these_voxels],weights=weights[these_voxels],ci_factor=ci_factor,outlier_num_stds=outlier_num_stds,detect_inliers=detect_inliers,return_d=True)
                ds.append(d)
            else:
                mean_y, se_y, p_y,N = self.CUO.bootstrap(y_data[these_voxels],weights=weights[these_voxels],ci_factor=ci_factor,outlier_num_stds=outlier_num_stds,detect_inliers=detect_inliers)
            mean_x, ci_x, p_x,N = self.CUO.bootstrap(x_data[these_voxels],weights=weights[these_voxels],ci_factor=ci_factor,outlier_num_stds=outlier_num_stds,detect_inliers=detect_inliers)                
            
            mean_y_data.append(mean_y)
            mean_x_data.append(mean_x)
            p_y_data.append(p_y)
            se_y_data.append(se_y)
            se_x_data.append(ci_x)
            Ns.append(np.sum(these_voxels))

        if return_d:
            return np.array(mean_x_data), np.array(se_x_data), np.array(mean_y_data), np.array(se_y_data), np.array(p_y_data), np.array(Ns), np.array(ds)
        else:
            return np.array(mean_x_data), np.array(se_x_data), np.array(mean_y_data), np.array(se_y_data), np.array(p_y_data), np.array(Ns)

    def create_mask(self,ecc,ecc_thresholds,r_squared_threshold,weights=0,sizes=0,outlier_num_stds=3,size_threshold=4):
        """
        Input:
        - ecc: array
        - weights: array

        Output:
        - mask: array

        This function recieves both eccentricities and weights and determines
        which values lie between the ecc_thresholds and above the 
        r_squared_threshold.
        """

        mask = np.ones_like(ecc).astype(bool)
        mask *= (ecc > ecc_thresholds[0])
        mask *= (ecc < ecc_thresholds[1])
        if type(weights) != int:
            mask *= (weights > r_squared_threshold)
        # # if we have sizes, only take those sizes within the
        # if type(sizes) != int:
        #   size_threshold = self.weighted_avg(sizes[mask],weights=weights[mask]) + outlier_num_stds*self.weighted_std(sizes[mask],weights=weights[mask])
        mask *= (sizes < size_threshold)

        return mask

    def create_mask_2_conditions(self,ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights,ecc_thresholds,
        r_squared_threshold,outlier_num_stds,mask_type='cond_1',size_threshold=4):
        """
        Input:
        - ecc: array
        - weights: array

        Output:
        - mask: array

        This function recieves both eccentricities and weights and determines
        which values lie between the ecc_thresholds and above the 
        r_squared_threshold.
        """

        ecc_mask_cond_0 = (ecc_cond_0 < 1e100)
        ecc_mask_cond_1 = (ecc_cond_1 > ecc_thresholds[0]) * (ecc_cond_1 < ecc_thresholds[1])

        weights_mask = (weights > r_squared_threshold)
        
        # now apply the size thresholds:
        size_mask_cond_0 = (sizes_cond_0 < size_threshold)
        size_mask_cond_1 = (sizes_cond_1 < size_threshold)

        # now add the size masks 
        mask =  ecc_mask_cond_1 * weights_mask * size_mask_cond_1 * size_mask_cond_0

        return mask

    def roi_data_from_hdf(self,h5file, run = '', roi_wildcard = 'v1', data_type = 'tf_psc_data',):
        """
        drags data from an already opened hdf file into a numpy array, 
        concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
        """

        this_run_group_name = run

        try:
            thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
            
            roi_names = []
            for roi_name in h5file.iter_nodes(where = '/' + this_run_group_name, classname = 'Group'):
                if len(roi_name._v_name.split('.')) > 1:
                    hemi, area = roi_name._v_name.split('.')
                    if roi_wildcard == area:
                        roi_names.append(roi_name._v_name)
                if roi_wildcard == roi_name._v_name:
                    roi_names.append(roi_name._v_name)
            if len(roi_names) == 0:
                return None
        except:
            # import actual data
            return None
        
        all_roi_data = []
        for roi_name in roi_names:
            thisRoi = h5file.get_node(where = '/' + this_run_group_name, name = roi_name, classname='Group')
            all_roi_data.append( eval('thisRoi.' + data_type + '.read()') )
        all_roi_data_np = np.hstack(all_roi_data).T

        return all_roi_data_np

    def get_arrows(self,field,n_eccen_bins,ecc_thresholds,n_polar_bins,
        cond_0_data,cond_1_data,weights,detect_outliers,outlier_num_stds,location='cond_1',min_vox_per_arrow=0):

        # create the bins, either fixed or variable (percentiles)
        ecc_boundaries = np.linspace(ecc_thresholds[0],ecc_thresholds[1],n_eccen_bins+1)
        ecc_bins = [[ecc_boundaries[bi],ecc_boundaries[bi+1]] for bi in range(len(ecc_boundaries)-1)]
        if field == 'quadrant':
            polar_boundaries = np.linspace(0,np.pi/2,n_polar_bins+1)
        elif field == 'whole_field':
            polar_boundaries = np.linspace(-np.pi,np.pi,n_polar_bins+1) 
        polar_bins = [[polar_boundaries[bi],polar_boundaries[bi+1]] for bi in range(len(polar_boundaries)-1)]

        # get polar and ecc values for bins
        cond_1_polar = np.arctan2(cond_1_data[1],cond_1_data[0])
        cond_1_eccen = np.linalg.norm(cond_1_data,axis=0)
        
        CUO = CustomStatUtilities() 
        
        # get the arrow data
        arrow_starts = []; arrow_diffs = []
        for this_ecc_bin in ecc_bins:
            for this_polar_bin in polar_bins:
                # check which voxels fall within both the eccen and polar bin range based on the bin_data we 
                these_voxels = ((cond_1_eccen > this_ecc_bin[0]) * (cond_1_eccen < this_ecc_bin[1]) * 
                                (cond_1_polar > this_polar_bin[0]) * (cond_1_polar < this_polar_bin[1]))

                # now get the data
                this_cond_1_data = cond_1_data[:,these_voxels]
                this_cond_0_data = cond_0_data[:,these_voxels]
                # compute the length of the difference vectors
                all_diffs = np.squeeze(np.linalg.norm([this_cond_0_data - this_cond_1_data],axis=1))        

                # exclude the outlier diffs
                if detect_outliers * (these_voxels.sum()>1):
                    inlier_diffs = CUO.detect_inliers_mad(all_diffs,outlier_num_stds)  
                    n_inliers = np.sum(inlier_diffs)    
                    these_cond_0_data = this_cond_0_data[:,inlier_diffs]
                    these_cond_1_data = this_cond_1_data[:,inlier_diffs]
                    these_weights = weights[these_voxels][inlier_diffs]
                else:
                    these_cond_0_data = this_cond_0_data
                    these_cond_1_data = this_cond_1_data
                    these_weights = weights[these_voxels]
                    n_inliers = these_voxels.sum()

                if n_inliers >= min_vox_per_arrow:
                    # and compute the average difference
                    avg_diff = np.average(these_cond_0_data-these_cond_1_data,weights=these_weights,axis=1)
                    arrow_diffs.append(avg_diff)
                    # then determine where we want the arrows to start
                    if location == 'centre_of_bin':
                        arrow_starts.append([np.mean(this_ecc_bin) * np.cos(np.mean(this_polar_bin)),
                            np.mean(this_ecc_bin) * np.sin(np.mean(this_polar_bin))])
                    elif location == 'cond_1':
                        arrow_starts.append(np.average(these_cond_1_data,weights=these_weights,axis=1))
                else:
                    arrow_starts.append([np.nan,np.nan])
                    arrow_diffs.append([np.nan,np.nan])


        arrow_starts = np.array(arrow_starts)
        arrow_diffs = np.array(arrow_diffs)        
        # now detect outlier arrows
        if detect_outliers:
            arrow_lengths = np.linalg.norm(arrow_diffs,axis=1)
            valid_arrows = CUO.detect_inliers_mad(arrow_lengths,outlier_num_stds)
            # convert to numpy arrays
            arrow_diffs[~valid_arrows] = [np.nan,np.nan]
            arrow_starts[~valid_arrows] = [np.nan,np.nan]

        return arrow_starts, arrow_diffs

    def bootstrap_bin_diff(self,x_data,y_data,y_data2,weights,n_bins,bin_range,
        outlier_num_stds,ci_factor,detect_inliers,reps,test_value=0,two_tailed=True,stat_type='t'):

        bin_edges = np.linspace(bin_range[0],bin_range[1],n_bins+1)
        bin0 = (x_data>bin_edges[0])*(x_data<bin_edges[1])
        bin1 = (x_data>bin_edges[-2])*(x_data<bin_edges[-1])

        # slope of the x or y change over polar angle
        y1 = np.average(y_data[bin0],weights=weights[bin0])
        y2 = np.average(y_data[bin1],weights=weights[bin1])
        center1 = y2-y1

        # slope of the ecc change over polar angle
        y1 = np.average(y_data2[bin0],weights=weights[bin0])
        y2 = np.average(y_data2[bin1],weights=weights[bin1])
        center2 = y2-y1   

        # now see whether the slope in x/y over polar angle is greater then ecc over polar angle
        center = center1 - center2

        # compute a cohen_d for that
        def weighted_cohend(y1,y2,w1,w2):


            ## from https://en.wikipedia.org/wiki/Effect_size#Cohen.27s_d

            n1 = len(y1)
            n2 = len(y2)

            # weighted means of both measures
            m1 = DescrStatsW(y1,weights=w1).mean
            m2 = DescrStatsW(y2,weights=w2).mean

            # estimate weighted variances of both measures
            s1 = (1/(n1-1)) * DescrStatsW(y1,weights=w1).sumsquares
            s2 = (1/(n2-1)) * DescrStatsW(y2,weights=w2).sumsquares

            # and pooled variance
            num = (n1-1)*s1 + (n2-1)*s2 
            denom = n1+n2-2
            s = np.sqrt(num/denom)
            
            cohen_d = (m2-m1)/s

            return cohen_d

        y1 = y_data[bin0] - y_data2[bin0]
        y2 = y_data[bin1] - y_data2[bin1]
        w1 = weights[bin0]
        w2 = weights[bin1]
        cohen_d = weighted_cohend(y1,y2,w1,w2)

        # and varinance through bootstrapping
        N = len(x_data)

        if stat_type == 't':
            def t_welch(x, y, tails=2):
                """Welch's t-test for two unequal-size samples, not assuming equal variances
                """
                # try:
                assert tails in (1,2), "invalid: tails must be 1 or 2, found %s"%str(tails)
                x, y = np.asarray(x), np.asarray(y)
                nx, ny = x.size, y.size
                vx, vy = x.var(), y.var()
                df = int((vx/nx + vy/ny)**2 / # Welch-Satterthwaite equation
                    ((vx/nx)**2 / (nx - 1) + (vy/ny)**2 / (ny - 1)))
                t_obs = (x.mean() - y.mean()) / np.sqrt(vx/nx + vy/ny)
                p_value = tails * sp.stats.t.sf(abs(t_obs), df)
                return t_obs,p_value
                # except:
                    # return 0,1
            t,p=t_welch(y_data[bin0],y_data[bin1])

        # now for estimate of variance:
        permute_indices = np.random.randint(0, len(x_data), size = (len(x_data), int(reps))).T
        bootstrap_distr = []
        for perm in permute_indices:
            bin0 = (x_data[perm]>bin_edges[0])*(x_data[perm]<bin_edges[1])
            bin1 = (x_data[perm]>bin_edges[-2])*(x_data[perm]<bin_edges[-1])
            if (np.sum(bin0) >0) * (np.sum(bin1)>0):

                # this is the slope of the x or y diff over polar angle:
                y1 = np.average(y_data[perm][bin0],weights=weights[perm][bin0])
                y2 = np.average(y_data[perm][bin1],weights=weights[perm][bin1])
                slope1 = y2-y1

                # this is the slope of the ecc diff over polar angle:
                y1 = np.average(y_data2[perm][bin0],weights=weights[perm][bin0])
                y2 = np.average(y_data2[perm][bin1],weights=weights[perm][bin1])
                slope2 = y2-y1               

                # what were after is whether the slope in x/y is greater than ecc 
                bootstrap_distr.append(slope1-slope2)

        ci = self.CUO.get_ci(bootstrap_distr,ci_factor)

        if stat_type == 'b':
            p = self.CUO.p_val_from_bootstrap_dist(bootstrap_distr,test_value,two_tailed)

        return center,ci, p, N,cohen_d




class GroupLevelPlots(object):
    
    def __init__(self,subjects,mask_ecc_thresholds,plot_ecc_thresholds,r_squared_threshold,stim_radius,outlier_num_stds,
        rois,ci_factor,rois_for_plot,results_frames,stats_frames,group_dir,roi_subplot_grid,
        roi_colors,roi_groups_for_plot,roi_group_subplot_grid,mask_type,rescale_factor,size_threshold,comparison_colors,condition_colors,detect_outliers,reps):

        self.subjects = subjects
        self.plot_ecc_thresholds = plot_ecc_thresholds
        self.mask_ecc_thresholds = mask_ecc_thresholds
        self.r_squared_threshold = r_squared_threshold
        self.stim_radius = stim_radius
        self.outlier_num_stds = outlier_num_stds
        self.rois = rois
        self.ci_factor = ci_factor
        self.rois_for_plot = rois_for_plot
        self.results_frames = results_frames
        self.stats_frames = stats_frames
        self.roi_subplot_grid = roi_subplot_grid
        self.roi_colors = roi_colors
        self.group_dir = group_dir
        self.roi_groups_for_plot = roi_groups_for_plot
        self.roi_group_subplot_grid = roi_group_subplot_grid
        self.mask_type = mask_type
        self.rescale_factor = rescale_factor
        self.size_threshold = size_threshold
        self.comparison_colors = comparison_colors
        self.condition_colors = condition_colors
        self.detect_outliers = detect_outliers
        self.reps=reps
        self.group_plot_dir = os.path.join(self.group_dir,'plots')
        self.bootstrap_reps = int(1e4)

        ## initiate general functions object
        self.functions = General_functions(reps=reps)
        self.CUO = CustomStatUtilities(reps=reps)    

    ########################
    #### CREATE PLOT DIR
    ########################

    def create_plot_dir(self,plot_type):
        """Creates an empty dir at self.group_dir/plot_type"""
        if not os.path.isdir(self.group_plot_dir): 
            os.mkdir(self.group_plot_dir)
        this_plot_dir = os.path.join(self.group_plot_dir,plot_type)
        if not os.path.isdir(this_plot_dir):
            os.mkdir(this_plot_dir)

    ############################
    ####### LOAD DATA
    ############################

    def load_data(self,behavior=False,PRF=False,Mapper=False,timecourses=False,PRF_CV=False,predictions=False,conditions=['All','Fix','Color','Speed','Stim'],
        subjects=['NA','JS','JW','TK','DE'],bootstrap_super_subject=False,permute_indices=[],HRF_params=False,load_hemispheres=False):
            
        """
        Input
        - PRF: bool 
        - Mapper: bool
        - timecourses: bool
        - conditions: list of strings
        - subjects: list of strings

        Output
        - All of the output variables will saved under the self object and will be formatted as nested dictionaries, 
            which an be accessed like: variable[subject][condition][roi]

        This function uses roi_data_from_hdf to pull data from the group hdf5.
        If PRF is True, it pulls the PRF parameters and stats and creates a self.all_results
        and self.all_stats variable to store them in. 
        if Mapper is True, it pulls the Mapper cope betas, and creates a self.all_mapper variable.
        if timecourses is True, it also pulls the averaged timecourses and puts them in self.all_timecourses

        If conditions is not passed, it will default to loading in all conditions.

        If subjects is not passed, it will default to all subjects.

        The 'super_subject' is created by pooling all data from all subjects passed to the function. 
        """


        if load_hemispheres:
            for roi in self.rois.keys():
                self.rois.update({'lh.%s'%roi:['lh.%s'%subroi for subroi in self.rois[roi]]})
                self.rois.update({'rh.%s'%roi:['rh.%s'%subroi for subroi in self.rois[roi]]})


        # open the hdf5 file
        hdf5_group_filename = os.path.join(self.group_dir,'group_level.hdf5')
        h5file = open_file(hdf5_group_filename, mode = "r", title = 'group_level')


        if behavior:
            self.all_staircase_values = {}
            self.all_staircase_times = {}
            self.all_behavior_values = {}
            self.all_behavior_times = {}
            for subject in subjects:

                print 'loading behavior results from hdf5 for subject %s...'%(subject)
                self.all_behavior_values[subject] = {}
                self.all_behavior_times[subject] = {}
                self.all_staircase_values[subject] = {}
                self.all_staircase_times[subject] = {}
                for this_condition in ['Color','Fix','Speed','Fix_no_stim']:
                    self.all_behavior_times[subject][this_condition] = {}
                    self.all_behavior_values[subject][this_condition] = {}
                    self.all_staircase_times[subject][this_condition] = {}
                    self.all_staircase_values[subject][this_condition] = {}
                    # see how many runs we have for this subject
                    if not this_condition == 'Fix_no_stim':
                        eccen_bins = range(3)
                    else:
                        eccen_bins = [0]
                    for eccen_bin in eccen_bins:
                        self.all_behavior_times[subject][this_condition][eccen_bin] = {}
                        self.all_behavior_values[subject][this_condition][eccen_bin] = {}
                        self.all_staircase_times[subject][this_condition][eccen_bin] = {}
                        self.all_staircase_values[subject][this_condition][eccen_bin] = {}
                        for runi in range(10):
                            try:
                                self.all_behavior_times[subject][this_condition][eccen_bin][runi] = h5file.get_node(where='/'+subject+'/'+this_condition,name=this_condition+'_response_times_'+str(eccen_bin) + '_run_' + str(runi),classname='Array').read()
                                self.all_behavior_values[subject][this_condition][eccen_bin][runi] = h5file.get_node(where='/'+subject+'/'+this_condition,name=this_condition+'_response_values_'+str(eccen_bin) + '_run_' + str(runi),classname='Array').read()
                                self.all_staircase_times[subject][this_condition][eccen_bin][runi] = h5file.get_node(where='/'+subject+'/'+this_condition,name=this_condition+'_staircase_times_'+str(eccen_bin) + '_run_' + str(runi),classname='Array').read()
                                self.all_staircase_values[subject][this_condition][eccen_bin][runi] = h5file.get_node(where='/'+subject+'/'+this_condition,name=this_condition+'_staircase_values_'+str(eccen_bin) + '_run_' + str(runi),classname='Array').read()
                            except:
                                continue

        if PRF:
            # pre allocate self variables
            self.all_results = {}
            self.all_stats = {}
            for subject in subjects:
                print 'loading prf results from hdf5 for subject %s...'%(subject)
                # add empty nested dicts for this subject
                self.all_results[subject] = {}
                self.all_stats[subject] = {}
                for this_condition in conditions:
                    # add empty nested dicts for this condition
                    self.all_results[subject][this_condition] = {}
                    self.all_stats[subject][this_condition] = {}
                    for roi in self.rois.keys():
                        # pre allocate temporary empty lists that we can extend
                        temp_results = [];temp_stats=[]
                        # now loop over subrois. These are the rois listed for each 'main' roi in the self.rois dict. 
                        # for instance, for V2 we have V2v and V2d 
                        for subroi in self.rois[roi]:

                            # extend the empty temporary list using self.functions.roi_data_from_hdf if there is data for this roi for this subject
                            if this_condition == 'Stim':
                                try:
                                    temp_results.extend(np.mean([self.functions.roi_data_from_hdf(h5file, run = subject, roi_wildcard = subroi, data_type = 'Speed_results'),
                                        self.functions.roi_data_from_hdf(h5file, run = subject, roi_wildcard = subroi, data_type =  'Color_results')],axis=0))
                                    temp_stats.extend(np.mean([self.functions.roi_data_from_hdf(h5file, run = subject, roi_wildcard = subroi, data_type = 'Speed_corrs')[:,self.stats_frames['r_squared']],
                                        self.functions.roi_data_from_hdf(h5file, run = subject, roi_wildcard = subroi, data_type = 'Color_corrs')[:,self.stats_frames['r_squared']]],axis=0))
                                except:
                                    pass
                            else:
                                try:
                                    temp_results.extend(self.functions.roi_data_from_hdf(h5file, run = subject, roi_wildcard = subroi, data_type = this_condition + '_results'))
                                    temp_stats.extend(self.functions.roi_data_from_hdf(h5file, run = subject, roi_wildcard = subroi, data_type = this_condition + '_corrs')[:,self.stats_frames['r_squared']])
                                except:
                                    pass

                        # now add another roi dict and store the temporary list there
                        self.all_results[subject][this_condition][roi] = temp_results
                        self.all_stats[subject][this_condition][roi] = temp_stats


            # Now, we'll use the same dictionary structure to create one 'super subject',
            # that contains all data from all subjects combined.
            print 'creating super subject...'
            subject = 'super_subject'
            # add empty nested dicts for this super subject
            self.all_results[subject] = {}
            self.all_stats[subject] = {}
            for this_condition in conditions:
                # add empty nested dicts for this condition in the super subject
                self.all_results[subject][this_condition] = {}
                self.all_stats[subject][this_condition] = {}
                # Because all sub rois have already been combined into main rois in the loops above,
                # we no longer need to loop over sub rois, but can directly pull the data from the 
                # main roi from the all_results and all_stats dicts we have just created
                for roi in self.rois.keys():
                    # again, create temporary lists
                    temp_results = [];temp_stats=[]
                    # now extend these lists with all subjects' data for this condition and roi
                    for sub_subject in subjects:
                        temp_results.extend(self.all_results[sub_subject][this_condition][roi])
                        temp_stats.extend(self.all_stats[sub_subject][this_condition][roi])
                    # and add a new roi dict for the super subject for this condition in the all_results and all_stats dicts
                    self.all_results[subject][this_condition][roi] = temp_results
                    self.all_stats[subject][this_condition][roi] = temp_stats
            
        # The same dictionary logic used above is also used to load in the Mapper and Timecourse data. 
        # Refer to the comments above for understanding the code below.

        if Mapper:
            result_types = ['t_stat','z_stat','copes','varcopes']
            self.all_mapper = {}
            for subject in subjects:
                self.all_mapper[subject] = {}
                print 'loading mapper results from hdf5 for subject %s...'%(subject)
                for result_type in result_types:
                    self.all_mapper[subject][result_type] = {}
                    for roi in self.rois.keys():
                        this_result = []
                        for subroi in self.rois[roi]:
                            try:
                                this_result.extend(self.functions.roi_data_from_hdf(h5file, run = subject, roi_wildcard = subroi, data_type = 'Mapper_%s'%result_type))
                            except:
                                pass
                        self.all_mapper[subject][result_type][roi] = this_result

            print 'creating super subject...'
            subject = 'super_subject'
            self.all_mapper[subject] = {}

            for result_type in result_types:
                self.all_mapper[subject][result_type] = {}
                for roi in self.rois.keys():
                    this_result = []
                    for sub_subject in subjects:
                        this_result.extend(self.all_mapper[sub_subject][result_type][roi])
                    self.all_mapper[subject][result_type][roi] = this_result


    ############################
    ####### PLOT FUNCTIONALITY
    ############################

    def analyze_behavior(self,conditions=['Fix','Color','Speed','Fix_no_stim'],these_subjects=['DE','NA','JS','JW','TK'],with_first_block=True):

        def convert_quest_sample( quest_sample):

            return 1 - (1/(np.e**quest_sample+1))

        self.load_data(behavior=True)   

        self.create_plot_dir('Figure_9')

        eccen_colors = np.array([colorsys.hsv_to_rgb(c,0.6,0.9) for c in np.linspace(0.0,1,4)])[:-1]

        n_TRs_per_run = 27
        n_trials_per_run = 28
        TR = 1.6 # can be approximate - only for plot purpose
        run_length_minutes = n_TRs_per_run*n_trials_per_run*TR

        all_response_values = np.zeros((len(these_subjects),len(conditions)*3))
        all_mean_staircase_values = np.zeros((len(these_subjects),len(conditions)*3))
        all_median_staircase_values = np.zeros((len(these_subjects),len(conditions)*3))

        for si,subject in enumerate(these_subjects):

            n_runs = len(self.all_behavior_times[subject]['Fix'][0])

            f = pl.figure(figsize=(5,5))
            pi = 1
            ci_ei_count = 0 

            for ci, this_condition in enumerate(conditions):
    
                s1 = f.add_subplot(len(conditions),2,pi)
                pl.title('Staircase values %s'%this_condition)
                pi += 1
                s2 = f.add_subplot(len(conditions),2,pi)
                pl.title('Moving accuracy %s'%this_condition)
                pl.axhline(0.83,c='k')
                pi += 1

                if this_condition == 'Fix_no_stim':
                    eccen_bins = [0]
                else:
                    eccen_bins = range(3)

                for ei,eccen_bin in enumerate(eccen_bins):


                    response_values = []
                    response_times = []
                    staircase_times = []
                    staircase_values = []
                    run_start_times = []
                    response_value_count = []
                    staircase_value_count = []

                    for runi in range(n_runs):

                        response_values.append(self.all_behavior_values[subject][this_condition][eccen_bin][runi])
                        response_times.append(self.all_behavior_times[subject][this_condition][eccen_bin][runi])

                        these_staircase_values = self.all_staircase_values[subject][this_condition][eccen_bin][runi]
                        if 'Fix' not in this_condition:
                            converted_staircase_values = convert_quest_sample(these_staircase_values)
                        else:
                            converted_staircase_values =  (convert_quest_sample(these_staircase_values) - 0.5) * 2.0

                        staircase_values.append(converted_staircase_values)
                        staircase_times.append(self.all_staircase_times[subject][this_condition][eccen_bin][runi])
                        run_start_times.append(run_length_minutes*runi)
                        response_value_count.append(len(response_values[runi]))
                        staircase_value_count.append(len(staircase_values[runi]))

                    response_value_count = np.array(response_value_count).astype(int)
                    staircase_value_count = np.array(staircase_value_count).astype(int)
                    response_values = np.hstack(response_values)
                    response_times = np.hstack(response_times)
                    staircase_values = np.hstack(staircase_values)
                    staircase_times = np.hstack(staircase_times)
                    moving_accuracy = [np.mean(response_values[:ti]) for ti in range(len(response_values))]

                    if ei == 0:
                        for rst in run_start_times:
                            s1.axvline(rst,c='k',lw=0.5)
                            s2.axvline(rst,c='k',lw=0.5)

                    for runi in range(n_runs):
                        if runi ==0:
                            s1.plot(staircase_times[:staircase_value_count[0]],staircase_values[:staircase_value_count[0]],c= eccen_colors[ei],alpha=0.5,lw=1)
                            s2.plot(response_times[:response_value_count[0]],moving_accuracy[:response_value_count[0]],c= eccen_colors[ei],alpha=0.5,lw=1)
                        elif runi ==1:
                            s1.plot(staircase_times[np.sum(staircase_value_count[:runi]):np.sum(staircase_value_count[:runi+1])],staircase_values[np.sum(staircase_value_count[:runi]):np.sum(staircase_value_count[:runi+1])],c= eccen_colors[ei],label='ecc bin %d'%ei,lw=1)
                            s2.plot(response_times[np.sum(response_value_count[:runi]):np.sum(response_value_count[:runi+1])],moving_accuracy[np.sum(response_value_count[:runi]):np.sum(response_value_count[:runi+1])],c= eccen_colors[ei],label='ecc bin %d'%ei,lw=1)
                        else:
                            s1.plot(staircase_times[np.sum(staircase_value_count[:runi]):np.sum(staircase_value_count[:runi+1])],staircase_values[np.sum(staircase_value_count[:runi]):np.sum(staircase_value_count[:runi+1])],c= eccen_colors[ei],lw=1)
                            s2.plot(response_times[np.sum(response_value_count[:runi]):np.sum(response_value_count[:runi+1])],moving_accuracy[np.sum(response_value_count[:runi]):np.sum(response_value_count[:runi+1])],c= eccen_colors[ei],lw=1)

                    # and append the values to the overall var for later stats
                    if not with_first_block:
                        all_response_values[si,ci_ei_count] = np.mean(response_values[response_value_count[0]:])
                        all_staircase_values[si,ci_ei_count] = np.mean(staircase_values[response_value_count[0]:])
                        name = 'without_first_block'
                    else:
                        all_response_values[si,ci_ei_count] = np.mean(response_values)
                        all_mean_staircase_values[si,ci_ei_count] = np.mean(staircase_values)
                        all_median_staircase_values[si,ci_ei_count] = np.median(staircase_values)
                        name = 'with_first_block'

                    ci_ei_count += 1 


                pl.sca(s1)
                pl.xticks(np.array(run_start_times) + 0.5*run_length_minutes,np.arange(len(run_start_times)))
                pl.xlabel('run')
                pl.xlim(0,run_length_minutes*n_runs)
                if 'Fix' not in this_condition:
                    pl.ylim(0.5,1)
                else:
                    pl.ylim(0,1)
                pl.legend(loc='best')
                sn.despine(offset=10)

                pl.sca(s2)
                pl.ylim(0.5,1)
                pl.xticks(np.array(run_start_times) + 0.5*run_length_minutes,np.arange(len(run_start_times)))
                pl.xlabel('run')
                pl.legend(loc='best')
                sn.despine(offset=10)

            # figure options
            pl.tight_layout()
            pl.savefig(os.path.join(self.group_plot_dir,'Figure_9','%s.pdf'%(subject)))
            pl.close()


        # now let's patch things together
        columns = np.ravel([['%s_%s'%(this_condition,this_eccen_bin) for this_eccen_bin in range(3)] for this_condition in conditions])
        # convert to pandas
        import pandas as pd
        accuracy_df = pd.DataFrame(all_response_values,index=these_subjects,columns=columns)
        accuracy_df.to_csv(os.path.join(self.group_plot_dir,'Figure_9','accuracy_%s.csv'%(name)))

        mean_staircase_df = pd.DataFrame(all_mean_staircase_values,index=these_subjects,columns=columns)
        mean_staircase_df.to_csv(os.path.join(self.group_plot_dir,'Figure_9','mean_staircases_%s.csv'%(name)))
        median_staircase_df = pd.DataFrame(all_median_staircase_values,index=these_subjects,columns=columns)
        median_staircase_df.to_csv(os.path.join(self.group_plot_dir,'Figure_9','median_staircases_%s.csv'%(name)))

        for data_type in ['accuracy','mean_staircases']:
            for stat_type in ['classical']:
                condition_colors ={
                'Color': colorsys.hsv_to_rgb(0,0,0.63),
                'Speed': colorsys.hsv_to_rgb(0,0,0.33),
                'TF': colorsys.hsv_to_rgb(0,0,0.33),
                'Fix': colorsys.hsv_to_rgb(0,0,0),
                'Fix_no_stim': colorsys.hsv_to_rgb(0,0,0),
                }
                f = pl.figure(figsize=(1.2,1.5))
                s = f.add_subplot(111)
                if data_type == 'accuracy':
                    data = all_response_values
                    conditions = ['Speed', 'Color', 'Fix', 'Fix_no_stim']
                    condition_labels = ['TF','Color','Fix']
                    pl.axhline(0.83,c='k',ls='-',lw=0.5)
                elif data_type == 'mean_staircases':
                    data = all_mean_staircase_values
                    conditions = ['Speed','Color']
                    condition_labels = ['TF','Color']
                elif data_type == 'median_staircases':
                    data = all_median_staircase_values
                    conditions = ['Speed','Color']
                    condition_labels = ['TF','Color']

                if stat_type == 'bootstrap':
                    mean_all_values, ci_all_values, p_all_values,N = self.CUO.bootstrap(data.T,ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=False,outlier_num_stds=self.outlier_num_stds)
                else:
                    mean_all_values = np.mean(data,axis=0)
                    # ci_all_values = np.std(data,axis=0) / np.sqrt(len(these_subjects)) * self.ci_factor
                    ci_all_values = np.array(DescrStatsW(data).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)).T

                s = f.add_subplot(111)
                for ci,condition in enumerate(conditions):

                    if condition != 'Fix_no_stim':
                        n_bins = 3
                    else:
                        n_bins = 1
                    for ecc_bin in range(n_bins):
                        coli = np.where(columns==condition+'_%d'%ecc_bin)[0][0]
                        if condition != 'Fix_no_stim':
                            x_loc = ecc_bin + ci/(len(conditions))+1
                        else:
                            x_loc = 0.25
                        pl.plot(x_loc,mean_all_values[coli],'o',ms=6,mec='w',mew=0.25,color=condition_colors[condition])
                        pl.plot([x_loc,x_loc],ci_all_values[coli],color='w',lw=0.75)
                        pl.plot([x_loc,x_loc],ci_all_values[coli],color=condition_colors[condition],lw=0.75,alpha=0.5)

                for ci,condition in enumerate(condition_labels):
                    pl.plot(-10,-10,'o',mec=condition_colors[condition],color=condition_colors[condition],label=condition_labels[ci],ms=6)
                
                sn.despine(offset=2)
                if data_type == 'accuracy':
                    pl.yticks([0.5,0.83,0.9])
                    pl.ylim(0.5,0.9)
                    pl.legend(loc='lower left',frameon=False)
                    pl.ylabel('accuracy')
                else:
                    pl.yticks([0.5,0.75,1])
                    pl.ylim(0.5,1)
                    pl.legend(loc='upper left',frameon=False)
                    pl.ylabel('ratio')

                pl.xticks(np.arange(4)+1/(len(conditions)),[0,1,2,3])
                pl.xlim(0,4)
                pl.xlabel('eccentricity bin')
                
                pl.tight_layout(pad=0)
                pl.savefig(os.path.join(self.group_plot_dir,'Figure_9','%s_%s_%s.pdf'%(name,stat_type,data_type)))


    def color_wheel(self):

        self.create_plot_dir('Figure_3')

        # first create the colormap
        import matplotlib.colors as mcolors

        def make_colormap(seq):
            """Return a LinearSegmentedColormap
            seq: a sequence of floats and RGB-tuples. The floats should be increasing
            and in the interval (0,1).
            """
            import matplotlib.colors as mcolors

            seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
            cdict = {'red': [], 'green': [], 'blue': []}
            for i, item in enumerate(seq):
                if isinstance(item, float):
                    r1, g1, b1 = seq[i - 1]
                    r2, g2, b2 = seq[i + 1]
                    cdict['red'].append([item, r1, r2])
                    cdict['green'].append([item, g1, g2])
                    cdict['blue'].append([item, b1, b2])
            return mcolors.LinearSegmentedColormap('CustomMap', cdict)

        c = mcolors.ColorConverter().to_rgb
        rbg = make_colormap(
        [c('red'),c('blue'),0.5, c('blue'),c('green')])
        fig = pl.figure()

        grid_size = 1001
        xs = np.tile(np.arange(-grid_size/2,grid_size/2,1),(grid_size,1))
        ys = np.tile(np.arange(grid_size/2,-grid_size/2,-1),(grid_size,1)).T

        angles = np.arctan2(xs,ys)
        angles[:,grid_size/2:] = np.nan
        angles[(disk((grid_size-1)/2)==False)] = np.nan

        pl.imshow(angles,cmap=rbg)

        pl.axis('off')

        pl.savefig(os.path.join(self.group_plot_dir,'Figure_3','color_wheel.pdf'),facecolor='k')

    def R2_distributions2(self,):

        self.create_plot_dir('R2_dist')
        # only load data if it has not already been added to the self object
        if ((not hasattr(self,'all_results')) + (not hasattr(self,'all_stats'))): 
            self.load_data(PRF=True)        

        for sub in np.hstack([self.subjects,'super_subject','over_subjects']):

            if sub == 'over_subjects':
                subjects_to_use = self.subjects
            else:
                subjects_to_use = [sub]

            f = pl.figure(figsize = (3,1.5))
            for mi,measure in enumerate(['mean','sum']):

                s = f.add_subplot(1,2,mi+1)
                for ri, roi in enumerate(self.rois_for_plot[:-1]):

                    # create empty array for this roi
                    all_weight_means = []

                    for si, subject in enumerate(subjects_to_use):

                        # get mask variables
                        sizes = np.array(self.all_results[subject]['Fix'][roi])[:,self.results_frames['sigma_center']] / self.rescale_factor
                        ecc = np.array(self.all_results[subject]['Fix'][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                        weights = np.squeeze(self.all_stats[subject]['Fix'][roi])

                        # create mask based on ecc, sizes and weights
                        mask = self.functions.create_mask(ecc,self.mask_ecc_thresholds,self.r_squared_threshold,weights,sizes,self.outlier_num_stds,self.size_threshold)

                        # apply mask
                        weights = weights[mask]

                        # mean, se, p,N = self.CUO.bootstrap(np.array(weights),ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=False,outlier_num_stds=self.outlier_num_stds)

                        # if measure == 'sum':
                            # mean *= len(weights)
                            # se *= len(weights)

                        all_weight_means.append(mean)

                    # if more than 1 subject, 
                    if len(subjects_to_use)>1:
                        mean = np.mean(all_weight_means)
                        se = DescrStatsW(np.array(all_weight_means)).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)
                        N = len(all_weight_means)
                        t,p = sp.stats.ttest_1samp(all_weight_means,0)

                    pl.plot(ri,mean,'o',ms=5,c=self.roi_colors[ri],mec='w')
                    pl.plot([ri,ri],se,lw=1.5,c='w')
                    pl.plot([ri,ri],se,lw=1,c=self.roi_colors[ri])
                    # pl.plot(ri,mean,'o',ms=5)

                sn.despine(offset=2)
                if measure == 'mean':
                    pl.ylim(0,0.5)
                else:
                    if sub != 'super_subject':
                        pl.ylim(0,300)
                pl.xlim(-0.5,len(self.rois_for_plot)-1.5)
                pl.xticks([])
                pl.ylabel('$R^2$ %s'%measure)

            # figure options 
            pl.tight_layout()
            # save
            pl.savefig(os.path.join(self.group_plot_dir,'R2_dist','R2_%s.pdf'%(sub)))
            pl.close()

    def R2_distributions(self,):

        self.create_plot_dir('R2_violin')
        # only load data if it has not already been added to the self object
        if ((not hasattr(self,'all_results')) + (not hasattr(self,'all_stats'))): 
            self.load_data(PRF=True)        

        for subject in np.hstack([self.subjects,'super_subject']):

            f = pl.figure(figsize = (2,1.5))
            s = f.add_subplot(111)
            pl.axhline(0.1,color='k')            
            dc = pd.DataFrame()
            for ri, roi in enumerate(self.rois_for_plot[:-1]):

                # get mask variables
                sizes = np.array(self.all_results[subject]['Fix'][roi])[:,self.results_frames['sigma_center']] / self.rescale_factor
                ecc = np.array(self.all_results[subject]['Fix'][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                weights = np.squeeze(self.all_stats[subject]['Fix'][roi])

                # create mask based on ecc, sizes and weights
                mask = self.functions.create_mask(ecc,self.mask_ecc_thresholds,self.r_squared_threshold,weights,sizes,self.outlier_num_stds,self.size_threshold)

                # apply mask
                weights = weights[mask]

                dc[roi] = pd.Series(weights)

            # sn.violinplot(x=np.ones(len(weights))*ri,y=weights,color=self.roi_colors[ri],orient='v',alpha=0.1)
            sn.violinplot(data=dc,palette=self.roi_colors,orient='v',alpha=0.1,scale='width',linewidth=0)
            # pl.plot(ri,mean,'o',ms=5,c=self.roi_colors[ri],mec='w')
            # pl.plot([ri,ri],se,lw=1.5,c='w')
            # pl.plot([ri,ri],se,lw=1,c=self.roi_colors[ri])
            # pl.plot(ri,mean,'o',ms=5)

            pl.ylim(0,0.8)

            sn.despine()
            pl.xticks([])

            # # pl.xlim(-0.5,len(self.rois_for_plot)-1.5)
            # # pl.xticks([])
            pl.ylabel('$R^2$')
            # if ri > 0:
            #     pl.axis('off')
            # pl.s
            # figure options 
            pl.tight_layout()
            # save
            pl.savefig(os.path.join(self.group_plot_dir,'R2_violin','R2_%s.pdf'%(subject)))
            pl.close()

    def mask_visualization(self,
        subjects = ['NA','JS','DE','JW','TK','super_subject',]):

        self.create_plot_dir('maskvis')
        # only load data if it has not already been added to the self object
        if ((not hasattr(self,'all_results')) + (not hasattr(self,'all_stats'))): 
            self.load_data(PRF=True)        

        for plot_type in ['all','last']:
            for sub in subjects:

                if sub == 'over_subjects':
                    subjects_to_use = self.subjects
                else:
                    subjects_to_use = [sub]

                if plot_type == 'all':
                    f = pl.figure(figsize = (3,1.5))
                elif plot_type == 'last':
                    f = pl.figure(figsize = (1.4,1.5))
                s = f.add_subplot(111)
                for ri, roi in enumerate(self.rois_for_plot[:-1]):

                    # create empty array for this roi
                    alltotals,allr2s, allr2eccs, allr2eccsizes = [],[],[],[]

                    for si, subject in enumerate(subjects_to_use):

                        # get mask variables
                        sizes = np.array(self.all_results[subject]['Fix'][roi])[:,self.results_frames['sigma_center']] / self.rescale_factor
                        ecc = np.array(self.all_results[subject]['Fix'][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                        weights = np.squeeze(self.all_stats[subject]['Fix'][roi])

                        valid_r2slow  = (weights>0.01)
                        valid_r2s = (weights>self.r_squared_threshold)
                        valid_eccs = (ecc<self.mask_ecc_thresholds[1])
                        valid_sizes = (sizes<self.size_threshold)

                        sum_total = valid_r2slow.sum()
                        sum_r2 = valid_r2s.sum()
                        sum_r2ecc = (valid_r2s*valid_eccs).sum()
                        sum_r2eccsize = (valid_r2s*valid_eccs*valid_sizes).sum()

                        alltotals.append(sum_total)
                        allr2s.append(sum_r2)
                        allr2eccs.append(sum_r2ecc)
                        allr2eccsizes.append(sum_r2eccsize)

                    # if more than 1 subject, 
                    if len(subjects_to_use)>1:
                        
                        sum_total = np.mean(alltotals)
                        se_total = DescrStatsW(np.array(alltotals)).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)

                        sum_r2 = np.mean(allr2s)
                        se_r2 = DescrStatsW(np.array(allr2s)).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)

                        sum_r2ecc = np.mean(allr2eccs)
                        se_r2ecc = DescrStatsW(np.array(allr2eccs)).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)

                        sum_r2eccsize = np.mean(allr2eccsizes)
                        se_r2eccsize = DescrStatsW(np.array(allr2eccsizes)).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)
                    
                        # print 'sub %s, roi %s %d (%.2f-%.2f) voxels remain where R2 > .01'%(sub,roi,sum_total,se_total[0],se_total[1])
                        # print 'sub %s, roi %s  %d (%.2f-%.2f) voxels remain where R2 > %.2f'%(sub,roi,sum_r2,se_r2[0],se_r2[1],self.r_squared_threshold)
                        # print 'sub %s, roi %s  %d (%.2f-%.2f) voxels remain where R2 > %.2f and ecc < %.2f'%(sub,roi,sum_r2ecc,se_r2ecc[0],se_r2ecc[1],self.r_squared_threshold,self.mask_ecc_thresholds[1])
                        # print 'sub %s, roi %s  %d (%.2f-%.2f) voxels remain where R2 > %.2f and ecc < %.2f and size < %.2f'%(sub,roi,sum_r2eccsize,se_r2eccsize[0],se_r2eccsize[1],self.r_squared_threshold,self.mask_ecc_thresholds[1],self.size_threshold)
                    
                    # if (roi == 'V1') * (subject == 'super_subject'):
                    #     width = 1
                    #     offset = -2
                    #     osm = 5
                    # else:
                    if plot_type == 'all':
                        width = 0.3
                        offset=0
                        osm = 1
                    elif plot_type == 'last':
                        width = 1
                        osm = 0
                        offset=-0.5

                    # first plot marker for r2 only
                    # pl.plot(ri-0.3,sum_total,'v',ms=5,c=self.roi_colors[ri],mec='w')
                    if plot_type == 'all':
                        pl.bar(ri-0.3*osm+offset,sum_total,color=self.roi_colors[ri],edgecolor='w',width=width)
                        if len(subjects_to_use)>1:
                            pl.plot([ri-0.3,ri-0.3],se_total,lw=1.5,c='w')
                            pl.plot([ri-0.3,ri-0.3],se_total,lw=1,c=self.roi_colors[ri])
                       
                        # first plot marker for r2 only
                        pl.bar(ri-0.1*osm+offset,sum_r2,color=self.roi_colors[ri],edgecolor='w',width=width)
                        # pl.plot(ri-0.1,sum_r2,'o',ms=5,c=self.roi_colors[ri],mec='w')
                        if len(subjects_to_use)>1:
                            pl.plot([ri-0.1,ri-0.1],se_r2,lw=1.5,c='w')
                            pl.plot([ri-0.1,ri-0.1],se_r2,lw=1,c=self.roi_colors[ri])
                        
                        # then also including ecc
                        # pl.plot(ri+0.1,sum_r2ecc,'s',ms=5,c=self.roi_colors[ri],mec='w')
                        pl.bar(ri+0.1*osm+offset,sum_r2ecc,color=self.roi_colors[ri],edgecolor='w',width=width)
                        if len(subjects_to_use)>1:
                            pl.plot([ri+0.1,ri+0.1],se_r2ecc,lw=1.5,c='w')
                            pl.plot([ri+0.1,ri+0.1],se_r2ecc,lw=1,c=self.roi_colors[ri])
                        
                    # then also including size
                    pl.bar(ri+0.3*osm+offset,sum_r2eccsize,color=self.roi_colors[ri],edgecolor='w',width=width)
                    # pl.plot(ri+0.3,sum_r2eccsize,'>',ms=5,c=self.roi_colors[ri],mec='w')
                    if len(subjects_to_use)>1:
                        pl.plot([ri+0.3,ri+0.3],se_r2eccsize,lw=1.5,c='w')
                        pl.plot([ri+0.3,ri+0.3],se_r2eccsize,lw=1,c=self.roi_colors[ri])                

                if sub != 'super_subject':
                    pl.ylim(0,1200)
                if roi == 'MT+':
                    pl.legend(loc='best')
                sn.despine(offset=2)

                pl.ylabel('# voxels')
                pl.xlim(-0.5,len(self.rois_for_plot)-1.5)
                pl.xticks([])
                # pl.ylabel(' %s'%measure)

                # figure options 
                pl.tight_layout()
                # save
                pl.savefig(os.path.join(self.group_plot_dir,'maskvis','maskvis_%s_%s.pdf'%(sub,plot_type)))
                pl.close()

    def var_over_var(self,
        conditions=['ALL','Fix','Color','Speed','Stim'],
        n_bins = 6,
        bin_type='fixed',
        plot_types = ['eccen_surf'],
        subjects=['over_subjects','super_subject']
        ):

        """
        Plots one variable as a function of another (see plot_types)

        Inputs:
        conditions: creates plot for every condition (ALL is from the ALL fit)
        plot_types: 'eccen_surf' or 'weights_over_ecc', or 'weights_over_size' 
        stat_method: 'super_subject', 'over_subjects', 'over_super_subjects'
        n_bins: amount of bins of x variable
        bin_type: 'fixed', or 'percentile'
        """
        self.create_plot_dir('Figure_3')
        # only load data if it has not already been added to the self object
        if ((not hasattr(self,'all_results')) + (not hasattr(self,'all_stats'))): 
            self.load_data(PRF=True)        
        for sub in subjects:

            if sub == 'over_subjects':
                subjects_to_use = self.subjects
            else:
                subjects_to_use = [sub]

            mpl.rc_file_defaults()

            # refresh plot dir
            # see which plot types to do
            for plot_type in plot_types:

                # for the different conditions specified
                for ci,condition in enumerate(conditions):

                    f=pl.figure(figsize = (3.2,1.2))
                    # initiate a counter for roi colors
                    ri = -1

                    if sub == 'super_subject':
                        all_ps = []
                    
                    # now loop over the roi groups
                    for rgi,roi_group in enumerate(np.sort(self.roi_groups_for_plot.keys())):
                        # reset the roi counter again when we hit the 'all' roi group
                        if roi_group == '4_%s'%self.rois_for_plot[-1]:
                            continue

                        # create subplot per roigroup
                        s = f.add_subplot(self.roi_group_subplot_grid[0],self.roi_group_subplot_grid[1]-1,rgi+1)
                        pl.title('%s'%(' '.join(roi_group.split('_')[1:])))
                        sn.despine(offset=2)
                        print ''

                        # now for all rois within the roi group
                        for subri, roi in enumerate(self.roi_groups_for_plot[roi_group]):
                            
                            # advance roi color counter
                            ri += 1

                            # initiate group lists
                            all_mean_x_data = []
                            all_mean_y_data = []
                            these_slopes = []
                            these_intercepts = []
                                    
                            print ''

                            # loop over subjects and get the data and either plot or save to group list
                            for si, subject in enumerate(subjects_to_use):

                                sys.stdout.write('Analyzing %s relation for %s, %s, subject %d/%d\r'%(plot_type,condition,roi,si+1,len(subjects_to_use)))
                                sys.stdout.flush()

                                # get mask variables
                                sizes = np.array(self.all_results[subject][condition][roi])[:,self.results_frames['sigma_center']] / self.rescale_factor
                                ecc = np.array(self.all_results[subject][condition][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                weights = np.squeeze(self.all_stats[subject][condition][roi])

                                # create mask based on ecc, sizes and weights
                                mask = self.functions.create_mask(ecc,self.mask_ecc_thresholds,self.r_squared_threshold,weights,sizes,self.outlier_num_stds,self.size_threshold)

                                # apply mask to variables of interest
                                sizes_masked, ecc_masked, weights_masked = sizes[mask], ecc[mask], weights[mask]

                                if plot_type == 'eccen_surf':
                                    y_data = sizes_masked
                                    x_data = ecc_masked
                                    weights = weights_masked
                                elif plot_type == 'weights_over_ecc':
                                    y_data = weights_masked
                                    x_data = ecc_masked
                                    weights = np.ones_like(weights_masked)
                                elif plot_type == 'weights_over_size':
                                    y_data = weights_masked
                                    x_data = sizes_masked
                                    weights = np.ones_like(weights_masked)          

                                # get binned data
                                if bin_type == 'percentile':
                                    x_means, x_cis, y_means, y_cis, y_ps, Ns = self.functions.percentile_bins(y_data=y_data,x_data=x_data,weights=weights,n_bins=n_bins,ci_factor=self.ci_factor,outlier_num_stds=self.outlier_num_stds,detect_inliers=self.detect_outliers)
                                elif bin_type == 'fixed':
                                    x_means, x_cis, y_means, y_cis, y_ps, Ns  = self.functions.fixed_bins(y_data=y_data,x_data=x_data,weights=weights,n_bins=n_bins,ci_factor=self.ci_factor,ecc_thresholds=self.plot_ecc_thresholds,outlier_num_stds=self.outlier_num_stds,detect_inliers=self.detect_outliers)

                                # append the data to the group list,
                                all_mean_x_data.append(x_means)
                                all_mean_y_data.append(y_means)

                                # compute linear fit for this participant
                                mean_slope, se_slope, p_slope,mean_intercept, se_intercept, p_intercept = self.CUO.bootstrap_linear_fit(x_data,y_data,
                                    weights=weights,ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds)


                                if not sub in ['over_subjects']:
                                    print 'roi %s sub %s mean slope: %.3f 95 ci: %s,p:%.3f, N: %d'%(roi,sub,mean_slope,se_slope,p_slope,len(y_data))
                                these_slopes.append(mean_slope)
                                these_intercepts.append(mean_intercept)
                                if sub == 'super_subject':
                                    all_ps.append(p_slope)

                            # if there's more than 1 subject, bootstrap over subjects
                            if len(subjects_to_use)>1:
                                mean_intercept = np.mean(these_intercepts)
                                se_intercept = DescrStatsW(these_intercepts).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)
                                mean_slope = np.mean(these_slopes)
                                se_slope = DescrStatsW(these_slopes).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)
                                t,p_slope = sp.stats.ttest_1samp(these_slopes,0)

                                y_means = np.nanmean(all_mean_y_data,axis=0)
                                y_cis = np.array([DescrStatsW(y[~np.isnan(y)]).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2) for y in np.array(all_mean_y_data).T])
                                x_means = np.nanmean(all_mean_x_data,axis=0)
                                x_cis =  np.array([DescrStatsW(x[~np.isnan(x)]).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2) for x in np.array(all_mean_x_data).T])
                                
                                print 'roi %s sub %s mean slope: %.3f 95 ci: %s,p:%.3f,N:%d'%(roi,sub,mean_slope,se_slope,p_slope,len(subjects_to_use))

                                # mean_intercept, se_intercept, p_intercept,N = self.CUO.bootstrap(np.array(these_intercepts),ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=False,outlier_num_stds=self.outlier_num_stds)
                                # mean_slope, se_slope, p_slope,N = self.CUO.bootstrap(np.array(these_slopes),ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=False,outlier_num_stds=self.outlier_num_stds)
                                # y_means,y_cis,y_ps,N = self.CUO.bootstrap(np.array(all_mean_y_data).T,ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=False,outlier_num_stds=self.outlier_num_stds)
                                # x_means,x_cis,x_ps,N = self.CUO.bootstrap(np.array(all_mean_x_data).T,ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=False,outlier_num_stds=self.outlier_num_stds)

                            fit_fn = np.poly1d([mean_slope,mean_intercept])
                            fit_fn_low = np.poly1d([se_slope[0],se_intercept[0]])               
                            fit_fn_high = np.poly1d([se_slope[1],se_intercept[1]])              

                            pl.plot(x_means,y_means,marker=['o','s','^'][subri],ms=4,lw = 0,c= self.roi_colors[ri],mec= 'w',mew=0.5,label=roi)
                            # for error bars, fill_between or error bars
                            for bi in range(n_bins):
                                if not np.isnan(x_means[bi]):
                                    pl.plot(np.repeat(x_means[bi],2),y_cis[bi],'-',lw = 0.6,c= 'w')
                                    pl.plot(np.repeat(x_means[bi],2),y_cis[bi],'-',lw = 0.6,c= self.roi_colors[ri],alpha=0.5)

                            pl.plot([np.min(x_data),np.max(x_data)],[fit_fn(np.min(x_data)),fit_fn(np.max(x_data))],c= self.roi_colors[ri],lw=1,ls='-')

                            pl.fill_between([np.min(x_data),np.max(x_data)],
                                [fit_fn_low(np.min(x_data)),fit_fn_low(np.max(x_data))],
                                [fit_fn_high(np.min(x_data)),fit_fn_high(np.max(x_data))],
                            color=self.roi_colors[ri],alpha=0.2)                            

                        # subplot options
                        if plot_type == 'eccen_surf':
                            if rgi==0:
                                pl.ylabel('pRF size (dva)')
                                pl.xlabel('pRF eccentricity (dva)')
                           
                            if sub == 'super_subject':
                                yticks = self.functions.find_yticks(s.get_ylim())   
                            else:
                                yticks = [
                                    [[0.2,1.6],['.2','1.6']],
                                    [[0,3.5],['0','3.5']],
                                    [[0.2,5],['.2','5']],
                                    ][rgi]

                            pl.yticks(yticks[0],yticks[1])

                            if rgi == 0:
                                pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])
                            else:
                                pl.xticks([])
                            pl.xlim(self.mask_ecc_thresholds)

                        elif plot_type == 'weights_over_ecc':
                            pl.ylabel('model R2')
                            pl.xlabel('pRF eccentricity (dva)')
                            pl.ylim(0,0.5)  
                        elif plot_type == 'weights_over_size':
                            pl.ylabel('model R2')
                            pl.ylim(0,0.5)
                            pl.xlabel('pRF size (dva)')
                            pl.xlim(0,7)

                    if subject == 'super_subject':
                        rejections,all_ps = mne.stats.fdr_correction(all_ps,alpha=.05)
                        print 'corrected ps for super_subject: %s'%all_ps

                    # figure options 
                    pl.tight_layout(w_pad = 0.0,h_pad=0.0,pad=0)
                    # save
                    pl.savefig(os.path.join(self.group_plot_dir,'Figure_3','%s_%s_%s_%.2f_outlier_detection_%s.pdf'%(plot_type,sub,condition,self.r_squared_threshold,self.detect_outliers)))
                    pl.close()

    def arrow_plot(self,
        comparisons = {'Stim - Fix': ['Stim','Fix'],'Speed - Fix':['Speed','Fix'],'Color - Fix':['Color','Fix']},
        subjects = ['super_subject'],
        fields = ['quadrant','whole_field'],
        bin_type = 'variable',
        bin_data = 'cond_1_data',
        n_eccen_bins = 8,
        n_polar_bins = 8,
        location = 'centre_of_bin',
        min_vox_per_arrow=0,
        ):

        """
        This function creates vector plots that indicate the mean difference
        vector throughout the visual field. 

        We'll bin the data based on fix position and look at the difference in the stimulus 
        conditions. This can be done by setting the 'bin_data' variable to cond_1. 
        """

        # only load data if it has not already been added to the self object
        if ((not hasattr(self,'all_results')) + (not hasattr(self,'all_stats'))): 
            self.load_data(PRF=True)
        # refresh plot dir
        self.create_plot_dir('Figure_4')  
        self.roi_subplot_grid = [2,5]
        mpl.rc_file_defaults()

        for subject in subjects:
            if subject == 'over_subjects':
                these_subjects = self.subjects
            else:
                these_subjects = [subject]  

            # create seperate plot for evey 'field': whole_field or quadrant
            for field in fields:
                # and do this for all condition comparisons
                for ci, comparison in enumerate(comparisons.keys()):
    

                    print 'creating arrow plot for %s comparison %s...'%(subject,comparison)

                    # create one figure per subject per comparison
                    f = pl.figure(figsize=(5,2.5))

                    # f = pl.figure(figsize=(self.roi_subplot_grid[1]*2,self.roi_subplot_grid[0]*2+1))
                    for ri,roi in enumerate(self.rois_for_plot):
                        all_arrow_starts = []
                        all_arrow_diffs = []
                        for sub_subject in these_subjects:

                            # and one subplot per roi
                            s = f.add_subplot(self.roi_subplot_grid[0],self.roi_subplot_grid[1],ri+1,aspect='equal')
                            
                            # get relevant variables for mask and rescale
                            sizes_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                            sizes_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                            ecc_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                            ecc_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                            weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])

                            # now use the mask function
                            mask = self.functions.create_mask_2_conditions(ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights,self.mask_ecc_thresholds,
                                self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)

                            # r2 same for all conditions, so no matter which one we take
                            weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])[mask]

                            # now, we'll get the vector data for both conditions
                            cond_0 = np.array([np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['xo']],
                                np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['yo']]]) * self.stim_radius
                            cond_1 = np.array([np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['xo']],
                                np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['yo']]]) * self.stim_radius
                    
                            # now, we want to rotate the fix vector so that it is in the upper right quadrant. 
                            # In order to do that, we can multiply it by it's sign (this is the same as taking the absolute)
                            # However, in order to rotate the stim vector accordingly, we also need to multiply it by the sign of fix. 
                            # This way, all fix vectors will be rotated so that they have positive x and y values, but the stim vector does 
                            # not necessary also have this. This means pRFs can still move out of the quadrant, which is what we want.
                            cond_1_shifted = cond_1 * np.sign(cond_1)
                            cond_0_shifted = cond_0 * np.sign(cond_1)

                            # set variables according to desired field
                            if field == 'quadrant':
                                cond_1_data = cond_1_shifted
                                cond_0_data = cond_0_shifted
                            elif field == 'whole_field':
                                cond_1_data = cond_1
                                cond_0_data = cond_0
                            
                            # get the arrow data
                            arrow_starts, arrow_diffs = self.functions.get_arrows(field,n_eccen_bins,self.mask_ecc_thresholds,n_polar_bins,
                                                            cond_0_data,cond_1_data,weights,self.detect_outliers,self.outlier_num_stds,location,min_vox_per_arrow)

                            all_arrow_starts.append(arrow_starts)
                            all_arrow_diffs.append(arrow_diffs)

                        if len(these_subjects) > 1:
                            arrow_starts = np.nanmean(all_arrow_starts,axis=0)
                            arrow_diffs = np.nanmean(all_arrow_diffs,axis=0)

                        for circle_edge in np.linspace(0,self.mask_ecc_thresholds[1],5)[1:]:
                            circle = pl.Circle((0,0),circle_edge,color='k',edgecolor='k',ls='dashed',fill=False,lw=0.5)
                            s.add_artist(circle)

                        circle = pl.Circle((0,0),self.stim_radius,color='k',edgecolor='k',fill=False,lw=0.5)
                        s.add_artist(circle)

                        # set ticks and labels
                        degree_sign= u'\N{DEGREE SIGN}'
                        if ri == (self.roi_subplot_grid[1]*(self.roi_subplot_grid[0]-1)+np.round(self.roi_subplot_grid[0]/2)-1):
                            pl.xticks([0,self.stim_radius])
                            pl.xlabel('x-position (dva)')   
                        else:
                            pl.xticks([])   
                        if ri == self.roi_subplot_grid[1]:
                            pl.yticks([0,self.stim_radius])
                            pl.ylabel('y-position (dva)')
                        else:
                            pl.yticks([])

                        # now start plotting
                        if field == 'whole_field':
                            # 0 lines
                            pl.axhline(0,ls='-',c='k',lw=0.25)
                            pl.axvline(0,ls='-',c='k',lw=0.25)
                            # plot arrow in for loop
                            for bi in range(len(arrow_starts)):
                                if np.isnan(arrow_starts[bi]).sum() == 0:
                                    head_width = 0.8*self.plot_ecc_thresholds[1]/6
                                    pl.arrow(arrow_starts[bi,0],arrow_starts[bi,1],arrow_diffs[bi,0],arrow_diffs[bi,1],color=self.roi_colors[ri],head_width=head_width,width=0.12,length_includes_head=True)

                            # and set limits
                            pl.xlim(-self.stim_radius,self.stim_radius)
                            pl.ylim(-self.stim_radius,self.stim_radius)
                        elif field == 'quadrant':   
                            # 0 lines
                            pl.axhline(0,ls='-',c='k',lw=0.25)
                            pl.axvline(0,ls='-',c='k',lw=0.25)  
                            # plot arrow in for loop
                            for bi in range(len(arrow_starts)):
                                if np.isnan(arrow_starts[bi]).sum() == 0:
                                    head_width = 0.1
                                    head_length = 0.1
                                    tail_width = 0.75
                                    if np.linalg.norm(arrow_starts[bi]) > self.mask_ecc_thresholds[1]:
                                        alpha=0.5
                                    else:
                                        alpha=1
                                    pl.arrow(arrow_starts[bi,0],arrow_starts[bi,1],arrow_diffs[bi,0],arrow_diffs[bi,1],color=self.roi_colors[ri],head_length=head_length,lw=tail_width,head_width=head_width,alpha=alpha,length_includes_head=False)
                            pl.xlim(0,self.stim_radius)
                            pl.ylim(0,self.stim_radius)
                        s.axis('off')

                    pl.tight_layout(w_pad = 0.2,h_pad=0,pad=0)#
                    pl.savefig(os.path.join(self.group_plot_dir,'Figure_4','%s_pos_change_%s_%s_minvoxs_%d_nbins_%d.pdf'%(subject,field,comparison,min_vox_per_arrow,n_eccen_bins*n_polar_bins)))

    def shift_explained_by_ecc(self,
        comparisons = {'Speed - Fix':['Speed','Fix'],'Color - Fix': ['Color','Fix'],'Color - Speed': ['Color','Speed']},
        measures = ['ecc_diff','x_diff','y_diff'],
        subjects=['over_subjects','super_subject','DE','JS','TK','JW','NA']):

        """
        This function examines how much of the pRF shift is expained by pRF x/y/ecc changes.
        """

        self.create_plot_dir('Figure_4')
        mpl.rc_file_defaults()
        self.load_data(PRF=True)
        for subject in subjects:
            if subject == 'over_subjects':
                these_subjects = self.subjects
            else:
                these_subjects = [subject]

            for ci, comparison in enumerate(comparisons):
                all_ps = []
                all_p_indices = []

                f = pl.figure(figsize=(3.75,2))
                s = f.add_subplot(111)
                

                rois = []
                ratios = []
                measure_ids  = []
                sub_ids = []
                max_y_values = []
                for mi,measure in enumerate(measures):
                    # pl.title(measure)     
                    for ri,roi in enumerate(self.rois_for_plot):
                        # print ''    
                        all_mean_data = []
                        all_diff_data = []
                        sig_in1 = []
                        sig_opp1 = []
                        sig_in2 = []
                        sig_opp2 = []
                        for si, sub_subject in enumerate(these_subjects):

                            # get relevant variables for mask and rescale
                            sizes_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                            sizes_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                            ecc_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                            ecc_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                            weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])

                            # now use the mask function
                            mask = self.functions.create_mask_2_conditions(ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights,self.mask_ecc_thresholds,
                                self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)

                            # r2 same for all conditions, so no matter which one we take
                            weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])[mask]

                            # get the data for both conditions
                            data_0 = np.array([np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['xo']],
                                np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['yo']]])* self.stim_radius
                            data_1 = np.array([np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['xo']],
                                np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['yo']]])* self.stim_radius

                            ecc_diff = np.abs(ecc_cond_0[mask] - ecc_cond_1[mask])
                            x_diff = np.abs(data_0[0]-data_1[0])
                            y_diff = np.abs(data_0[1]-data_1[1])
                            norm_diff_vector = np.abs(np.linalg.norm(data_0 - data_1,axis=0))

                            # now get desired measure
                            data = eval(measure)/norm_diff_vector

                            # print out whether ecc explains more than x_diff
                            if (measure == 'ecc_diff'):

                                these_data = data-(x_diff/norm_diff_vector)
                                mean_diff, se_diff, p_diff, N_diff,cohen_d = self.CUO.bootstrap(these_data,weights=weights,ci_factor=self.ci_factor,test_value=0,
                                    two_tailed=True,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds,return_d=True)
                                all_diff_data.append(mean_diff)
                                sig_in1.append((p_diff<.05)*(mean_diff>0))
                                sig_opp1.append((p_diff<.05)*(mean_diff<0))

                            # print out whether ecc explains more than x_diff
                            elif (measure == 'x_diff'):

                                these_data = data-(y_diff/norm_diff_vector)
                                mean_diff, se_diff, p_diff, N_diff,cohen_d   = self.CUO.bootstrap(these_data,weights=weights,ci_factor=self.ci_factor,test_value=0,
                                    two_tailed=True,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds,return_d=True)                                
                                all_diff_data.append(mean_diff)
                                sig_in2.append((p_diff<.05)*(mean_diff>0))
                                sig_opp2.append((p_diff<.05)*(mean_diff<0))

                            mean_data, se_data, p, N = self.CUO.bootstrap(np.array(data),weights=weights,ci_factor=self.ci_factor,test_value=0,
                                two_tailed=True,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds)


                            all_mean_data.append(mean_data)

                            if (subject == 'over_subjects') * (roi != 'combined'):

                                rois.append(roi)
                                ratios.append(mean_data)
                                measure_ids.append(measure)
                                sub_ids.append(sub_subject)

                        # if we have more than one subject compute average over subjects
                        if len(these_subjects) > 1:
                        
                            valid_data = np.array(all_mean_data)[~np.isnan(all_mean_data)]
                            mean_data = np.mean(valid_data)
                            se_data = DescrStatsW(valid_data).tconfint_mean(alpha=(stats.norm.sf(self.ci_factor))*2)

                            # if measure is x_diff or ecc_diff, we computed a diff
                            if all_diff_data != []:

                                mean_diff, se_diff, t_diff, p_diff, cohen_d, N_diff = self.CUO.stats_from_data(all_diff_data,self.ci_factor)

                                if (measure == 'ecc_diff'):
                                    print '%s %s ecc>x sig in %d/%d subs'%(subject, roi, np.sum(sig_in1),len(these_subjects))
                                    print '%s %s ecc<x sig in %d/%d subs'%(subject, roi, np.sum(sig_opp1),len(these_subjects))
                                    all_p_indices.append('%s %s ecc > x'%(subject,roi))
                                    print '%s %s ecc > x: %.2f, t:%.3f, p: %.3f, N: %d, cohen d: %.2f '%(subject,roi,mean_diff,t_diff,p_diff,N_diff,cohen_d)
                                elif (measure == 'x_diff'):
                                    all_p_indices.append('%s %s x > y'%(subject,roi))
                                    print '%s %s x > y: %.2f, t:%.3f, p: %.3f, N: %d, cohen d: %.2f '%(subject,roi,mean_diff,t_diff,p_diff,N_diff,cohen_d)
                                    print '%s %s x>y sig in %d/%d subs'%(subject, roi, np.sum(sig_in2),len(these_subjects))
                                    print '%s %s x<y sig in %d/%d subs'%(subject, roi, np.sum(sig_opp2),len(these_subjects))

                        # if we had more than one subjects, p_diff is now defined directly above (the ttest),
                        # otherwise it still stems from the above per subject bootstrap
                        if all_diff_data != []:
                            all_ps.append(p_diff)
                            if (measure == 'ecc_diff'):
                                all_p_indices.append('%s %s ecc > x'%(subject,roi))
                                print '%s %s ecc > x: %.2f, p: %.3f, N: %d, cohen d: %.2f '%(subject,roi,mean_diff,p_diff,N_diff,cohen_d)
                            elif (measure == 'x_diff'):
                                all_p_indices.append('%s %s x > y'%(subject,roi))
                                print '%s %s x > y: %.2f, p: %.3f, N: %d, cohen d: %.2f '%(subject,roi,mean_diff,p_diff,N_diff,cohen_d)

                        max_y_values.append(se_data[1])

                        x_spacing = 1/(len(measures)+1)
                        x = ri+mi*x_spacing
                        pl.plot(x,mean_data,['o','s','^'][mi],ms=5,mec=self.roi_colors[ri],color=self.roi_colors[ri])
                        pl.plot([x,x],se_data,c='w',lw=0.6)
                        pl.plot([x,x],se_data,color=self.roi_colors[ri],alpha=0.5,lw=0.6)

                # only do FDR correction for the super subject
                if subject == 'super_subject':
                    rejections,all_ps = mne.stats.fdr_correction(all_ps,alpha=.05)

                for pi,p_val in enumerate(all_ps):

                    if p_val < .05:

                        ri = np.hstack([np.arange(len(self.rois_for_plot)),np.arange(len(self.rois_for_plot))])[pi].astype(int)
                        mi = np.hstack([np.zeros(len(self.rois_for_plot)),np.ones(len(self.rois_for_plot))])[pi].astype(int)

                        # plot bracket:
                        x1 = ri+mi*x_spacing
                        x2 = ri+(mi+1)*x_spacing

                        # find highest y of the two:
                        max_y = np.max([max_y_values[mi*len(self.rois_for_plot)+ri],max_y_values[(mi+1)*len(self.rois_for_plot)+ri]])
                        y_offset_data = 0.02
                        bracket_height = 0.01
                        y_low = max_y+y_offset_data
                        y_high = y_low+bracket_height

                        # plot the bracket
                        pl.plot([x1,x1],[y_low,y_high],ls='-',lw=0.5,c='k')
                        pl.plot([x2,x2],[y_low,y_high],ls='-',lw=0.5,c='k')
                        pl.plot([x1,x2],[y_high,y_high],ls='-',lw=0.5,c='k')

                        # plot the stars
                        text_y_offset = 0
                        num_stars = np.sum([p_val<.05,p_val<.01,p_val<.001])
                        pl.text(np.mean([x1,x2]),y_high+text_y_offset,'*'*num_stars,horizontalalignment='center')

                if 'vs' in measure:
                    pl.ylim(0,0.25)
                    pl.yticks([0,0.25],[0,25])
                else:
                    if subject == 'super_subject':
                        pl.ylim(0.25,1)
                        pl.yticks(np.linspace(0.25,1,4))
                    else:
                        pl.ylim(.2,1.1)
                        pl.yticks(np.linspace(0.2,1,9))
                
                pl.xticks(np.arange(len(self.rois_for_plot))+x_spacing,self.rois_for_plot,rotation=45)
                pl.xlim(-0.25,len(self.rois_for_plot))

                for ri, roi in enumerate(self.rois_for_plot):
                    if np.mod(ri,2) == 1:
                        pl.fill_between([ri-x_spacing,ri+1-x_spacing],s.get_ylim()[0],s.get_ylim()[1],color='k',alpha=.05)

                # axis options
                sn.despine(offset=2)
                [pl.plot(-100,1,['o','s','^'][mi],color='k',alpha=0.2,label=['pRF ecc','pRF x','pRF y'][mi]) for mi, measure in enumerate(measures)]
                legend = pl.legend(loc='lower left',frameon=True,fancybox=True)
                pl.ylabel('proportion of shift magnitude')

                pl.axhline(0,c='k',lw=0.5)
                # figure options
                pl.tight_layout(w_pad = 0.0,h_pad=0.0,pad=0)
                # save
                pl.savefig(os.path.join(self.group_plot_dir,'Figure_4','ecc_shift_ratio_%s_%s_R2thresh_%.2f_outliers_rejected_%s.pdf'%(comparison,subject,self.r_squared_threshold,self.detect_outliers)))
                pl.close()

    def pRF_distributions(self,condition='Fix',subjects=['over_subjects','super_subject']):
        """
        This function computes distribution non-uniformity over polar angle
        """

        import pycircstat as pcs

        self.load_data(PRF=True)

        for subject in subjects:
            if subject == 'over_subjects':
                these_subjects = self.subjects
            elif subject == 'super_subject':
                these_subjects = ['super_subject']
            else:
                these_subjects = subjects

                all_zs = []
            for this_subject in these_subjects:
                for ri,roi in enumerate(self.rois_for_plot):    
                    # get mask variables
                    sizes = np.array(self.all_results[this_subject][condition][roi])[:,self.results_frames['sigma_center']] / self.rescale_factor
                    ecc = np.array(self.all_results[this_subject][condition][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                    weights = np.squeeze(self.all_stats[this_subject][condition][roi])

                    # create mask based on ecc, sizes and weights
                    mask = self.functions.create_mask(ecc,self.mask_ecc_thresholds,self.r_squared_threshold,weights,sizes,self.outlier_num_stds,self.size_threshold)

                    # get the fix data
                    fix_data = np.array([np.array(self.all_results[this_subject][condition][roi])[mask,self.results_frames['xo']],
                        np.array(self.all_results[this_subject][condition][roi])[mask,self.results_frames['yo']]]) * self.stim_radius
            
                    # convert to polar angles
                    polar_angles = np.arctan2(fix_data[1,:],fix_data[0,:])
                    # colapse angles, so that double mode is converted to single mode circular distribution
                    angles = self.CUO.collapse_angles_symmetrically(polar_angles)

                    # now we can check whether this distribution is uniform or not
                    ray_p,z = pcs.rayleigh(angles)
                    # if ray_p < .05:
                    # all_ps.append(ray_p)
                    # all_zs.append(z)
                    print '%s %s rayleigh z: %.3f, p%.3f, N: %d'%(this_subject,roi,z,ray_p,len(angles))

    def shift_vector_explained_by_x_y_ecc_over_polar_angle(self,
        comparisons = {'Stim - Fix': ['Stim','Fix']},
        subjects=['super_subject'],
        dist_types = ['x_diff','y_diff','ecc_diff'],
        n_bins = 6,
        data_units=['voxels','arrows'],#'voxels'])#,'voxels'
        n_eccen_bins = 8,
        n_polar_bins = 8,
        location='cond_1',
        min_vox_per_arrow=1,
        ):       

        """
        This function plots the changes in a pRF parameter (x/y/ecc) 
        as a function of quadrant visual field collapsed polar angle.
        """

        self.create_plot_dir('Figure_4')
        regular_data_loaded=False
        self.load_data(PRF=True)
        for subject in subjects:
            if subject == 'over_subjects':
                these_subjects = self.subjects
            else:
                these_subjects = [subject]
            for comparison in comparisons:
                for data_unit in data_units:
                    these_roi_data={}
                    for dist_type in dist_types:
                        these_roi_data[dist_type]={}
                        for ri,roi in enumerate(['combined']):    
                            these_roi_data[dist_type][roi]={}
                            
                            these_mean_x_data = []
                            these_mean_y_data = []
                            these_diffs = []
                            pos_in = []
                            neg_in = []
                            for si, sub_subject in enumerate(these_subjects):
                                
                                print 'computing %s vs polar_angle slopes for %s in %s, sub-subject %s'%(dist_type,subject,roi,sub_subject)

                                # get relevant variables for mask and rescale
                                sizes_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                sizes_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                ecc_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                ecc_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])

                                # now use the mask function
                                mask = self.functions.create_mask_2_conditions(ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights,self.mask_ecc_thresholds,
                                    self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)

                                # r2 same for all conditions, so no matter which one we take
                                weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])[mask]

                                # now, we'll get the vector data for both conditions
                                cond_0 = np.array([np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['xo']],
                                    np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['yo']]]) * self.stim_radius
                                cond_1 = np.array([np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['xo']],
                                    np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['yo']]]) * self.stim_radius
                        
                                # rotate data into quadrant
                                cond_1_shifted = cond_1 * np.sign(cond_1)
                                cond_0_shifted = cond_0 * np.sign(cond_1)

                                # set variables according to desired field
                                cond_1_data = cond_1_shifted
                                cond_0_data = cond_0_shifted
                                polar_range = [0,np.pi/2]

                                if data_unit == 'voxels':

                                    # get the variables of interest
                                    norm_change_vector = np.linalg.norm(cond_0_data-cond_1_data,axis=0)
                                    x_diff = np.abs(cond_0_data[0,:] - cond_1_data[0,:])
                                    y_diff = np.abs(cond_0_data[1,:] - cond_1_data[1,:])
                                    ecc_diff = np.abs(np.linalg.norm(cond_0_data,axis=0)-np.linalg.norm(cond_1_data,axis=0))
                                    polar_angles = np.arctan2(cond_1_data[1,:],cond_1_data[0,:])
                                    stat_type = 'b'

                                elif data_unit == 'arrows':

                                    # get the arrow data
                                    arrow_starts, arrow_diffs = self.functions.get_arrows('quadrant',n_eccen_bins,self.mask_ecc_thresholds,n_polar_bins,
                                                                    cond_0_data,cond_1_data,weights,self.detect_outliers,self.outlier_num_stds,location,min_vox_per_arrow)

                                    # calculate the change in x, y and ecc
                                    x_diff = np.abs(arrow_diffs[:,0])
                                    y_diff = np.abs(arrow_diffs[:,1])
                                    arrow_ends_ecc =  np.linalg.norm(arrow_starts+arrow_diffs,axis=1)
                                    arrow_start_ecc = np.linalg.norm(arrow_starts,axis=1)
                                    ecc_diff = np.abs(arrow_ends_ecc - arrow_start_ecc)
                                    norm_change_vector = np.linalg.norm(arrow_diffs,axis=1)
                                    polar_angles = np.arctan2(arrow_starts[:,1],arrow_starts[:,0])
                                    weights = np.ones(np.shape(x_diff)[0])

                                    stat_type = 't'

                                # get diff as factor of total change vector length
                                x_data = x_diff/norm_change_vector

                                # pick data:                                 
                                data = eval(dist_type)/norm_change_vector

                                mean_diff,se_diff,p_diff, N_diff,cohen_d= self.functions.bootstrap_bin_diff(x_data=polar_angles,y_data=data,y_data2=x_data,weights=weights,n_bins=n_bins,bin_range=polar_range,
                                    outlier_num_stds=self.outlier_num_stds,ci_factor=self.ci_factor,detect_inliers=False,reps=self.reps,stat_type=stat_type)
                                these_diffs.append(mean_diff)

                                pos_in.append((p_diff<.05)*(mean_diff>0))
                                neg_in.append((p_diff<.05)*(mean_diff<0))

                                # bins
                                x_means, x_cis, y_means, y_cis, y_p, N = self.functions.fixed_bins(x_data=polar_angles,y_data=data,weights=weights,n_bins=n_bins,ecc_thresholds=polar_range,
                                    outlier_num_stds=self.outlier_num_stds,ci_factor=self.ci_factor,detect_inliers=False)

                                these_mean_y_data.append(y_means)
                                these_mean_x_data.append(x_means)

                            # then, if we're asking for the average, compute that:
                            if len(these_subjects)>1:
                                if dist_type == 'y_diff':
                                    print '%s %s pos slope diff in %d/%d'%(subject,roi,np.sum(pos_in),len(these_subjects))
                                    print '%s %s neg slope diff  in %d/%d'%(subject,roi,np.sum(neg_in),len(these_subjects))

                                    mean_diff, se_diff, t_diff, p_diff, cohen_d, N_diff = self.CUO.stats_from_data(these_diffs,self.ci_factor)
                                    
                                y_means, y_cis, y_t, y_p, y_d, N = self.CUO.stats_from_data(np.array(these_mean_y_data).T,self.ci_factor)                               
                                x_means, x_cis, x_t, x_p, x_d, N = self.CUO.stats_from_data(np.array(these_mean_x_data).T,self.ci_factor)                               

                            if dist_type == 'y_diff':
                                # append values to array
                                these_roi_data[dist_type][roi]['mean_diff'] = mean_diff
                                these_roi_data[dist_type][roi]['se_diff'] = se_diff
                                these_roi_data[dist_type][roi]['p_diff'] = p_diff

                                if len(these_subjects)==1:
                                    print '%s %s roi=%s, over %s diff = %.3f,N=%d,p=%.3f,d=%.3f'%(subject,dist_type,roi,data_unit,mean_diff,N_diff,p_diff,cohen_d)
                                else:
                                    print '%s %s roi=%s, over %s diff = %.3f,t=%.3f,N=%d,p=%.3f,d=%.3f'%(subject,dist_type,roi,data_unit,mean_diff,t_diff,N_diff,p_diff,cohen_d)
                                    these_roi_data[dist_type][roi]['t_diff'] = t_diff

                            # now for the scatters:
                            these_roi_data[dist_type][roi]['x_means'] = x_means
                            these_roi_data[dist_type][roi]['x_cis'] = x_cis
                            these_roi_data[dist_type][roi]['y_means'] = y_means
                            these_roi_data[dist_type][roi]['y_cis'] = y_cis

                    # only fdr correct for super subject
                    if subject == 'super_subject':
                        all_ps = []
                        for mi,mc in enumerate(['y_diff']):
                            all_ps.append(these_roi_data[mc]['combined']['p_diff'])
                        rejections,corrected_ps = mne.stats.fdr_correction(all_ps,alpha=.05)
                        for mi,mc in enumerate(['y_diff']):
                            these_roi_data[mc]['combined']['p_diff'] = corrected_ps[mi]

                    f = pl.figure(figsize=(1,1))               
                    s=f.add_subplot(1,1,1)
                    for mi,mc in enumerate(['y_diff']):
                        pl.axhline(0,color='k',lw=0.5)
                        mean_diff = these_roi_data[mc]['combined']['mean_diff']
                        se_diff = these_roi_data[mc]['combined']['se_diff']
                        p_diff = these_roi_data[mc]['combined']['p_diff']

                        if p_diff < .05:
                            ecolor = 'w'
                            color = self.roi_colors[-1]
                        else:
                            color = 'w'
                            ecolor = self.roi_colors[-1]

                        pl.plot([mi,mi],se_diff,ls='-',lw=0.5,color=self.roi_colors[-1])
                        pl.plot(mi,mean_diff,'o',ms=6,mew=0.5,mec=ecolor,color=color)
                        sn.despine(offset=2)

                    yticks = self.functions.find_yticks(s.get_ylim())   
                    # pl.yticks(yticks[0],yticks[1])
                    pl.yticks([0,.5],['0','.5'])
                    pl.xticks([])
                    pl.xlim(-0.5,0.5)
                    # figure options
                    pl.tight_layout(pad=0,h_pad=0,w_pad=0)
                    # save
                    pl.savefig(os.path.join(self.group_plot_dir,'Figure_4','%s_%s_%s_slopes.pdf'%(subject,data_unit,comparison)))
                    pl.close()

                    f = pl.figure(figsize=(0.75,1.6))               
                    for mi,mc in enumerate(dist_types):

                        s=f.add_subplot(3,1,mi+1)
                        sn.despine(offset=2)
                        pl.axhline(0,color='k',lw=0.5)
                        pl.axvline(0,color='k',lw=0.5)

                        x_data = these_roi_data[mc]['combined']['x_means']
                        x_ci = these_roi_data[mc]['combined']['x_cis']
                        y_data = these_roi_data[mc]['combined']['y_means']
                        y_ci = these_roi_data[mc]['combined']['y_cis']

                        nandata = ~np.isnan(x_data)
                        y_ci = np.array([ci if ci!=[] else [np.nan,np.nan] for i,ci in enumerate(y_ci)])

                        color = self.roi_colors[-1]
                        pl.plot(x_data[nandata],y_data[nandata],color=color,lw=1,alpha=1)
                        # try:
                        try:
                            pl.fill_between(x_data[nandata],y_ci[:,0][nandata],y_ci[:,1][nandata],color=color,alpha=0.25)
                        except:
                            pl.fill_between(x_data[nandata],y_ci[:,0],y_ci[:,1],color=color,alpha=0.25)

                        # now for the subplot options
                        if data_unit == 'voxels':
                            if (subject == 'super_subject') + (subject == 'over_subjects'):
                                lims = [0.55,0.75]
                                ticks = ['.55','.75']
                            else:
                                lims = [0.4,.8]
                                ticks = ['.4','.8']
                        elif data_unit == 'arrows':
                            if (subject == 'super_subject') + (subject == 'over_subjects'):
                                lims = [.3,1]
                                ticks = ['.3','1']
                            else:
                                lims = [0.2,1]
                                ticks = ['.2','1']

                        pl.ylim(lims)
                        pl.xlim(polar_range)
                        pl.yticks(lims,ticks)
                        pl.xticks([])#polar_range)

                    # figure options
                    pl.tight_layout(pad=0,h_pad=0,w_pad=0)
                    # save
                    pl.savefig(os.path.join(self.group_plot_dir,'Figure_4','%s_%s_%s.pdf'%(subject,data_unit,comparison)))
                    pl.close()

    def diff_over_ecc(self,
        comparisons = {'Stim - Fix': ['Stim','Fix']},#'Color - Fix': ['Color','Fix'],'Speed - Fix': ['Speed','Fix']},#,'Color - Speed':['Color','Speed']},,
        measures=['ecc','size'],#,'amp_center'],#,'size'],
        n_bins=5,
        bin_type='fixed',
        diff_type = 'abs',
        subjects = ['DE','JS','NA','TK','JW','super_subject'],
        over = 'ecc',
        sig_test = False,
        ):

        if ('feature_pref' in measures) + ('fami' in measures):
            figno = 'Figure_8'
        else:
            figno = 'Figure_5'

        self.create_plot_dir(figno)

        self.load_data(PRF=True,Mapper=True)
        for subject in subjects:
            if subject == 'over_subjects':
                these_subjects = self.subjects
            else:
                these_subjects = [subject]     

            for measure in measures:
                # let's loop over comparisons 
                for ci, comparison in enumerate(comparisons.keys()):

                    all_roi_data = {}
                    all_ps = []
                    all_p_indices = []
                    all_Ns = []
                    all_ds = []

                    # loop over roi groups
                    for rgi,roi_group in enumerate(np.sort(self.roi_groups_for_plot.keys())):

                        # then loop over all subrois
                        for subri, roi in enumerate(self.roi_groups_for_plot[roi_group]):
                            all_roi_data[roi] = {}
                            
                            all_mean_x_data = []
                            all_mean_y_data = []

                            # print ''

                            pos_in = []
                            neg_in = []
                            Ns = []
                            # then loop over subjects
                            for si, sub_subject in enumerate(these_subjects):

                                # sys.stdout.write('Computing %s difference in %s for %s, subject %s\r'%(measure,roi,comparison,subject))
                                # sys.stdout.flush()

                                # get relevant variables for mask and rescale
                                sizes_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                sizes_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                ecc_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                ecc_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])

                                # now use the mask function
                                mask = self.functions.create_mask_2_conditions(ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights,self.mask_ecc_thresholds,
                                    self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)

                                weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][1]][roi])[mask]

                                # get amps
                                amps_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['amp_center']]
                                amps_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['amp_center']]

                                # get data masked
                                cond_0_xo = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['xo']] * self.stim_radius
                                cond_0_yo = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['yo']] * self.stim_radius 
                                cond_0_data = np.array([cond_0_xo,cond_0_yo])           
                                cond_1_xo = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['xo']] * self.stim_radius
                                cond_1_yo = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['yo']] * self.stim_radius
                                cond_1_data = np.array([cond_1_xo,cond_1_yo])   
                                                            
                                # get eccen values for bins
                                cond_0_eccen = np.linalg.norm(cond_0_data,axis=0)
                                cond_1_eccen = np.linalg.norm(cond_1_data,axis=0)

                                # this is for computing the fami
                                these_data = {}
                                for condition in ['Stim','Color','Speed','Fix']:
                                    these_data[condition] = {}
                                    for m in ['sigma_center','ecc']:
                                        these_data[condition][m] = np.array(self.all_results[sub_subject][condition][roi])[mask,self.results_frames[m]]/self.rescale_factor
                                col,speed,fami = self.functions.compute_FBA(these_data,weights,['sigma_center','ecc'],self.outlier_num_stds)#,diff_type)

                                # get fix ecc
                                if over == 'ecc':
                                    fix_ecc = ecc_cond_1[mask]
                                    bin_lims = self.mask_ecc_thresholds
                                elif over == 'size':
                                    fix_ecc = sizes_cond_1[mask]
                                    bin_lims = [0,np.max(fix_ecc)]
                                elif over == 'feature_pref':
                                    fix_ecc = np.array(self.all_mapper[sub_subject]['copes'][roi])[mask,10]
                                    bin_lims = [np.min(fix_ecc),np.max(fix_ecc)]

                                if measure == 'r_squared':
                                    diff = weights
                                    weights = np.ones(len(weights))
                                else:
                                    # add all measures to dict:
                                    # cond_0_ecc = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['ecc']]/self.rescale_factor
                                    # cond_1_ecc = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['ecc']]/self.rescale_factor
                                    ecc_diff = cond_0_eccen - cond_1_eccen
                                    cond_0_size = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames['sigma_center']]/self.rescale_factor
                                    cond_1_size = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames['sigma_center']]/self.rescale_factor
                                    size_diff = cond_0_size - cond_1_size
                                    combined_diff = np.linalg.norm([ecc_diff,size_diff],axis=0)


                                    if measure == 'ecc':
                                        diff = ecc_diff
                                    elif measure == 'count':
                                        diff = weights
                                        weights = np.ones(mask.sum())
                                    elif measure == 'fix_size':
                                        diff = cond_1_size
                                    elif measure == 'size':
                                        diff = size_diff
                                    elif measure == 'combined':
                                        diff = ami
                                    elif measure == 'amp_center':
                                        diff = amps_cond_0 - amps_cond_1
                                    elif measure == 'fami':
                                        diff = fami
                                    elif measure == 'feature_pref':
                                        diff = np.array(self.all_mapper[sub_subject]['copes'][roi])[mask,10]
                                    elif measure == 'feature_pref_z':
                                        diff = np.array(self.all_mapper[sub_subject]['z_stat'][roi])[mask,10]

                                # get binned data
                                if bin_type == 'percentile':
                                    x_means, x_cis, y_means, y_cis, y_p,Ns,ds = self.functions.percentile_bins(y_data=diff,x_data=fix_ecc,weights=weights,n_bins=n_bins,
                                        outlier_num_stds=self.outlier_num_stds,ci_factor=self.ci_factor,detect_inliers=self.detect_outliers)
                                elif bin_type == 'fixed':
                                    x_means, x_cis, y_means, y_cis, y_p,Ns,ds = self.functions.fixed_bins(y_data=diff,x_data=fix_ecc,weights=weights,n_bins=n_bins,
                                        ecc_thresholds=bin_lims,outlier_num_stds=self.outlier_num_stds,ci_factor=self.ci_factor,detect_inliers=self.detect_outliers,return_d=True)

                                pos_in.append((y_p<.05)*(y_means>0))
                                neg_in.append((y_p<.05)*(y_means<0))

                                # and append to group variable
                                all_mean_x_data.append(x_means)
                                all_mean_y_data.append(y_means)

                            # then average over hemispheres and subjects if not using super_subject
                            if len(these_subjects) > 1:
                                # shell()                               

                                y_means, y_cis, y_ts, y_p, ds, Ns = self.CUO.stats_from_data(np.array(all_mean_y_data).T,self.ci_factor)  
                                x_means, x_cis, x_ts, x_p, ds, Ns = self.CUO.stats_from_data(np.array(all_mean_x_data).T,self.ci_factor)                     
                                
                                print '%s %s pos in %s/%s'%(subject,roi,np.sum(pos_in,axis=0),Ns)
                                print '%s %s neg in %s/%s'%(subject,roi,np.sum(neg_in,axis=0),Ns)                                

                                all_roi_data[roi]['ts'] = y_ts 


                            all_roi_data[roi]['x_means'] = x_means
                            all_roi_data[roi]['y_mean'] = y_means
                            all_roi_data[roi]['cis'] = y_cis
                            all_roi_data[roi]['ps'] = y_p 
                            all_roi_data[roi]['Ns'] = Ns 
                            all_roi_data[roi]['ds'] = ds 
                            all_Ns.append(Ns)  
                            all_ps.append(y_p)
                            all_p_indices.append(roi)
                            all_ds.append(ds)

                    if subject == 'super_subject':
                        # determine FDR
                        rejections,corrected_ps = mne.stats.fdr_correction(all_ps,alpha=.05)
                        print 'alpha .05 FDR corrections: %s'%rejections
                        rejections_01,corrected_ps_01 = mne.stats.fdr_correction(all_ps,alpha=.01)
                        print 'alpha .01 FDR corrections: %s'%rejections_01
                        rejections_001,corrected_ps_001 = mne.stats.fdr_correction(all_ps,alpha=.001)
                        print 'alpha .001 FDR corrections: %s'%rejections_001
                        # # put back in dict
                        for pi, rej in enumerate(rejections):
                            all_roi_data[all_p_indices[pi]]['rej'] = rej
                            all_roi_data[all_p_indices[pi]]['p_corr'] = corrected_ps[pi]
                            decimal_ps = ['%.3f'%float(p) for p in all_ps[pi]]
                            decimal_ds = ['%.2f'%float(d) for d in all_ds[pi]]
                            # print '%s, Ns=%s, ps=%s'%(all_p_indices[pi],all_Ns[pi],decimal_ps)#,decimal_ds)

                    f=pl.figure(figsize = (3.75,1))
                    # loop over roi groups
                    ri = -1
                    for rgi,roi_group in enumerate(np.sort(self.roi_groups_for_plot.keys())):
                        # subplot per roi group
                        s = f.add_subplot(self.roi_group_subplot_grid[0],self.roi_group_subplot_grid[1],rgi+1,adjustable='box-forced',)
                        if sig_test:
                            pl.axhline(0,c='k',lw=0.5,ls='-')
                        # then loop over all subrois
                        # this_roi_diffs = []
                        for this_ri, roi in enumerate(self.roi_groups_for_plot[roi_group]):
                            ri +=1
                            x_means = all_roi_data[roi]['x_means']
                            y_means = all_roi_data[roi]['y_mean']
                            y_cis = all_roi_data[roi]['cis']
                            Ns = all_roi_data[roi]['Ns']
                            ds = all_roi_data[roi]['ds']
                            ps = all_roi_data[roi]['ps']

                            if subject == 'over_subjects':
                                ts = all_roi_data[roi]['ts']

                                print '%s %s diffs: %s p: %s Ns: %s ds:%s ts:%s'%(subject,roi,['%.3f'%y_mean for y_mean in y_means],['%.3f'%p for p in ps],Ns,['%.3f'%d for d in ds],['%.3f'%t for t in ts])
                            else:
                                print '%s %s diffs: %s p: %s Ns: %s ds:%s'%(subject,roi,['%.3f'%y_mean for y_mean in y_means],['%.3f'%p for p in ps],Ns,['%.3f'%d for d in ds])

                            min_n = 1
                            x_means[Ns<min_n] = np.nan
                            y_means[Ns<min_n] = np.nan
                            y_cis = [ci if Ns[i]>min_n else [np.nan,np.nan] for i,ci in enumerate(y_cis)]
                            # plot line between dots
                            pl.plot(x_means,y_means,ls='-',lw = 1,ms=0,c= self.roi_colors[ri])#,mec= self.roi_colors[ri])
                            # loop over bins
                            for bi in range(n_bins):
                                fill_color = self.roi_colors[ri]
                                marker_edge_color = 'w'
                                if sig_test:
                                    if subject =='super_subject':
                                        if (all_roi_data[roi]['rej'][bi]):
                                            ms = 6
                                        else:
                                            ms = 2
                                    else:
                                        if (ps[bi]<.05):
                                            ms = 6
                                        else:
                                            ms = 2
                                else:
                                    ms = 6

                                # if Ns[bi] > min_n:
                                # y_cis = [ci if ci!=[] else [np.nan,np.nan] for ci in y_cis]
                                pl.plot(x_means[bi],y_means[bi],ms=ms,marker=['o','s','^'][this_ri],c= fill_color,mec= marker_edge_color,mew=0.5,label=roi)
                                pl.plot([x_means[bi],x_means[bi]],y_cis[bi],linestyle='-',lw = 0.75,color = 'w')
                                pl.plot([x_means[bi],x_means[bi]],y_cis[bi],linestyle='-',lw = 0.75,color = self.roi_colors[ri],alpha=0.5)


                        sn.despine(offset=2)
                        if rgi==0:
                            if measure in ['ecc','size']:
                                pl.ylabel(r'$\Delta$ %s (dva)'%(measure))
                            elif measure =='combined':
                                pl.ylabel('ami')
                            elif measure == 'fami':
                                pl.ylabel('fami')
                            else:
                                pl.ylabel(measure)

                        if over == 'ecc':
                            pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])
                            pl.xlim(self.mask_ecc_thresholds)
                        else:
                            pl.xticks(self.functions.find_yticks(s.get_xlim())[0],self.functions.find_yticks(s.get_xlim())[1])

                        if measure == 'ecc':
                            if subject == 'super_subject':
                                ylim = [0.1,0.7,1.4,0.1][rgi]
                            elif subject == 'over_subjects':
                                ylim = [0.2,1.4,2.8,0.2][rgi]
                            else:
                                ylim = [0.2,3.5,4,0.25][rgi]                            
                            pl.ylim(-ylim,ylim)
                            pl.yticks([-ylim,0,ylim],[str(-ylim),'0',str(ylim)])
                        elif measure == 'size':
                            if subject == 'super_subject':
                                ylim = [0.1,0.3,1.1,0.1][rgi]
                            elif subject == 'over_subjects':
                                ylim = [0.2,0.6,2.5,0.1][rgi]
                            else:
                                ylim = [0.15,1.5,2.5,0.1][rgi]                                   
                            pl.ylim(-ylim,ylim)
                            pl.yticks([-ylim,0,ylim],[str(-ylim),'0',str(ylim)])
                        # elif (measure == 'feature_pref') * (over == 'ecc')
                        #     ylim = 
                        #     pl.ylim(-ylim,ylim)
                        #     pl.yticks([-ylim,0,ylim],[str(-ylim),'0',str(ylim)])
                        else:
                            pl.yticks(self.functions.find_yticks(s.get_ylim())[0],self.functions.find_yticks(s.get_ylim())[1])

                    # figure options 
                    pl.tight_layout(w_pad = 0.,h_pad=0.0,pad=0)
                    # save
                    pl.savefig(os.path.join(self.group_plot_dir,figno,'%s_%s_%s_over_Fix_%s_%s_%.2f_outlier_rejection_%s.pdf'%(comparison,measure,diff_type,over,subject,self.r_squared_threshold,self.detect_outliers)))
                    pl.close()

    def correlate_diffs(self,
        comparisons = {'Color - Fix': ['Color','Fix'],'Speed - Fix':['Speed','Fix'],'Speed - Color':['Speed','Color']},
        ecc_size_relation_condition='Fix',
        subjects = ['DE','NA','JS','TK','JW'],
        n_bins=20,
        target_regions=['IPS0'],
        stat_methods=['over_subjects'],
        correlation_types = ['pearson','spearman'],
        measure_comparisons = [['ecc','size'],['ecc','amp_center'],['size','amp_center']],
        corr_overs = ['voxels','bins']):
        """
        This function computes correlation between PRF ecc and size changes. 
        """

        self.create_plot_dir('Figure_5')
        self.load_data(PRF=True)
        for subject in subjects:
            if subject == 'over_subjects':
                these_subjects = self.subjects
            else:
                these_subjects = [subject]

            for corr_over in corr_overs:
                for correlation_type in correlation_types:
                    # now let's loop over comparisons and plot the size_diff over ecc_diff with the ecc-size relations of every roi over it 
                    all_roi_data = {}
                    all_ps = []
                    all_Ns = []
                    all_corrs = []
                    all_p_indices = []
                    each_corr = []
                    each_roi = []
                    each_sub_id = []

                    for comparison in comparisons:
                        all_roi_data[comparison] = {}

                        for measure_comparison in measure_comparisons:
                            all_roi_data[comparison][measure_comparison] = {}

                            for ri,roi in enumerate(self.rois_for_plot):    
                                all_roi_data[comparison][measure_comparison][roi] = {}

                                these_corrs = []
                                these_slopes = []
                                these_intercepts = []
                                these_es_slopes = []
                                these_es_intercepts = []                                
                                these_mean_x_data = []
                                these_mean_y_data = []                  
                                # loop over hemispheres if wanted
                                print ''

                                pos_in=[]
                                neg_in=[]

                                for si, sub_subject in enumerate(these_subjects):

                                    sys.stdout.write('Correlating %s diffs for %s, %s, sub_subject %d/%d\r'%(measure_comparison,comparison,roi,si+1,len(these_subjects)))
                                    sys.stdout.flush()
                                    # print 'Creating size-diff vs ecc-diff correlation plot for %s, %s, sub_subject %d/%d'%(comparison,roi,si+1,len(these_subjects))#,diff_type)
                                    
                                    # get relevant variables for mask and rescale
                                    sizes_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                    sizes_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                    ecc_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                    ecc_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                    weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])

                                    # now use the mask function
                                    mask = self.functions.create_mask_2_conditions(ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights,self.mask_ecc_thresholds,
                                        self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)                  

                                    weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][1]][roi])[mask]

                                    # now get the differences for both measures
                                    measure_0_diff = (np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames[measure_comparisons[measure_comparison][0]]] -
                                        np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames[measure_comparisons[measure_comparison][0]]])
                                    # rescale when measure is not amp_center:
                                    if measure_comparisons[measure_comparison][0] in ['ecc','sigma_center']:
                                        measure_0_diff /= self.rescale_factor
                                    
                                    # second measure
                                    measure_1_diff = (np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[mask,self.results_frames[measure_comparisons[measure_comparison][1]]] - 
                                        np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[mask,self.results_frames[measure_comparisons[measure_comparison][1]]]) 
                                    # rescale when measure is not amp_center:
                                    if measure_comparisons[measure_comparison][1] in ['ecc','sigma_center']:
                                        measure_1_diff /= self.rescale_factor

                                    # now detect outliers if wanted
                                    if self.detect_outliers:
                                        measure_0_inliers = self.CUO.detect_inliers_mad(measure_0_diff)
                                        measure_1_inliers = self.CUO.detect_inliers_mad(measure_0_diff)
                                        inliers = measure_0_inliers * measure_1_inliers
                                        measure_0_diff = measure_0_diff[inliers]
                                        measure_1_diff = measure_1_diff[inliers]
                                        weights = weights[inliers]
                                    # compute percentile bins for both changes for visualisation purposes
                                    x_means, x_cis, y_means, y_cis, y_p, Ns, ds= self.functions.percentile_bins(x_data=measure_0_diff,y_data=measure_1_diff,weights=weights,n_bins=n_bins,
                                        outlier_num_stds=self.outlier_num_stds,ci_factor=self.ci_factor,detect_inliers=self.detect_outliers)
                                    these_mean_y_data.append(y_means)
                                    these_mean_x_data.append(x_means)

                                    # compute linear fit for this participant
                                    if corr_over == 'bins':
                                        corr_x_data = x_means
                                        corr_y_data = y_means
                                        detect_further_outliers = False
                                        weights = None
                                    elif corr_over == 'voxels':
                                        corr_x_data = measure_0_diff
                                        corr_y_data = measure_1_diff
                                        detect_further_outliers = True

                                    mean_slope, se_slope, p_slope,mean_intercept, se_intercept, p_intercept = self.CUO.bootstrap_linear_fit(corr_x_data,corr_y_data,
                                        weights=weights,ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=detect_further_outliers,outlier_num_stds=self.outlier_num_stds)
                                    these_slopes.append(mean_slope)
                                    these_intercepts.append(mean_intercept)

                                    # mean_es_slope, se_es_slope, p_es_slope,mean_es_intercept, se_es_intercept, p_es_intercept = self.CUO.bootstrap_linear_fit(ecc_cond_1[mask],sizes_cond_1[mask],
                                    #     weights=weights,ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=detect_further_outliers,outlier_num_stds=self.outlier_num_stds)
                                    # these_es_slopes.append(mean_es_slope)
                                    # these_es_intercepts.append(mean_es_intercept)

                                    # compute pearson correlation per participant
                                    if corr_over == 'voxels':
                                        mean_corr, se_corr, p_corr, N_corr = self.CUO.bootstrap_correlation(corr_x_data,corr_y_data,weights=weights,corr_type=correlation_type,
                                            ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=detect_further_outliers,outlier_num_stds=self.outlier_num_stds)
                                    elif corr_over == 'bins':
                                        if correlation_type == 'pearson':
                                            mean_corr, p_corr = sp.stats.pearsonr(corr_x_data,corr_y_data)
                                        elif correlation_type == 'spearman':
                                            mean_corr, p_corr = sp.stats.spearmanr(corr_x_data,corr_y_data)
                                        N_corr = n_bins

                                    these_corrs.append(mean_corr)
                                    # print 'roi %s, %s correlation: R=%.2f,p=%.4f'%(roi,measure_comparison,mean_corr,p_corr)

                                    pos_in.append((p_corr<.05)*(mean_corr>0))
                                    neg_in.append((p_corr<.05)*(mean_corr<0))

                                    if (subject == 'over_subjects') * (roi != 'combined'):
                                        each_corr.append(mean_corr)
                                        each_roi.append(roi)
                                        each_sub_id.append(sub_subject)


                                # then, if we're asking for the average, compute that:
                                if len(these_subjects)>1:
                                    print '%s %s,corr pos in %d/%d subs'%(subject,roi,np.sum(pos_in),len(these_subjects))
                                    print '%s %s,corr neg in %d/%d subs'%(subject,roi,np.sum(neg_in),len(these_subjects))

                                    mean_intercept, se_intercept, t, p_intercept, cohen_d, N = self.CUO.stats_from_data(these_intercepts,self.ci_factor)
                                    mean_slope, se_slope, t, p_slope, cohen_d, N = self.CUO.stats_from_data(these_slopes,self.ci_factor)
                                    mean_corr, se_corr, t_corr, p_corr, cohen_d, N_corr = self.CUO.stats_from_data(these_corrs,self.ci_factor)
                                    # re-calculate t and p on fisher transformed correlations:
                                    t_corr, p_corr = sp.stats.ttest_1samp(np.arctanh(np.array(these_corrs)),0)

                                    y_means, y_cis, y_t, y_p, y_d, N = self.CUO.stats_from_data(np.array(these_mean_y_data).T,self.ci_factor)                               
                                    x_means, x_cis, x_t, x_p, x_d, N = self.CUO.stats_from_data(np.array(these_mean_x_data).T,self.ci_factor)                               

                                    print '%s %s, corr: %.3f, t(%d): %.3f, p: %.3f'%(subject,roi,mean_corr,N_corr,t_corr,p_corr)

                                # append corr values
                                all_roi_data[comparison][measure_comparison][roi]['mean_corr'] = mean_corr
                                # all_roi_data[comparison][measure_comparison][roi]['se_corr'] = se_corr
                                all_roi_data[comparison][measure_comparison][roi]['p'] = p_corr
                                if len(these_subjects)>1:
                                    all_roi_data[comparison][measure_comparison][roi]['t'] = t_corr
                                all_ps.append(p_corr)
                                all_Ns.append(N_corr)
                                all_corrs.append(mean_corr)
                                all_p_indices.append([comparison,measure_comparison,roi])
                                # append slope values
                                all_roi_data[comparison][measure_comparison][roi]['mean_slope'] = mean_slope
                                all_roi_data[comparison][measure_comparison][roi]['se_slope'] = se_slope
                                # all_roi_data[comparison][measure_comparison][roi]['mean_es_slope'] = mean_es_slope
                                # all_roi_data[comparison][measure_comparison][roi]['se_es_slope'] = se_es_slope
                                # append intercept values
                                all_roi_data[comparison][measure_comparison][roi]['mean_intercept'] = mean_intercept
                                all_roi_data[comparison][measure_comparison][roi]['se_intercept'] = se_intercept
                                # all_roi_data[comparison][measure_comparison][roi]['mean_es_intercept'] = mean_es_intercept
                                # all_roi_data[comparison][measure_comparison][roi]['se_es_intercept'] = se_es_intercept
                                # append x for scatter
                                all_roi_data[comparison][measure_comparison][roi]['x_means'] = x_means
                                all_roi_data[comparison][measure_comparison][roi]['x_cis'] = x_cis
                                # append y for scatter
                                all_roi_data[comparison][measure_comparison][roi]['y_means'] = y_means
                                all_roi_data[comparison][measure_comparison][roi]['y_cis'] = y_cis

                    # determine FDR
                    if subject == 'super_subject':
                        rejections05,corrected_ps05 = mne.stats.fdr_correction(all_ps,alpha=.05)
                        rejections01,corrected_ps01 = mne.stats.fdr_correction(all_ps,alpha=.01)
                        rejections001,corrected_ps001 = mne.stats.fdr_correction(all_ps,alpha=.001)

                        corr_d = ['%.3f'%c for c in all_corrs]
                        p_d = ['%.3f'%p for p in all_ps]
                        print 'Ns:%s\nps:%s\nrej05:%s\nrej01:%s\nrej001:%s\ncorrs:%s'%(all_Ns,p_d,np.array(self.rois_for_plot)[rejections05],np.array(self.rois_for_plot)[rejections01],np.array(self.rois_for_plot)[rejections001],corr_d)

                    # first, let's create a scatter plot of the comparisons
                    for comparison in comparisons:

                        for mi,mc in enumerate(measure_comparisons):

                            f = pl.figure(figsize=(3.6,1.6))                
                            for ri, roi in enumerate(self.rois_for_plot):

                                s = f.add_subplot(2,5,ri+1)
                                pl.title(roi)
                                sn.despine(offset=2)
                                pl.axhline(0,color='k',lw=0.25)
                                pl.axvline(0,color='k',lw=0.25)

                                x_data = all_roi_data[comparison][mc][roi]['x_means']
                                x_ci = all_roi_data[comparison][mc][roi]['x_cis']
                                y_data = all_roi_data[comparison][mc][roi]['y_means']
                                y_ci = all_roi_data[comparison][mc][roi]['y_cis']

                                # first plot the dots
                                pl.plot(x_data,y_data,'o',ms=4,color=self.roi_colors[ri],mec='w',mew=0.5)
                                # then error bars
                                for bi in range(n_bins):
                                    pl.plot([x_data[bi],x_data[bi]],y_ci[bi],lw=0.5,color='w')
                                    # pl.plot(x_ci[bi],[y_data[bi],y_data[bi]],lw=0.5,color='w')
                                    pl.plot([x_data[bi],x_data[bi]],y_ci[bi],lw=0.5,color=self.roi_colors[ri],alpha=0.5)
                                    # pl.plot(x_ci[bi],[y_data[bi],y_data[bi]],lw=0.5,color=self.roi_colors[ri],alpha=0.5)

                                # now add line on top
                                intercept = all_roi_data[comparison][mc][roi]['mean_intercept']
                                slope = all_roi_data[comparison][mc][roi]['mean_slope']
                                fit_fn = np.poly1d([slope,intercept])               
                                pl.plot([np.min(x_data),np.max(x_data)],[fit_fn(np.min(x_data)),fit_fn(np.max(x_data))],lw=1,color=self.roi_colors[ri])

                                # # and add shaded region
                                intercept_ci = all_roi_data[comparison][mc][roi]['se_intercept']
                                slope_ci = all_roi_data[comparison][mc][roi]['se_slope']
                                fit_fn_low = np.poly1d([slope_ci[0],intercept_ci[0]])             
                                fit_fn_high = np.poly1d([slope_ci[1],intercept_ci[1]])                

                                pl.fill_between([np.min(x_data),np.max(x_data)],
                                    [fit_fn_low(np.min(x_data)),fit_fn_low(np.max(x_data))],
                                    [fit_fn_high(np.min(x_data)),fit_fn_high(np.max(x_data))],
                                    color=self.roi_colors[ri],alpha=0.2)
                                
                                # now for the subplot options
                                if subject in ['super_subject','over_subjects']:
                                    yticks = self.functions.find_yticks(s.get_ylim())
                                    xticks = self.functions.find_yticks(s.get_xlim())
                                else:
                                    yticks = [
                                        [[-0.1,0,0.1],['-.1','0','.1']],        #V1
                                        [[-0.1,0,0.07],['-.1','0','.07']],    #V2
                                        [[-0.2,0,0.1],['-.2','0','.1']],        #V3
                                        [[-0.3,0,0.3],['-.3','0','.3']],        #V4
                                        [[-0.7,0,1.0],['-.7','0','1']],        #VO
                                        [[-0.6,0,0.8],['-.6','0','.8']],        #LO
                                        [[-0.4,0,0.8],['-.4','0','.8']],        #V3AB
                                        [[-1.0,0,2.0],['-1','0','2']],        #IPS0
                                        [[-1.0,0,3.0],['-1','0','3']],        #MT+
                                        [[-.15,0,0.15],['-.15','0','.15']],        #combined
                                        ][ri]                                    
                                
                                    xticks = [
                                        [[-0.35,0,0.2],['-.35','0','.2']],        #V1
                                        [[-0.3,0,0.2],['-.3','0','.2']],    #V2
                                        [[-0.3,0,0.6],['-.3','0','.6']],        #V3
                                        [[-0.3,0,1.2],['-.3','0','1.2']],        #V4
                                        [[-.5,0,2.0],['-0.5','0','2']],        #VO
                                        [[-0.4,0,1.4],['-.4','0','1.4']],        #LO
                                        [[-.6,0,2.5],['-.6','0','2.5']],        #V3AB
                                        [[-.5,0,3.0],['-.5','0','3']],        #IPS0
                                        [[-0.5,0,4.0],['-.5','0','4']],        #MT+
                                        [[-.4,0,1.0],['-.4','0','1']],        #combined
                                        ][ri]            

                                pl.yticks(yticks[0],yticks[1])
                                pl.xticks(xticks[0],xticks[1])

                                if roi not in ['VO','V3AB','IPS0','MT+']:
                                    pl.xticks(self.functions.find_yticks(s.get_xlim())[0],self.functions.find_yticks(s.get_xlim())[1])
                                else:
                                    pl.xticks(self.functions.find_yticks(s.get_xlim())[0][1:],self.functions.find_yticks(s.get_xlim())[1][1:])


                                if ri == 5:
                                    pl.xlabel(r'$\Delta$ '+ mc.split('-')[0])
                                    pl.ylabel(r'$\Delta$ '+mc.split('-')[1])

                            # figure options
                            pl.tight_layout(pad=0,h_pad=0,w_pad=0)
                            # save
                            pl.savefig(os.path.join(self.group_plot_dir,'Figure_5','%s_%s_%s_scatter_R2thresh_%.2f_outliers_rejected_%s_corr_over_%s.pdf'%(comparison,mc,subject,self.r_squared_threshold,self.detect_outliers,corr_over)))
                            pl.close()


    def diff_over_ecc_col_tf(self,
        comparisons = {'Stim - Fix': ['Stim','Fix']},
        measures=['ecc','size','ami'],
        n_bins=5,
        bin_type='fixed',
        diff_type = 'abs',
        # stat_methods=['over_subjects','super_subject','over_super_subjects'],
        subjects = ['DE','JS','NA','TK','JW','super_subject'],
        plot_types = ['per_roi','across_rois']):#these_rois=['combined']):

        self.create_plot_dir('Figure_7')    
        self.load_data(PRF=True)

        for plot_type in plot_types:

            for subject in subjects:
                if subject == 'over_subjects':
                    these_subjects = self.subjects
                else:
                    these_subjects = [subject]     
                
                if subject == 'super_subject': 
                    if plot_type == 'per_roi':
                        these_rois = self.rois_for_plot
                        these_colors = self.roi_colors
                    elif plot_type == 'across_rois':
                        these_rois = self.rois_for_plot[:-1]
                        these_colors = self.roi_colors[:-1]
                else:
                    if plot_type == 'per_roi':
                        these_rois = [self.rois_for_plot[-1]]
                        these_colors = [self.roi_colors[-1]]
                    elif plot_type == 'across_rois':
                        break

                # save all ps, so we can do FDR them
                all_roi_data = {}
                all_ps = []
                all_p_indices = []  

                for measure in measures:
                    all_roi_data[measure] = {}

                    # loop over rois
                    for ri,roi in enumerate(these_rois):
                        all_roi_data[measure][roi] = {}


                        for ci, comparison in enumerate(comparisons.keys()):
                            # and initialize group lvl variables
                            all_mean_x_data = []
                            all_mean_y_data = []
                            all_roi_data[measure][roi][comparison] = {}

                            print ''

                            # then loop over subjects
                            for si, sub_subject in enumerate(these_subjects):
                                sys.stdout.write('Computing %s difference in %s for %s, subject %d/%d\r'%(measure,roi,comparison,si+1,len(these_subjects)))
                                sys.stdout.flush()  

                                # get relevant variables for mask and rescale
                                sizes_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                sizes_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                ecc_cond_0 = np.array(self.all_results[sub_subject][comparisons[comparison][0]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                ecc_cond_1 = np.array(self.all_results[sub_subject][comparisons[comparison][1]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                weights = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][0]][roi])

                                # now use the mask function
                                mask = self.functions.create_mask_2_conditions(ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights,self.mask_ecc_thresholds,
                                    self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)

                                # mask weights 
                                weights_masked = np.squeeze(self.all_stats[sub_subject][comparisons[comparison][1]][roi])[mask]
                                
                                # get fix ecc
                                fix_ecc = ecc_cond_1[mask]

                                # get differences
                                ecc = ecc_cond_0[mask] - ecc_cond_1[mask]
                                size = sizes_cond_0[mask] - sizes_cond_1[mask]
                                # combined = np.linalg.norm([ecc_cond_0[mask] - ecc_cond_1[mask]sizes_cond_0[mask] - sizes_cond_1[mask]],axis=0)

                                if measure == 'ecc':
                                    diff = ecc
                                elif measure == 'size':
                                    diff = size
                                elif 'ami' in measure:
                                    # this is for computing the fami
                                    these_data = {}
                                    for condition in ['Stim','Color','Speed','Fix']:
                                        these_data[condition] = {}
                                        for m in ['sigma_center','ecc']:
                                            these_data[condition][m] = np.array(self.all_results[sub_subject][condition][roi])[mask,self.results_frames[m]]/self.rescale_factor
                                    col,speed,fami = self.functions.compute_FBA(these_data,weights_masked,['sigma_center','ecc'],self.outlier_num_stds)
                                    if measure == 'ami':
                                        if 'Speed' == comparisons[comparison][0]:
                                            diff = speed
                                        elif 'Color' == comparisons[comparison][0]:
                                            diff = col
                                    elif measure == 'fami':
                                        diff = fami

                                x_means, x_cis, y_means, y_cis, y_p,N= self.functions.fixed_bins(y_data=diff,x_data=fix_ecc,weights=weights_masked,n_bins=n_bins,
                                        ecc_thresholds=self.mask_ecc_thresholds,outlier_num_stds=self.outlier_num_stds,ci_factor=self.ci_factor,detect_inliers=self.detect_outliers)

                                # and append to group variable
                                all_mean_x_data.append(x_means)
                                all_mean_y_data.append(y_means)

                            # then, if we're asking for the average, compute that:
                            if len(these_subjects)>1:
                                y_means, y_cis, y_t, y_p, y_d, N = self.CUO.stats_from_data(np.array(all_mean_y_data).T,self.ci_factor)                               
                                x_means, x_cis, x_t, x_p, x_d, N = self.CUO.stats_from_data(np.array(all_mean_x_data).T,self.ci_factor)                               

                            all_roi_data[measure][roi][comparison]['x_means'] = x_means
                            all_roi_data[measure][roi][comparison]['y_mean'] = y_means
                            all_roi_data[measure][roi][comparison]['cis'] = y_cis
                            all_roi_data[measure][roi][comparison]['ps'] = y_p  
                            all_ps.append(y_p)
                            all_p_indices.append([measure,roi,comparison])

                # determine FDR 
                rejections,corrected_ps = mne.stats.fdr_correction(all_ps)
                # put back in dict
                for pi, rej in enumerate(rejections):
                    all_roi_data[all_p_indices[pi][0]][all_p_indices[pi][1]][all_p_indices[pi][2]]['rej'] = rej
                    all_roi_data[all_p_indices[pi][0]][all_p_indices[pi][1]][all_p_indices[pi][2]]['p_corr'] = corrected_ps[pi]

                if plot_type == 'per_roi':
                    # # now plot    
                    for ri,roi in enumerate(these_rois):
                        f=pl.figure(figsize = (3.3,2))
                        for mi,measure in enumerate(measures):
                            s = f.add_subplot(2,2,[1,2,3,4][mi])#[1,3,2,4][mi])
                            pl.axhline(0,c='k',lw=0.5,ls='-')
                            if measure in ['ecc','size','ami']:
                                for ci,comparison in enumerate(comparisons):
                                    x_means = all_roi_data[measure][roi][comparison]['x_means']
                                    y_means = all_roi_data[measure][roi][comparison]['y_mean']
                                    y_cis = all_roi_data[measure][roi][comparison]['cis']
                                    # plot line
                                    pl.plot(x_means,y_means,ls='-',lw = 1,c= self.comparison_colors[comparison])
                                    pl.fill_between(x_means,y_cis[:,0],y_cis[:,1],color = self.comparison_colors[comparison],alpha=0.25)
                            elif measure == 'fami':
                                    x_means = all_roi_data[measure][roi][comparison]['x_means']
                                    y_means = all_roi_data[measure][roi][comparison]['y_mean']
                                    y_cis = all_roi_data[measure][roi][comparison]['cis']
                                    # plot line
                                    pl.plot(x_means,y_means,ls='-',lw = 1,c= these_colors[ri])
                                    pl.fill_between(x_means,y_cis[:,0],y_cis[:,1],color = these_colors[ri],alpha=0.25)                        

                            sn.despine(offset=2)


                            if mi ==2:
                                pl.xlabel('Attend Fixation eccentricity (dva)')
                            pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])
                            pl.xlim(self.mask_ecc_thresholds)

                            if roi == 'combined':
                                if measure in 'ecc':
                                    pl.ylabel(r'$\Delta$ %s (dva)'%(measure))
                                    if subject in ['super_subject','over_subjects']:
                                        pl.ylim(-.15,.15)
                                        pl.yticks([-.15,0,.15],['-.15','0','.15'])
                                    else:
                                        pl.ylim(-.25,.25)
                                        pl.yticks([-.25,0,.25],['-.25','0','.25'])                            
                                elif measure == 'size':
                                    pl.ylabel(r'$\Delta$ %s (dva)'%(measure))
                                    if subject in ['super_subject','over_subjects']:
                                        pl.ylim(-.05,.05)
                                        pl.yticks([-.05,0,.05],['-.05','0','.05'])
                                    else:
                                        pl.ylim(-.10,.10)
                                        pl.yticks([-.10,0,.10],['-.10','0','.10'])     
                                elif measure == 'ami':
                                    pl.ylabel('AMI (z-score)')                       
                                    if subject in ['super_subject','over_subjects']:
                                        pl.yticks(self.functions.find_yticks(s.get_ylim())[0],self.functions.find_yticks(s.get_ylim())[1])
                                    else:
                                        pl.ylim(0,2)
                                        pl.yticks([0,2],['0','2'])     
                                elif measure == 'fami':
                                    pl.ylabel('feature AMI (a.u.)')    
                                    if subject in ['super_subject','over_subjects']:
                                        max_y = np.max(np.abs(self.functions.find_yticks(s.get_ylim())[0]))
                                        pl.yticks([-max_y,0,max_y],['-%.2f'%max_y,'0','%.2f'%max_y])
                                    else:
                                        pl.ylim(-.15,.15)
                                        pl.yticks([-.15,0,.15],['-.15','0','.15'])  

                            elif roi != 'combined':
                                if measure in ['ecc','size']:
                                    pl.ylabel(r'$\Delta$ %s (dva)'%(measure))
                                    max_y = np.max(np.abs(self.functions.find_yticks(s.get_ylim())[0]))
                                    pl.yticks([-max_y,0,max_y],['-%.2f'%max_y,'0','%.2f'%max_y])
                                elif measure == 'ami':
                                    pl.ylabel('AMI (z-score)')                       
                                    pl.yticks(self.functions.find_yticks(s.get_ylim())[0],self.functions.find_yticks(s.get_ylim())[1])
                                elif measure == 'fami':
                                    pl.ylabel('feature AMI (a.u.)')    
                                    max_y = np.max(np.abs(self.functions.find_yticks(s.get_ylim())[0]))
                                    pl.yticks([-max_y,0,max_y],['-%.2f'%max_y,'0','%.2f'%max_y])

                        pl.tight_layout()

                        # save
                        pl.savefig(os.path.join(self.group_plot_dir,'Figure_7','%s_%s_%s_over_Fix_ecc_%s_R2thresh_%.2f_outliers_detected_%s.pdf'%(roi,comparison,diff_type,subject,self.r_squared_threshold,self.detect_outliers)))
                        pl.close()

                elif plot_type == 'across_rois':
                    f=pl.figure(figsize = (3.2,1.2))
                    ri = -1
                    for rgi,roi_group in enumerate(np.sort(self.roi_groups_for_plot.keys())[:-1]):

                        s = f.add_subplot(self.roi_group_subplot_grid[0],self.roi_group_subplot_grid[1]-1,rgi+1)
                        pl.axhline(0,c='k',lw=0.5,ls='-')
                        pl.title('%s'%(' '.join(roi_group.split('_')[1:])))

                        # now for all rois within the roi group
                        for subri, roi in enumerate(self.roi_groups_for_plot[roi_group]):
                            
                            # advance roi color counter
                            ri += 1

                            x_means = all_roi_data['fami'][roi][comparison]['x_means']
                            y_means = all_roi_data['fami'][roi][comparison]['y_mean']
                            y_cis = all_roi_data['fami'][roi][comparison]['cis']

                            # plot line
                            pl.plot(x_means,y_means,ls='-',lw = 1,c= self.roi_colors[ri])
                            pl.fill_between(x_means,y_cis[:,0],y_cis[:,1],color = self.roi_colors[ri],alpha=0.25)                              

                        sn.despine(offset=2)

                        pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])
                        pl.xlim(self.mask_ecc_thresholds)

                        if rgi == 0:
                            pl.ylabel('feature AMI (a.u.)')    
                        # if rgi == 1:
                            # pl.xlabel('Attend Fix eccentricity (dva)')
                        max_y = np.max(np.abs(self.functions.find_yticks(s.get_ylim())[0]))
                        pl.yticks([-max_y,0,max_y],['-%.2f'%max_y,'0','%.2f'%max_y])

                    pl.tight_layout()
                    # save
                    pl.savefig(os.path.join(self.group_plot_dir,'Figure_7','all_rois_fami_over_ecc_%s.pdf'%subject))
                    pl.close()


    def mapper_fba_ecc(self,
        mcs = [10],
        mcs_labels = ['Color>TF'],
        mapper_measures = ['copes'],
        measure_types = ['combined'],#,'ecc','size'],
        measures=['ecc','sigma_center'],
        create_per_roi_fami_bar_plot = False,
        create_over_roi_scatter_plot = False,
        create_over_ecc_corrs = False,
        subjects = [ 'over_subjects','super_subject','DE','NA','JS','JW','TK']
        ):

        """
        This function will create bar plots of Mapper results per ROI
        """

        mpl.rc_file_defaults()

        if create_per_roi_fami_bar_plot or create_over_roi_scatter_plot:
            figno = 'Figure_7'
        else:
            figno = 'Figure_8'

        self.create_plot_dir(figno)

        self.load_data(PRF=True,Mapper=True,load_hemispheres=True)
        subjects_to_use = ['super_subject','NA','JS','TK','DE','JW']
        # subjects_to_use = ['super_subject']
        hemi_iterable = ['']

        all_roi_data = {}
        for mapper_measure in mapper_measures:
            for ci, mc in enumerate(mcs):            
                for measure_type in measure_types:
                    all_roi_data[measure_type] = {}

                    for ri, roi in enumerate(self.rois_for_plot):
                        # loop over hemispheres if wanted
                        for hemi in hemi_iterable:
                            this_roi = hemi+roi
                            
                            all_roi_data[measure_type][this_roi] = {}

                            print ''

                            for si, subject in enumerate(subjects_to_use):

                                all_roi_data[measure_type][this_roi][subject] = {}

                                sys.stdout.write('getting data for %s mc - %s FBA corr in %s, subject %d/%d\r'%(mcs_labels[ci],measure_type,this_roi,si+1,len(subjects_to_use)))
                                sys.stdout.flush()

                                # get mask variables
                                sizes_stim = np.array(self.all_results[subject]['Stim'][this_roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                sizes_fix = np.array(self.all_results[subject]['Fix'][this_roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
                                ecc_stim = np.array(self.all_results[subject]['Stim'][this_roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                ecc_fix = np.array(self.all_results[subject]['Fix'][this_roi])[:,self.results_frames['ecc']]/ self.rescale_factor
                                weights = np.squeeze(self.all_stats[subject]['Stim'][this_roi])

                                # now use the mask function
                                mask = self.functions.create_mask_2_conditions(ecc_stim,ecc_fix,sizes_stim,sizes_fix,weights,self.mask_ecc_thresholds,
                                        self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)

                                # mask weights 
                                weights = weights[mask]

                                # get mapper data:
                                mapper = np.array(self.all_mapper[subject][mapper_measure][this_roi])[mask,mc]

                                # now get the ami and fami
                                # first get all measures
                                these_data = {}
                                for condition in ['Stim','Color','Speed','Fix']:
                                    these_data[condition] = {}
                                    for measure in measures:
                                        temp = np.array(self.all_results[subject][condition][this_roi])[mask,self.results_frames[measure]] 
                                        if measure in ['sigma_center','ecc']:
                                            temp/=self.rescale_factor
                                        these_data[condition][measure] = temp

                                if measure_type == 'combined':
                                    col,speed,fami = self.functions.compute_FBA(these_data,weights,['sigma_center','ecc'],self.outlier_num_stds)
                                elif measure_type == 'ecc':
                                    col,speed,fami = self.functions.compute_FBA(these_data,weights,['ecc'],self.outlier_num_stds)
                                elif measure_type == 'size':
                                    col,speed,fami = self.functions.compute_FBA(these_data,weights,['sigma_center'],self.outlier_num_stds)

                                # inliers_feature_preference = self.CUO.detect_inliers_mad(mapper,self.outlier_num_stds)
                                # inliers_fami = self.CUO.detect_inliers_mad(fami,self.outlier_num_stds)
                                # combined_inliers = inliers_feature_preference* inliers_fami

                                # write to dict
                                all_roi_data[measure_type][this_roi][subject]['ecc'] = ecc_fix[mask]#[combined_inliers]
                                all_roi_data[measure_type][this_roi][subject]['size'] = sizes_fix[mask]#[combined_inliers]
                                all_roi_data[measure_type][this_roi][subject]['weights'] = weights#[combined_inliers]
                                all_roi_data[measure_type][this_roi][subject]['mapper'] = mapper#[combined_inliers]
                                all_roi_data[measure_type][this_roi][subject]['fami_%s'%measure_type] = fami#[combined_inliers]


        if create_per_roi_fami_bar_plot:
            subjects_to_use = ['DE','NA','JS','JW','TK','over_subjects','super_subject']#'super_subject',
            these_data = {}
            for mapper_measure in mapper_measures:
                for hemi in hemi_iterable:
                    for measure_type in measure_types:
                        for mi,mc in enumerate(mcs):
                            for si, subject in enumerate(subjects_to_use):

                                these_data[subject] = {}
                               
                                for ri, roi in enumerate(self.rois_for_plot):

                                    # preallocate for this roi
                                    these_data[subject][roi] = {}
                                    roi=hemi+roi
                                    # compute stuff if not avg subject:
                                    if subject != 'over_subjects':

                                        weights = all_roi_data[measure_type][roi][subject]['weights']
                                        y_data = all_roi_data[measure_type][roi][subject]['fami_%s'%measure_type]

                                        y_mean, y_ci, y_p,N_y,d_y = self.CUO.bootstrap(y_data,ci_factor=self.ci_factor,
                                            weights = weights, two_tailed=True,test_value=0,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds,return_d=True)

                                        these_data[subject][roi]['y_mean'] = y_mean
                                        these_data[subject][roi]['y_ci'] = y_ci
                                        these_data[subject][roi]['y_p'] = y_p                                        
                                        these_data[subject][roi]['y_d'] = d_y
                                        these_data[subject][roi]['y_N'] = N_y                                   
                            
                                    # if the data is the avg, compute from already computed stuff:
                                    elif subject == 'over_subjects':

                                        temp = np.array([these_data[s][roi]['y_mean'] for s in self.subjects])

                                        mean, ci, t, p, d, N = self.CUO.stats_from_data(temp,self.ci_factor)

                                        these_data[subject][roi]['y_mean'] = mean
                                        these_data[subject][roi]['y_ci'] = ci
                                        these_data[subject][roi]['y_p'] = p
                                        these_data[subject][roi]['y_N'] = N
                                        these_data[subject][roi]['y_d'] = d
                                        these_data[subject][roi]['y_t'] = t


                            for si, subject in enumerate(subjects):

                                # now that we have the relevant data, let's plot
                                f=pl.figure(figsize = (1.5,1)) 
                                s = f.add_subplot(111)
                                sn.despine(offset=2)
                                pl.axhline(0,c='k',lw=0.5,ls='-')

                                # only fdr correct super subject pvals
                                all_y_ps = np.array([these_data[subject][roi]['y_p'] for ri, roi in enumerate(self.rois_for_plot)])
                                
                                if subject == 'super_subject':
                                    rej05 = mne.stats.fdr_correction(all_y_ps,alpha=.05)[0]
                                    rej01 = mne.stats.fdr_correction(all_y_ps,alpha=.01)[0]
                                    rej001 = mne.stats.fdr_correction(all_y_ps,alpha=.001)[0]
                                else:
                                    rej05 = all_y_ps<.05
                                    rej01 = all_y_ps<.01
                                    rej001 = all_y_ps<.001

                                # now print out which 
                                for ri, roi in enumerate(self.rois_for_plot):

                                    p_y = all_y_ps[ri]
                                    y_data = these_data[subject][roi]['y_mean']
                                    y_ci = these_data[subject][roi]['y_ci']
                                    y_N = these_data[subject][roi]['y_N']
                                    y_d = these_data[subject][roi]['y_d']

                                    stars = '*'*np.sum([rej05[ri],rej01[ri],rej001[ri]])
                                    if subject == 'over_subjects':
                                        t = these_data[subject][roi]['y_t']
                                        print '%s %s fami = %.3f, p = %.3f%s, t = %.3f, cohen_d = %.3f, N = %d'%(subject,roi,y_data,p_y,stars,t,y_d,y_N)
                                    else:
                                        print '%s %s fami = %.3f, p = %.3f%s, cohen_d = %.3f, N = %d'%(subject,roi,y_data,p_y,stars,y_d,y_N)


                                    if rej05[ri]:
                                        fill_color = self.roi_colors[ri]
                                    else:
                                        fill_color = 'w'
                                        edge_color =self.roi_colors[ri]

                                    # plot the bar for the mean 
                                    pl.bar(ri,y_data,color=fill_color,edgecolor=self.roi_colors[ri])
                                    # x error bars
                                    pl.plot([ri+.4,ri+.4],y_ci,lw=0.75,color='w')
                                    pl.plot([ri+.4,ri+.4],y_ci,lw=0.75,color=self.roi_colors[ri],alpha=0.5)

                                if subject not in ['super_subject']:
                                    pl.ylim(-0.2,0.35)
                                    pl.yticks([-0.2,0,0.35],['-.2','0','.35'])
                                else:
                                    # pl.ylim(-0.05,0.20)
                                    # pl.yticks([-0.05,0,0.20],['-.05','0','.20'])                                    
                                    pl.yticks(self.functions.find_yticks(s.get_ylim())[0],self.functions.find_yticks(s.get_ylim())[1])

                                pl.ylabel('FAMI')
                                # figure options 
                                pl.tight_layout(w_pad = 0.0,h_pad=0.0,pad=0)
                                # save
                                pl.savefig(os.path.join(self.group_plot_dir,figno,'FAMI_barplot_%s_%s.pdf'%(subject,measure_type)))
                                pl.close()  

                            # print out for how many subjects there is significance
                            for ri, roi in enumerate(self.rois_for_plot):
                                sig_pos = 0
                                sig_neg = 0
                                for subject in self.subjects:
                                    if these_data[subject][roi]['y_p'] < .05:
                                        if these_data[subject][roi]['y_mean'] > 0:
                                            sig_pos +=1
                                        elif these_data[subject][roi]['y_mean'] < 0:
                                            sig_neg +=1

                                print '%s, %d/%d pos, %d/%d neg'%(roi,sig_pos,len(self.subjects),sig_neg,len(self.subjects))

                            # now calculate anova over subjects
                            import pyvttbl as pt

                            rois = []; subs = []; famis= []; conds = []
                            for ri, roi in enumerate(self.rois_for_plot):
                                if not roi == 'combined':
                                    for subject in self.subjects:
                                        for cond in [0,1]:
                                            rois.append(roi)
                                            subs.append(subject)
                                            if cond == 1:
                                                famis.append(these_data[subject][roi]['y_mean'])
                                            else:
                                                famis.append(0)
                                            conds.append(cond)

                            df = pt.DataFrame()
                            df['roi'] = rois
                            df['sub'] = subs
                            df['fami'] = famis
                            df['cond'] = conds

                            aov = df.anova('fami', sub='sub', wfactors=['roi','cond'])
                            print aov


        if create_over_roi_scatter_plot:
            # Figure 6 B
            comparisons = [
            # ['fami',['ecc']],
            ['fami',['mapper']],
            # ['fami',['mapper','ecc']],
            # ['fami',['mapper','size']],
            # ['fami',['fami','mapper']],
            # ['fami',['fami','ecc']],
            ]

            for comp in comparisons:

                subjects_to_use = ['super_subject','DE','NA','JS','JW','TK','over_subjects']

                these_data = {}
                for mapper_measure in mapper_measures:
                    for hemi in hemi_iterable:
                        for measure_type in measure_types:
                            for mi,mc in enumerate(mcs):
                                for si, subject in enumerate(subjects_to_use):

                                    these_data[subject] = {}
                                   
                                    for ri, roi in enumerate(self.rois_for_plot):

                                        # preallocate for this roi
                                        these_data[subject][roi] = {}
                                        roi=hemi+roi
                                        # compute stuff if not avg subject:
                                        if subject != 'over_subjects':

                                            weights = all_roi_data[measure_type][roi][subject]['weights']
                                           
                                            y_data = all_roi_data[measure_type][roi][subject][comp[0]+'_%s'%measure_type]
                                            y_mean, y_ci, y_p,N_y,d_y = self.CUO.bootstrap(y_data,ci_factor=self.ci_factor,
                                                weights = weights, two_tailed=True,test_value=0,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds,return_d=True)
                                            
                                            these_data[subject][roi]['y_mean'] = y_mean
                                            these_data[subject][roi]['y_ci'] = y_ci
                                            these_data[subject][roi]['y_p'] = y_p                                        
                                            these_data[subject][roi]['y_d'] = d_y
                                            these_data[subject][roi]['y_N'] = N_y

                                            # then the data along the x axis
                                            # if it's length is 1, we ask for a single measure
                                            if len(comp[1]) == 1:
                                                x_data = all_roi_data[measure_type][roi][subject][comp[1][0]]
                                                x_mean, x_ci, x_p,N_x,d_x = self.CUO.bootstrap(x_data,
                                                    weights = weights, ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds,return_d=True)
                                                these_data[subject][roi]['x_d'] = d_x

                                            # however, if the length is 2, we ask for correlation instead of average value
                                            elif len(comp[1]) == 2:
                                                x_data1 = all_roi_data[measure_type][roi][subject][comp[1][0]]
                                                x_data2 = all_roi_data[measure_type][roi][subject][comp[1][1]]

                                                x_mean, x_ci, x_p, N_x = self.CUO.bootstrap_correlation(
                                                    x_data1,x_data2,corr_type='spearman',weights = weights, ci_factor=self.ci_factor,two_tailed=True,test_value=0,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds)                                        
                                                these_data[subject][roi]['x_d'] = 0

                                            these_data[subject][roi]['x_mean'] = x_mean
                                            these_data[subject][roi]['x_ci'] = x_ci
                                            these_data[subject][roi]['x_p'] = x_p                                            
                                            these_data[subject][roi]['x_N'] = x_p                                            
                                
                                        # if the data is the avg, compute from already computed stuff:
                                        elif subject == 'over_subjects':

                                            for m in ['x','y']:
                                                temp = np.array([these_data[s][roi]['%s_mean'%m] for s in self.subjects])
                                                # if (len(comp[1])==2)*(m=='x'):
                                                #     # now we've asked for a correlation, we must fisher transform data for p value
                                                #     mean_, ci_, p,N,d  = self.CUO.bootstrap(np.arctanh(temp),ci_factor=self.ci_factor,test_value=0,
                                                #         two_tailed=True,detect_inliers=False,outlier_num_stds=self.outlier_num_stds,return_d =True)
                                                #     t,p =   
                                                #     mean, ci, p_,N_,d_  = self.CUO.bootstrap(temp,ci_factor=self.ci_factor,test_value=0,
                                                #         two_tailed=True,detect_inliers=False,outlier_num_stds=self.outlier_num_stds,return_d =True)  
                                                # else:   

                                                mean, ci, t, p, d, N = self.CUO.stats_from_data(temp,self.ci_factor)

                                                these_data[subject][roi]['%s_mean'%m] = mean
                                                these_data[subject][roi]['%s_ci'%m] = ci
                                                these_data[subject][roi]['%s_p'%m] = p
                                                these_data[subject][roi]['%s_N'%m] = N
                                                these_data[subject][roi]['%s_d'%m] = d
                                                these_data[subject][roi]['%s_t'%m] = t


                                all_spearmans = []
                                all_pearsons = []
                                for si, subject in enumerate(subjects_to_use):

                                    if len(comp[1]) ==2:
                                        print '\n\n%s\nplotting %s vs %s-%s corr for %s\n%s\n\n'%('-'*25,comp[0],comp[1][0],comp[1][1],subject,'-'*25)
                                    elif len(comp[1]) ==1:
                                       print '\n\n%s\nplotting %s vs %s for %s\n%s\n\n'%('-'*25,comp[0],comp[1][0],subject,'-'*25)

                                    # now that we have the relevant data, let's plot
                                    f=pl.figure(figsize = (1.5,1)) 
                                    s = f.add_subplot(111)
                                    sn.despine(offset=2)
                                    pl.axhline(0,c='k',lw=0.5,ls='-')


                                    # only fdr correct super subject pvals
                                    all_y_ps = np.array([these_data[subject][roi]['y_p'] for ri, roi in enumerate(self.rois_for_plot)])
                                    all_x_ps = np.array([these_data[subject][roi]['x_p'] for ri, roi in enumerate(self.rois_for_plot)])
                                    if subject == 'super_subject':
                                        all_x_ps = mne.stats.fdr_correction(all_x_ps,alpha=.05)[1]
                                        all_y_ps = mne.stats.fdr_correction(all_y_ps,alpha=.05)[1]

                                    all_roi_x = []
                                    all_roi_y = []
                                    # now print out which 
                                    for ri, roi in enumerate(self.rois_for_plot):

                                        p_x = all_x_ps[ri]
                                        p_y = all_y_ps[ri]

                                        mew = 0.5
                                        ms = 6
                                        if p_y < 0.05:
                                            fill_color = self.roi_colors[ri]
                                            edge_color = 'w'
                                        else:
                                            fill_color = 'w'
                                            edge_color = self.roi_colors[ri]

                                        x_data = these_data[subject][roi]['x_mean']
                                        y_data = these_data[subject][roi]['y_mean']
                                        x_ci = these_data[subject][roi]['x_ci']
                                        y_ci = these_data[subject][roi]['y_ci']
                                        x_N = these_data[subject][roi]['x_N']
                                        y_N = these_data[subject][roi]['y_N']
                                        x_d = these_data[subject][roi]['x_d']
                                        y_d = these_data[subject][roi]['y_d']
                                        # print 'FAMI mean %s: %.2f, N=%d,p: %.3f,cohend: %.3f'%(roi,y_data,y_N,float(p_y),y_d)
                                        # print '%s mean %s: %.2f, N=%d,p: %.3f,cohend: %.3f'%(comp[1],roi,x_data,x_N,float(p_x),x_d)

                                        # plot the marker for the mean 
                                        pl.plot(x_data,y_data,['o','s','^'][mi],color=fill_color,ms=ms,mec=edge_color,mew=mew,alpha=1)
                                        # y error bars
                                        pl.plot([x_data,x_data],y_ci,lw=0.75,color='w')
                                        pl.plot([x_data,x_data],y_ci,lw=0.75,color=self.roi_colors[ri],alpha=0.5)
                                        # x error bars
                                        pl.plot(x_ci,[y_data,y_data],lw=0.75,color='w')
                                        pl.plot(x_ci,[y_data,y_data],lw=0.75,color=self.roi_colors[ri],alpha=0.5)

                                        if roi != 'combined':
                                            all_roi_x.append(x_data)
                                            all_roi_y.append(y_data)

                                    if subject != 'over_subjects':
                                        # now correlate the x and y data over rois for this subject (both spearman and pearson)
                                        pearsonr,pearsonp = sp.stats.pearsonr(all_roi_x,all_roi_y)
                                        print 'pearson correlation over rois: corr: %.4f, p: %.3f'%(pearsonr,pearsonp)   
                                        spearmanr,spearmanp = sp.stats.spearmanr(all_roi_x,all_roi_y) 
                                        print 'spearman correlation over rois: corr: %.4f, p: %.3f'%(spearmanr,spearmanp)   
                                        all_spearmans.append(spearmanr)
                                        all_pearsons.append(pearsonr)
                                    else:
                                        pearsonr = np.mean(all_pearsons)
                                        pearsont,pearsonp = sp.stats.ttest_1samp(np.arctanh(all_pearsons),0)
                                        pearsond = np.mean(all_pearsons)/np.std(all_pearsons)

                                        spearmanr = np.mean(all_spearmans)
                                        spearmant,spearmanp = sp.stats.ttest_1samp(np.arctanh(all_spearmans),0)
                                        spearmand = np.mean(all_spearmans)/np.std(all_spearmans)
                                        print 'pearson correlation over rois: corr: %.4f, t: %.4f, d: %.4f,p: %.3f'%(pearsonr,pearsont,pearsond,pearsonp)   
                                        print 'spearman correlation over rois: corr: %.4f, t: %.4f, d: %.4f,p: %.3f'%(spearmanr,spearmant,spearmand,spearmanp)   

                                    if subject in ['super_subject','over_subjects']:
                                        # pl.xlim(-2.5,0.5)
                                        # pl.xticks([-2.5,0,0.5],['-2.5','0','.5'])
                                        # pl.ylim(-0.05,0.20)
                                        # pl.yticks([-0.05,0,0.20],['-.05','0','.20'])
                                        # pl.yticks(self.functions.find_yticks(s.get_ylim())[0],self.functions.find_yticks(s.get_ylim())[1])
                                        pl.yticks([0,0.2],['0','.2'])
                                        pl.ylim(0,0.2)
                                    else:
                                        pl.xlim(-3.5,1.5)
                                        pl.xticks([-3.5,0,1.5],['-3.5','0','1.5'])
                                        pl.ylim(-0.2,0.35)
                                        pl.yticks([-0.2,0,0.35],['-.2','0','.35'])                                        

                                    pl.xticks(self.functions.find_yticks(s.get_xlim())[0],self.functions.find_yticks(s.get_xlim())[1])                                        

                                    # pl.yticks(self.functions.find_yticks(s.get_ylim())[0],self.functions.find_yticks(s.get_ylim())[1])
                                    # pl.xticks(self.functions.find_yticks(s.get_xlim())[0],self.functions.find_yticks(s.get_xlim())[1])

                                    xwidth = np.diff(s.get_xlim())
                                    ywidth = np.diff(s.get_ylim())
                                    pl.text(s.get_xlim()[0]+0.1*xwidth,s.get_ylim()[1]-.2*ywidth,r'R: %.2f'%pearsonr + '\n p: %.3f'%pearsonp)
                                    pl.text(s.get_xlim()[0]+0.5*xwidth,s.get_ylim()[1]-.2*ywidth,r'$\rho$: %.2f'%spearmanr + '\n p: %.3f'%spearmanp)

                                    if len(comp[1]) ==2:
                                        xlabel = '%s-%s corr'%(comp[1][0],comp[1][1])
                                    elif len(comp[1]) == 1:
                                        xlabel = '%s'%(comp[1][0])                                        
                                    pl.ylabel('FAMI')
                                    # figure options 
                                    pl.tight_layout(w_pad = 0.0,h_pad=0.0,pad=0)
                                    # save
                                    pl.savefig(os.path.join(self.group_plot_dir,figno,'FAMI_vs_%s_for_%s_%s.pdf'%(comp,subject,measure_type)))
                                    pl.close()      


        if create_over_ecc_corrs:

            comparisons = [
            ['ecc','mapper'],
            # ['ecc','fami'],
            # ['mapper','fami'],
            # ['size','mapper'],
            ]

            subjects_to_use = ['super_subject','DE','NA','JS','JW','TK','over_subjects']
            for corr_type in ['spearman']:
                for mapper_measure in mapper_measures:
                    for hemi in hemi_iterable:
                        for measure_type in measure_types:
                            for mi,mc in enumerate(mcs):
                                these_data = {}
                                corr_df = {key: [] for key in ['subject','roi','corr','cond']}
                                for si, subject in enumerate(subjects_to_use):

                                    these_data[subject] = {}


                                    for ri, roi in enumerate(self.rois_for_plot[:-1]):

                                        print 'now computing %s correlation for %s in %s'%(corr_type,roi,subject)
                                        # preallocate for this roi
                                        these_data[subject][roi] = {}
                                        roi=hemi+roi


                                        if subject != 'over_subjects':

                                            weights = all_roi_data[measure_type][roi][subject]['weights']
                                           
                                            for comparison in comparisons:

                                                these_data[subject][roi]['_'.join(comparison)] = {}

                                                # get these data to compare
                                                x_data = copy.copy(all_roi_data[measure_type][roi][subject][comparison[0]])
                                                y_data = copy.copy(all_roi_data[measure_type][roi][subject][comparison[1]])

                                                # regress out the effects of eccentricity on both variables
                                                # use a 2nd order polynomial for possible non-linear effects of ecc
                                                # if '_'.join(comparison) == 'mapper_fami':
                                                    # ecc = all_roi_data[roi][subject]['ecc']
                                                    # x_data -= np.polyval(np.polyfit(ecc,x_data,2),ecc)
                                                    # y_data -= np.polyval(np.polyfit(ecc,y_data,1,w=weights),ecc)

                                                # if corr_type == 'median_split':
                                                #     # now correlate
                                                #     corr,corr_ci, corr_p, corr_N,cohen_d = self.CUO.bootstrap_median_split(x_data,y_data,weights,
                                                #         self.outlier_num_stds,self.ci_factor,self.detect_outliers,self.reps,test_value=0,two_tailed=True)
                                                # else:
                                                corr, corr_ci, corr_p, corr_N = self.CUO.bootstrap_correlation(
                                                        x_data,y_data,corr_type=corr_type,weights = weights, ci_factor=self.ci_factor,
                                                        two_tailed=True,test_value=0,detect_inliers=self.detect_outliers,outlier_num_stds=self.outlier_num_stds)                                        

                                                # now append all results
                                                these_data[subject][roi]['_'.join(comparison)]['corr'] = corr
                                                these_data[subject][roi]['_'.join(comparison)]['corr_ci'] = corr_ci
                                                these_data[subject][roi]['_'.join(comparison)]['corr_p'] = corr_p                                        
                                                these_data[subject][roi]['_'.join(comparison)]['corr_N'] = corr_N
                                           
                                            # add to dict if not super subject
                                            if subject != 'super_subject':

                                                corr_df['subject'].append(subject)
                                                corr_df['roi'].append(roi)
                                                corr_df['corr'].append(corr)
                                                corr_df['cond'].append(0)

                                                corr_df['subject'].append(subject)
                                                corr_df['roi'].append(roi)
                                                corr_df['corr'].append(0)
                                                corr_df['cond'].append(1)

                                        # if the data is the avg, compute from already computed stuff:
                                        elif subject == 'over_subjects':

                                            for comparison in comparisons:

                                                these_data[subject][roi]['_'.join(comparison)] = {}

                                                temp = np.array([these_data[s][roi]['_'.join(comparison)]['corr'] for s in self.subjects])

                                                mean, ci, t, p, d, N = self.CUO.stats_from_data(temp,self.ci_factor)
                                                # fisher transform correlations for p value
                                                t,p = sp.stats.ttest_1samp(np.arctanh(temp),0)

                                                these_data[subject][roi]['_'.join(comparison)]['corr'] = mean
                                                these_data[subject][roi]['_'.join(comparison)]['corr_ci'] = ci
                                                these_data[subject][roi]['_'.join(comparison)]['corr_p'] = p
                                                these_data[subject][roi]['_'.join(comparison)]['corr_N'] = N


                           
                                for si, subject in enumerate(subjects_to_use):#,'over_subjects']):
                                    for compi,comparison in enumerate(comparisons):

                                        # only fdr correct super subject pvals
                                        all_ps = np.array([these_data[subject][roi]['_'.join(comparison)]['corr_p'] for ri, roi in enumerate(self.rois_for_plot[:-1])])
                                        

                                        if subject == 'super_subject':
                                            rej05 = mne.stats.fdr_correction(all_ps,alpha=.05)[0]
                                            rej01 = mne.stats.fdr_correction(all_ps,alpha=.01)[0]
                                            rej001 = mne.stats.fdr_correction(all_ps,alpha=.001)[0]
                                        else:
                                            rej05 = all_ps<.05
                                            rej01 = all_ps<.01
                                            rej001 = all_ps<.001
                                       
                                        # if subject not in ['super_subject','over_subjects']:

                                        #     pos_in 
                                        
                                        f = pl.figure(figsize=(1,1))
                                        
                                        s = f.add_subplot(1,1,1)
                                        pl.axhline(0,lw=0.5,c='k')

                                        for ri, roi in enumerate(self.rois_for_plot[:-1]):


                                            mean = these_data[subject][roi]['_'.join(comparison)]['corr']
                                            ci = these_data[subject][roi]['_'.join(comparison)]['corr_ci']
                                            p = these_data[subject][roi]['_'.join(comparison)]['corr_p']
                                            N = these_data[subject][roi]['_'.join(comparison)]['corr_N']
                                            
                                            stars = '*'*np.sum([rej05[ri],rej01[ri],rej001[ri]])
                                            print '%s %s %s %s correlation = %.3f, p = %.3f%s, N = %d'%(comparison,subject,roi,corr_type,mean,p,stars,N)

                                            if rej05[ri]:
                                                fill_color = self.roi_colors[ri]
                                            else:
                                                fill_color = 'w'

                                            pl.plot(ri,mean,color=fill_color,marker = ['o','s','D','<','>','v','*','d','^','o'][ri],mec=self.roi_colors[ri],label='_'.join(comparison))
                                            pl.plot([ri,ri],ci,lw=1.5,c='w')
                                            pl.plot([ri,ri],ci,lw=1,c=self.roi_colors[ri])


                                        sn.despine(offset=2)

                                        if subject == 'super_subject':
                                            pl.yticks([-.5,0,.5],['-.5','0','.5'])
                                        else:
                                            pl.yticks([-.8,0,.8],['-.8','0','.8'])
                                        pl.xlim(-0.5,len(self.rois_for_plot[:-1])-0.5)

                                        # figure options 
                                        pl.tight_layout(w_pad = 0.0,h_pad=0.0,pad=0)
                                        # save
                                        pl.savefig(os.path.join(self.group_plot_dir,figno,'correlations_%s_%s_%s.pdf'%(subject,corr_type,'_'.join(comparison))))
                                        pl.close()    

                                # first factor of roi and fictitious 'cond' needed to get offset of correlation in rm anova
                                import pyvttbl as pt
                                df = pt.DataFrame(corr_df)
                                aov = df.anova('corr', sub='subject', wfactors=['roi','cond'])
                                print(aov)

                                # and results for individual subjects
                                for compi,comparison in enumerate(comparisons):
                                    for ri, roi in enumerate(self.rois_for_plot[:-1]):
                                        sig_ps = np.array([these_data[subject][roi]['_'.join(comparison)]['corr_p']<.05 for subject in self.subjects])
                                        pos_corrs = np.array([these_data[subject][roi]['_'.join(comparison)]['corr']>0 for subject in self.subjects])
                                        neg_corrs = np.array([these_data[subject][roi]['_'.join(comparison)]['corr']<0 for subject in self.subjects])

                                        print '%s %s neg in %d/%d'%(comparison,roi,np.sum(sig_ps*neg_corrs),len(self.subjects))
                                        print '%s %s pos in %d/%d'%(comparison,roi,np.sum(sig_ps*pos_corrs),len(self.subjects))


    def AF_fitting(self,AF_shape='bar_convolved',fit_method='grid',initialize_on='custom',these_subjects=['DE','NA','JS','TK','JW'],conditions = ['Stim','Fix'],res=131,AF_plot_dir='',
        AF_surround_ratios=[2],AF_surround_amps=[-0.5],n_eccen_bins=8,n_polar_bins=8,
        AF_intercepts=[0,1,10],model_area_increase_factor=1,AF_fix_sizes=[1,2,3],n_jobs=20,
        AF_slopes='positive',circle_mask=False,center_method='something',upsample_factor=10,
        model_tag = 'OG',rectify=False,overlap=False,AF_slopelist=[0],jacknife=False,init_fix_from_stim=False,fix_init_model_name='none',per_subject=False):

        # create model name
        if rectify:
            model_tag += '_rect'
        if overlap:
            model_tag += '_overlap'

        # create fixed bins for the quadrant shift plot to minimize
        ecc_boundaries = np.linspace(self.mask_ecc_thresholds[0],self.mask_ecc_thresholds[1],n_eccen_bins+1)
        ecc_bins = [[ecc_boundaries[bi],ecc_boundaries[bi+1]] for bi in range(len(ecc_boundaries)-1)]
        polar_boundaries = np.linspace(0,np.pi/2,n_polar_bins+1)
        polar_bins = [[polar_boundaries[bi],polar_boundaries[bi+1]] for bi in range(len(polar_boundaries)-1)]                   

        all_subject_arrays = [] #-> inserting 'these subjects' in this list will include a super subject when jacknife = True
        if jacknife:
            model_dir = os.path.join(self.group_plot_dir,'AF_modeling_super_subjects/')
            for left_out_subject in these_subjects:
                this_subject_array = copy.copy(these_subjects)
                popped_subject = this_subject_array.pop(np.where(np.array(self.subjects)==left_out_subject)[0][0])
                all_subject_arrays.append(this_subject_array)
        if per_subject:
            all_subject_arrays = [[sub] for sub in these_subjects]
            model_dir = os.path.join(self.group_plot_dir,'AF_modeling_over_subjects/')

        model_name = '%s_%s_%s_AFslope_%s'%(AF_shape,model_tag,AF_slopes,self.rois_for_plot[0])
        this_plot_dir = os.path.join(model_dir,model_name)
        if not os.path.isdir(this_plot_dir):os.mkdir(this_plot_dir) 

        print 'commencing AF fitting with model: %s'%model_name
        
        roi = self.rois_for_plot[0]
       
        all_results = {}
        all_params = {}
        for this_subject_array in all_subject_arrays:   

            print 'now fitting on pooled data from %s'%(this_subject_array)

            super_subject_name = '_'.join(this_subject_array)

            if init_fix_from_stim:
                
                # load in the stim results to initialize fix AF size
                # model_dir = os.path.join(self.group_plot_dir,'AF_modeling_current_paper')
                # for roi in self.rois_for_plot:
                this_model_name = fix_init_model_name + '_%s'%roi
                this_model_dir = os.path.join(model_dir,this_model_name)

                these_data = pickle.load(open(os.path.join(this_model_dir,'%s_results_Stim_.pickle'%(fit_method)),'rb'))
                these_params = pickle.load(open(os.path.join(this_model_dir,'%s_params_Stim_.pickle'%(fit_method)),'rb'))

                n_models = len(these_data[super_subject_name])

                if jacknife:

                    best_model = np.argmax([these_data[super_subject_name][mi]['r_squared'] for mi in range(n_models)])

                elif per_subject:
                    # first compute observed ecc diff
                    these_results = these_data[super_subject_name][0]
                    ecc_1 = these_results['ecc_cond_1']
                    ecc_diff = these_results['ecc_cond_0']-these_results['ecc_cond_1']
                    weights = these_results['weights']
                    observed_ecc_diffs = []

                    for bi in range(n_eccen_bins):    
                        these_voxels = (ecc_1 > ecc_boundaries[bi]) * (ecc_1 < ecc_boundaries[bi+1])
                        if these_voxels.sum() > 0:
                            observed_ecc_diffs.append(np.average(ecc_diff[these_voxels],weights=weights[these_voxels]))
                        else:
                            observed_ecc_diffs.append(np.nan)
                    
                    errors = []
                    # then also for each model
                    for model in range(n_models):

                        # get results for this model
                        these_results =  these_data[super_subject_name][model]

                        # predicted
                        ecc_0 = np.linalg.norm([these_results['bar_RFx'],these_results['bar_RFy']],axis=0)
                        ecc_1 = np.linalg.norm([these_results['fix_RFx'],these_results['fix_RFy']],axis=0)
                        ecc_diff = ecc_0 - ecc_1

                        weights = these_results['weights']

                        predicted_ecc_diffs = []
                        for bi in range(n_eccen_bins):    
                            these_voxels = (ecc_1 > ecc_boundaries[bi]) * (ecc_1 < ecc_boundaries[bi+1])
                            if these_voxels.sum() > 0:
                                predicted_ecc_diffs.append(np.average(ecc_diff[these_voxels],weights=weights[these_voxels]))
                            else:
                                predicted_ecc_diffs.append(np.nan)
                        errors.append(np.sum((np.array(predicted_ecc_diffs)-np.array(observed_ecc_diffs))**2))

                    # best_model_this_subject = np.argmax([these_data[subject][mi]['r_squared'] for mi in range(n_models)])
                    best_model = np.argmin(errors)
                        
                these_params = these_params['_'.join(this_subject_array)][best_model]
                AF_fix_sizes = [these_params['fix_AF_size']]

            # only load data if it has not already been added to the self object
            # if ((not hasattr(self,'all_results')) + (not hasattr(self,'all_stats'))): 
            self.load_data(PRF=True,subjects=this_subject_array)

            # get relevant variables for mask and rescale
            sizes_cond_0 = np.array(self.all_results['super_subject'][conditions[0]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
            sizes_cond_1 = np.array(self.all_results['super_subject'][conditions[1]][roi])[:,self.results_frames['sigma_center']]/ self.rescale_factor
            ecc_cond_0 = np.array(self.all_results['super_subject'][conditions[0]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
            ecc_cond_1 = np.array(self.all_results['super_subject'][conditions[1]][roi])[:,self.results_frames['ecc']]/ self.rescale_factor
            weights_cond_0 = np.squeeze(self.all_stats['super_subject'][conditions[0]][roi])
            weights_cond_1 = np.squeeze(self.all_stats['super_subject'][conditions[1]][roi])

            # now use the mask function
            mask = self.functions.create_mask_2_conditions(ecc_cond_0,ecc_cond_1,sizes_cond_0,sizes_cond_1,weights_cond_0,weights_cond_1,self.mask_ecc_thresholds,
                self.r_squared_threshold,self.outlier_num_stds,self.mask_type,self.size_threshold)

            # get weights
            weights = np.min([np.squeeze(self.all_stats['super_subject'][conditions[0]][roi]),
                np.squeeze(self.all_stats['super_subject'][conditions[1]][roi])],axis=0)[mask]

            # get data masked
            cond_0_xo = np.array(self.all_results['super_subject'][conditions[0]][roi])[mask,self.results_frames['xo']] * self.stim_radius
            cond_0_yo = np.array(self.all_results['super_subject'][conditions[0]][roi])[mask,self.results_frames['yo']] * self.stim_radius 
            cond_0_data = np.array([cond_0_xo,cond_0_yo])           
            cond_1_xo = np.array(self.all_results['super_subject'][conditions[1]][roi])[mask,self.results_frames['xo']] * self.stim_radius
            cond_1_yo = np.array(self.all_results['super_subject'][conditions[1]][roi])[mask,self.results_frames['yo']] * self.stim_radius
            cond_1_data = np.array([cond_1_xo,cond_1_yo])   
           
            cond_0_ecc = np.array(self.all_results['super_subject'][conditions[0]][roi])[mask,self.results_frames['ecc']]/ self.rescale_factor
            cond_1_ecc = np.array(self.all_results['super_subject'][conditions[1]][roi])[mask,self.results_frames['ecc']]/ self.rescale_factor
            cond_1_size = np.array(self.all_results['super_subject'][conditions[1]][roi])[mask,self.results_frames['sigma_center']] / self.rescale_factor

            # determine the arrows that go into the arrow plot
            # shift data to one quadrant
            sign_cond_1_data = np.sign(cond_1_data)
            cond_1_data = cond_1_data * sign_cond_1_data
            cond_0_data = cond_0_data * sign_cond_1_data    
            
            # get eccen values for bins
            cond_0_eccen = np.linalg.norm(cond_0_data,axis=0)
            cond_1_eccen = np.linalg.norm(cond_1_data,axis=0)

            # # get the arrows
            # arrow_starts, diff_vectors = self.functions.get_arrows('quadrant',n_eccen_bins,self.mask_ecc_thresholds,n_polar_bins,
            #                                 cond_0_data,cond_1_data,weights,self.detect_outliers,self.outlier_num_stds,'cond_1')

            # put these data in a dictionary
            these_results = {}
            these_results['cond_1_xo'], these_results['cond_1_yo'] = cond_1_xo,cond_1_yo
            these_results['flipped_cond_0_data'], these_results['flipped_cond_1_data'] = cond_0_data,cond_1_data
            these_results['sign_cond_1_data'] = sign_cond_1_data
            these_results['ecc_cond_0'], these_results['ecc_cond_1'] = cond_0_eccen,cond_1_eccen
            # these_results['cond_1_sizes'], these_results['diff_vectors'] = cond_1_size, diff_vectors
            these_results['cond_1_sizes'] = cond_1_size
            these_results['weights'] = weights

            all_results[super_subject_name] = these_results


            if fit_method == 'grid':
                
                grid = []

                # setup AF slopes if we want to:
                AF_sizes = []
                min_bar_width = 0.5
                for AF_intercept in AF_intercepts:
                    max_slope = (AF_intercept-min_bar_width)/self.stim_radius
                    for fix_AF_size in AF_fix_sizes:
                        # if AF_slopes == 'negative':
                        #     AF_sizes.append([fix_AF_size,intercept,0])
                        #     AF_sizes.append([fix_AF_size,intercept,-max_slope/2])
                        #     AF_sizes.append([fix_AF_size,intercept,-max_slope])
                        # elif AF_slopes == 'positive':
                        #     AF_sizes.append([fix_AF_size,intercept,0])
                        #     AF_sizes.append([fix_AF_size,min_bar_width+((intercept-min_bar_width)/2),max_slope/2])
                        #     AF_sizes.append([fix_AF_size,min_bar_width,max_slope])
                        if AF_slopes == 'positive':
                            for AF_slope in AF_slopelist:
                                AF_sizes.append([fix_AF_size,AF_intercept,AF_slope])
                        elif AF_slopes == 'zero':
                            AF_sizes.append([fix_AF_size,AF_intercept,0])

                # add surround amplitude     
                for AF_surround_ratio in AF_surround_ratios:                               
                    for AF_surround_amp in AF_surround_amps:
                        for AF_size_combo in AF_sizes:
                            grid.append(np.hstack([AF_size_combo,AF_surround_ratio,AF_surround_amp]))  

                import time as t
                print '%s: now performing grid search over %d elements in %s (%d voxels)'%(t.ctime(),len(grid),roi,mask.sum())

                # initialize empty vars not needed for grid search
                print 'using params:'
                print 'AF_fix_sizes: %s'%['%.2f'%value for value in np.unique(np.array(AF_sizes)[:,0])]
                print 'AF_bar_intercepts: %s'%['%.2f'%value for value in np.unique(np.array(AF_sizes)[:,1])]
                print 'AF_bar_slopes: %s'%['%.2f'%value for value in np.unique(np.array(AF_sizes)[:,2])]
                print 'AF_surround_amps: %s'%['%.2f'%value for value in AF_surround_amps]
                print 'AF_surround_ratio: %s'%['%.2f'%value for value in AF_surround_ratios]

                output_results=True
                n_grid_jobs = n_jobs
                n_voxel_jobs = 1
                # now do the grid search in parallel over searches
                # give message every 24 jobs
                results = Parallel(n_jobs = n_grid_jobs, verbose = int(len(grid)/24))(delayed(AF_fit_residual)
                (params,AF_shape,fit_method,res,self.stim_radius,copy.copy(these_results),AF_plot_dir,output_results,n_voxel_jobs,
                        model_area_increase_factor,circle_mask,center_method,upsample_factor,self.mask_ecc_thresholds,
                        self.detect_outliers,self.outlier_num_stds,n_eccen_bins,n_polar_bins,rectify,overlap)
                        for params in grid)     

                # put results in larger variables
                all_results[super_subject_name] = results

                # store parameters:
                params = []
                for this_grid in grid:
                    these_params = {}
                    these_params['fix_AF_size'] = this_grid[0]
                    these_params['bar_AF_intercept'] = this_grid[1]
                    these_params['bar_AF_slope'] = this_grid[2]
                    these_params['AF_surround_ratio'] = this_grid[3]
                    these_params['AF_surround_amp'] = this_grid[4]

                    params.append(these_params)

                    # add to all params
                all_params[super_subject_name] = params

            elif fit_method == 'fit':

                import time as t
                t1 = t.time()
                print '%s: now minimizing AF parameters in parallel over voxels in %s'%(t.ctime(),roi)

                if initialize_on == 'grid':
                    initial_params = pickle.load(open(os.path.join(initial_plot_dir,'grid_params_%s.pickle'%(comparison)),'rb'))
                elif initialize_on == 'custom':
                    initial_params = {}
                    initial_params['fix_AF_size'] = AF_fix_sizes[0]
                    initial_params['bar_AF_intercept'] = AF_intercepts[0]
                    initial_params['bar_AF_slope'] = 0
                    initial_params['AF_surround_amp'] = AF_surround_amps[0]
                    initial_params['AF_surround_ratio'] = AF_surround_ratios[0]

                ## initiate parameters:
                params = Parameters()
                params.add('fix_AF_size', value=initial_params['fix_AF_size'],min=1e-2,max=15)
                params.add('bar_AF_intercept', value= initial_params['bar_AF_intercept'],min=1e-2,max=15)
                params.add('bar_AF_slope',value=initial_params['bar_AF_slope'],min=1e-2,max=10)
                params.add('AF_surround_amp',value=initial_params['AF_surround_amp'],min=-0.5,max=0)
                params.add('AF_surround_ratio',value=initial_params['AF_surround_ratio'],min=0,max=10)

                if AF_slopes == 'zero':
                    params['bar_AF_slope'].vary = False
                params['AF_surround_ratio'].vary = False
                if initial_params['AF_surround_amp'].value == 0:
                    params['AF_surround_amp'].vary = False

                minimize(AF_fit_residual, params, args=(), kws={
                    'AF_shape':AF_shape,
                    'stim_radius':stim_radius,
                    'outlier_num_stds':outlier_num_stds,
                    'these_results':these_results,
                    'fit_method':fit_method,
                    'output_results':False,
                    'n_jobs':n_jobs,
                    'model_area_increase_factor':model_area_increase_factor,
                    'circle_mask':circle_mask,
                    'center_method':center_method,
                    'upsample_factor':upsample_factor,
                    'mask_ecc_thresholds':mask_ecc_thresholds,
                    'detect_outliers':detect_outliers,
                    'n_eccen_bins':n_eccen_bins,
                    'n_polar_bins':n_polar_bins,
                    'rectify':rectify
                    }
                    ,method='powell')

                # recreate results
                output_results = True
                results = AF_fit_residual(params,AF_shape,fit_method,res,stim_radius,these_results,dm_dir,output_results,n_jobs,
                    model_area_increase_factor,circle_mask,center_method,upsample_factor,mask_ecc_thresholds,
                    detect_outliers,outlier_num_stds,n_eccen_bins,n_polar_bins)    

                print '%s: done minimizing AF parameters (took about %d min)'%(t.ctime(),(t.time()-t1)/60)

                # store parameters:
                param_dict = {}
                param_dict['fix_AF_size'] = params['fix_AF_size'].value
                param_dict['bar_AF_intercept'] = params['bar_AF_intercept'].value
                param_dict['bar_AF_slope'] = params['bar_AF_slope'].value
                param_dict['AF_surround_amp'] = params['AF_surround_amp'].value
                param_dict['AF_surround_ratio'] = params['AF_surround_ratio'].value

                all_results[super_subject_name][roi] = results
                all_params[super_subject_name][roi] = param_dict

        param_postFix = ''#'bar%s_fix%s_amp%s'%(AF_intercepts,AF_fix_sizes,AF_surround_amps)
        # and save the results 
        pickle.dump( all_results, open(os.path.join(this_plot_dir,'%s_results_%s_%s.pickle'%(fit_method,conditions[0],param_postFix)), "wb" ) )
        pickle.dump( all_params, open(os.path.join(this_plot_dir,'%s_params_%s_%s.pickle'%(fit_method,conditions[0],param_postFix)), "wb" ) )

    def compare_AF_models(self,
        model_name = 'whatever',
        conditions = ['Stim'],
        subjects = ['super_subject'],
        ecc_diff_n_bins = 4,
        save_postfix = '',
        ecc_plot = False,
        AF_bar_param_plot = False,
        arrow_plot = False,
        RSS_matrix_plot = False,
        ecc_plot_rois=['V3'],
        markers = ['o','s','^','v','*'],
        surround_suppression_params = False,
        ecc_diff_corr_plot = False,
        AF_fit = False,
        jacknife = True,
        per_subject = False,
        AF_mechanics_plot = False,
        ):

        if AF_fit:
            prefix = 'fit'
        else:
            prefix = 'grid'

        all_subject_arrays = []#[subjects]
        if jacknife:
            model_dir = os.path.join(self.group_plot_dir,'AF_modeling_super_subjects/')
            figdir = 'Figure_6_super_sub'
            for left_out_subject in subjects:
                this_subject_array = copy.copy(subjects)
                popped_subject = this_subject_array.pop(np.where(np.array(self.subjects)==left_out_subject)[0][0])
                all_subject_arrays.append(this_subject_array)
                diff_vector_n_ecc_bins = 8
                diff_vector_n_polar_bins = 8
                min_vox_per_arrow = 1
        if per_subject:
            all_subject_arrays = [[sub] for sub in subjects]
            model_dir = os.path.join(self.group_plot_dir,'AF_modeling_over_subjects/')
            figdir = 'Figure_6_over_sub'
            diff_vector_n_ecc_bins = 8
            diff_vector_n_polar_bins = 8
            min_vox_per_arrow = 1

        all_pickle_data = {}
        all_pickle_params = {}
        for condition in conditions:
            all_pickle_data[condition] = {}
            all_pickle_params[condition] = {}
            for roi in ecc_plot_rois:
                this_model_name = model_name + '_%s'%roi
                # creat folder for this model type
                this_model_dir = os.path.join(model_dir,this_model_name)
                print 'loading AF fit results and stats for model %s, roi %s'%(model_name,roi)
                all_pickle_data[condition][roi] = pickle.load(open(os.path.join(this_model_dir,'%s_results_%s_.pickle'%(prefix,condition)),'rb'))
                all_pickle_params[condition][roi] = pickle.load(open(os.path.join(this_model_dir,'%s_params_%s_.pickle'%(prefix,condition)),'rb'))

        bin_edges = np.linspace(self.mask_ecc_thresholds[0],self.mask_ecc_thresholds[1],ecc_diff_n_bins+1)

        self.create_plot_dir(figdir)   
        for condition in conditions:

            # determine previous model dir
            all_data = {}
            # for model in models:
            all_data = {}
            all_data['results'] = {}
            # all_data['stats'] = {}
            all_data['params'] = {}
            # all_data['RSS_matrix'] = {}

            for roi in ecc_plot_rois:
                print 'selecting best params for model %s, roi %s'%(model_name,roi)

                all_data['params'][roi] = {}
                all_data['results'][roi] = {}

                these_data =  all_pickle_data[condition][roi]
                these_params = all_pickle_params[condition][roi]

                # get params and results
                for this_subject_array in all_subject_arrays:

                    subject = '_'.join(this_subject_array)
                    n_models = len(these_data[subject])

                    # first get the arrows for the observed data
                    these_results =  these_data[subject][0]

                    # get data 
                    cond_0_data =  these_results['flipped_cond_0_data']
                    cond_1_data =  these_results['flipped_cond_1_data']

                    weights = these_results['weights']

                    # # get the arrow data
                    arrow_starts, arrow_diffs = self.functions.get_arrows('quadrant',diff_vector_n_ecc_bins,self.mask_ecc_thresholds,diff_vector_n_polar_bins,
                                                    cond_0_data,cond_1_data,weights,self.detect_outliers,self.outlier_num_stds,'cond_1',min_vox_per_arrow=min_vox_per_arrow)
                    valid_arrows = ~np.isnan(arrow_diffs[:,0])
                    arrow_diffs = arrow_diffs[valid_arrows]

                    # and now per model the arrows
                    errors = []
                    for mi in range(n_models):
                        # pdone = model/n_models*100
                        # sys.stdout.write('Determining best model for subject %d/%d %d%s\r'%(si+1,len(all_subject_arrays),pdone,'%'))
                        # sys.stdout.flush()  
                        these_results =  these_data[subject][mi]

                        cond_1_data = np.vstack([these_results['fix_RFx'],these_results['fix_RFy']]) * these_results['sign_cond_1_data']
                        cond_0_data = np.vstack([these_results['bar_RFx'],these_results['bar_RFy']]) * these_results['sign_cond_1_data']

                        weights = these_results['weights']

                        arrow_starts_predicted, arrow_diffs_predicted = self.functions.get_arrows('quadrant',diff_vector_n_ecc_bins,self.mask_ecc_thresholds,diff_vector_n_polar_bins,
                                                            cond_0_data,cond_1_data,weights,False,self.outlier_num_stds,'cond_1',min_vox_per_arrow=min_vox_per_arrow)

                        # use valid arrows from data to select arrows:
                        arrow_diffs_predicted = arrow_diffs_predicted[valid_arrows]

                        # calculate errors
                        error_lengths = np.linalg.norm(arrow_diffs-arrow_diffs_predicted,axis=1)

                        # get rid of any nans
                        error_lengths = error_lengths[~np.isnan(error_lengths)]

                        # now create measure of model fidelity
                        errors.append(np.mean(error_lengths))

                    best_model_this_subject = np.argmin(errors)

                    all_data['results'][roi][subject] = these_data[subject][best_model_this_subject]
                    all_data['params'][roi][subject] = these_params[subject][best_model_this_subject]

            subs = ['_'.join(sa) for sa in all_subject_arrays if not len(sa)==len(subjects)]            
            if jacknife:
                subarr = ['over_subjects']
            else:
                subarr = (['over_subjects']+subs)
                # subarr = subs
            for subject in subarr:
                if subject == 'over_subjects':
                    these_subjects = subs
                elif subject == 'super_subject':
                    these_subjects = ['_'.join(subjects)]
                else:
                    these_subjects = [subject]

                if AF_mechanics_plot:

                    for roi in self.rois_for_plot:
                    # for roi in ['V3','IPS0']:
                        print 'creating AF mechanics plot for %s, %s'%(subject,roi)

                        fix_sd_ecc_over_sd_eccs_x = []
                        fix_sd_ecc_over_sd_eccs_y = []
                        stim_sd_ecc_over_sd_eccs_x = []
                        stim_sd_ecc_over_sd_eccs_y = []
                        stim_fix_ecc_over_fix_eccs_x = []
                        stim_fix_ecc_over_fix_eccs_y = []
                        for sub_subject in these_subjects:

                            # get the different eccentricities
                            these_results =  all_data['results'][roi][sub_subject]  
                            weights = these_results['weights']
                            SD_ecc = np.linalg.norm([these_results['SD_RFx'],these_results['SD_RFy']],axis=0)
                            stim_ecc = np.linalg.norm([these_results['bar_RFx'],these_results['bar_RFy']],axis=0)
                            fix_ecc = np.linalg.norm([these_results['fix_RFx'],these_results['fix_RFy']],axis=0)

                            # now bin those over sd and fix ecc
                            # first, fix-sd ecc
                            mean_x, x_cis, mean_y, y_cis, y_ps,N = self.functions.fixed_bins(y_data=fix_ecc-SD_ecc,x_data=fix_ecc,weights=weights,n_bins=ecc_diff_n_bins,
                                ci_factor=self.ci_factor,ecc_thresholds=self.mask_ecc_thresholds,outlier_num_stds=self.outlier_num_stds,detect_inliers=self.detect_outliers)
                            fix_sd_ecc_over_sd_eccs_x.append(mean_x) 
                            fix_sd_ecc_over_sd_eccs_y.append(mean_y) 

                            # then, stim-sd ecc
                            mean_x, x_cis, mean_y, y_cis, y_ps,N = self.functions.fixed_bins(y_data=stim_ecc-SD_ecc,x_data=fix_ecc,weights=weights,n_bins=ecc_diff_n_bins,
                                ci_factor=self.ci_factor,ecc_thresholds=self.mask_ecc_thresholds,outlier_num_stds=self.outlier_num_stds,detect_inliers=self.detect_outliers)
                            stim_sd_ecc_over_sd_eccs_x.append(mean_x) 
                            stim_sd_ecc_over_sd_eccs_y.append(mean_y) 

                            # now stim-fix ecc
                            mean_x, x_cis, mean_y, y_cis, y_ps,N = self.functions.fixed_bins(y_data=stim_ecc-fix_ecc,x_data=fix_ecc,weights=weights,n_bins=ecc_diff_n_bins,
                                ci_factor=self.ci_factor,ecc_thresholds=self.mask_ecc_thresholds,outlier_num_stds=self.outlier_num_stds,detect_inliers=self.detect_outliers)
                            stim_fix_ecc_over_fix_eccs_x.append(mean_x) 
                            stim_fix_ecc_over_fix_eccs_y.append(mean_y) 

                        f = pl.figure(figsize=(2.5,1.25))
                        lw = 1.5
                        hl = 0.01
                        c =  colorsys.hsv_to_rgb(0,0,0.5)
                        c2 =  colorsys.hsv_to_rgb(0,0,0.75)
                        # now mean over subjects
                        s = pl.subplot(121)
                        pl.plot(np.nanmean(fix_sd_ecc_over_sd_eccs_x,axis=0),np.nanmean(fix_sd_ecc_over_sd_eccs_y,axis=0),c=c,ls='-',lw=lw,label='fix-SD')
                        pl.plot(np.nanmean(stim_sd_ecc_over_sd_eccs_x,axis=0),np.nanmean(stim_sd_ecc_over_sd_eccs_y,axis=0),c=c2,ls='-',lw=lw,label='stim-SD')
                        ylims = self.functions.find_yticks(s.get_ylim(),ndec=1)
                        pl.ylim([ylims[0][0],ylims[0][-1]])
                        pl.yticks(ylims[0],ylims[1])
                        pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])
                        pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])
                        pl.axhline(0,c='k',lw=0.5)
                        pl.ylabel(r'$\Delta$ ecc (dva)')
                        sn.despine(offset=2)

                        # and second subplot
                        s = pl.subplot(122)
                        sn.despine(offset=2)
                        pl.axhline(0,c='k',lw=0.5)
                        pl.plot(np.nanmean(stim_fix_ecc_over_fix_eccs_x,axis=0),np.nanmean(stim_fix_ecc_over_fix_eccs_y,axis=0),c=c,lw=lw,label='stim-fix',ls='-')
                        if roi == 'V3':
                            pl.yticks([-.1,0,.1],['-.1','0','.1'])
                            pl.ylim([-.1,.1])
                        elif roi == 'IPS0':
                            pl.yticks([0,1],['0','1.0'])
                            pl.ylim([0,1])
                        else:
                            ylims = self.functions.find_yticks(s.get_ylim(),ndec=1)
                            pl.ylim([ylims[0][0],ylims[0][-1]])
                            pl.yticks(ylims[0],ylims[1])
                        pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])

                        pl.tight_layout(w_pad = 0.,h_pad=0.0,pad=0)
                        pl.savefig(os.path.join(self.group_plot_dir,figdir,'AF_mechanics_plot_%s_%s_%s_%s.pdf'%(condition,subject,save_postfix,roi)))
                        pl.close()

                if ecc_plot:
                    #############
                    ## ECC PLOT
                    #############
                    print 'creating ecc plot for %s'%subject

                    f=pl.figure(figsize = (3.75,1))
                    ri = -1
                    ecc_diff_n_bins=4
                    for rgi,roi_group in enumerate(np.sort(self.roi_groups_for_plot.keys())):
                        # s = f.add_subplot(self.roi_group_subplot_grid[0],self.roi_group_subplot_grid[1]-1,rgi+1,adjustable='box-forced',)
                        s = f.add_subplot(self.roi_group_subplot_grid[0],self.roi_group_subplot_grid[1],rgi+1,adjustable='box-forced',)
                        # pl.title('%s'%(' '.join(roi_group.split('_')[1:])))
                        pl.axhline(0,c='k',lw=0.5,ls='-')
                        # then loop over all subrois
                        for this_ri, roi in enumerate(self.roi_groups_for_plot[roi_group]):
                            ri+=1
                            ############
                            # first the observed data
                            ############

                            all_mean_x = []
                            all_mean_y = []
                            for sub_subject in these_subjects:

                                these_results =  all_data['results'][roi][sub_subject]

                                ecc_1 = these_results['ecc_cond_1']
                                ecc_diff = these_results['ecc_cond_0']-these_results['ecc_cond_1']
                                weights = these_results['weights']

                                # get binned data
                                mean_x, x_cis, mean_y, y_cis, y_ps,N = self.functions.fixed_bins(y_data=ecc_diff,x_data=ecc_1,weights=weights,n_bins=ecc_diff_n_bins,
                                    ci_factor=self.ci_factor,ecc_thresholds=self.mask_ecc_thresholds,outlier_num_stds=self.outlier_num_stds,detect_inliers=self.detect_outliers)

                                all_mean_x.append(mean_x)
                                all_mean_y.append(mean_y)

                            if len(these_subjects)>1:

                                mean_x, x_cis, y_t, x_p, ds, N = self.CUO.stats_from_data(np.array(all_mean_x).T,self.ci_factor)                               
                                mean_y, y_cis, x_t, y_p, ds, N = self.CUO.stats_from_data(np.array(all_mean_y).T,self.ci_factor)                                            
                                
                            # plot mean
                            pl.plot(mean_x,mean_y,marker=['o','s','^'][this_ri],markersize=4,mec='w',lw=0,color=self.roi_colors[ri],mew=0.5)
                            for bi in range(ecc_diff_n_bins):
                                if not y_cis != []:
                                    pl.plot(np.repeat(mean_x[bi],2),y_cis[bi],'-',lw = 0.5,color='w') 
                                    pl.plot(np.repeat(mean_x[bi],2),y_cis[bi],'-',lw = 0.5,color=self.roi_colors[ri],alpha=0.5) 


                            ############
                            # now for the predicted data
                            ############

                            all_mean_x = []
                            all_mean_y = []
                            for sub_subject in these_subjects:

                                these_results =  all_data['results'][roi][sub_subject]

                                ecc_0 = np.linalg.norm([these_results['bar_RFx'],these_results['bar_RFy']],axis=0)
                                ecc_1 = np.linalg.norm([these_results['fix_RFx'],these_results['fix_RFy']],axis=0)
                                ecc_diff = ecc_0 - ecc_1

                                weights = these_results['weights']

                                # get binned data
                                mean_x, x_cis, mean_y, y_cis, y_ps,Ns = self.functions.fixed_bins(y_data=ecc_diff,x_data=ecc_1,weights=weights,n_bins=ecc_diff_n_bins,
                                    ci_factor=self.ci_factor,ecc_thresholds=self.mask_ecc_thresholds,outlier_num_stds=self.outlier_num_stds)
                                
                                all_mean_x.append(mean_x)
                                all_mean_y.append(mean_y)

                            if len(these_subjects)>1:
                                mean_x, x_cis, y_t, x_p, ds, N = self.CUO.stats_from_data(np.array(all_mean_x).T,self.ci_factor)                               
                                mean_y, y_cis, x_t, y_p, ds, N = self.CUO.stats_from_data(np.array(all_mean_y).T,self.ci_factor)                                            
                                
                            # plot
                            min_n=1
                            pl.plot(mean_x,mean_y,linestyle='-',color=self.roi_colors[ri],lw=1)
                            y_cis = np.array([ci if Ns[i]>min_n else [np.nan,np.nan] for i,ci in enumerate(y_cis)])
                            pl.fill_between(mean_x,y_cis[:,0],y_cis[:,1],color=self.roi_colors[ri],alpha=0.2)


                        sn.despine(offset=2)
                        if rgi==0:
                            pl.ylabel(r'$\Delta$ ecc (dva)')
                            # pl.xlabel('Fix ecc (dva)')
                        pl.xticks(self.mask_ecc_thresholds,['0',str(self.mask_ecc_thresholds[1])])
                        pl.xlim(self.mask_ecc_thresholds)
                        # ylim = [0.15,0.7,1.4,0.15][rgi]
                        if subject != 'over_subjects':
                            ylim = [0.2,3.5,4,0.25][rgi]
                        else:
                            ylim = np.max(np.abs(s.get_ylim()))                        
                        pl.ylim(-ylim,ylim)
                        pl.yticks([-ylim,0,ylim],[str(-ylim),'0',str(ylim)])
                        # pl.yticks(self.functions.find_yticks(s.get_ylim())[0],self.functions.find_yticks(s.get_ylim())[1])

                    pl.tight_layout(w_pad = 0.,h_pad=0.0,pad=0)
                    # pl.savefig(os.path.join(self.group_plot_dir,figdir,'%s_%s_ecc_plot_%s_%s_%s_%s.pdf'%(model_name,prefix,'_'.join(ecc_plot_rois),save_postfix,condition,subject)))
                    pl.savefig(os.path.join(self.group_plot_dir,figdir,'ecc_plot_%s_%s_%s.pdf'%(condition,subject,save_postfix)))
                    pl.close()

                if AF_bar_param_plot * (subject == 'over_subjects'):
                    print 'creating AF_bar_param_plot for %s'%subject
                    #############
                    ## AF param plot
                    #############
                    roi_indices = [np.where(this_roi==np.array(self.rois_for_plot))[0][0] for this_roi in ecc_plot_rois]
                    xs = np.array(self.mask_ecc_thresholds)

                    # bar plot parameters
                    f = pl.figure(figsize=(3.75,1.5))
                    s1 = f.add_subplot(121)
                    pl.title('fixation attention field')
                    s2 = f.add_subplot(122)
                    pl.title('stimulus attention field')
                    for ri, roi in enumerate(ecc_plot_rois):
                        bar_AFs=[]
                        fix_AFs=[]
                        for sub_subject in these_subjects:

                            fix_AFs.append(all_data['params'][roi][sub_subject]['fix_AF_size'])
                            bar_AFs.append(all_data['params'][roi][sub_subject]['bar_AF_intercept'])

                        fix_AF = np.nanmean(fix_AFs)
                        bar_AF = np.nanmean(bar_AFs)
                        # fix_AF, fix_AFs_ci, p,N = self.CUO.bootstrap(np.array(fix_AFs),detect_inliers=False)
                        # bar_AF, bar_AF_ci, p,N = self.CUO.bootstrap(np.array(bar_AFs),detect_inliers=False)

                        bar_width = 0.9

                        this_color = self.roi_colors[roi_indices[ri]]

                        pl.sca(s1)
                        pl.bar(ri,fix_AF,color=this_color,edgecolor=this_color)
                        pl.plot((np.arange(len(these_subjects))*0.05)+(ri+0.3),fix_AFs,'o',ms=2,alpha=1,c='w')
                        pl.plot((np.arange(len(these_subjects))*0.05)+(ri+0.3),fix_AFs,'o',ms=2,alpha=0.2,c=this_color)

                        pl.sca(s2)
                        pl.bar(ri,bar_AF+bar_width,color=this_color,edgecolor=this_color)
                        pl.plot((np.arange(len(these_subjects))*0.05)+(ri+0.3),np.array(bar_AFs)+bar_width,'o',ms=2,alpha=1,c='w')
                        pl.plot((np.arange(len(these_subjects))*0.05)+(ri+0.3),np.array(bar_AFs)+bar_width,'o',ms=2,alpha=0.2,c=this_color)

                    pl.sca(s1)

                    pl.xticks([])
                    pl.yticks([0,3])
                    sn.despine(offset=2 )
                    pl.ylim(0,3)
                    pl.ylabel('size (dva)')

                    pl.sca(s2)

                    pl.xticks([])
                    pl.yticks([0,3])
                    sn.despine(offset=2 )
                    pl.ylim(0,3)

                    pl.tight_layout()#w_pad = 0.,h_pad=0.0,pad=0)
                    # pl.savefig(os.path.join(self.group_plot_dir,figdir,'%s_%s_best_AF_params_bar_%s_%s_%s_%s.pdf'%(condition,model_name,prefix,save_postfix,'_'.join(ecc_plot_rois),subject)),transparent=False)
                    pl.savefig(os.path.join(self.group_plot_dir,figdir,'param_plot_%s_%s_%s.pdf'%(condition,subject,save_postfix)))                  
                    pl.close()  

                if arrow_plot:
                    ###############
                    ## ARROW PLOTS
                    ############### 
                    print 'creating arrow plot for %s'%subject
                    roi_indices = [np.where(this_roi==np.array(self.rois_for_plot))[0][0] for this_roi in ecc_plot_rois]
                    # f = pl.figure(figsize=(np.floor(sqrt(len(ecc_plot_rois)))*2,np.ceil(sqrt(len(ecc_plot_rois)))*2))
                    # f = pl.figure(figsize=(3.75,1.875))
                    f = pl.figure(figsize=(5,2.5))
                    # f = pl.figure(figsize=(3.5,3.5))

                    for ri, roi in zip(roi_indices,ecc_plot_rois):
                        # s = f.add_subplot(np.ceil(sqrt(len(ecc_plot_rois))),np.floor(sqrt(len(ecc_plot_rois))),ri+1,aspect='equal')
                        # s = f.add_subplot(3,3,ri+1,aspect='equal')
                        s = f.add_subplot(2,5,ri+1,aspect='equal')
                        # pl.title(roi,color=self.roi_colors[ri],fontweight='bold')                        

                        all_arrow_starts = []
                        all_arrow_diffs = []

                        for sub_subject in these_subjects:

                            these_results =  all_data['results'][roi][sub_subject]

                            # get data 
                            cond_0_data =  these_results['flipped_cond_0_data']
                            cond_1_data =  these_results['flipped_cond_1_data']

                            weights = these_results['weights']

                            # # get the arrow data
                            arrow_starts, arrow_diffs = self.functions.get_arrows('quadrant',diff_vector_n_ecc_bins,self.mask_ecc_thresholds,diff_vector_n_polar_bins,
                                                            cond_0_data,cond_1_data,weights,self.detect_outliers,self.outlier_num_stds,'cond_1',min_vox_per_arrow)

                            all_arrow_starts.append(arrow_starts)
                            all_arrow_diffs.append(arrow_diffs)

                        if len(these_subjects) > 1:
                            arrow_starts = np.nanmean(all_arrow_starts,axis=0)
                            arrow_diffs = np.nanmean(all_arrow_diffs,axis=0)

                        for circle_edge in np.linspace(0,self.mask_ecc_thresholds[1],5)[1:]:
                            circle = pl.Circle((0,0),circle_edge,color='k',edgecolor='k',ls='dashed',fill=False,lw=0.5)
                            s.add_artist(circle)

                        pl.yticks([0,self.stim_radius])
                        pl.xticks([0,self.stim_radius])

                        # plot data arrows in for loop
                        head_width = 0.1
                        head_length = 0.1
                        tail_width = 0.75
                        for bi in range(len(arrow_starts)):
                            pl.arrow(arrow_starts[bi,0],arrow_starts[bi,1],arrow_diffs[bi,0],arrow_diffs[bi,1],color='k',lw=tail_width,head_length=head_length,head_width=head_width,alpha=1,edgecolor='w',length_includes_head=False)
                        
                        pl.axhline(0,color='k',linewidth=0.5)
                        pl.axvline(0,color='k',linewidth=0.5)
                        pl.xlim(-0.5,self.stim_radius*1.1)
                        pl.ylim(-0.5,self.stim_radius*1.1)
                        sn.despine(ax=s)


                        all_arrow_starts = []
                        all_arrow_diffs = []
                        
                        for sub_subject in these_subjects:
                            these_results =  all_data['results'][roi][sub_subject]

                            cond_1_data = np.vstack([these_results['fix_RFx'],these_results['fix_RFy']]) * these_results['sign_cond_1_data']
                            cond_0_data = np.vstack([these_results['bar_RFx'],these_results['bar_RFy']]) * these_results['sign_cond_1_data']

                            weights = these_results['weights']

                            # get arrow data
                            arrow_starts, arrow_diffs = self.functions.get_arrows('quadrant',diff_vector_n_ecc_bins,self.mask_ecc_thresholds,diff_vector_n_polar_bins,
                                                            cond_0_data,cond_1_data,weights,self.detect_outliers,self.outlier_num_stds,'cond_1',min_vox_per_arrow)
                            all_arrow_starts.append(arrow_starts)
                            all_arrow_diffs.append(arrow_diffs)

                        if len(these_subjects) > 1:
                            arrow_starts = np.nanmean(all_arrow_starts,axis=0)
                            arrow_diffs = np.nanmean(all_arrow_diffs,axis=0)

                        for circle_edge in [0,self.stim_radius]:
                            circle = pl.Circle((0,0),circle_edge,color='k',edgecolor='k',fill=False)
                            s.add_artist(circle)

                        circle = pl.Circle((0,0),self.mask_ecc_thresholds[1],color='k',edgecolor='k',ls='dashed',fill=False)
                        s.add_artist(circle)


                        pl.yticks([])
                        pl.xticks([])

                        for bi in range(len(arrow_starts)):
                            pl.arrow(arrow_starts[bi,0],arrow_starts[bi,1],arrow_diffs[bi,0],arrow_diffs[bi,1],color=self.roi_colors[ri],head_length=head_length,lw=tail_width,head_width=head_width,alpha=1,edgecolor='w',length_includes_head=False)
                        
                        pl.axhline(0,color='k',linewidth=0.5)
                        pl.axvline(0,color='k',linewidth=0.5)
                        pl.xlim(0,self.stim_radius)
                        pl.ylim(0,self.stim_radius)
                        sn.despine(ax=s)

                    pl.tight_layout()
                    # pl.savefig(os.path.join(self.group_plot_dir,figdir,'%s_%s_arrow_plot_%s_%s_%s.pdf'%(model_name,prefix,ecc_plot_rois,save_postfix,subject)))
                    pl.savefig(os.path.join(self.group_plot_dir,figdir,'arrow_plot_%s_%s_%s.pdf'%(condition,subject,save_postfix)))
                    pl.close()


    def compare_AF_params_between_conditions(self,
        model_name = 'whatever',
        conditions = ['Color','Speed'],
        AF_fit = False,
        save_postfix = '',
        ecc_plot_rois=['V3'],
        markers = ['o','*'],
        ecc_diff_n_bins = 4,
        bar_param_plot = False,
        jacknife = True,
        per_subject=False,
        subjects = ['DE','TK','NA','JS','JW'],
        ):

        if AF_fit:
            prefix = 'fit'
        else:
            prefix = 'grid'

        all_subject_arrays = []#[subjects]
        if jacknife:
            model_dir = os.path.join(self.group_plot_dir,'AF_modeling_super_subjects/')
            figdir = 'Figure_7_super_sub'
            for left_out_subject in subjects:
                this_subject_array = copy.copy(subjects)
                popped_subject = this_subject_array.pop(np.where(np.array(self.subjects)==left_out_subject)[0][0])
                all_subject_arrays.append(this_subject_array)
                diff_vector_n_ecc_bins = 8
                diff_vector_n_polar_bins = 8
                min_vox_per_arrow = 1               
        if per_subject:
            all_subject_arrays = [[sub] for sub in subjects]
            model_dir = os.path.join(self.group_plot_dir,'AF_modeling_over_subjects/')
            figdir = 'Figure_7_over_sub'
            diff_vector_n_ecc_bins = 8
            diff_vector_n_polar_bins = 8
            min_vox_per_arrow = 1

        self.create_plot_dir(figdir)   

        # determine previous model dir
        all_data = {}
        # for model in models:
        all_data = {}
        all_data['results'] = {}
        all_data['params'] = {}
     
        # now that we've got the best Stim-Fix result, only regard those models with that fix af size in the color-speed comparison:
        for condition in conditions:
            all_data['results'][condition] = {}
            all_data['params'][condition] = {}
            for roi in ecc_plot_rois:
                print 'loading %s AF fit results and stats for model %s, roi %s'%(condition,model_name,roi)

                all_data['params'][condition][roi] = {}
                all_data['results'][condition][roi] = {}

                this_model_name = model_name + '_%s'%roi
                # creat folder for this model type
                this_model_dir = os.path.join(model_dir,this_model_name)

                these_data = pickle.load(open(os.path.join(this_model_dir,'%s_results_%s_.pickle'%(prefix,condition)),'rb'))
                these_params = pickle.load(open(os.path.join(this_model_dir,'%s_params_%s_.pickle'%(prefix,condition)),'rb'))

                # get params and results
                for si,this_subject_array in enumerate(all_subject_arrays):
                    print 'Determining best model for subject %d/%d'%(si+1,len(all_subject_arrays))
                    subject = '_'.join(this_subject_array)
                    n_models = len(these_data[subject])

                    # first get the arrows for the observed data
                    these_results =  these_data[subject][0]

                    # get data 
                    cond_0_data =  these_results['flipped_cond_0_data']
                    cond_1_data =  these_results['flipped_cond_1_data']

                    weights = these_results['weights']

                    # # get the arrow data
                    arrow_starts, arrow_diffs = self.functions.get_arrows('quadrant',diff_vector_n_ecc_bins,self.mask_ecc_thresholds,diff_vector_n_polar_bins,
                                                    cond_0_data,cond_1_data,weights,self.detect_outliers,self.outlier_num_stds,'cond_1',min_vox_per_arrow=min_vox_per_arrow)
                    valid_arrows = ~np.isnan(arrow_diffs[:,0])                    
                    arrow_diffs = arrow_diffs[valid_arrows]

                    # and now per model the arrows
                    errors = []
                    for mi in range(n_models):
                        # pdone = model/n_models*100
                        # sys.stdout.write('Determining best model for subject %d/%d %d%s\r'%(si+1,len(all_subject_arrays),pdone,'%'))
                        # sys.stdout.flush()

                        these_results =  these_data[subject][mi]

                        cond_1_data = np.vstack([these_results['fix_RFx'],these_results['fix_RFy']]) * these_results['sign_cond_1_data']
                        cond_0_data = np.vstack([these_results['bar_RFx'],these_results['bar_RFy']]) * these_results['sign_cond_1_data']

                        weights = these_results['weights']

                        # get arrow data
                        arrow_starts_predicted, arrow_diffs_predicted = self.functions.get_arrows('quadrant',diff_vector_n_ecc_bins,self.mask_ecc_thresholds,diff_vector_n_polar_bins,
                                                        cond_0_data,cond_1_data,weights,False,self.outlier_num_stds,'cond_1',min_vox_per_arrow=min_vox_per_arrow)
                        arrow_diffs_predicted = arrow_diffs_predicted[valid_arrows]


                        # calculate errors
                        error_lengths = np.linalg.norm(arrow_diffs-arrow_diffs_predicted,axis=1)

                        # get rid of nans
                        if np.sum(np.isnan(error_lengths)) > 0:
                            print 'WARNING STILL %d NANS IN ERRORS'%np.sum(np.isnan(error_lengths))
                        error_lengths = error_lengths[~np.isnan(error_lengths)]

                        # now create measure of model fidelity
                        errors.append(np.mean(error_lengths))

                    best_model_this_subject = np.argmin(errors)

                    all_data['results'][condition][roi][subject] = these_data[subject][best_model_this_subject]
                    all_data['params'][condition][roi][subject] = these_params[subject][best_model_this_subject]

        shell()
        subs = ['_'.join(sa) for sa in all_subject_arrays]
        # for subject in (['over_subjects']+subs):
        for subject in (['over_subjects']):
            if subject == 'over_subjects':
                these_subjects = subs
            else:
                these_subjects = [subject]

            func = np.median
            roi_indices = [np.where(this_roi==np.array(self.rois_for_plot))[0][0] for this_roi in ecc_plot_rois]
            xs = np.array(self.mask_ecc_thresholds)

            ylim = .4

            # bar plot parameters
            for mi,measure in enumerate(['bar_AF_intercept','fix_AF_size']):
                # f = pl.figure(figsize=(1.53,1.3))
                f = pl.figure(figsize=(1.5,1.5))
                s = f.add_subplot(1,1,1)#,adjustable='box-forced')#,aspect='equal')
                
                if mi ==0:
                    pl.title('stimulus attention field')
                else:
                    pl.title('fixation attention field')
                pl.axhline(0,color='k',lw=0.5)
                all_AFs = []
                all_subids = []
                all_rois = []
                all_conditions = []

                all_diffs= np.zeros((len(these_subjects),len(ecc_plot_rois)))
                all_colors= np.zeros((len(these_subjects),len(ecc_plot_rois)))
                all_speeds= np.zeros((len(these_subjects),len(ecc_plot_rois)))
                for ri, roi in enumerate(ecc_plot_rois):
                    this_color = self.roi_colors[roi_indices[ri]]
                    for si,sub_subject in enumerate(these_subjects):

                        bar_AF_color = all_data['params']['Color'][roi][sub_subject][measure]
                        bar_AF_speed = all_data['params']['Speed'][roi][sub_subject][measure]

                        if roi != 'combined':
                            # append color value
                            all_AFs.append(bar_AF_color)
                            all_subids.append(sub_subject)
                            all_rois.append(roi)
                            all_conditions.append('color')

                            # append speed value
                            all_AFs.append(bar_AF_speed)
                            all_subids.append(sub_subject)
                            all_rois.append(roi)
                            all_conditions.append('speed')

                        all_colors[si,ri]=bar_AF_color
                        all_speeds[si,ri]=bar_AF_speed

                        diff = bar_AF_speed - bar_AF_color
                        all_diffs[si,ri] = diff

                        # pl.plot(ri+.3+(.05*si),diff,'o',ms=1.5,alpha=1,c='w')
                        # pl.plot(ri+.3+(.05*si),diff,'o',ms=1.5,alpha=0.2,c=this_color)
                  
                    mean_measure, ci, t_diff, pq, dq, Nq = self.CUO.stats_from_data(all_diffs[:,ri],self.ci_factor)

                    # mean_measure, ci, pq,Nq,dq  = self.CUO.bootstrap(all_diffs[:,ri],detect_inliers=False,return_d=True)#,ci_factor=1)

                    # if pq<.05:
                    fill_color = this_color
                    # else:
                    #     fill_color = 'w'
                    pl.bar(ri,mean_measure,color=fill_color,edgecolor=this_color)
                    pl.plot([ri+0.4,ri+0.4],ci,c='w',lw=1)
                    pl.plot([ri+0.4,ri+0.4],ci,c=self.roi_colors[ri],lw=1,alpha=0.4)

                all_sub_diffs = func(all_diffs[:,:-1],axis=1)

                mean_measure, ci, t_diff, p, d, N = self.CUO.stats_from_data(all_sub_diffs,self.ci_factor)
                # mean_measure, ci, p,N,d  = self.CUO.bootstrap(all_sub_diffs,detect_inliers=False,return_d=True)
                pl.plot([0,8.8],[mean_measure,mean_measure],color='k')
                pl.fill_between([0,8.8],ci[0],ci[1],color='k',alpha=0.3,zorder=10)

                if subject == 'over_subjects':

                    t,p = sp.stats.ttest_1samp(all_sub_diffs,0)
                    mean_measure = np.mean(all_sub_diffs)
                    d = np.mean(all_sub_diffs)/np.std(all_sub_diffs)
                    N= len(all_sub_diffs)

                    print '%s on %.3f above 0 over subjects (N=%d) t: %.3f,p: %.3f, cohen_d: %.3f'%(measure,mean_measure,N,t,p,d)

                    comb_mean = all_diffs[:,-1]
                    t,p = sp.stats.ttest_1samp(comb_mean,0)
                    mean_measure = np.mean(comb_mean)
                    N= len(comb_mean)
                    d = np.mean(comb_mean)/np.std(comb_mean)
                    print '%s AF size on %.3f above in combined ROI (N=%d) t: %.3f,p: %.3f, cohen_d: %.3f'%(measure,mean_measure,N,t,p,d)


                pl.xticks([])
                # pl.ylim([0,3])
                sn.despine(offset=2 )
                # pl.ylim(s.get_ylim()[0],s.get_ylim()[1]+0.4)
                pl.ylim([-ylim,ylim])
                pl.yticks([-ylim,0,ylim],[str(-ylim),'0',str(ylim)])
                # if mi ==0:            
                #     pl.ylabel(r'TF - Color size (dva)')


                pl.tight_layout()#w_pad = 0.,h_pad=0.0,pad=0)
                pl.savefig(os.path.join(self.group_plot_dir,figdir,'%s_%s_AF_size_differences.pdf'%(measure,subject)),transparent=False)
                pl.close() 


        # and create reduced plot with only median over ROIs and the combined ROI
        subs = ['_'.join(sa) for sa in all_subject_arrays]
        # for subject in (['over_subjects']+subs):
        for subject in (['over_subjects']):
            if subject == 'over_subjects':
                these_subjects = subs
            else:
                these_subjects = [subject]

            func = np.median
            roi_indices = [np.where(this_roi==np.array(self.rois_for_plot))[0][0] for this_roi in ecc_plot_rois]
            xs = np.array(self.mask_ecc_thresholds)

            ylim = .15

            f = pl.figure(figsize=(1.5,1))
            # bar plot parameters
            for mi,measure in enumerate(['bar_AF_intercept','fix_AF_size']):
                # f = pl.figure(figsize=(1.53,1.3))
                s = f.add_subplot(1,2,mi+1)#,adjustable='box-forced')#,aspect='equal')
                
                if mi ==0:
                    pl.title('stimulus AF')
                else:
                    pl.title('fixation AF')

                pl.axhline(0,color='k',lw=0.5)
                all_AFs = []
                all_subids = []
                all_rois = []
                all_conditions = []

                all_diffs= np.zeros((len(these_subjects),len(ecc_plot_rois)))
                all_colors= np.zeros((len(these_subjects),len(ecc_plot_rois)))
                all_speeds= np.zeros((len(these_subjects),len(ecc_plot_rois)))
                for ri, roi in enumerate(ecc_plot_rois):
                    this_color = self.roi_colors[roi_indices[ri]]
                    for si,sub_subject in enumerate(these_subjects):

                        bar_AF_color = all_data['params']['Color'][roi][sub_subject][measure]
                        bar_AF_speed = all_data['params']['Speed'][roi][sub_subject][measure]

                        if roi != 'combined':
                            # append color value
                            all_AFs.append(bar_AF_color)
                            all_subids.append(sub_subject)
                            all_rois.append(roi)
                            all_conditions.append('color')

                            # append speed value
                            all_AFs.append(bar_AF_speed)
                            all_subids.append(sub_subject)
                            all_rois.append(roi)
                            all_conditions.append('speed')

                        all_colors[si,ri]=bar_AF_color
                        all_speeds[si,ri]=bar_AF_speed

                        diff = bar_AF_speed - bar_AF_color
                        all_diffs[si,ri] = diff

                        # pl.plot(ri+.3+(.05*si),diff,'o',ms=1.5,alpha=1,c='w')
                        # pl.plot(ri+.3+(.05*si),diff,'o',ms=1.5,alpha=0.2,c=this_color)
                  
                    mean_measure, ci, t_diff, pq, dq, Nq = self.CUO.stats_from_data(all_diffs[:,ri],self.ci_factor)

                    # mean_measure, ci, pq,Nq,dq  = self.CUO.bootstrap(all_diffs[:,ri],detect_inliers=False,return_d=True)#,ci_factor=1)

                    if pq<.05:
                        fill_color = this_color
                        edge_color = 'w'                        
                    else:
                        fill_color = 'w'
                        edge_color = this_color

                    if roi == 'combined':
                        pl.bar(mi,mean_measure,color=fill_color,edgecolor=edge_color,width=0.4)
                        pl.plot([mi+0.2,mi+0.2],ci,c='w',lw=1)
                        pl.plot([mi+0.2,mi+0.2],ci,c=self.roi_colors[ri],lw=1,alpha=0.4)

                all_sub_diffs = func(all_diffs[:,:-1],axis=1)

                mean_measure, ci, t_diff, p, d, N = self.CUO.stats_from_data(all_sub_diffs,self.ci_factor)
                # mean_measure, ci, p,N,d  = self.CUO.bootstrap(all_sub_diffs,detect_inliers=False,return_d=True)
                # pl.plot([0,8.8],[mean_measure,mean_measure],color='k')
                # pl.fill_between([0,8.8],ci[0],ci[1],color='k',alpha=0.3,zorder=10)
                if p < .05:
                    fill_color = 'k'
                    edge_color = 'w'
                else:
                    fill_color = 'w'
                    edge_color = 'k'
                pl.bar(mi+0.5,mean_measure,color=fill_color,edgecolor=edge_color,width=0.4)
                pl.plot([mi+0.7,mi+0.7],ci,c='w',lw=1)
                pl.plot([mi+0.7,mi+0.7],ci,c='k',lw=1,alpha=0.4)

                if subject == 'over_subjects':

                    t,p = sp.stats.ttest_1samp(all_sub_diffs,0)
                    mean_measure = np.mean(all_sub_diffs)
                    d = np.mean(all_sub_diffs)/np.std(all_sub_diffs)
                    N= len(all_sub_diffs)

                    print '%s on %.3f above 0 over subjects (N=%d) t: %.3f,p: %.3f, cohen_d: %.3f'%(measure,mean_measure,N,t,p,d)

                    comb_mean = all_diffs[:,-1]
                    t,p = sp.stats.ttest_1samp(comb_mean,0)
                    mean_measure = np.mean(comb_mean)
                    N= len(comb_mean)
                    d = np.mean(comb_mean)/np.std(comb_mean)
                    print '%s AF size on %.3f above in combined ROI (N=%d) t: %.3f,p: %.3f, cohen_d: %.3f'%(measure,mean_measure,N,t,p,d)


                pl.xticks([])
                # pl.ylim([0,3])
                sn.despine(offset=2 )
                pl.ylim(s.get_ylim()[0],s.get_ylim()[1])
                pl.ylim([-ylim,ylim])
                if mi == 0:
                    pl.yticks([-ylim,0,ylim],[str(-ylim),'0',str(ylim)])
                    pl.ylabel(r'TF - Color size (dva)')
                else:
                    pl.yticks([])                    


            pl.tight_layout()#w_pad = 0.,h_pad=0.0,pad=0)
            pl.savefig(os.path.join(self.group_plot_dir,figdir,'%s_AF_size_differences_reduced.pdf'%(subject)),transparent=False)
            pl.close() 


        # subs = ['_'.join(sa) for sa in all_subject_arrays]
        # # for subject in (['over_subjects']+subs):
        # for subject in (['over_subjects']):
        #     if subject == 'over_subjects':
        #         these_subjects = subs
        #     else:
        #         these_subjects = [subject]

        #     func = np.median
        #     roi_indices = [np.where(this_roi==np.array(self.rois_for_plot))[0][0] for this_roi in ecc_plot_rois]
        #     xs = np.array(self.mask_ecc_thresholds)

        #     ylim = .4

        #     # bar plot parameters
        #     # f = pl.figure(figsize=(1.53,1.3))
        #     f = pl.figure(figsize=(1.5,1.5))
        #     s = f.add_subplot(1,1,1)#,adjustable='box-forced')#,aspect='equal')
            
        #     if mi ==0:
        #         pl.title('stimulus attention field')
        #     else:
        #         pl.title('fixation attention field')
        #     pl.axhline(0,color='k',lw=0.5)
        #     all_AFs = []
        #     all_subids = []
        #     all_rois = []
        #     all_conditions = []

        #     all_diffs= np.zeros((len(these_subjects),len(ecc_plot_rois)))
        #     all_colors= np.zeros((len(these_subjects),len(ecc_plot_rois)))
        #     all_speeds= np.zeros((len(these_subjects),len(ecc_plot_rois)))
        #     for ri, roi in enumerate(ecc_plot_rois):
        #         this_color = self.roi_colors[roi_indices[ri]]
        #         for si,sub_subject in enumerate(these_subjects):

        #             bar_AF_color = all_data['params']['Color'][roi][sub_subject]['bar_AF_intercept'] - all_data['params']['Color'][roi][sub_subject]['fix_AF_size']
        #             bar_AF_speed  = all_data['params']['Speed'][roi][sub_subject]['bar_AF_intercept'] - all_data['params']['Speed'][roi][sub_subject]['fix_AF_size']

        #             if roi != 'combined':
        #                 # append color value
        #                 all_AFs.append(bar_AF_color)
        #                 all_subids.append(sub_subject)
        #                 all_rois.append(roi)
        #                 all_conditions.append('color')

        #                 # append speed value
        #                 all_AFs.append(bar_AF_speed)
        #                 all_subids.append(sub_subject)
        #                 all_rois.append(roi)
        #                 all_conditions.append('speed')

        #             all_colors[si,ri]=bar_AF_color
        #             all_speeds[si,ri]=bar_AF_speed

        #             diff = bar_AF_speed - bar_AF_color
        #             all_diffs[si,ri] = diff

        #             # pl.plot(ri+.3+(.05*si),diff,'o',ms=1.5,alpha=1,c='w')
        #             # pl.plot(ri+.3+(.05*si),diff,'o',ms=1.5,alpha=0.2,c=this_color)
              
        #         mean_measure, ci, t_diff, pq, dq, Nq = self.CUO.stats_from_data(all_diffs[:,ri],self.ci_factor)

        #         # mean_measure, ci, pq,Nq,dq  = self.CUO.bootstrap(all_diffs[:,ri],detect_inliers=False,return_d=True)#,ci_factor=1)

        #         # if pq<.05:
        #         fill_color = this_color
        #         # else:
        #         #     fill_color = 'w'
        #         pl.bar(ri,mean_measure,color=fill_color,edgecolor=this_color)
        #         pl.plot([ri+0.4,ri+0.4],ci,c='w',lw=1)
        #         pl.plot([ri+0.4,ri+0.4],ci,c=self.roi_colors[ri],lw=1,alpha=0.4)

        #     all_sub_diffs = func(all_diffs[:,:-1],axis=1)

        #     mean_measure, ci, t_diff, p, d, N = self.CUO.stats_from_data(all_sub_diffs,self.ci_factor)
        #     # mean_measure, ci, p,N,d  = self.CUO.bootstrap(all_sub_diffs,detect_inliers=False,return_d=True)
        #     pl.plot([0,8.8],[mean_measure,mean_measure],color='k')
        #     pl.fill_between([0,8.8],ci[0],ci[1],color='k',alpha=0.3,zorder=10)

        #     if subject == 'over_subjects':

        #         t,p = sp.stats.ttest_1samp(all_sub_diffs,0)
        #         mean_measure = np.mean(all_sub_diffs)
        #         d = np.mean(all_sub_diffs)/np.std(all_sub_diffs)
        #         N= len(all_sub_diffs)

        #         print 'on %.3f above 0 over subjects (N=%d) t: %.3f,p: %.3f, cohen_d: %.3f'%(mean_measure,N,t,p,d)

        #         comb_mean = all_diffs[:,-1]
        #         t,p = sp.stats.ttest_1samp(comb_mean,0)
        #         mean_measure = np.mean(comb_mean)
        #         N= len(comb_mean)
        #         d = np.mean(comb_mean)/np.std(comb_mean)
        #         print 'AF size on %.3f above in combined ROI (N=%d) t: %.3f,p: %.3f, cohen_d: %.3f'%(mean_measure,N,t,p,d)


        #     pl.xticks([])
        #     # pl.ylim([0,3])
        #     sn.despine(offset=2 )
        #     # pl.ylim(s.get_ylim()[0],s.get_ylim()[1]+0.4)
        #     pl.ylim([-ylim,ylim])
        #     pl.yticks([-ylim,0,ylim],[str(-ylim),'0',str(ylim)])
        #     # if mi ==0:            
        #     #     pl.ylabel(r'TF - Color size (dva)')


        #     pl.tight_layout()#w_pad = 0.,h_pad=0.0,pad=0)
        #     pl.savefig(os.path.join(self.group_plot_dir,figdir,'%s_AF_size_differences.pdf'%(subject)),transparent=False)
        #     pl.close() 


# 