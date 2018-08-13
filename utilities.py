from __future__ import division

# import python moduls
from IPython import embed as shell
import numpy as np
from skimage.morphology import disk
import copy
import mne
import pickle
import time as t
import os, sys
from statsmodels.stats.weightstats import DescrStatsW
from wpca import WPCA
import pycircstat as pcs
import matplotlib as mpl
import pylab as pl
import seaborn as sn
from scipy import stats
sn.set(style="white")
mpl.rc_file_defaults()


class CustomStatUtilities(object):
    """
    This class contains all custom written general functionalities.
    """

    def __init__(self,reps=1e3):

        self.reps = reps

    def GLM(self,dm,data,c):
        """
        dm should be of shape [timepoints,reg]
        data should be of shape [timepoints, voxels]
        c should be of shape [n_contrasts,n_regressors]
        
        returns:
        prediction
        betas
        tstat
        """
        
        from numpy.linalg import pinv
        import scipy.stats as stats


        # # convert variables to matrices
        dm = np.mat(dm)
        data = np.mat(data)
        c = np.mat(c)

        # # betas
        betas = pinv(dm.T * dm) * dm.T * data

        # #then, determine difference in regressor scalling
        cope = c*betas # this is size [now num_contrasts,num_voxels] [2,3]

        # # find estimate of the error variance
        df = (dm.shape[0] - dm.shape[1])
        prediction = dm*betas

        # sse, take diagonal is taking sse per voxel (off diagnoals are combinations of voxels)
        residuals = data-prediction
        sse = np.diag(residuals.T*residuals)[np.newaxis,:]
        # normalize sse for df
        error_var = (sse / df)

        # now find the design variance per contrast, again take diagonal, off diagonals is variance of contrast combinations
        design_var = np.diag(c * pinv(dm.T * dm) * c.T)[:,np.newaxis]

        # variance per cope and per voxel:
        varcope = np.sqrt(error_var * design_var)

        # and compute the t-stat
        t_stat = cope/varcope

        # to ensure that zstats will not reach infinity, use this conversion:
        # (see http://www.stats.uwo.ca/faculty/aim/2010/JSSSnipets/V23N1.pdf)
        # This is because a p of .9999956 will be less precies than .0000044, 
        # as the latter is internally represented as 4.4e-6,
        # which leaves much more room for decimals.
        # To get pvals close to zero, make sure the t-stat is negative
        # and the cumulative distribution is taken up to that point
        p = 2*stats.t.cdf(-np.abs(t_stat), df = df)
        ts = np.array(np.sign(t_stat))
        z = -ts*stats.norm.ppf(p)

        return np.array(prediction),np.array(betas),np.array(t_stat),np.array(cope),np.array(varcope),np.array(p),np.array(z)
        

    def detect_inliers_mad(self,data,outlier_num_stds=3,generate_diagnostic=False):
        
        """
        Detects inliers based on the Median Absolute Deviation, see:
        http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
        https://en.wikipedia.org/wiki/Median_absolute_deviation

        It calculates the MAD for left and right of median separately, 
        ensuring that it can handle skewed data.
    
        :param data: input data
        :type data: 1-D array
        :param outlier_num_stds: number of standard deviations to include
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the outlier rejections
        :type generate_diagnostics: bool

        :return inliers: returns boolean array of inliers
        :return type: same shape as data
        """

        # first calculate the distances for the median for left and right of median separately:
        mad_left = np.median(np.abs(data[data<np.median(data)]-np.median(data)))
        mad_right = np.median(np.abs(data[data>np.median(data)]-np.median(data)))

        # constant assuming normality
        k = 1.4826 

        # then determine thresholds 
        left_threshold = k*mad_left*outlier_num_stds
        right_threshold = k*mad_right*outlier_num_stds

        # find which values fall below the thresholds
        left_inliers = (np.abs(data[data<np.median(data)]-np.median(data))<left_threshold)
        right_inliers = (np.abs(data[data>np.median(data)]-np.median(data))<right_threshold)

        # and put them together
        inliers = np.ones_like(data).astype(bool)
        inliers[data<np.median(data)] = left_inliers
        inliers[data>np.median(data)] = right_inliers

        if generate_diagnostic:
            f = pl.figure(figsize=(10,5))
            s = f.add_subplot(121)
            pl.title('input data')
            pl.hist(data,100)
            sn.despine(offset=2)
            s = f.add_subplot(122 )
            pl.title('input data without outliers')
            pl.hist(data[inliers],100)
            sn.despine(offset=2)
            pl.tight_layout(pad=0)
            pl.savefig('/home/vanes/temp/plots/outlier_rejection_mad_%d.pdf'%np.random.randint(1e8    ))
            pl.close()

        return inliers

    def bootstrap(self,data,center_estimate='mean',weights=None,test_value=0,ci_factor = 1.96,two_tailed=True,detect_inliers=True,outlier_num_stds =3,generate_diagnostic=False,return_d=False ):
        """
        This finds a distribution of (weighted) average or regular median of the data,
        returning the median of this distribution along with std from the median.

        :param data: input data
        :type data: array of shape: [n_variables,n_observations]
        :param center_estimate: central measure: mean or median
        :type center_estimate: string
        :param weights: possible input weights when calculating mean
        :type weights: 1-D array with same n_observations as data
        :param reps: amount of bootstrap repetitions
        :type reps: int
        :param test_value: value to test bootstrap distribution against
        :type test_value: float
        :param ci_factor: z-score to multiply std of bootstrap distr with
        :type ci_factor: float
        :param two_tailed: when True, returns two-sided p-val, else returns one-sided p-val
        :type two_tailed: bool
        :param detect_inliers: uses outlier detection when True
        :type detect_inliers: bool
        :param outlier_num_stds: number of stds from median in outlier detection
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the bootstrap distr
        :type generate_diagnostics: bool

        :return center: central estimate of bootstrap distribution
        :type center: float
        :return ci: ci of central estimate
        :type ci: float
        :return p-val: p-val for difference with test_value
        :type p-val: float

        """

        means = []
        ps = []
        Ns = []
        cis = []
        cohen_d = []
        # put single array in iterable container
        if np.ndim(data) == 1:
            data = [data]

        # loop over different variables
        for di in range(np.shape(data)[0]):

            # if there's no data in here, or when all values are nans, set results to nans:
            if (len(data[di][np.invert(np.isnan(data[di]))]) == 0 ):
                means.append(np.nan)
                ps.append(np.nan)
                cohen_d.append(np.nan)
                Ns.append(np.nan)

            # else do bootstrap
            else:
                
                # set weights to one if none provided 
                if weights is None:
                    these_weights = np.ones(len(data[di]))
                else:
                    these_weights = copy.copy(weights)

                # remove nan values from data and weights
                valid_values = np.invert(np.isnan(data[di]))
                these_data = data[di][valid_values]
                these_weights = these_weights[valid_values]

                # remove outliers from data and weights
                if detect_inliers:
                    inliers = self.detect_inliers_mad(these_data,outlier_num_stds)#,weights=weights,center_estimate=center_estimate)
                    # inliers = self.detect_inliers_std(these_data,outlier_num_stds,weights=weights)
                    these_data = these_data[inliers]
                    these_weights = these_weights[inliers]

                Ns.append(len(these_data))

                # get random ints for random indices
                permute_indices = np.random.randint(0, len(these_data), size = (len(these_data), int(self.reps)))

                # now average over all these random draws 
                if center_estimate == 'mean':
                    # in weighted fashion in case of average
                    bootstrap_distr = np.average(these_data[permute_indices],weights=these_weights[permute_indices],axis=0)
                elif center_estimate == 'median':
                    # or regular median
                    bootstrap_distr = np.median(these_data[permute_indices],axis=0)
                elif center_estimate == 'std':
                    bootstrap_distr = np.std(these_data[permute_indices],axis=0)

                # calculate p-val
                ps.append(self.p_val_from_bootstrap_dist(bootstrap_distr,test_value,two_tailed))

                # get ci
                cis.append(self.get_ci(bootstrap_distr,ci_factor))

                # return standard deviation of bootstrap distro for plotting
                if center_estimate == 'mean':
                    means.append(np.average(these_data,weights=these_weights))
                elif center_estimate == 'median':
                    means.append(np.median(these_data))
                elif center_estimate == 'std':
                    means.append(np.std(these_data))
                # ses.append(np.std(bootstrap_distr)*ci_factor)
                
                if generate_diagnostic:
                    f = pl.figure(figsize=(5,5))
                    s = f.add_subplot(111)
                    pl.title('input data')
                    pl.hist(bootstrap_distr,100)
                    pl.axvline(np.average(these_data,weights=these_weights),color='k',label='center',lw=5)
                    pl.axvline(np.average(these_data,weights=these_weights)+np.std(bootstrap_distr)*ci_factor,color='r',label='ci',lw=5)
                    pl.axvline(np.average(these_data,weights=these_weights)-np.std(bootstrap_distr)*ci_factor,color='r',label='ci',lw=5)
                    pl.legend(loc='best')
                    sn.despine(offset=2)
                    pl.savefig('/home/vanes/temp/plots/bootstrap_distr_outlier_detection_%s_%d.pdf'%(detect_inliers,np.random.randint(1e8)))
                    pl.close()

                # calculate cohen's d:
                cohen_d.append(( np.average(these_data,weights=these_weights) - test_value ) / DescrStatsW(data=these_data, weights=these_weights).std)

        if return_d:
            return np.squeeze(means), np.squeeze(cis), np.squeeze(ps), np.squeeze(Ns), np.squeeze(cohen_d)
        else:
            return np.squeeze(means), np.squeeze(cis), np.squeeze(ps), np.squeeze(Ns)

    def bootstrap_median_split(self,x_data,y_data,weights,outlier_num_stds,ci_factor,detect_inliers,reps,test_value=0,two_tailed=True):

        # set weights to one if none provided 
        if weights is None:
            these_weights = np.ones(len(x_data))
        else:
            these_weights = copy.copy(weights)

        # remove nan values from data and weights
        valid_values = np.invert(np.isnan(x_data))
        these_x_data = x_data[valid_values]
        these_y_data = y_data[valid_values]
        these_weights = these_weights[valid_values]

        # remove outliers from data and weights
        if detect_inliers:
            x_inliers = self.detect_inliers_mad(these_x_data,outlier_num_stds)
            y_inliers = self.detect_inliers_mad(these_y_data,outlier_num_stds)
            inliers = x_inliers*y_inliers

            # and apply
            these_x_data = these_x_data[inliers]
            these_y_data = these_y_data[inliers]
            these_weights = these_weights[inliers]

        # now find the permute indices (the random voxel indices values to draw) 
        permute_indices = np.random.randint(0, len(these_x_data), size = (len(these_x_data), int(self.reps)))

        split_value = np.median(these_x_data)

        bootstrap_diffs = []
        for perm in permute_indices:
            this_perm_x = these_x_data[perm]
            this_perm_weights = these_weights[perm]
            this_perm_y = these_y_data[perm]

            below_median_x = (this_perm_x<split_value)
            above_median_x = (this_perm_x>split_value)

            mean_below_median = np.average(this_perm_y[below_median_x],weights=this_perm_weights[below_median_x])
            mean_above_median = np.average(this_perm_y[above_median_x],weights=this_perm_weights[above_median_x])

            diff = mean_above_median - mean_below_median

            bootstrap_diffs.append(diff)


        # calculate p-val
        p = self.p_val_from_bootstrap_dist(bootstrap_diffs,test_value,two_tailed)

        # get ci
        ci = self.get_ci(bootstrap_diffs,ci_factor)

        # determine number of samples
        N = len(these_weights)

        # get overall diff
        below_median_x = (these_x_data<split_value)
        above_median_x = (these_x_data>split_value)

        y1 = these_y_data[below_median_x]
        y2 = these_y_data[above_median_x]
        w1 = these_weights[below_median_x]
        w2 = these_weights[above_median_x]

        mean_below_median = np.average(y1,weights=w1)
        mean_above_median = np.average(y2,weights=w2)

        diff = mean_above_median - mean_below_median

        # determine cohen_d
        def weighted_cohend(y1,y2,w1,w2):

            from statsmodels.stats.weightstats import DescrStatsW

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

        cohen_d = weighted_cohend(y1,y2,w1,w2)

        return diff,ci,p,N,cohen_d


    def bootstrap_correlation(self,x_data,y_data,weights=None,corr_type='pearson',test_value=0,ci_factor = 1.96,two_tailed=True,detect_inliers=True,outlier_num_stds =3,generate_diagnostic=False):
        """
        Fits linear regression to data, either weighted or not. 
        Returns bootstrapped CIs for slope and intercept

        :param x_data: input x_data
        :type x_data: 1-D array 
        :param y_data: input y_data
        :type y_data: 1-D array with same shape as x_data
        :param weights: possible input weights when calculating mean
        :type weights: 1-D array with same n_observations as data
        :param reps: amount of bootstrap repetitions
        :type reps: int
        :param test_value: value to test bootstrap distribution against
        :type test_value: float
        :param ci_factor: z-score to multiply std of bootstrap distr with
        :type ci_factor: float
        :param two_tailed: when True, returns two-sided p-val, else returns one-sided p-val
        :type two_tailed: bool
        :param detect_inliers: uses outlier detection when True
        :type detect_inliers: bool
        :param outlier_num_stds: number of stds from median in outlier detection
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the bootstrap distr
        :type generate_diagnostics: bool

        :return center: central estimate of bootstrap distribution
        :type center: float
        :return ci: ci of central estimate
        :type ci: float
        :return p-val: p-val for difference with test_value
        :type p-val: float

        """
        # remove nans from signal if present

        # set weights to one if none provided 
        if weights is None:
            weights = np.ones(len(x_data))

        if detect_inliers:
            x_inliers = self.detect_inliers_mad(x_data,outlier_num_stds)
            y_inliers = self.detect_inliers_mad(y_data,outlier_num_stds)
            x_data = x_data[x_inliers*y_inliers]
            y_data = y_data[x_inliers*y_inliers]
            weights = weights[x_inliers*y_inliers]

        N = len(x_data)

        # get random ints for random indices
        permute_indices = np.random.randint(0, len(x_data), size = (len(x_data), int(self.reps))).T

        # rank transform data if spearman is requested
        if corr_type == 'spearman':
            x_data = stats.rankdata(x_data)
            y_data = stats.rankdata(y_data)

        bootstrap_distr=[]
        bootstrap_distr_z=[]
        # loop over permutes
        for fold in permute_indices:
            r = DescrStatsW(data=np.vstack([x_data[fold],y_data[fold]]).T, weights=weights[fold]).corrcoef[0,1]
            z = np.arctanh(r)
            bootstrap_distr.append(r)
            bootstrap_distr_z.append(z)

        # calculate p-val
        p = self.p_val_from_bootstrap_dist(bootstrap_distr_z,test_value,two_tailed)

        # compute central corr on all data
        corr = DescrStatsW(data=np.vstack([x_data,y_data]).T, weights=weights).corrcoef[0,1]

        # return standard deviation of bootstrap distro as CI
        # corr_ci = np.std(bootstrap_distr)*ci_factor
        corr_ci = self.get_ci(bootstrap_distr,ci_factor)

        if generate_diagnostic:
            f = pl.figure(figsize=(5,5))
            s = f.add_subplot(111)
            pl.title('input data')
            pl.hist(bootstrap_distr,100)
            pl.axvline(corr,color='k',label='center',lw=5)
            pl.axvline(corr+corr_ci,color='r',label='ci',lw=5)
            pl.axvline(corr-corr_ci,color='r',label='ci',lw=5)
            pl.legend(loc='best')
            sn.despine(offset=2)
            pl.savefig('/home/vanes/temp/plots/corr_bootstrap_distr_outlier_detection_%s_%d.pdf'%(detect_inliers,np.random.randint(1e8)))
            pl.close()

        return corr, corr_ci, p, N

    def bootstrap_correlation_difference(self,x_data_1,x_data_2,y_data_1,y_data_2,weights=None,corr_type='pearson',test_value=0,ci_factor = 1.96,two_tailed=True,detect_inliers=True,outlier_num_stds =3,generate_diagnostic=False):
        """
        
        This function performs either pearson or spearman correlation on two sets of data.
        Within each fold, it does so for two sets of data (same permutation, different variables), and subtracts the correlation coefficients.
        Finally, it tests whether the correlations are different from each other (whether difference in correlations is different from 0).

        :param x_data_1: input x_data
        :type x_data_1: 1-D array 
        :param y_data_1: input y_data
        :type y_data_1: 1-D array with same shape as x_data
        :param x_data_2: input x_data
        :type x_data_2: 1-D array 
        :param y_data_2: input y_data
        :type y_data_2: 1-D array with same shape as x_data
        :param weights: possible input weights when calculating mean
        :type weights: 1-D array with same n_observations as data
        :param reps: amount of bootstrap repetitions
        :type reps: int
        :param test_value: value to test bootstrap distribution against
        :type test_value: float
        :param ci_factor: z-score to multiply std of bootstrap distr with
        :type ci_factor: float
        :param two_tailed: when True, returns two-sided p-val, else returns one-sided p-val
        :type two_tailed: bool
        :param detect_inliers: uses outlier detection when True
        :type detect_inliers: bool
        :param outlier_num_stds: number of stds from median in outlier detection
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the bootstrap distr
        :type generate_diagnostics: bool

        :return center: central estimate of bootstrap distribution
        :type center: float
        :return ci: ci of central estimate
        :type ci: float
        :return p-val: p-val for difference with test_value
        :type p-val: float

        """

        # set weights to one if none provided 
        if weights is None:
            weights = np.ones(len(x_data))

        if detect_inliers:
            x_1_inliers = self.detect_inliers_mad(x_data_1,outlier_num_stds)
            y_1_inliers = self.detect_inliers_mad(y_data_1,outlier_num_stds)
            x_2_inliers = self.detect_inliers_mad(x_data_2,outlier_num_stds)
            y_2_inliers = self.detect_inliers_mad(y_data_2,outlier_num_stds)    
            all_inliers = x_1_inliers*x_2_inliers*y_1_inliers*y_2_inliers
            x_data_1 = x_data_1[all_inliers]
            y_data_1 = y_data_1[all_inliers]
            x_data_2 = x_data_2[all_inliers]
            y_data_2 = y_data_2[all_inliers]
            weights = weights[all_inliers]

        N = len(x_data_1)

        # get random ints for random indices
        permute_indices = np.random.randint(0, len(x_data_1), size = (len(x_data_1), int(self.reps))).T

        # rank transform data if spearman is requested
        if corr_type == 'spearman':
            x_data_1 = stats.rankdata(x_data_1)
            y_data_1 = stats.rankdata(y_data_1)
            x_data_2 = stats.rankdata(x_data_2)
            y_data_2 = stats.rankdata(y_data_2)

        bootstrap_distr=[]
        bootstrap_distr_z=[]
        # loop over permutes
        for fold in permute_indices:
            r_1 = DescrStatsW(data=np.vstack([x_data_1[fold],y_data_1[fold]]).T, weights=weights[fold]).corrcoef[0,1]
            r_2 = DescrStatsW(data=np.vstack([x_data_2[fold],y_data_2[fold]]).T, weights=weights[fold]).corrcoef[0,1]
            bootstrap_distr.append(r_1-r_2)

            # fisher transform correlations
            zr1 = np.arctanh(r_1)
            zr2 = np.arctanh(r_2)
            bootstrap_distr_z.append(zr1-zr2)

        # calculate p-val
        p = self.p_val_from_bootstrap_dist(bootstrap_distr_z,test_value,two_tailed)

        # compute central corr on all data
        r1 = DescrStatsW(data=np.vstack([x_data_1,y_data_1]).T, weights=weights).corrcoef[0,1]
        r2 = DescrStatsW(data=np.vstack([x_data_2,y_data_2]).T, weights=weights).corrcoef[0,1]
        r_diff = r1-r2

        # return standard deviation of bootstrap distro as CI
        r_diff_ci = np.std(bootstrap_distr)*ci_factor

        if generate_diagnostic:
            f = pl.figure(figsize=(5,5))
            s = f.add_subplot(111)
            pl.title('input data')
            pl.hist(bootstrap_distr,100)
            pl.axvline(corr,color='k',label='center',lw=5)
            pl.axvline(corr+corr_ci,color='r',label='ci',lw=5)
            pl.axvline(corr-corr_ci,color='r',label='ci',lw=5)
            pl.legend(loc='best')
            sn.despine(offset=2)
            pl.savefig('/home/vanes/temp/plots/corr_bootstrap_distr_outlier_detection_%s_%d.pdf'%(detect_inliers,np.random.randint(1e8)))
            pl.close()

        return r_diff, r_diff_ci, p,N

    def bootstrap_linear_fit(self,x_data,y_data,weights=None,test_value=0,ci_factor = 1.96,two_tailed=True,detect_inliers=True,outlier_num_stds =3,generate_diagnostic=False):
        """
        Fits linear regression to data, either weighted or not. 
        Returns bootstrapped CIs for slope and intercept

        :param x_data: input x_data
        :type x_data: 1-D array 
        :param y_data: input y_data
        :type y_data: 1-D array with same shape as x_data
        :param weights: possible input weights when calculating mean
        :type weights: 1-D array with same n_observations as data
        :param reps: amount of bootstrap repetitions
        :type reps: int
        :param test_value: value to test bootstrap distribution against
        :type test_value: float
        :param ci_factor: z-score to multiply std of bootstrap distr with
        :type ci_factor: float
        :param two_tailed: when True, returns two-sided p-val, else returns one-sided p-val
        :type two_tailed: bool
        :param detect_inliers: uses outlier detection when True
        :type detect_inliers: bool
        :param outlier_num_stds: number of stds from median in outlier detection
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the bootstrap distr
        :type generate_diagnostics: bool

        :return center: central estimate of bootstrap distribution
        :type center: float
        :return ci: ci of central estimate
        :type ci: float
        :return p-val: p-val for difference with test_value
        :type p-val: float

        """
        # # set weights to one if none provided 
        if weights is None:
            weights = np.ones(len(x_data))

        if detect_inliers:
            x_inliers = self.detect_inliers_mad(x_data,outlier_num_stds)
            y_inliers = self.detect_inliers_mad(y_data,outlier_num_stds)
            x_data = x_data[x_inliers*y_inliers]
            y_data = y_data[x_inliers*y_inliers]
            weights = weights[x_inliers*y_inliers]

        # get random ints for random indices
        permute_indices = np.random.randint(0, len(x_data), size = (len(x_data), int(self.reps))).T

        slope_bootstrap_distr=[]
        intercept_bootstrap_distr=[]
        # loop over permutes
        for fold in permute_indices:
            # compute weighted linear fit
            slope,intercept=np.polyfit(x_data[fold], y_data[fold], 1,w=weights[fold])
            slope_bootstrap_distr.append(slope)
            intercept_bootstrap_distr.append(intercept)

        # get centers for slope and intercept by fitting to all data
        mean_slope,mean_intercept=np.polyfit(x_data, y_data, 1,w=weights)
        # slope distribution diagnostics
        ci_slope = self.get_ci(slope_bootstrap_distr,ci_factor)
        # ci_slope = np.std(slope_bootstrap_distr)*ci_factor
        p_slope = self.p_val_from_bootstrap_dist(slope_bootstrap_distr,test_value,two_tailed)

        # intercept distribution diagnostics
        # ci_intercept = np.std(intercept_bootstrap_distr)*ci_factor
        ci_intercept = self.get_ci(intercept_bootstrap_distr,ci_factor)
        p_intercept = self.p_val_from_bootstrap_dist(intercept_bootstrap_distr,test_value,two_tailed)

        if generate_diagnostic:
            f = pl.figure(figsize=(10,5))
            s = f.add_subplot(121)
            pl.title('slope data')
            pl.hist(slope_bootstrap_distr,100)
            pl.axvline(mean_slope,color='k',label='center',lw=5)
            pl.axvline(mean_slope+ci_slope,color='r',label='ci',lw=5)
            pl.axvline(mean_slope-ci_slope,color='r',label='ci',lw=5)
            pl.legend(loc='best')
            sn.despine(offset=2)
            s = f.add_subplot(122)
            pl.title('intercept data')
            pl.hist(intercept_bootstrap_distr,100)
            pl.axvline(mean_intercept,color='k',label='center',lw=5)
            pl.axvline(mean_intercept+ci_intercept,color='r',label='ci',lw=5)
            pl.axvline(mean_intercept-ci_intercept,color='r',label='ci',lw=5)
            pl.legend(loc='best')
            sn.despine(offset=2)            
            pl.savefig('/home/vanes/temp/plots/linfit_bootstrap_distr_outlier_detection_%s_%d.pdf'%(detect_inliers,np.random.randint(1e8)))
            pl.close()

        return mean_slope, ci_slope, p_slope, mean_intercept, ci_intercept, p_intercept

    def stats_from_data(self,data,ci_factor,test_value=0):
        """
        computes descriptive statistics of data
        if dimension of data is not 1 dimensional,
        it computes this for every row in the data
        """

        if np.ndim(data)==1:
            valid_data = np.array(data)[~np.isnan(data)]
            N = len(valid_data)
            mean = np.mean(valid_data)
            se = DescrStatsW(valid_data).tconfint_mean(alpha=(stats.norm.sf(ci_factor))*2)
            d = np.mean(valid_data)/np.std(valid_data)
            t,p = stats.ttest_1samp(valid_data,test_value)
        else:
            mean=[];se=[];d=[];N=[];t=[];p=[]
            for subdata in data:
                valid_data = np.array(subdata)[~np.isnan(subdata)]
                N.append(len(valid_data))
                mean.append(np.mean(valid_data))
                se.append(DescrStatsW(valid_data).tconfint_mean(alpha=(stats.norm.sf(ci_factor))*2))
                d.append(np.mean(valid_data)/np.std(valid_data))
                this_t,this_p = stats.ttest_1samp(valid_data,test_value)
                t.append(this_t)
                p.append(this_p)

        return np.array(mean), np.array(se), np.array(t), np.array(p), np.array(d), np.array(N)

    def p_val_from_bootstrap_dist(self,distribution,test_value=0,two_tailed=True):
        """
        Finds p-value for hypothesis that the distribution is not different 
        from the test_value.

        :param distribution: distribution of bootstrapped parameter
        :type distribution: 1-D array
        :param test_value: value to test distribution against
        :type test_value: float
        :param two_tailed: if True, returns two-tailed test, else one-tailed
        :type two_tailed: bool

        :return p-val: p-val
        :type p-val: float
        """

        # see which part of the distribution falls below / above test value:
        proportion_smaller_than_test_value = np.sum(np.array(distribution) < test_value) / len(distribution)
        proportion_larger_than_test_value = np.sum(np.array(distribution) > test_value) / len(distribution)

        # take minimum value as p-val:
        p = np.min([proportion_smaller_than_test_value,proportion_larger_than_test_value])
        
        # this yields a one-tailed test, so multiply by 2 if we want a two-tailed p-val:
        if two_tailed:
            p*=2

        return p

    def get_ci(self,distribution,ci_factor):

        # convert ci factor to percentile
        perc_low = (stats.norm.sf(ci_factor))*100
        perc_high = (1 - (stats.norm.sf(ci_factor)))*100
        ci_low = np.percentile(distribution,perc_low)
        ci_high = np.percentile(distribution,perc_high)

        return [ci_low,ci_high]

    def collapse_angles_symmetrically(self,angles):
        """
        This function collapses angles symmetrically, such that
        if the data had two opposite peaks (say at 0pi and pi),
        the data will now have 1 single peak.
        This is also called the angle-double procedure, desciption can be found here:
        http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf

        :param angles: input angles
        :type angles: 1-D array

        :return angles: flipped angles
        :type angles: 1-D array
        """

        collapsed_angles = copy.copy(angles)
        # first, we'll need to recode the values, so that they run from 0to2pi, instead of from -pitopi:
        collapsed_angles[collapsed_angles<0] += 2*np.pi
        # the, we'll need to double all the angles
        collapsed_angles *= 2
        # then, all the angles with values above 2*np.pi need to brought back to the circle:
        collapsed_angles[collapsed_angles>(np.pi*2)] -= 2*np.pi

        return collapsed_angles

    def angular_linear_correlation(self,angles,data,weights=None,double_peak=False):
        """
        This function computes an angular-linear correlation.
        When expecting the data to have two symmetrical, opposite peaks (e.g.
        a non-direction selective effect such as horizontal vs. vertical
        instead of up vs down), double_peak should be set to True.

        :param angles: input angles
        :type angles: 1-D array
        :param data: input data
        :type data: 1-D array, same shape as angles
        :param weights: weights to use for correlation
        :type weights: 1-D array, same shape as angles
        :param double_peak: when True, angles are doubled
        :type double_peak: bool

        :return corr: circular correlation
        :type corr: float

        """

        # set weights to one 
        if weights is None:
            weights = np.ones_like(angles)

        # In cases of expected periodicity (e.g. data peaks at two opposite angles),
        # the angular data should be scaled:
        if double_peak:
            angles = self.collapse_angles_symmetrically(angles)

        # use formula from the pycircstat package to calculate circular correlation:
        rxs = DescrStatsW(data=np.vstack([data,np.sin(angles)]).T, weights=weights).corrcoef[0,1]
        rxc = DescrStatsW(data=np.vstack([data,np.cos(angles)]).T, weights=weights).corrcoef[0,1]
        rcs = DescrStatsW(data=np.vstack([np.sin(angles),np.cos(angles)]).T, weights=weights).corrcoef[0,1]

        # rxs = self.functions.wpearson(these_data,np.sin(doubled_angles),weights)
        # rxc = self.functions.wpearson(these_data,np.cos(doubled_angles),weights)
        # rcs = self.functions.wpearson(np.sin(doubled_angles),np.cos(doubled_angles),weights)
        # compute angular-linear correlation (equ. 27.47)
        corr = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))

        return corr

    def bootstrap_linear_angular_correlation(self,angles,data,weights=None,test_value=0,ci_factor = 1.96,two_tailed=True,detect_inliers=True,outlier_num_stds =3,generate_diagnostic=True,double_peak=True):
        """
        Computes linear angular correlation for bootstrapped samples of data.
        Returns bootstrapped CIs for circular correlation.

        :param angles: input angles
        :type angles: 1-D array 
        :param data: input data
        :type data: 1-D array with same shape as angles
        :param weights: possible input weights when calculating mean
        :type weights: 1-D array with same n_observations as data
        :param reps: amount of bootstrap repetitions
        :type reps: int
        :param test_value: value to test bootstrap distribution against
        :type test_value: float
        :param ci_factor: z-score to multiply std of bootstrap distr with
        :type ci_factor: float
        :param two_tailed: when True, returns two-sided p-val, else returns one-sided p-val
        :type two_tailed: bool
        :param detect_inliers: uses outlier detection when True
        :type detect_inliers: bool
        :param outlier_num_stds: number of stds from median in outlier detection
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the bootstrap distr
        :type generate_diagnostics: bool

        :return center: central estimate of bootstrap distribution
        :type center: float
        :return ci: ci of central estimate
        :type ci: float
        :return p-val: p-val for difference with test_value
        :type p-val: float

        """
        # set weights to one if none provided 

        # detect inliers only based on data
        if detect_inliers:
            inliers = self.detect_inliers_mad(data,outlier_num_stds)
            angles = data[inliers]
            data = data[inliers]
            if weights != None:
                weights = weights[inliers]

        if weights is None:
            weights = np.ones_like(data)

        N = len(data)

        # get random ints for random indices
        permute_indices = np.random.randint(0, len(data), size = (len(data), int(self.reps))).T

        bootstrap_distr=[]
        bootstrap_distr_z=[]
        # loop over permutes
        for fold in permute_indices:
            # compute weighted correaltion
            r = self.angular_linear_correlation(angles[fold],data[fold],weights[fold],double_peak=double_peak)
            zr = np.arctanh(r)
            bootstrap_distr.append(r)
            bootstrap_distr_z.append(zr)

        # calculate p-val
        p = self.p_val_from_bootstrap_dist(bootstrap_distr_z,test_value,two_tailed)

        # compute central corr on all data
        corr = self.angular_linear_correlation(angles,data,weights,double_peak=double_peak)
        # return standard deviation of bootstrap distro as CI
        # corr_ci = np.std(bootstrap_distr)*ci_factor
        corr_ci = self.get_ci(bootstrap_distr,ci_factor)

        if generate_diagnostic:
            f = pl.figure(figsize=(5,5))
            s = f.add_subplot(111)
            pl.title('input data')
            pl.hist(bootstrap_distr,100)
            pl.axvline(corr,color='k',label='center',lw=5)
            pl.axvline(corr+corr_ci,color='r',label='ci',lw=5)
            pl.axvline(corr-corr_ci,color='r',label='ci',lw=5)
            pl.legend(loc='best')
            sn.despine(offset=2)
            pl.savefig('/home/vanes/temp/plots/angular_corr_bootstrap_distr_outlier_detection_%s_%d.pdf'%(detect_inliers,np.random.randint(1e8)))
            pl.close()

        return corr, corr_ci, p,N


    def bootstrap_diff_linear_angular_correlation(self,angles,data1,data2,weights=None,test_value=0,ci_factor = 1.96,two_tailed=True,detect_inliers=True,outlier_num_stds =3,generate_diagnostic=True,double_peak=True):
        """
        Computes difference in linear angular correlation
        between data1-angles and data2-angles.

        :param angles: input angles
        :type angles: 1-D array 
        :param data1: input data variable 1
        :type data1: 1-D array with same shape as angles
        :param data2: input data variable 2
        :type data2: 1-D array with same shape as angles
        :param weights: possible input weights when calculating mean
        :type weights: 1-D array with same n_observations as data
        :param reps: amount of bootstrap repetitions
        :type reps: int
        :param test_value: value to test bootstrap distribution against
        :type test_value: float
        :param ci_factor: z-score to multiply std of bootstrap distr with
        :type ci_factor: float
        :param two_tailed: when True, returns two-sided p-val, else returns one-sided p-val
        :type two_tailed: bool
        :param detect_inliers: uses outlier detection when True
        :type detect_inliers: bool
        :param outlier_num_stds: number of stds from median in outlier detection
        :type outlier_num_stds: float
        :param generate_diagnostics: if True, make plot of the bootstrap distr
        :type generate_diagnostics: bool

        :return center: central estimate of bootstrap distribution
        :type center: float
        :return ci: ci of central estimate
        :type ci: float
        :return p-val: p-val for difference with test_value
        :type p-val: float

        """
        # set weights to one if none provided 

        if weights is None:
            weights = np.ones_like(data1)

        N = len(data1)

        # get random ints for random indices
        permute_indices = np.random.randint(0, len(data1), size = (len(data1), int(self.reps))).T

        bootstrap_distr=[]
        bootstrap_distr_fisher=[]
        # for statistical difference, we want to fisher transform the data:
        for fold in permute_indices:
            # compute weighted correaltion
            r1 = self.angular_linear_correlation(angles[fold],data1[fold],weights[fold],double_peak=double_peak)
            r2 = self.angular_linear_correlation(angles[fold],data2[fold],weights[fold],double_peak=double_peak)
            bootstrap_distr.append(r1-r2)
            # fisher transform correlations for significance test
            z1=np.arctanh(r1)
            z2=np.arctanh(r2)
            bootstrap_distr_fisher.append(z1-z2)

        # calculate p-val
        p = self.p_val_from_bootstrap_dist(bootstrap_distr_fisher,test_value,two_tailed)

        # compute central corr on all data
        r1 = self.angular_linear_correlation(angles,data1,weights,double_peak=double_peak)
        r2 = self.angular_linear_correlation(angles,data2,weights,double_peak=double_peak)
        diff_r = r1-r2        

        # return standard deviation of bootstrap distro as CI
        # corr_ci = np.std(bootstrap_distr)*ci_factor
        corr_ci = self.get_ci(bootstrap_distr,ci_factor)

        if generate_diagnostic:
            f = pl.figure(figsize=(5,5))
            s = f.add_subplot(111)
            pl.title('input data')
            pl.hist(bootstrap_distr,100)
            pl.axvline(corr,color='k',label='center',lw=5)
            pl.axvline(corr+corr_ci,color='r',label='ci',lw=5)
            pl.axvline(corr-corr_ci,color='r',label='ci',lw=5)
            pl.legend(loc='best')
            sn.despine(offset=2)
            pl.savefig('/home/vanes/temp/plots/angular_corr_bootstrap_distr_outlier_detection_%s_%d.pdf'%(detect_inliers,np.random.randint(1e8)))
            pl.close()

        return diff_r, corr_ci, p,N


