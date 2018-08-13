from __future__ import division

import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import numpy as np
import colorsys
import seaborn as sn
sn.set(style="white")
mpl.rc_file_defaults()
import os
from AFSimulations import *
from GroupLevelPlotting import GroupLevelPlots
from GroupLevelPlotting import General_functions
from scipy.stats import binned_statistic

size_slope = 0.5
fix_RF_x = 0.2
fix_RF_y = 0
fix_RF_size = 1

fix_AF_size = 2
bar_AF_size = 1.1

res = 131
maif = 3
stim_radius = 3.6

n_bins = 4

AF_dir = os.path.join('/home/shared/2015/visual/PRF_2/AF_simulations/')
AFs = convolve_bar_with_AF(AF_size_intercept=bar_AF_size,AF_size_slope=0,model_area_increase_factor=maif,res=res,plot_dir=AF_dir,stim_radius=stim_radius,circle_mask=True,AF_surround_amp=0,AF_surround_ratio=1)[0]


fix_AF = twodgauss(xo=0, yo=0, sigma_x=fix_AF_size,max_eccentricity=stim_radius*maif,res=res,circle_mask=True)
sizess = [[0.5,0.2],[0.5,0.75]]
names = ['V3','MT+']
for ri in range(len(names)):
	sizes = sizess[ri]
	name = names[ri]
	random_eccs = np.abs(np.random.normal(0,stim_radius*0.6,200))
	valid_pRFs = (np.abs(random_eccs)<stim_radius)
	random_eccs = random_eccs[valid_pRFs]
	n_pRFs = valid_pRFs.sum()
	random_polars = np.random.uniform(-np.pi,np.pi,n_pRFs)
	random_xs = random_eccs * np.cos(random_polars)
	random_ys = random_eccs * np.sin(random_polars)
	random_sizes = sizes[0]+sizes[1]*random_eccs

	pRF_fix_poss = []
	pRF_stim_poss = []
	SD_poss = []
	for i in range(n_pRFs):
	    x = random_xs[i]
	    y = random_ys[i]
	    s = random_sizes[i]
	    fix_RF = twodgauss(xo=x, yo=y, sigma_x=s,max_eccentricity=stim_radius*maif,res=res,circle_mask=True)
	    SD = fix_RF/fix_AF
	    SD[np.isnan(SD)] = 0

	    # # now simulate attention to bar effect on this SD:
	    bar_RF = np.mean([(SD*AF)/np.max(SD*AF) for AF in AFs],axis=0)
	    bar_RF[disk(int((res-1)/2))==0] = 0    

	    pRF_stim_poss.append(find_profile_maxval(bar_RF,stim_radius*maif,use_circle_mask=True,upsample_factor=10))
	    pRF_fix_poss.append(find_profile_maxval(fix_RF,stim_radius*maif,use_circle_mask=True,upsample_factor=10))
	    SD_poss.append(find_profile_maxval(SD,stim_radius*maif,use_circle_mask=True,upsample_factor=10))


	GF = General_functions()

	predicted_fix_eccs = np.linalg.norm(pRF_fix_poss,axis=1)
	predicted_stim_eccs = np.linalg.norm(pRF_stim_poss,axis=1)
	SD_eccs = np.linalg.norm(SD_poss,axis=1)
	    
	y = binned_statistic(SD_eccs, predicted_fix_eccs-SD_eccs, bins=4)[0]
	x = binned_statistic(SD_eccs, SD_eccs, bins=4)[0]

	# color1 = colorsys.hsv_to_rgb(0,0,0.75)
	# color2 = colorsys.hsv_to_rgb(0,0,0.5)
	# color3 = colorsys.hsv_to_rgb(0,0,0.25)


	pl.figure(figsize=(2.5,1.25))
	# pl.subplot(131)
	# y = binned_statistic(predicted_fix_eccs, random_sizes, bins=n_bins)[0]
	# x = binned_statistic(predicted_fix_eccs, predicted_fix_eccs, bins=n_bins)[0]
	# pl.plot(x,y,lw=1,c='k')	
	# pl.xticks([0,3.6])
	# pl.yticks([0,4])

	pl.subplot(121)
	# plot(SD_eccs,predicted_fix_eccs-SD_eccs,'o',c=color1,label='Fixation pRF-SD',alpha=0.1)
	y = binned_statistic(SD_eccs, predicted_fix_eccs-SD_eccs, bins=n_bins)[0]
	x = binned_statistic(SD_eccs, SD_eccs, bins=n_bins)[0]
	pl.plot(x,y,c='k',lw=1,ls='-',label='fix-SD')
	# plot(SD_eccs,predicted_stim_eccs-SD_eccs,'o',c=color1,label='Stimulus pRF - SD',alpha=0.1)
	y = binned_statistic(SD_eccs, predicted_stim_eccs-SD_eccs, bins=n_bins)[0]
	x = binned_statistic(SD_eccs, SD_eccs, bins=n_bins)[0]
	ylims = GF.find_yticks(pl.gca().get_ylim(),ndec=2)
	pl.ylim([ylims[0][0],ylims[0][-1]])
	pl.yticks(ylims[0],ylims[1])
	pl.plot(x,y,c='k',lw=1,ls=':',label='stim-SD')
	pl.legend(loc='best',frameon=False)
	pl.xticks([0,3.6])
	pl.axhline(0,c='k',lw=0.5)

	pl.xlabel('SD ecc (dva)')
	pl.ylabel(r'$\Delta$ ecc (dva)')
	pl.subplot(122)

	sn.despine(offset=2)
	# plot(predicted_fix_eccs,predicted_stim_eccs-predicted_fix_eccs,'o',c=color3,alpha=0.1)
	y = binned_statistic(predicted_fix_eccs, predicted_stim_eccs-predicted_fix_eccs, bins=n_bins)[0]
	x = binned_statistic(predicted_fix_eccs, predicted_fix_eccs, bins=n_bins)[0]
	pl.axhline(0,c='k',lw=0.5)
	pl.plot(x,y,c='k',lw=1,label='stim-fix',ls='--')
	pl.xlabel('Fix ecc (dva)')
	pl.legend(loc='best',frameon=False)
	# ylabel('stim-fix ecc')
	# ylims = GF.find_yticks(pl.gca().get_ylim(),ndec=2)
	if ri == 0:
		pl.yticks([-.15,0,.15],['-.15','0','.15'])
		pl.ylim([-.15,.15])
	elif ri ==1:
		pl.yticks([0,3.5],['0','3.5'])
		pl.ylim([0,3.5])
	# pl.yticks(ylims[0],ylims[1])
	pl.tight_layout()
	pl.xticks([0,3.6])

	pl.savefig('/home/shared/2015/visual/PRF_2/data/_group_level/plots/Figure_5/understanding_AF_%s.pdf'%name)






