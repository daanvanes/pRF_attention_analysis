from IPython import embed as shell
import numpy as np
import matplotlib.pyplot as pl
from FitPRFModel import gpf
import seaborn as sn
import colorsys

dm = np.load('/home/shared/PRF_2/data/DE/DE_010615/processed/mri/PRF/raw_design_matrix_101x101_All.npy')[:300]
g = gpf(design_matrix = dm, max_eccentricity = 3.6, n_pixel_elements = 101, rtime = 1.6, ssr = 1,slice_no=0)
size = 1.5
sim_data = g.hrf_model_prediction(-3,3,size,[1,0,0])[0]
sim_data = sim_data / np.max(sim_data) * 5
sim_data += np.random.randn(len(dm))/2

pred_params = [[0,0,size],[3,-3,size],[-3,3,size]]


predictions = {}
for pred, params, in zip(range(len(pred_params)),pred_params):
	temp = g.hrf_model_prediction(params[0],params[1],params[2],[1,0,0])[0]
	temp = temp/np.max(temp)*5
	predictions[pred] = temp


colors = np.array([colorsys.hsv_to_rgb(c,0.7,0.8) for c in np.linspace(0,1,len(pred_params)+1)])[:-1]

f = pl.figure(figsize=(4,3))
s = f.add_subplot(111)
pl.plot(sim_data[35:59],':ok',label='data',markersize=6,lw=2)
pl.axhline(0,lw=1,c='k')
sn.despine(offset=10)
pl.ylabel('BOLD (%)')
pl.xlabel('time (s)')
pl.xticks([0,24],[0,24*1.6])
pl.yticks([0,5])
pl.tight_layout()
pl.savefig('/home/shared/PRF_2/data/_group_level/pRF_fit_figure_data_1_trial.pdf')

f = pl.figure(figsize=(6,3))
s = f.add_subplot(111)
pl.plot(sim_data,':ok',label='data',markersize=3,lw=1)
pl.axhline(0,lw=1,c='k')
sn.despine(offset=10)
pl.ylabel('BOLD (%)')
pl.xlabel('time (s)')
pl.xticks([0,300],[0,300*1.6])
pl.yticks([0,5])
pl.tight_layout()
pl.savefig('/home/shared/PRF_2/data/_group_level/pRF_fit_figure_data.pdf')


for i in range(len(pred_params)):
	f = pl.figure(figsize=(6,3))
	s = f.add_subplot(111)
	pl.plot(sim_data,':ok',label='data',markersize=3,lw=1)
	pl.plot(predictions[i],color=colors[i])
	sn.despine(offset=10)
	pl.axhline(0,lw=1,c='k')
	pl.ylabel('BOLD (%)')
	pl.xlabel('time (s)')
	pl.xticks([0,300],[0,300*1.6])
	pl.yticks([0,5])
	pl.tight_layout()
	pl.savefig('/home/shared/PRF_2/data/_group_level/pRF_fit_figure_data_pred_%d.pdf'%i)

f = pl.figure(figsize=(6,3))
s = f.add_subplot(111)
pl.plot(sim_data,':ok',label='data',markersize=3,lw=1)
TRs_per_trial = 29
trial_type_1_points = np.vstack([np.arange(TRs_per_trial*0,TRs_per_trial*1),np.arange(TRs_per_trial*2,TRs_per_trial*3),np.arange(TRs_per_trial*5,TRs_per_trial*6),np.arange(TRs_per_trial*6,TRs_per_trial*7)])
trial_type_2_points = np.vstack([np.arange(TRs_per_trial*1,TRs_per_trial*2),np.arange(TRs_per_trial*3,TRs_per_trial*4),np.arange(TRs_per_trial*4,TRs_per_trial*5),np.arange(TRs_per_trial*9,TRs_per_trial*10)])
for trial1, trial2 in zip(trial_type_1_points,trial_type_2_points):
	pl.plot(np.arange(300)[trial1],predictions[2][trial1],color=colors[0])
sn.despine(offset=10)
pl.axhline(0,lw=1,c='k')
pl.ylabel('BOLD (%)')
pl.xlabel('time (s)')
pl.xticks([0,300],[0,300*1.6])
pl.yticks([0,5])
pl.tight_layout()
pl.savefig('/home/shared/PRF_2/data/_group_level/pRF_fit_figure_data_pred_model_1.pdf')

f = pl.figure(figsize=(6,3))
s = f.add_subplot(111)
pl.plot(sim_data,':ok',label='data',markersize=3,lw=1)
TRs_per_trial = 29
trial_type_1_points = np.vstack([np.arange(TRs_per_trial*0,TRs_per_trial*1),np.arange(TRs_per_trial*2,TRs_per_trial*3),np.arange(TRs_per_trial*5,TRs_per_trial*6),np.arange(TRs_per_trial*6,TRs_per_trial*7)])
trial_type_2_points = np.vstack([np.arange(TRs_per_trial*1,TRs_per_trial*2),np.arange(TRs_per_trial*3,TRs_per_trial*4),np.arange(TRs_per_trial*4,TRs_per_trial*5),np.arange(TRs_per_trial*9,TRs_per_trial*10)])
for trial1, trial2 in zip(trial_type_1_points,trial_type_2_points):
	pl.plot(np.arange(300)[trial2],predictions[2][trial2],color=colors[1])
sn.despine(offset=10)
pl.axhline(0,lw=1,c='k')
pl.ylabel('BOLD (%)')
pl.xlabel('time (s)')
pl.xticks([0,300],[0,300*1.6])
pl.yticks([0,5])
pl.tight_layout()
pl.savefig('/home/shared/PRF_2/data/_group_level/pRF_fit_figure_data_pred_model_2.pdf')

