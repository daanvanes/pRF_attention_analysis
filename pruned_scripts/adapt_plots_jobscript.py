from __future__ import division
import numpy as np
from IPython import embed as shell
import re
import os

#################################
os.system('cd ~/job_submissions/')
jobscript_fn = '/home/vanes/job_submissions/plots_jobscript'

###############################

# for AF fitting
# rois = ['combined']#['V1','V2','V3','V3AB','hV4','VO','LO','IPS0','MT+']
# comparisons = ['Color','Speed']#['Color','Speed']#'Stim','Stim',
subject_methods= ['over_subjects','super_subjects']#'over_subjects',

# for AF plot
# rois = ['nan']
# comparisons = ['nan']#['Color','Speed']#'Stim',
# subject_methods= ['over_subjects','super_subjects']
# rois = [['V1','V2','V3','V3AB','hV4','VO','LO','IPS0','MT+']

for subject_method in subject_methods:
	for error_method in ['mean','median']:
		for error_over in ['arrows','voxels']:
			for error_diff in ['rel','abs']:
				for iffs in [0,1]:
	# for comparison in comparisons:
		# for roi in rois:
					if error_over == 'arrows':
						bins = [[8,8],[6,6],[6,4],[4,4]]
						# bins = [[6,4]]
						minvoxss = [1,2,3]
					else:
						bins = [[8,8]]
						minvoxss = [1]

					for eccbins,polbins in bins:
						for minvoxs in minvoxss:

							jobscript = open(jobscript_fn)
							working_string = jobscript.read()
							jobscript.close()

							RE_dict =  {
							# '---roi---': roi,
							# '---comparison---': comparison,#'%.2f'%AF_intercept,
							'---error_method---':error_method,
							'---error_over---':error_over,
							'---error_diff---':error_diff,
							'---subject_method---': subject_method,#'%.2f'%AF_intercept,
							'---eccbins---': str(eccbins),#'%.2f'%AF_intercept,
							'---polbins---': str(polbins),#'%.2f'%AF_intercept,
							'---minvoxs---': str(minvoxs),#'%.2f'%AF_intercept,
							'---iffs---': str(iffs),#'%.2f'%AF_intercept,
							# '---AF_fix_size---': '%.2f'%AF_fix_size,
							# '---AF_surround_amp---': '%.2f'%AF_surround_amp,
							# '---model_tag---': str(model_tag),
							}

							for e in RE_dict:
								rS = re.compile(e)
								working_string = re.sub(rS, RE_dict[e], working_string)

							of = open(jobscript_fn+'_tmp', 'w')
							of.write(working_string)
							of.close()

							os.system('sbatch ' + jobscript_fn +'_tmp')

