import numpy as np
from IPython import embed as shell
import re
import os

#################################
os.system('cd ~/job_submissions/')

subjects = ['DE']#['JS','JW','TK','NA']#,'TK','DE']	
slices = [10]#np.arange(30) #[15]#[8,9,10,11,12,13,14,15,16,17,18,19,21]#
n_jobs = 12

jobscript_fn = '/home/vanes/job_submissions/jobscript'

###############################

for subject in subjects:

	for slice_no in slices:

		jobscript = open(jobscript_fn)
		working_string = jobscript.read()
		jobscript.close()

		RE_dict =  {
		'---subject---': 				subject, 
		'---n_jobs---':					str(n_jobs),
		'---slice_no---': 				str(slice_no)
		}

		for e in RE_dict:
			rS = re.compile(e)
			working_string = re.sub(rS, RE_dict[e], working_string)
		
		of = open(jobscript_fn+'_tmp', 'w')
		of.write(working_string)
		of.close()

		os.system('sbatch ' + jobscript_fn +'_tmp')

