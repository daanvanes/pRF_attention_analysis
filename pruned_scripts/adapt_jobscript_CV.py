import numpy as np
from IPython import embed as shell
import re
import os

#################################
os.system('cd ~/job_submissions/')

subjects = ['JS','JW','TK','DE']#'NA',
slices = [1]#np.arange(30)
n_jobs = -1#12
change_type = 'leave_one_in'

runs = {
	'TK': [0,1,2,3,4,5],
	'DE': [0,1,2,3,4,5,6],
	'JS': [0,1,2,3,4,5,6,7],	
	'JW': [0,1,2,3,4,5],
	'NA': [0,1,2,3,4,5]	
}

jobscript_fn = '/home/vanes/job_submissions/jobscript_CV'

###############################

for subject in subjects:

	for run_num in runs[subject]:

		for slice_no in slices:

			jobscript = open(jobscript_fn)
			working_string = jobscript.read()
			jobscript.close()

			RE_dict =  {
			'---subject---': 				subject, 
			'---n_jobs---':					str(n_jobs),
			'---slice_no---': 				str(slice_no),
			'---change_type---':			change_type,
			'---run_num---':				str(run_num),
			}

			for e in RE_dict:
				rS = re.compile(e)
				working_string = re.sub(rS, RE_dict[e], working_string)
			
			of = open(jobscript_fn+'_tmp', 'w')
			of.write(working_string)
			of.close()

			os.system('sbatch ' + jobscript_fn +'_tmp')

