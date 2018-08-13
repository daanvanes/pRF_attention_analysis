import os, sys
from IPython import embed as shell
import numpy as np

os.system('cd ~/job_submissions/')

for jobno in np.arange(int(sys.argv[1]),int(sys.argv[2])+1,1):
	os.system('scancel ' + str(jobno))
