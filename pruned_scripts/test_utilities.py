from __future__ import division

# import python moduls
from IPython import embed as shell
import numpy as np
from utilities import CustomStatUtilities
from statsmodels.stats.weightstats import DescrStatsW
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as pl
mpl.rc_file_defaults()

CUO = CustomStatUtilities()	

###################################################
## first let's test the circular statistics
###################################################

# to do that, let's create a uniform amount of angles:
angles = np.random.uniform(0,np.pi*2,1e4)
# then, create some data that is highly correlated with this angle:
data = np.sin(angles)+np.random.rand(1e4)*1
# plot them against each other:

# let's compute the circular correlation:
corr = CUO.angular_linear_correlation(angles,data,double_peak=False)

f = pl.figure()
s = f.add_subplot(111)
pl.title(corr,fontsize=14)
pl.plot(angles,data,'o')

