# README

This software package contains the analysis presented in the manuscript 'Spatial sampling in human visual cortex is modulated by both spatial and feature-based attention', currently available at: https://www.biorxiv.org/content/early/2018/03/24/147223.

The functionality in this package are called by the run*.py scripts. 

#### run_subjects.py
This script runs:
* fMRI preprocessing
* Feature Mapper GLM
* HRF estimation
* Gaze data preprocessing
* pRF fitting

The preprocessing interfaces with FSL and other utilities through use of a general package of fMRI analysis tools created by our lab (tknapen.github.io), which can be found at: https://github.com/VU-Cog-Sci/analysis_tools

### pRF HDF5 file

The results of these procedures are stored in the group_level.hdf5 file, which can be found at https://figshare.com/s/ef7669dc02f1ecb1d8f7. 

If you are unfamiliar with hdf5 files, you can find information about the format here: https://support.hdfgroup.org/HDF5/whatishdf5.html. The files can be inspected using an hdf5 viewer like hdfCompass: https://support.hdfgroup.org/projects/compass/. You can access and import the data into Matlab using hdf5read (https://nl.mathworks.com/help/matlab/ref/hdf5read.html), and in Python using h5py (https://www.h5py.org). 

The file is organized according to subjects (DE/NA/JS/JW/TK). Each subject then has subfields for each ROI (like lh.V1). These ROI fields contain 6 fields corresponding to the pRF results (statistics about the fit, (corrs) and resulting pRF parameters (results)) fitted on data from the Attend Fixation (Fix), Attend Color (Colors) and Attend Temporal Frequency (Speed) conditions:

* Colors_corrs
* Colors_results
* Fix_corrs
* Fix_results
* Speed_corrs
* Speed_results

The _corrs fields contain 5 values for each voxel that describe the goodness of fit between the pRF model prediction and the observed timecourses:

0. residual sum of squares
1. R-squared
2. pearson correlation
3. kendalls tau
4. spearman correlation

The _results fields contains 24 values for each voxel that describe the pRF model. Many of these fields were added at some point in the development of the fit procedure, but are not useful to you end user. The main and useful pRF parameters include (counting in Pythonian fashion from 0):

* pRF center x location: 1
* pRF center y location: 13
* pRF center polar angle: 2
* pRF center eccentricity: 11
* pRF size (1 sd): 17
* prediction amplitude: 20
* prediction baseline: 5

Due to an error in the definition of the stimulus radius in the fit procedure, pRF x, y, eccentricity and size are wrongly scaled. To resolve this, multiply all values with 0.48 to arrive at values in degrees of visual angle. 

The HRF_params field contains 3 hrf parameters for each voxel. These values are identical for each voxel, as we used the median HRF across the 1000 most responsive voxels as the subject-specific HRF (see Methods). The values are scaling factors of the 0th, 1st and 2nd derivative of the SPM canonical HRF.

The Mapper_* fields contain values that resulted from the Feature Preference Mapper. The feature mapper was a 2x2 design with the factor color (black and white (bw) or color (col)), and temporal frequency (moving or static).

Each of these subfields (copes, t_stat, varcopes, z_stat) contains statistics for each of the following contrasts (the contrast used in our analysis as the relative color compared to temporal frequency index is contrast 10):

0. bw_static > baseline
1. bw_moving > baseline
2. col_static > baseline
3. col_moving > baseline
4. col_moving > col_static
5. col_moving > bw_static
6. (bw_moving+col_moving) > (bw_static+col_static)
7. col_moving > bw_moving
8. col_static > bw_static
9. (col_moving+col_static) > (bw_moving+bw_static)
10. col_static > bw_moving
11. bw_moving > bw_static
12. test contrast (not interpretable)

In the 

#### run_plots.py
This script analyzes the resulting pRF parameters. This includes plots and statistics concerning pRF parameter differences between attentional conditions. In addition, it runs and visualizes the attentional gain field model. It does so based on the following files:

* 'frames' that is a lookup table for which pRF parameter goes with what index
    - https://figshare.com/s/3c0e424591e7cbbf5178
* the pRF and Mapper results in hdf5 format
    - https://figshare.com/s/ef7669dc02f1ecb1d8f7
* stimulus design matrix for AF fitting
    - https://figshare.com/s/e869425d412121f68848 

#### run_eye_plots.py
This script performs the gaze analysis, rotating eye position in direction of the bar. It does so based on an hdf5 file that can be downloaded from: https://figshare.com/s/a2f79644813675992185. 

Any questions regarding this repository can be directed to: daan.van.es@gmail.com. 

## license

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If you use the Software for your own research, cite the paper.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    