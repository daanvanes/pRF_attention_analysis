# README

This software package contains the analysis presented in the manuscript 'Spatial sampling in human visual cortex is modulated by both spatial and feature-based attention', currently available at: https://www.biorxiv.org/content/early/2018/03/24/147223.

The functionality in this package are called by the run*.py scripts. 

The fitted parameters from the Feature Preference Mapper GLM and the pRF parameters are available online. The raw timecourse niftis are not, therefore the preprocessing (run_subjects.py) cannot be repeated, but the statistical analyses and figures created in our manuscript can be repeated using the run_plots.py and run_eye_plots.py scripts (see below). 

#### run_subjects.py
This script runs:
* fMRI preprocessing
* Feature Mapper GLM
* HRF estimation
* Gaze data preprocessing
* pRF fitting

The preprocessing interfaces with FSL and other utilities through use of a general package of fMRI analysis tools created by our lab (tknapen.github.io), which can be found at: https://github.com/VU-Cog-Sci/analysis_tools

### pRF HDF5 file

The results of the preprocessing procedures run in run_subjects.py are stored in the group_level.hdf5 file, which can be found at https://figshare.com/articles/prf_fit_and_mapper_glm_results/5488276.

If you are unfamiliar with hdf5 files, you can find information about the format here: https://support.hdfgroup.org/HDF5/whatishdf5.html. The files can be inspected using an hdf5 viewer like hdfCompass: https://support.hdfgroup.org/projects/compass/. You can access and import the data into Matlab using hdf5read (https://nl.mathworks.com/help/matlab/ref/hdf5read.html), and in Python using h5py (https://www.h5py.org). 

The file is organized according to subjects. Each subject then has subfields for each ROI (like lh.V1). These ROI fields contain 6 fields corresponding to the pRF results (statistics about the fit, (corrs) and resulting pRF parameters (results)) fitted on data from the Attend Fixation (Fix), Attend Color (Colors) and Attend Temporal Frequency (Speed) conditions:

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

The _results fields contains 26 values for each voxel that describe the pRF model. Many of these fields were added at some point in the development of the fit procedure, but are not useful to you end user. The main and useful pRF parameters include (counting in Pythonian fashion from 0):

* pRF center x location: 1
* pRF center y location: 13
* pRF center polar angle: 2
* pRF center eccentricity: 11
* pRF size (1 sd): 17
* prediction amplitude: 20
* prediction baseline: 5

Due to a misspecification of the stimulus radius in the fit procedure, pRF x, y, eccentricity and size are wrongly scaled. To resolve this, multiply all values with 0.48 to arrive at values in degrees of visual angle. 

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

The behavioral data is stored in the Color / Fix / Fix_no_stim / and Speed groups. The structure of the subfields here is given by the following: {condition}_{type}_{ecc_bin}_run_{runindex}. Condition refers to (Color / Fix / Fix_no_stim /  Speed); type can be either response_times, response_values or staircase_values. For you end use, the response time and staircase values are not very useful, as (1) the offset of the timings here is not given, and (2) these staircase values were converted to other values in the experimental software. More information about the staircase values and response times can be obtained from the gaze data hdf5 (see below). Yet, the response_values variable is directly interpretable and corresponds to accuracy. The ecc_bin refers to the distance to the fovea in 3 bins (0/1/2). The runindex refers to fMRI run. 

### Gaze data HDF5

The gaze data recorded by the Eyelink 1000 are stored in a file named group_level.hdf5 file, which can be found at https://figshare.com/articles/prf_fit_and_mapper_glm_results/5488276.

This file contains a separate subfield for each experimental run. Within these fields, are the following subfields. 

The raw gaze data is stored in a subfield called 'block_0' (the 'block' subfield is redundant). Withink block_0, the relevant data is stored in 'block_0_values'. The subfield 'block0_items' provides the column names.

In the remaining subfields, the column names are provided in the hdf5:

1. blinks_from_message_file. This subfield contains the blinks detected by the Eyelink software. 
2. buttons. This subfield contains buttons pressed in timestamps relative to the experiment (exp_timestamp) and in eyelink time (EL_timestamp)
3. fixation_from_message_file. This subfield contains the fixations detected by the Eyelink software.
4. mapper_transients: empty field
5. parameters: these are the trial specific parameters from the experiment (trial refers to single bar pass)
* BY_color: refers to the subject specific BY to RG luminance gain, where 1 would mean equal luminance.
* BY_comparison_color: used by the color matcher procedure to determine subject specific relative luminances
* PRF_ITI_in_TR: time between two bar passes in TR
* PRF_period_in_TR: bar pass duration in TR
* RG_color: baseline color to which the BY color is determined using the BY_color ratio
* TR: duration in s
* bar_width_ratio: ratio of bar width to stimulus diameter
* element_size: individual Gabor width in pixels
* element_spatial_frequency: Gabor spatial frequency in pixels
* fast_speed: temporal frequency of fast moving Gabors in Hz
* mapper_ITI_in_TR: time between two mapper stimuli (i.e. full-field colors vs temporal frequency experiment) 
* minimum_pulse_gap: minimum of exponential distribution (in s) that determines a transient change within a certain stimulus dimension (i.e. color / temporal frequency / fixation)
* num_elements: number of Gabor elements that made up the bar stimulus
* num_fns_trials: number of 'fix no stim' trials: empty trials with fixation (i.e. no bar). A trial here refers to a full bar pass.
* num_trials: refers to number of Feature Mapper trials per stimulus dimension (i.e. color / tf)
* orientation: bar movement direction in radians
* redraws_per_TR: number of updates per TR of Gabor position within the bar, Gabor spatial frequency, and Gabor color and temporal frequency
* slow_speed: temporal frequency of slow moving Gabors in Hz
* stim_size: relative to screen height 
* task_index: the dimension to attend (0:TF,1:Color,2:Fix,3:Fix_no_stim)
* task_rate: refers to mean of exponential distribution (in s) that determines a transient change within a certain stimulus dimension (i.e. color / temporal frequency / fixation)
6. saccades_from_message_file: This subfield contains the saccades detected by the Eyelink software. 
7. sounds: empty subfield
8. transients: changes within a certain stimulus dimension. 
* transient_type: 0:TF,1:Color,2:Fix 
* ecc_bin: bar stimulus ecc bin 
* phase: 0-1 runs from start to end of bar sweep
* value: staircase value used by experiment to determine stimulus intensity
9. trial_phases: progression of trial phases 
* trial_phase_trial: trial number
* trial_phase_index: 0=instruction only present during first trial, 1=1s delay, 2= wait for TR, 3=bar sweep, 4=ITI
10. trial_start_exp_timestamp: overview of timing of each trial

#### run_plots.py
This script analyzes the resulting pRF parameters. This includes plots and statistics concerning pRF parameter differences between attentional conditions. In addition, it runs and visualizes the attentional gain field model. It does so based on the following files:

* 'frames' that is a lookup table for which pRF parameter goes with what index
    - https://figshare.com/articles/frames_for_plotting_pRF_results/5488246
* the pRF and Mapper results in hdf5 format
    - https://figshare.com/articles/prf_fit_and_mapper_glm_results/5488276
* stimulus design matrix for AF fitting
    - https://figshare.com/articles/Design_matrix_for_AF_model/5488234

#### run_eye_plots.py
This script performs the gaze analysis, rotating eye position in direction of the bar. It does so based on an hdf5 file that can be downloaded from: https://figshare.com/articles/eye_analysis_hdf5/5488687. 

Any questions regarding this repository can be directed to: daan.van.es@gmail.com. 

## license

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If you use the Software for your own research, cite the paper.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    