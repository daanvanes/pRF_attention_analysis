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

The results of these procedures are stored in the group_level.hdf5 file, which can be found at https://figshare.com/s/ef7669dc02f1ecb1d8f7. The preprocessing interfaces with FSL and other utilities through use of a general package of fMRI analysis tools created by our lab (tknapen.github.io), which can be found at: https://github.com/VU-Cog-Sci/analysis_tools

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

