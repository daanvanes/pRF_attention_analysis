#!/bin/bash
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 23
#SBATCH -p normal
cd /home/vanes/PRF_2_analysis
python run_plots.py ---coordinate_system--- ---AF_slopes--- ---restriction--- ---logtype--- ---SD_AF_projection--- ---roi--- ---error_mesure---
