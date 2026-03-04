#!/bin/bash


python3 rotating_coordinate_generation.py
python3 JCMGrid.py

ESMF_RegridWeightGen -s grid_JCM_T31.SCRIP.nc -d rotating_gaussian_grid_4deg.SCRIP.nc -m bilinear -w weight_algo-bilinear_JCM_T31_to_RG4deg.nc -i
ESMF_RegridWeightGen -d grid_JCM_T31.SCRIP.nc -s rotating_gaussian_grid_4deg.SCRIP.nc -m bilinear -w weight_algo-bilinear_RG4deg_to_JCM_T31.nc -i

