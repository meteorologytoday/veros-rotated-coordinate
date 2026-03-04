#!/bin/bash


python3 rotating_coordinate_generation.py
python3 JCMGrid.py

mkdir weights

for method in bilinear conserve ; do
    ESMF_RegridWeightGen -s grid_JCM_T31.SCRIP.nc -d rotating_gaussian_grid_4.00deg.SCRIP.nc -m $method -w weights/weight_algo-${method}_JCM_T31_to_RG4.00deg.nc 
    ESMF_RegridWeightGen -d grid_JCM_T31.SCRIP.nc -s rotating_gaussian_grid_4.00deg.SCRIP.nc -m $method -w weights/weight_algo-${method}_RG4.00deg_to_JCM_T31.nc 
done
