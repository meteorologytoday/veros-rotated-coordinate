#!/bin/bash


python3 rotating_coordinate_generation.py
python3 JCMGrid.py
grid_dir=grid_data
weight_dir=regrid_data

mkdir $weight_dir
mkdir $grid_dir

for method in bilinear conserve ; do
    ESMF_RegridWeightGen -s $grid_dir/grid_JCM_T31.SCRIP.nc -d $grid_dir/rotating_gaussian_grid_4.00deg.SCRIP.nc -m $method -w $weight_dir/weight_algo-${method}_JCM_T31_to_RG4.00deg.nc 
    ESMF_RegridWeightGen -d $grid_dir/grid_JCM_T31.SCRIP.nc -s $grid_dir/rotating_gaussian_grid_4.00deg.SCRIP.nc -m $method -w $weight_dir/weight_algo-${method}_RG4.00deg_to_JCM_T31.nc 
done
