#!/bin/bash

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm
grid_dir=grid_data

set -x

for method in bilinear conserve ; do
    ESMF_RegridWeightGen -s $grid_dir/JCM_T31.SCRIP.nc -d $grid_dir/rotating_gaussian_grid_4.00deg.SCRIP.nc -m $method -w $grid_dir/weight_algo-${method}_JCM_T31_to_RG4.00deg.nc 
    ESMF_RegridWeightGen -d $grid_dir/JCM_T31.SCRIP.nc -s $grid_dir/rotating_gaussian_grid_4.00deg.SCRIP.nc -m $method -w $grid_dir/weight_algo-${method}_RG4.00deg_to_JCM_T31.nc
done
