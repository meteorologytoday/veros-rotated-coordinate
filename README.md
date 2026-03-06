# Goal

This repo is to rotate the gaussian grid coordinate in veros such that the coordinate poles sit in Greenland and Antarctica to avoid singularities. The physics part is done by reassigning the Coriolis parameter to the rotated coordinate system.

## Install

```
pip install global-land-mask
pip install veros
pip install esmf-regrid
pip install matplotlib
```

## Running

```
# Produce land-sea mask file and weight by running
bash 01_generate_grids_and_weights.sh

# Run veros
veros run veros_case_setup.py

# Demonstrate regird
python3 run_regridding.py
```

## Land-sea Mask

![Landsea mask](https://github.com/meteorologytoday/veros-rotated-coordinate/blob/main/figure/rotating_gaussian_landsea_mask.svg)

## To-do
Work out vector regridding between JCM and Veros grid

