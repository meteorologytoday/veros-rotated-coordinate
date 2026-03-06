# Goal

This repo is to rotate the gaussian grid coordinate in veros such that the coordinate poles sit in Greenland and Antarctica to avoid singularities. The physics part is done by reassigning the Coriolis parameter to the rotated coordinate system.

## Install

```
pip install matplotlib
pip install veros
pip install esmf-regrid
```

## Running

```
# Produce land-sea mask file and weight by running
bash 01_generate_grids_and_weights.sh

# Run veros
veros run veros_case_setup.py
```
