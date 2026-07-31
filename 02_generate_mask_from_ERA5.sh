#!/bin/bash

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm

set -x

python3 src/convert_era5_landsea_mask.py
