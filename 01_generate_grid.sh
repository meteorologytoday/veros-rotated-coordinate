#!/bin/bash

export PYTHONPATH=/home/tienyiao/projects_local/project_jax-esm/jem_repo/jax-esm
python3 src/rotating_coordinate_generation.py
python3 src/JCMGrid.py
