# Goal

This repo is to rotate the gaussian grid coordinate in veros such that the coordinate poles sit in Greenland and Antarctica to avoid singularities. The physics part is done by reassigning the Coriolis parameter to the rotated coordinate system.

1. Produce land-sea mask file by running `python3 rotating_coordinate_generation.py`
2. Run veros with `python3 run_test.py`
