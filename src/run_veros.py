import glob
import os
from pathlib import Path

from tqdm import tqdm
from veros import runtime_settings
print("Setting veros.runtime_settings...")
setattr(runtime_settings, "backend", "numpy")
setattr(runtime_settings, "force_overwrite", True)
setattr(runtime_settings, 'device', 'cpu')
from veros_case_setup import generateVerosSetup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

total_time = 86400 * 10
VerosCaseSetup = generateVerosSetup(
    scrip_grid_file=str(DATA_DIR / "RotatedGaussianLatLon.SCRIP.nc"),
    landsea_mask_file=str(DATA_DIR / "landsea_mask_fraction_RotatedGaussianLatLon.nc"),
    jcm_grid_file=str(DATA_DIR / "JCM_T31.SCRIP.nc"),
    a2o_weight_file=str(DATA_DIR / "weight_algo-conserve_JCM_T31_to_RotatedGaussianLatLon.nc"),
)
ocn_model = VerosCaseSetup()

# `force_overwrite` doesn't stop h5netcdf choking on a pre-existing output
# file's leftover dimension scales, so remove stale outputs first.
for f in glob.glob("output_veros.*.nc"):
    print(f"Deleting stale output file: {f}")
    os.remove(f)

print("Setup ocean model")
ocn_model.setup()
settings = ocn_model.state.settings

print("Step ocean model")
total_steps = int(total_time / settings.dt_tracer )
for step in tqdm(range(total_steps)):
    ocn_model.step(ocn_model.state)

