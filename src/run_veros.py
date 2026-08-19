import glob
import os
from pathlib import Path

import netCDF4
import numpy as np
from tqdm import tqdm
from veros import runtime_settings
print("Setting veros.runtime_settings...")
setattr(runtime_settings, "backend", "numpy")
setattr(runtime_settings, "force_overwrite", True)
setattr(runtime_settings, 'device', 'cpu')
from veros_case_setup import generateVerosSetup

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def write_snapshot(state, out_path):
    """Dump a handful of prognostic/forcing fields to `out_path`.

    Veros' own diagnostics (snapshot/averages/etc.) are disabled in
    `veros_case_setup.py` because writing *any* netCDF file through
    h5netcdf/h5py crashes ("Unspecified error in H5DSis_scale") once the
    streamfunction solver's per-island ILU solves have run -- an
    environment-level HDF5/JAX interaction, not specific to this setup.
    Writing directly with the `netCDF4` package (a separate C library
    binding, unaffected by whatever corrupts h5py's state) sidesteps that
    entirely.
    """
    vs = state.variables
    tau = int(vs.tau)
    xt, yt, xu, yu, zt = (np.asarray(vs.xt[2:-2]), np.asarray(vs.yt[2:-2]),
                          np.asarray(vs.xu[2:-2]), np.asarray(vs.yu[2:-2]), np.asarray(vs.zt))

    with netCDF4.Dataset(out_path, "w") as f:
        for name, coord in (("xt", xt), ("yt", yt), ("xu", xu), ("yu", yu), ("zt", zt)):
            f.createDimension(name, coord.size)
            f.createVariable(name, "f8", (name,))[:] = coord

        f.createVariable("temp", "f8", ("xt", "yt", "zt"))[:] = np.asarray(vs.temp[2:-2, 2:-2, :, tau])
        f.createVariable("salt", "f8", ("xt", "yt", "zt"))[:] = np.asarray(vs.salt[2:-2, 2:-2, :, tau])
        f.createVariable("u", "f8", ("xu", "yt", "zt"))[:] = np.asarray(vs.u[2:-2, 2:-2, :, tau])
        f.createVariable("v", "f8", ("xt", "yu", "zt"))[:] = np.asarray(vs.v[2:-2, 2:-2, :, tau])
        f.createVariable("surface_taux", "f8", ("xu", "yt"))[:] = np.asarray(vs.surface_taux[2:-2, 2:-2])
        f.createVariable("surface_tauy", "f8", ("xt", "yu"))[:] = np.asarray(vs.surface_tauy[2:-2, 2:-2])
        f.createVariable("forc_temp_surface", "f8", ("xt", "yt"))[:] = np.asarray(vs.forc_temp_surface[2:-2, 2:-2])
        f.createVariable("forc_salt_surface", "f8", ("xt", "yt"))[:] = np.asarray(vs.forc_salt_surface[2:-2, 2:-2])

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
for f in glob.glob("output_veros.snapshot.*.nc"):
    print(f"Deleting stale output file: {f}")
    os.remove(f)

print("Setup ocean model")
ocn_model.setup()
settings = ocn_model.state.settings

print("Step ocean model")
total_steps = int(total_time / settings.dt_tracer)
steps_per_snapshot = max(1, int(86400.0 / settings.dt_tracer))  # ~daily
for step in tqdm(range(total_steps)):
    ocn_model.step(ocn_model.state)
    if (step + 1) % steps_per_snapshot == 0 or step == total_steps - 1:
        day = (step + 1) * settings.dt_tracer / 86400.0
        write_snapshot(ocn_model.state, f"output_veros.snapshot.{day:06.2f}.nc")

