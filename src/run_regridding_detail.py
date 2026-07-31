import xarray as xr
from ESMF_regrid import ESMFRegridder
import numpy as np

JCM_ds = xr.load_dataset("grid_data/JCM_T31.SCRIP.nc")
RG_ds = xr.load_dataset("grid_data/rotating_gaussian_grid_4.00deg.SCRIP.nc")

JCM_shape = JCM_ds["grid_shape"].to_numpy()
RG_shape = RG_ds["grid_shape"].to_numpy()

regridder_JCM2RG = ESMFRegridder(
    weight_file = "grid_data/weight_algo-bilinear_JCM_T31_to_RG4.00deg.nc",
    src_shape = JCM_shape,
    dst_shape = RG_shape,
)

regridder_RG2JCM = ESMFRegridder(
    weight_file = "grid_data/weight_algo-bilinear_RG4.00deg_to_JCM_T31.nc",
    src_shape = RG_shape,
    dst_shape = JCM_shape,
)

RG_imask = RG_ds["grid_landseamask"].to_numpy().reshape(RG_shape)
RG_landseamask_data = np.zeros(RG_shape)
RG_landseamask_data[RG_imask == 0] = 1.0
RG_data = RG_landseamask_data.copy()
data_regridded_RG2JCM = regridder_RG2JCM(RG_data)

JCM_imask = JCM_ds["grid_landseamask"].to_numpy().reshape(JCM_shape)
JCM_landseamask_data = np.zeros(JCM_shape)
JCM_landseamask_data[JCM_imask == 0] = 1.0
JCM_data = np.ones(JCM_shape)
data_regridded_JCM2RG = regridder_JCM2RG(JCM_data)

def zero2nan(arr):
    _arr = np.array(arr)
    _arr[_arr == 0] = np.nan
    return _arr


JCM_regrid_total_missing = zero2nan(np.array((JCM_landseamask_data != 0) & (data_regridded_RG2JCM == 0)).astype(np.float32))
RG_regrid_total_missing = zero2nan(np.array((RG_landseamask_data != 0) & (data_regridded_JCM2RG == 0)).astype(np.float32))

JCM_regrid_partial_missing = zero2nan(np.array((JCM_landseamask_data != 0) & (np.abs(data_regridded_RG2JCM - 1) > 1e-3)).astype(np.float32))
RG_regrid_partial_missing = zero2nan(np.array((RG_landseamask_data != 0) & (np.abs(data_regridded_JCM2RG - 1) > 1e-3)).astype(np.float32))

import matplotlib as mplt
mplt.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

fig, ax = plt.subplots(2, 5, figsize=(16, 8), squeeze=False)

for i, (_d, _title) in enumerate([
    (RG_data, "RG_data"),
    (JCM_landseamask_data, "Target grid's landsea mask"),
    (data_regridded_RG2JCM, "RG_data to JCM grid"),
    (JCM_regrid_partial_missing, "Partially missing\n(RG => JCM)"),
    (JCM_regrid_total_missing, "Totally missing\n(RG => JCM)"),

    (JCM_data, "JCM_data"),
    (RG_landseamask_data, "Target grid's landsea mask"),
    (data_regridded_JCM2RG, "JCM_data to RG grid"),
    (RG_regrid_partial_missing, "Partially missing\n(JCM => RG)"),
    (RG_regrid_total_missing, "Totally missing\n(JCM => RG)"),
]):
    _ax = ax.flatten()[i]
    im =_ax.imshow(_d, cmap="GnBu", vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=_ax)
    _ax.set_title(_title)
    _ax.invert_yaxis()


fig.savefig("mask_transform.svg", dpi=300)
