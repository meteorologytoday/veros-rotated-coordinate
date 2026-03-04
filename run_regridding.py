import xarray as xr
from ESMF_regrid import ESMFRegridder
import numpy as np

JCM_shape = xr.load_dataset("grid_JCM_T31.SCRIP.nc")["grid_shape"].to_numpy()
RG_shape = xr.load_dataset("rotating_gaussian_grid_4.00deg.SCRIP.nc")["grid_shape"].to_numpy()

regridder_forward = ESMFRegridder(
    weight_file = "weights/weight_algo-bilinear_JCM_T31_to_RG4.00deg.nc",
    src_shape = JCM_shape,
    dst_shape = RG_shape,
)

regridder_backward = ESMFRegridder(
    weight_file = "weights/weight_algo-bilinear_RG4.00deg_to_JCM_T31.nc",
    src_shape = RG_shape,
    dst_shape = JCM_shape,
)


data = xr.open_dataset("data/atm_sample.nc")["specific_humidity"].isel(time=0, level=0).to_numpy()
#data = xr.open_dataset("data/atm_sample.nc")["surface_flux.tskin"].isel(time=0).to_numpy()

data_regridded = regridder_forward(data)
data_recovered = regridder_backward(data_regridded)
data_difference = data_recovered - data
difference_std = np.std(data_difference)

print(f"difference_std / max(abs(data)) = {difference_std / np.amax(np.abs(data)) * 100} %")

import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

ax[0, 0].imshow(data.transpose(), cmap="GnBu")
ax[0, 1].imshow(data_regridded, cmap="GnBu")
ax[1, 0].imshow(data_recovered.transpose(), cmap="GnBu")
ax[1, 1].imshow(data_difference.transpose(), cmap="bwr")
for _ax in ax.flatten():
    _ax.invert_yaxis()

ax[0, 0].set_title("JCM T31")
ax[0, 1].set_title("Rotated Gaussian 4deg")
ax[1, 0].set_title("Recovered JCM T31")
ax[1, 1].set_title("Difference")
#fig.suptitle(f"Rotate ${rotation_degree:.1f}^{{\\circ}}$ along longitude ${rotation_along_longitude_degree:.1f}^{{\\circ}}$ (right-hand rule)")

#fig.savefig("rotating_gaussian_landsea_mask.svg")
plt.show()


