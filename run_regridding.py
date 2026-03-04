import xarray as xr
from ESMF_regrid import ESMFRegridder
import jax.numpy as jnp



JCM_shape = xr.load_dataset("grid_JCM_T31.SCRIP.nc")["grid_shape"].to_numpy()
RG_shape = xr.load_dataset("rotating_gaussian_grid_4deg.SCRIP.nc")["grid_shape"].to_numpy()

regridder = ESMFRegridder(
    weight_file = "weight_algo-bilinear_JCM_T31_to_RG4deg.nc",
    src_shape = JCM_shape,
    dst_shape = RG_shape,
)

#data = xr.open_dataset("data/atm_sample.nc")["specific_humidity"].isel(time=0, level=-1).to_numpy()
data = xr.open_dataset("data/atm_sample.nc")["surface_flux.tskin"].isel(time=0).to_numpy()


import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

fig, ax = plt.subplots(2, 1, figsize=(10, 6))

ax[0].imshow(data.transpose(), cmap="GnBu")
ax[1].imshow(regridder(data), cmap="GnBu")
for _ax in ax.flatten():
    _ax.invert_yaxis()

ax[0].set_title("JCM T31")
ax[1].set_title("Rotated Gaussian 4deg")
#fig.suptitle(f"Rotate ${rotation_degree:.1f}^{{\\circ}}$ along longitude ${rotation_along_longitude_degree:.1f}^{{\\circ}}$ (right-hand rule)")

#fig.savefig("rotating_gaussian_landsea_mask.svg")
plt.show()


