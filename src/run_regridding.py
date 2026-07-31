import xarray as xr
from ESMF_regrid import ESMFRegridder
import numpy as np

JCM_shape = xr.load_dataset("grid_data/JCM_T31.SCRIP.nc")["grid_shape"].to_numpy()
RG_shape = xr.load_dataset("grid_data/rotating_gaussian_grid_4.00deg.SCRIP.nc")["grid_shape"].to_numpy()

regridder_forward = ESMFRegridder(
    weight_file = "grid_data/weight_algo-bilinear_JCM_T31_to_RG4.00deg.nc",
    src_shape = JCM_shape,
    dst_shape = RG_shape,
)

regridder_backward = ESMFRegridder(
    weight_file = "grid_data/weight_algo-bilinear_RG4.00deg_to_JCM_T31.nc",
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
ax[1, 1].set_title(f"Std of difference / max(abs(data)) = {np.round(difference_std / np.amax(np.abs(data)) * 100)} %")
#fig.suptitle(f"Rotate ${rotation_degree:.1f}^{{\\circ}}$ along longitude ${rotation_along_longitude_degree:.1f}^{{\\circ}}$ (right-hand rule)")

#fig.savefig("rotating_gaussian_landsea_mask.svg")
plt.show()


# ============================================================================
# Vector remap verification
# ============================================================================

import jax.numpy as jnp
from vector_remap import VectorRemapper

grid_shape = tuple(int(x) for x in RG_shape)
remapper = VectorRemapper.from_grid_file(
    grid_file="grid_data/rotating_gaussian_grid_4.00deg.nc",
    grid_shape=grid_shape,
)

print("\n" + "=" * 60)
print("Vector remap verification on RG 4-degree grid")
print("=" * 60)

# Case 1: pure eastward (u_geo, v_geo) = (1, 0)
# Expected rotated: (cos_alpha, -sin_alpha)
u1_geo = jnp.ones(grid_shape)
v1_geo = jnp.zeros(grid_shape)
u1_rot, v1_rot = remapper.remap_to_rotated(u1_geo, v1_geo)
err1_u = float(jnp.max(jnp.abs(u1_rot - remapper.cos_alpha)))
err1_v = float(jnp.max(jnp.abs(v1_rot - (-remapper.sin_alpha))))
print(f"\nCase 1: (u_geo, v_geo) = (1, 0)  ->  expect (cos_alpha, -sin_alpha)")
print(f"  max |u_rot - cos_alpha|    = {err1_u:.2e}")
print(f"  max |v_rot - (-sin_alpha)| = {err1_v:.2e}")

# Case 2: pure northward (u_geo, v_geo) = (0, 1)
# Expected rotated: (sin_alpha, cos_alpha)
u2_geo = jnp.zeros(grid_shape)
v2_geo = jnp.ones(grid_shape)
u2_rot, v2_rot = remapper.remap_to_rotated(u2_geo, v2_geo)
err2_u = float(jnp.max(jnp.abs(u2_rot - remapper.sin_alpha)))
err2_v = float(jnp.max(jnp.abs(v2_rot - remapper.cos_alpha)))
print(f"\nCase 2: (u_geo, v_geo) = (0, 1)  ->  expect (sin_alpha, cos_alpha)")
print(f"  max |u_rot - sin_alpha| = {err2_u:.2e}")
print(f"  max |v_rot - cos_alpha| = {err2_v:.2e}")

# Case 3: arbitrary (u_geo, v_geo) = (0.6, 0.8)
# Verify round-trip and magnitude preservation (rotation is orthogonal)
u3_geo = jnp.full(grid_shape, 0.6)
v3_geo = jnp.full(grid_shape, 0.8)
u3_rot, v3_rot = remapper.remap_to_rotated(u3_geo, v3_geo)
u3_back, v3_back = remapper.remap_to_geo(u3_rot, v3_rot)
err3_roundtrip = float(jnp.max(jnp.abs(u3_back - u3_geo) + jnp.abs(v3_back - v3_geo)))
err3_magnitude = float(jnp.max(jnp.abs(
    jnp.sqrt(u3_rot**2 + v3_rot**2) - jnp.sqrt(u3_geo**2 + v3_geo**2)
)))
print(f"\nCase 3: (u_geo, v_geo) = (0.6, 0.8)  arbitrary")
print(f"  round-trip max error = {err3_roundtrip:.2e}")
print(f"  magnitude max error  = {err3_magnitude:.2e}")
