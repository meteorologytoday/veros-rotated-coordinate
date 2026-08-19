"""Minimal, dependency-free applier for pre-computed ESMF regridding weight
files (as produced by ESMF_RegridWeightGen / the repo's
03_generate_remapping_weights step), so this repo doesn't need the `jem`
package installed just to regrid a field from one SCRIP grid to another.

Convention: both the source and destination arrays are 2-D, shaped
(n_lon, n_lat), and correspond to the weight file's SCRIP grids flattened
lon-fastest -- i.e. `array.reshape(-1, order="F")` recovers the flat
`n_a`/`n_b` ordering ESMF's `row`/`col` indices index into. This matches
`native_lat`/`native_lon`/`grid_cos_angle` etc. elsewhere in this repo
(see e.g. veros_case_setup.py's `GridInfo`).
"""

import numpy as np
import xarray as xr


class ESMFRegridder:
    def __init__(self, weight_file: str):
        ds = xr.open_dataset(weight_file)
        self.src_shape = tuple(int(n) for n in ds["src_grid_dims"].to_numpy())  # (n_lon, n_lat)
        self.dst_shape = tuple(int(n) for n in ds["dst_grid_dims"].to_numpy())  # (n_lon, n_lat)
        self.row = ds["row"].to_numpy() - 1  # ESMF is 1-based
        self.col = ds["col"].to_numpy() - 1
        self.S = ds["S"].to_numpy()
        self.dst_size = int(np.prod(self.dst_shape))

    def __call__(self, src_data: np.ndarray) -> np.ndarray:
        src_flat = np.asarray(src_data).reshape(-1, order="F")
        dst_flat = np.zeros(self.dst_size, dtype=src_flat.dtype)
        np.add.at(dst_flat, self.row, self.S * src_flat[self.col])
        return dst_flat.reshape(self.dst_shape, order="F")
