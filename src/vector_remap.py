"""
Remap vector components between a rotated coordinate frame and geographical
(true east/north) components.

The rotation angle alpha at each grid point is defined by:
    cos_alpha = rotated_east · true_east
    sin_alpha = rotated_east · true_north

These are stored per grid point and can be loaded from a netCDF file
produced by rotating_coordinate_generation.py (non-standard extension).

Remapping formulas (rotated -> geo):
    u_geo =  u_rot * cos_alpha - v_rot * sin_alpha
    v_geo =  u_rot * sin_alpha + v_rot * cos_alpha

The inverse (geo -> rotated) is the transpose of the above matrix.

Vector components with leading batch dimensions (e.g. time, depth) are
supported via broadcasting against cos_alpha and sin_alpha.

JAX compatibility: cos_alpha and sin_alpha are stored as jax arrays, so
remap_to_geo and remap_to_rotated are jit-compatible and support autodiff.
"""

import numpy as np
import jax.numpy as jnp
import xarray as xr
from typing import Tuple


class VectorRemapper:
    """
    Remap vector components between a rotated coordinate frame and
    geographical (true east/north) components.

    JAX-compatible: all internal arrays are jax arrays, so the remap
    methods can be used inside jit-compiled functions.
    """

    def __init__(self, cos_alpha: jnp.ndarray, sin_alpha: jnp.ndarray):
        """
        Parameters
        ----------
        cos_alpha : array-like
            cos of the rotation angle on the grid, shape (nlat, nlon).
        sin_alpha : array-like
            sin of the rotation angle on the grid, same shape as cos_alpha.
        """
        self.cos_alpha = jnp.asarray(cos_alpha)
        self.sin_alpha = jnp.asarray(sin_alpha)

        if self.cos_alpha.shape != self.sin_alpha.shape:
            raise ValueError(
                f"cos_alpha shape {self.cos_alpha.shape} does not match "
                f"sin_alpha shape {self.sin_alpha.shape}"
            )

    @classmethod
    def from_grid_file(
        cls,
        grid_file: str,
        grid_shape: Tuple[int, ...],
        cos_alpha_name: str = "grid_cos_alpha",
        sin_alpha_name: str = "grid_sin_alpha",
    ) -> "VectorRemapper":
        """
        Load rotation angles from a grid netCDF file.

        Parameters
        ----------
        grid_file : str
            Path to the netCDF file containing the rotation angle variables.
        grid_shape : tuple of int
            Shape (nlat, nlon) to reshape the loaded variables into.
            Handles both flattened (1-D) and already-2-D storage.
        cos_alpha_name : str
            Name of the cosine variable in the file.
            Default: ``"grid_cos_alpha"``.
        sin_alpha_name : str
            Name of the sine variable in the file.
            Default: ``"grid_sin_alpha"``.
        """
        ds = xr.open_dataset(grid_file)
        try:
            cos_alpha = ds[cos_alpha_name].values.reshape(grid_shape)
            sin_alpha = ds[sin_alpha_name].values.reshape(grid_shape)
        finally:
            ds.close()

        return cls(cos_alpha, sin_alpha)

    def remap_to_geo(
        self,
        u_rot: jnp.ndarray,
        v_rot: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Remap vector from rotated frame to geographical (true east/north).

        Parameters
        ----------
        u_rot, v_rot : array-like
            Vector components in the rotated coordinate frame.
            Any leading batch dimensions (e.g. time, depth) are supported
            via broadcasting.

        Returns
        -------
        u_geo, v_geo : jnp.ndarray
            Vector components in true east/north directions.
        """
        u_geo = u_rot * self.cos_alpha - v_rot * self.sin_alpha
        v_geo = u_rot * self.sin_alpha + v_rot * self.cos_alpha
        return u_geo, v_geo

    def remap_to_rotated(
        self,
        u_geo: jnp.ndarray,
        v_geo: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Remap vector from geographical (true east/north) to rotated frame.

        This is the transpose (inverse) of remap_to_geo.

        Parameters
        ----------
        u_geo, v_geo : array-like
            Vector components in true east/north directions.

        Returns
        -------
        u_rot, v_rot : jnp.ndarray
            Vector components in the rotated coordinate frame.
        """
        u_rot =  u_geo * self.cos_alpha + v_geo * self.sin_alpha
        v_rot = -u_geo * self.sin_alpha + v_geo * self.cos_alpha
        return u_rot, v_rot


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    # --- Option 1: load rotation angles from the grid file ---
    # remapper = VectorRemapper.from_grid_file(
    #     grid_file="grid_data/rotating_gaussian_grid_1.00deg.nc",
    #     grid_shape=(180, 360),
    #     # cos_alpha_name="grid_cos_alpha",  # defaults
    #     # sin_alpha_name="grid_sin_alpha",
    # )

    # --- Option 2: supply rotation angles directly ---
    nlat, nlon = 6, 8
    alpha = jnp.linspace(0, jnp.pi / 4, nlat * nlon).reshape(nlat, nlon)
    remapper = VectorRemapper(
        cos_alpha=jnp.cos(alpha),
        sin_alpha=jnp.sin(alpha),
    )

    # Single snapshot: shape (nlat, nlon)
    u_rot = jnp.ones((nlat, nlon))
    v_rot = jnp.zeros((nlat, nlon))

    u_geo, v_geo = remapper.remap_to_geo(u_rot, v_rot)
    print("remap_to_geo   u_geo[0,0]:", u_geo[0, 0], " v_geo[0,0]:", v_geo[0, 0])

    # Round-trip check
    u_back, v_back = remapper.remap_to_rotated(u_geo, v_geo)
    print("round-trip max error:", float(jnp.max(jnp.abs(u_back - u_rot))))

    # Batched: shape (time, nlat, nlon) — broadcasting just works
    u_rot_t = jnp.ones((10, nlat, nlon))
    v_rot_t = jnp.zeros((10, nlat, nlon))
    u_geo_t, v_geo_t = remapper.remap_to_geo(u_rot_t, v_rot_t)
    print("batched output shape:", u_geo_t.shape)

    # jit-compiled usage
    remap_jit = jax.jit(remapper.remap_to_geo)
    u_geo_jit, v_geo_jit = remap_jit(u_rot, v_rot)
    print("jit output shape:", u_geo_jit.shape)
