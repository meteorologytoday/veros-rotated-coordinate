#!/usr/bin/env python

"""
Adapted from jax-esm's
examples/02_experimental/03_jcm_veros_earth/veros_case_setup.py to run
standalone (no JCM/coupler) on this repo's rotated Gaussian grid. Surface
forcing (wind stress, heat flux, freshwater flux) is prescribed on the
(unrotated) JCM T31 grid in true-east/true-north components, then remapped
onto the rotated ocean grid -- see `ForcingInfo` below.
"""

import veros
print(f"Veros path: {veros.__file__:s}")

from veros import VerosSetup, veros_routine
from veros.core.operators import numpy as npx, update, at
from typing import Sequence

import numpy as np
import jax.numpy as npx
import xarray as xr

from esmf_regrid import ESMFRegridder


class GridInfo:

    scrip_grid_file: str
    landsea_mask_file: str
    landsea_mask_threshold: float
    info: dict

    def __init__(self, scrip_grid_file: str, landsea_mask_file: str, landsea_mask_threshold: float):
        self.scrip_grid_file = scrip_grid_file
        self.landsea_mask_threshold = landsea_mask_threshold
        self.landsea_mask_file = landsea_mask_file
        self.get_grid_info()

    def get_grid_info(self):

        # `scrip_grid_file` is a SCRIP grid file (e.g. RotatedGaussianLatLon.SCRIP.nc)
        # whose pole may be rigidly rotated away from Earth's true pole. It carries
        # both the grid's native (pre-rotation) Gaussian lat-lon axis --
        # `native_lat`/`native_lon` centres and `native_lat_bounds`/`native_lon_bounds`
        # cell faces -- and each cell's true (post-rotation) geographic location
        # (`grid_center_lat`/`grid_center_lon`). The native axis sizes and shapes
        # the Veros grid itself (a rigid rotation preserves the sphere exactly, so
        # the grid's own metric terms -- dxt/dyt, cost, cosu, tantr, area -- are
        # correct when computed in the native frame). The Coriolis parameter is
        # different: it depends on position relative to Earth's actual spin axis,
        # which is fixed in space regardless of the coordinate mesh chosen to
        # discretize the domain, so it must use the true (post-rotation) latitude
        # instead -- see `set_coriolis` below.
        grid_ds = xr.open_dataset(self.scrip_grid_file)
        nlon, nlat = grid_ds["grid_dims"].to_numpy()
        nx, ny = int(nlon), int(nlat)

        native_lat = grid_ds["native_lat"].to_numpy()  # (ny,) centres, non-uniform (Gaussian)
        native_lon = grid_ds["native_lon"].to_numpy()  # (nx,) centres, uniform

        # Veros reconstructs cell-centre vs.yt/vs.xt from vs.dyt/vs.dxt via a
        # leapfrog-style recursion (veros.core.numerics.u_centered_grid) that is
        # only exact when spacing[j] equals the *central* difference of the
        # target centres, (centre[j+1]-centre[j-1])/2 -- not the cell width
        # implied by native_lat_bounds/native_lon_bounds. For the (uniform)
        # longitude axis those coincide, but for the (non-uniform, Gaussian)
        # latitude axis the pole-clamped outermost bounds break that relation,
        # which -- left uncorrected -- introduces up to ~1.5 degrees of error at
        # the two polar-cap rows (verified numerically), enough to meaningfully
        # bias cos/tan there. Build spacing from centre differences instead, and
        # solve for x_origin/y_origin by exactly replicating Veros's own
        # reconstruction (host-side, degrees) so vs.yt/vs.xt come out identical
        # to native_lat/native_lon.
        def _spacing_from_centers(centers):
            d = np.empty_like(centers)
            d[1:-1] = (centers[2:] - centers[:-2]) / 2.0
            d[0] = centers[1] - centers[0]
            d[-1] = centers[-1] - centers[-2]
            return d

        def _calibrate_origin(spacing, first_center, cyclic):
            """Solve for the origin Veros needs so that its own u_centered_grid
            reconstruction from `spacing` places the first interior centre
            exactly at `first_center`. Mirrors
            veros.core.numerics.calc_grid_spacings_kernel's ghost-cell fill and
            u_centered_grid exactly."""
            n = spacing.size
            padded = np.zeros(n + 4)
            padded[2:-2] = spacing
            if cyclic:
                padded[-2:] = padded[2:4]
                padded[:2] = padded[-4:-2]
            else:
                padded[-2:] = padded[-3]
                padded[:2] = padded[2]
            yu = np.zeros(n + 4)
            yu[1:] = np.cumsum(padded[1:])
            yt = np.zeros(n + 4)
            yt[0] = yu[0] - padded[0] * 0.5
            yt[1:] = 2 * yu[:-1]
            alt = np.ones(n + 4)
            alt[::2] = -1
            yt = alt * np.cumsum(alt * yt)
            return first_center - yt[2] + yu[2]

        dyt = _spacing_from_centers(native_lat)                              # (ny,)
        dxt = float(native_lon[1] - native_lon[0])                           # scalar, uniform
        y_origin: float = float(_calibrate_origin(dyt, native_lat[0], cyclic=False))
        x_origin: float = float(_calibrate_origin(np.full(nx, dxt), native_lon[0], cyclic=True))

        # True (post-rotation) geographic latitude of each (j, i) cell -- used for
        # the Coriolis parameter only, not for grid geometry.
        true_lat = grid_ds["grid_center_lat"].to_numpy().reshape(ny, nx)
        true_lat_xy = true_lat.transpose()  # (j, i) -> (xt, yt)

        # Grid rotation angle: cos/sin of the angle (radians, positive
        # anticlockwise) from true east to this grid's local x-axis, needed
        # to rotate true-east/true-north vector forcing (e.g. wind) into the
        # grid's own local frame -- see `ForcingInfo` below. Flat
        # (grid_size,), SCRIP lon-fastest; reshape with order="F" to
        # (n_lon, n_lat).
        cos_angle = grid_ds["grid_cos_angle"].to_numpy().reshape((nx, ny), order="F")
        sin_angle = grid_ds["grid_sin_angle"].to_numpy().reshape((nx, ny), order="F")

        # TODO: real bathymetry is still needed here -- this only gives a
        # binary land/sea mask, so every ocean column gets the same flat
        # bottom depth (kbot) rather than actual varying ocean depth.
        # ERA5-derived fractional land-sea mask on the same SCRIP grid; convention: 1 = land.
        mask_ds = xr.open_dataset(self.landsea_mask_file)
        lsm = mask_ds["lsm"]
        if "valid_time" in lsm.dims:
            lsm = lsm.isel(valid_time=0, drop=True)
        lsm = lsm.to_numpy().reshape(ny, nx)
        is_land = lsm >= self.landsea_mask_threshold
        landsea_mask = (1 - is_land.astype(int)).transpose()  # -> ocean=1/land=0, (xt, yt); Veros kbot wants 0 = land

        self.nx = nx
        self.ny = ny
        self.dyt = dyt
        self.dxt = dxt
        self.y_origin = y_origin
        self.x_origin = x_origin
        self.true_lat_xy = true_lat_xy
        self.landsea_mask = landsea_mask
        self.cos_angle = cos_angle
        self.sin_angle = sin_angle


class ForcingInfo:
    """Idealized surface forcing (wind stress, heat flux, freshwater flux),
    prescribed on the *unrotated* JCM T31 grid in true-east/true-north
    components -- as it would be handed off by an atmosphere model -- then
    remapped onto the (rotated) ocean grid with a conservative ESMF weight
    file.

    The forcing shapes are schematic latitude-only profiles (trade winds /
    mid-latitude westerlies, tropical warming / polar cooling, an ITCZ-like
    freshwater pattern), not observational data -- good enough to drive a
    recognizable circulation for this runnable example.
    """

    def __init__(
        self,
        jcm_grid_file: str,
        a2o_weight_file: str,
        grid_info: GridInfo,
        rho_0: float = 1024.0,
    ):
        self.jcm_grid_file = jcm_grid_file
        self.a2o_weight_file = a2o_weight_file
        self.grid_info = grid_info
        self.rho_0 = rho_0
        self._compute()

    def _compute(self):
        # JCM's grid is unrotated, so its own latitude *is* the true
        # geographic latitude -- unlike the ocean's rotated grid, there is
        # no native/true distinction to make here.
        jcm_ds = xr.open_dataset(self.jcm_grid_file)
        nlon_a, nlat_a = (int(n) for n in jcm_ds["grid_dims"].to_numpy())
        lat_a = jcm_ds["grid_center_lat"].to_numpy().reshape((nlon_a, nlat_a), order="F")
        lat_rad = np.deg2rad(lat_a)
        weight = np.cos(lat_rad)  # ~area weight (lon-lat grid), used to zero global means below

        # 10 m wind (true-east/true-north, m/s): easterly trade winds near
        # the equator, mid-latitude westerlies. No meridional component.
        U0 = 6.0  # m/s
        u0 = -U0 * np.cos(3.0 * lat_rad)
        v0 = np.zeros_like(u0)

        # Net surface heat flux into the ocean (W/m^2): warm in the tropics,
        # cool at the poles. The area-weighted global mean is removed so it
        # integrates to ~0 (avoids a long-term energy drift under a purely
        # prescribed, non-restoring flux).
        Q0 = 150.0  # W/m^2
        qnet_raw = Q0 * np.cos(lat_rad)
        qnet = qnet_raw - np.sum(qnet_raw * weight) / np.sum(weight)

        # Freshwater flux, precip minus evaporation (m/s): ITCZ-like excess
        # precipitation near the equator, subtropical evaporation belts.
        # Global mean removed for the same reason as qnet.
        P0 = 3.0e-8  # m/s (~1 m/yr)
        pme_raw = P0 * (np.exp(-(lat_a / 8.0) ** 2) - 0.5 * np.exp(-((np.abs(lat_a) - 25.0) / 10.0) ** 2))
        pme = pme_raw - np.sum(pme_raw * weight) / np.sum(weight)

        # ----- remap atmosphere (JCM T31) -> ocean (rotated Gaussian) grid -----
        a2o = ESMFRegridder(self.a2o_weight_file)
        u0_o = a2o(u0)
        v0_o = a2o(v0)
        qnet_o = a2o(qnet)
        pme_o = a2o(pme)

        # `a2o` interpolates u0/v0 component-wise, which is fine since JCM's
        # grid is unrotated (u0/v0 are true-east/true-north everywhere on
        # it). The result is then rotated into the ocean grid's local
        # (rotated) frame using its per-cell grid_cos_angle/grid_sin_angle
        # -- same convention as
        # jax-esm/.../03_jcm_veros_earth/model_setup.py's `interaction`.
        cos_angle = self.grid_info.cos_angle
        sin_angle = self.grid_info.sin_angle
        wind_x = cos_angle * u0_o + sin_angle * v0_o
        wind_y = -sin_angle * u0_o + cos_angle * v0_o

        drag_coefficient = 1e-3  # dimensionless
        air_density = 1.22  # kg/m^3
        wind_speed = np.sqrt(wind_x ** 2 + wind_y ** 2)
        self.surface_taux = drag_coefficient * air_density * wind_speed * wind_x  # N/m^2, ocean grid's local frame
        self.surface_tauy = drag_coefficient * air_density * wind_speed * wind_y  # N/m^2, ocean grid's local frame

        # W/m^2 -> m degC/s (Veros' `forc_temp_surface` units): divide by
        # rho_0 * cp_0. cp_0 is the standard oceanic specific heat capacity
        # value used e.g. by veros.setups.global_4deg.
        cp_0 = 3991.86795711963  # J/(kg K)
        self.forc_temp_surface = qnet_o / (self.rho_0 * cp_0)  # m degC/s

        self.pme = pme_o  # m/s; forc_salt_surface is computed per-step from this (needs live vs.salt)


def generateVerosSetup(
    scrip_grid_file: str,
    landsea_mask_file: str,
    jcm_grid_file: str,
    a2o_weight_file: str,
    landsea_mask_threshold: float = 0.5,
    ddz: Sequence[float] = [50.0, 70.0, 100.0, 140.0, 190.0, 240.0, 290.0, 340.0, 390.0, 440.0, 490.0, 540.0, 590.0, 640.0, 690.0],
    dt_mom: float = 1800.0,
    dt_tracer: float = 1800.0,
    runlen: float = 86400.0 * 365,
    cold_start_ocean_temperature_reference_K: float = 15.0,
):

    grid_info = GridInfo(
        scrip_grid_file=scrip_grid_file,
        landsea_mask_file=landsea_mask_file,
        landsea_mask_threshold=landsea_mask_threshold,
    )
    forcing_info = ForcingInfo(
        jcm_grid_file=jcm_grid_file,
        a2o_weight_file=a2o_weight_file,
        grid_info=grid_info,
    )

    ddz = npx.array(ddz)
    nz = len(ddz)

    class VerosCaseSetup(VerosSetup):
        """A standalone (uncoupled) Veros setup on a rotated Gaussian
        lat-lon grid whose pole is displaced off Earth's true pole (e.g.
        onto Greenland/Antarctica) to avoid the numerical singularity at a
        true geographic pole. Grid geometry (dxt/dyt/x_origin/y_origin) is
        computed in the grid's own *native* (pre-rotation) frame, while the
        Coriolis parameter uses each cell's *true* (post-rotation)
        geographic latitude -- see `GridInfo.get_grid_info` above.

        Surface forcing (wind stress, heat flux, freshwater flux) is
        prescribed on the unrotated JCM T31 grid in true-east/true-north
        components and remapped onto this grid -- see `ForcingInfo` above.
        There is no atmospheric coupling; the forcing is static.
        """

        @veros_routine
        def set_parameter(self, state):
            settings = state.settings
            settings.identifier = "output_veros"
            settings.description = "Rotated Gaussian grid Veros setup, forced from the JCM T31 grid"

            # NOTE: leaving this False (linear free surface, no streamfunction
            # solve) makes Veros register a zero-size "isle" dimension, which
            # crashes h5netcdf's HDF5 dimension-scale creation the moment any
            # diagnostic (e.g. "averages") tries to write an output file --
            # so keep the streamfunction solver on despite the extra cost.
            settings.enable_streamfunction = True
            settings.enable_nan_checks = False

            settings.nx, settings.ny, settings.nz = grid_info.nx, grid_info.ny, nz
            settings.dt_mom = dt_mom
            settings.dt_tracer = dt_tracer
            settings.runlen = runlen

            settings.x_origin = grid_info.x_origin
            settings.y_origin = grid_info.y_origin

            settings.coord_degree = True
            settings.enable_cyclic_x = True

            settings.enable_neutral_diffusion = True
            settings.K_iso_0 = 1000.0
            settings.K_iso_steep = 500.0
            settings.iso_dslope = 0.005
            settings.iso_slopec = 0.01
            settings.enable_skew_diffusion = True

            settings.enable_hor_friction = True
            settings.A_h = ((grid_info.dxt + grid_info.dyt.mean()) / 2 * settings.degtom) ** 3 * 2e-11
            settings.enable_hor_friction_cos_scaling = True
            settings.hor_friction_cosPower = 1

            settings.enable_bottom_friction = True
            settings.r_bot = 1e-5

            settings.enable_implicit_vert_friction = True

            settings.enable_tke = True
            settings.c_k = 0.1
            settings.c_eps = 0.7
            settings.alpha_tke = 30.0
            settings.mxl_min = 1e-8
            settings.tke_mxl_choice = 2
            settings.kappaM_min = 2e-4
            settings.kappaH_min = 2e-5
            settings.enable_kappaH_profile = True

            settings.K_gm_0 = 1000.0
            settings.enable_eke = True
            settings.eke_k_max = 1e4
            settings.eke_c_k = 0.4
            settings.eke_c_eps = 0.5
            settings.eke_cross = 2.0
            settings.eke_crhin = 1.0
            settings.eke_lmin = 100.0
            settings.enable_eke_superbee_advection = True
            settings.enable_eke_isopycnal_diffusion = True

            settings.enable_idemix = False

            settings.eq_of_state_type = 1

        @veros_routine
        def set_grid(self, state):
            vs = state.variables
            # Ghost cells ([:2]/[-2:]) are filled automatically from the
            # interior by calc_grid_spacings_kernel -- only the interior
            # needs to be set here (see e.g. global_flexible's set_grid).
            vs.dxt = update(vs.dxt, at[2:-2], grid_info.dxt)
            vs.dyt = update(vs.dyt, at[2:-2], grid_info.dyt)  # per-row array: native Gaussian latitude spacing is non-uniform
            vs.dzt = update(vs.dzt, at[...], ddz[::-1])  # ocean grid starts from below

        @veros_routine
        def set_coriolis(self, state):
            vs = state.variables
            settings = state.settings
            # Coriolis depends on position relative to Earth's true spin axis,
            # not the grid's own (rotated) latitude -- see note in
            # GridInfo.get_grid_info above.
            vs.coriolis_t = update(
                vs.coriolis_t, at[2:-2, 2:-2], 2 * settings.omega * npx.sin(grid_info.true_lat_xy / 180.0 * settings.pi)
            )

        @veros_routine
        def set_topography(self, state):
            vs = state.variables
            x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")

            vs.kbot = npx.zeros_like(x)
            vs.kbot = update(
                vs.kbot,
                at[2:-2, 2:-2],
                grid_info.landsea_mask,
            )

        @veros_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            # initial conditions
            vs.temp = update(
                vs.temp,
                at[...],
                ((1 - vs.zt[None, None, :] / vs.zw[0]) * cold_start_ocean_temperature_reference_K * vs.maskT)[..., None],
            )
            vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

            # Wind stress: static, prescribed on the JCM T31 grid and
            # remapped onto this grid -- see `ForcingInfo`.
            vs.surface_taux = update(
                vs.surface_taux, at[2:-2, 2:-2], forcing_info.surface_taux * vs.maskU[2:-2, 2:-2, -1]
            )
            vs.surface_tauy = update(
                vs.surface_tauy, at[2:-2, 2:-2], forcing_info.surface_tauy * vs.maskV[2:-2, 2:-2, -1]
            )

            if settings.enable_tke:
                vs.forc_tke_surface = update(
                    vs.forc_tke_surface,
                    at[2:-2, 2:-2],
                    npx.sqrt(
                        (0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0) ** 2
                        + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0) ** 2
                    )
                    ** (1.5),
                )

            if settings.enable_idemix:
                vs.forc_iw_bottom = 1e-6 * vs.maskW[:, :, -1]
                vs.forc_iw_surface = 1e-7 * vs.maskW[:, :, -1]

        @veros_routine
        def set_forcing(self, state):
            vs = state.variables

            # Heat flux: static, prescribed on the JCM T31 grid and remapped
            # onto this grid -- see `ForcingInfo`.
            vs.forc_temp_surface = update(
                vs.forc_temp_surface, at[2:-2, 2:-2], forcing_info.forc_temp_surface * vs.maskT[2:-2, 2:-2, -1]
            )

            # Freshwater flux -> virtual salt flux. Prescribed as a
            # precip-minus-evaporation rate (m/s, `ForcingInfo.pme`) on the
            # JCM T31 grid and remapped onto this grid; converted to a salt
            # flux using the live surface salinity each step (standard
            # "virtual salting" -- P/E carries no salt itself, it only
            # dilutes/concentrates what's already there).
            vs.forc_salt_surface = update(
                vs.forc_salt_surface,
                at[2:-2, 2:-2],
                -vs.salt[2:-2, 2:-2, -1, vs.tau] * forcing_info.pme * vs.maskT[2:-2, 2:-2, -1],
            )

        @veros_routine
        def set_diagnostics(self, state):
            # NOTE: any diagnostic that writes a netCDF output file crashes
            # here -- `h5netcdf`/HDF5 fails with "Unspecified error in
            # H5DSis_scale" the moment *any* file's first dimension is
            # created, but only *after* the streamfunction solver has run
            # its per-island ILU solves during `.setup()`. Verified this
            # isn't specific to Veros's own output file (an unrelated,
            # freshly opened h5netcdf file fails identically at that point
            # in the process), so it's an environment-level HDF5/JAX
            # interaction, not something under this setup's control.
            # jax-esm's own `03_jcm_veros_earth/veros_case_setup.py` hits
            # the same wall and works around it the same way: disable
            # diagnostics output entirely.
            diagnostics = state.diagnostics
            diagnostics.clear()

        @veros_routine
        def after_timestep(self, state):
            pass

    return VerosCaseSetup
