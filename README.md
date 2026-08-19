# Goal

This repo runs [Veros](https://veros.readthedocs.io) on a Gaussian
lat-lon grid whose pole has been rigidly rotated off Earth's true pole
(onto Greenland/Antarctica) to avoid the numerical singularity a true
geographic pole causes. The rotation is purely a change of coordinate
mesh: grid geometry (`dxt`/`dyt`/`x_origin`/`y_origin`, computed in the
grid's own *native*, pre-rotation frame) comes out identical to an
unrotated Gaussian grid, while the Coriolis parameter is computed from
each cell's *true* (post-rotation) geographic latitude.

The grid, land-sea mask, and ESMF regridding weight files under `data/`
were produced by an earlier, now-removed grid-generation pipeline; this
repo currently contains only the runnable Veros example that consumes
them.

## Layout

- `data/` -- pre-built SCRIP grid files, fractional land-sea masks, and
  ESMF regridding weights. The files actually used by the example are:
  - `RotatedGaussianLatLon.SCRIP.nc` -- the rotated ocean grid (carries
    both its native pre-rotation lat/lon axis and each cell's true
    post-rotation lat/lon and rotation angle).
  - `landsea_mask_fraction_RotatedGaussianLatLon.nc` -- ERA5-derived
    fractional land-sea mask on that grid.
  - `JCM_T31.SCRIP.nc` -- the unrotated JCM T31 atmosphere grid that
    surface forcing is prescribed on.
  - `weight_algo-conserve_JCM_T31_to_RotatedGaussianLatLon.nc` -- ESMF
    conservative regridding weights from the JCM grid onto the ocean grid.

  (`DisplacedPoleGrid.*`, `landsea_mask_fraction_JCM_T31.nc`,
  `terrain_JCM_T31.nc`, and the `*_DisplacedPoleGrid_*`/`*_to_JCM_T31.nc`
  weight files are left over from earlier experiments and aren't read by
  anything here.)
- `src/veros_case_setup.py` -- `generateVerosSetup(...)`, which builds a
  `VerosSetup` for the rotated grid: grid geometry and land-sea mask via
  `GridInfo`, idealized surface forcing via `ForcingInfo`.
- `src/esmf_regrid.py` -- a small, dependency-free `ESMFRegridder` that
  applies a precomputed ESMF weight file (sparse matrix multiply), so
  this repo doesn't need the `jem` package installed just to regrid.
- `src/run_veros.py` -- the entry point: builds the setup from the files
  in `data/`, runs `.setup()` and a stepping loop, and writes periodic
  snapshot output.

## Forcing

There's no atmospheric coupling here -- surface wind stress, heat flux,
and freshwater flux are idealized, latitude-only, schematic profiles
(trade winds / mid-latitude westerlies, tropical warming / polar
cooling, an ITCZ-like precipitation pattern). They're prescribed on the
*unrotated JCM T31 grid* in true-east/true-north components, then:

1. remapped onto the rotated ocean grid with the conservative ESMF
   weights (`ESMFRegridder`);
2. the wind vector is rotated into the ocean grid's local frame using
   its per-cell `grid_cos_angle`/`grid_sin_angle` (from
   `RotatedGaussianLatLon.SCRIP.nc`);
3. converted to wind stress via a simple bulk drag formula, and to
   `forc_temp_surface`/`forc_salt_surface` via a fixed `cp_0` and
   virtual salting.

See `ForcingInfo` in `src/veros_case_setup.py` for the exact formulas.

## Running

```
python3 src/run_veros.py
```

This builds the model on the 96x48x15 rotated grid and integrates for
10 simulated days (`total_time` in `run_veros.py`), writing one
`output_veros.snapshot.<day>.nc` file per simulated day. Each file has
a single-record, unlimited `time` dimension, so the daily files
concatenate cleanly:

```python
import xarray as xr
ds = xr.open_mfdataset(
    "output_veros.snapshot.*.nc", concat_dim="time", combine="nested"
)
```

### Performance

On the `numpy` runtime backend (set in `run_veros.py`), one 1800s time
step on this grid takes roughly 70-90s, so the full 10-day/480-step run
takes on the order of 10-11 hours. The `jax` runtime backend hasn't been
tried here yet and would likely be faster.

### Known environment quirks (and their workarounds)

- **`enable_streamfunction = False` breaks output.** With the linear
  free-surface solver, Veros registers a zero-size `isle` HDF5
  dimension, which crashes `h5netcdf`'s dimension-scale creation the
  moment any diagnostic writes a file. `veros_case_setup.py` therefore
  keeps `enable_streamfunction = True` (elliptic solver + a per-island
  ILU solve during `.setup()`).
- **Any `h5netcdf`/`h5py`-based netCDF write crashes once that ILU
  solve has run** -- `RuntimeError: Unspecified error in H5DSis_scale`.
  This reproduces even for a brand-new, unrelated file opened fresh, so
  it looks like an environment-level HDF5/JAX interaction rather than
  something specific to this setup (jax-esm's own coupled Veros example
  hits the same wall and works around it the same way). `set_diagnostics`
  therefore calls `diagnostics.clear()`, and `run_veros.py` writes its
  own snapshot output directly with the separate `netCDF4` package
  instead, which is unaffected.

## To-do

- Real bathymetry: `set_topography` currently only uses a binary
  land-sea mask, so every ocean column gets the same flat bottom depth
  rather than actual varying ocean depth.
- Try the `jax` runtime backend for speed.
