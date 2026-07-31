"""
Coarsen the high-resolution ERA5 land-sea mask onto the JCM grid and onto
the rotated-coordinate (rotating Gaussian) grid, by simple area averaging:
every ERA5 0.25deg pixel center is binned into whichever target grid cell
contains it, and the target cell's land fraction is the mean of the ERA5
land-fraction values ("lsm") that fall inside it.

For the JCM grid this is a plain lon/lat box binning.

For the rotated grid, each cell is a quadrilateral in true (geographic)
lon/lat -- not an axis-aligned box -- because the grid was built by
rotating a plain lat/lon grid (see rotating_coordinate_generation.py).
Rather than doing point-in-quadrilateral tests, each ERA5 point is rotated
*backward* by the same rotation used to build the grid, which maps it
exactly onto the pre-rotation plain lat/lon grid. Binning there with plain
box lookup is equivalent to binning into the true rotated cells, because
rotation is a continuous, distance-preserving bijection: it maps the
pre-rotation cell region onto the rotated cell region exactly.

No ESMF / SCRIP files are involved.
"""
import numpy as np
import xarray as xr
from pathlib import Path

from era5_grid import load_era5_landsea_mask
from JCMGrid import generate_JCMGrid
from rotating_coordinate_generation import my_rotate_in_spherical

ERA5_FILE = "ERA5_landsea_mask.nc"
OUTPUT_DIR = Path("landsea_mask_data")

JCM_RESOLUTION = 31

RG_RESOLUTIONS_DEG = [1, 2, 3, 4]
RG_ROTATION_ALONG_LONGITUDE_DEGREE = -42 + 90
RG_ROTATION_DEGREE = 12.0

LAND_THRESHOLD = 0.5


def bin_average(lon_rad, lat_rad, values, lon_bounds_rad, lat_bounds_rad):
    """
    Average `values` sampled at points (lon_rad, lat_rad) into the regular
    grid cells defined by ascending lon_bounds_rad (periodic on 2*pi) and
    ascending lat_bounds_rad. Returns (mean, count) of shape
    (n_lat, n_lon), with NaN in `mean` where a cell received no points.
    """
    n_lon = len(lon_bounds_rad) - 1
    n_lat = len(lat_bounds_rad) - 1

    two_pi = 2 * np.pi
    lon0 = lon_bounds_rad[0]
    lon = np.mod(lon_rad - lon0, two_pi) + lon0

    i_idx = np.searchsorted(lon_bounds_rad, lon, side="right") - 1
    j_idx = np.searchsorted(lat_bounds_rad, lat_rad, side="right") - 1

    # Guard against float roundoff landing exactly on the outer lon edge
    i_idx = np.clip(i_idx, 0, n_lon - 1)
    valid = (j_idx >= 0) & (j_idx < n_lat)

    flat_idx = j_idx[valid] * n_lon + i_idx[valid]
    sums = np.bincount(flat_idx, weights=values[valid], minlength=n_lat * n_lon)
    counts = np.bincount(flat_idx, minlength=n_lat * n_lon)

    mean = np.full(n_lat * n_lon, np.nan)
    nonzero = counts > 0
    mean[nonzero] = sums[nonzero] / counts[nonzero]

    return mean.reshape(n_lat, n_lon), counts.reshape(n_lat, n_lon)


def nearest_era5_value(lon_query_rad, lat_query_rad, lon_c_deg, lat_c_deg, lsm_data):
    """
    Nearest-neighbor lookup on the regular ERA5 grid (lsm_data has shape
    (nlon, nlat)). Used only to fill the rare target cell that receives
    zero ERA5 samples (this happens for a handful of 1x1deg cells right
    at the poles, where a one-degree-wide cell covers a physically tiny
    area and can miss every 0.25deg pixel center by parity).
    """
    lon_query_deg = np.rad2deg(lon_query_rad) % 360
    lat_query_deg = np.rad2deg(lat_query_rad)
    dlon = lon_c_deg[1] - lon_c_deg[0]
    dlat = lat_c_deg[1] - lat_c_deg[0]
    i_idx = np.clip(np.round((lon_query_deg - lon_c_deg[0]) / dlon).astype(int), 0, len(lon_c_deg) - 1)
    j_idx = np.clip(np.round((lat_query_deg - lat_c_deg[0]) / dlat).astype(int), 0, len(lat_c_deg) - 1)
    return lsm_data[i_idx, j_idx]


def fill_empty_cells(land_fraction, counts, query_lon_rad, query_lat_rad, lon_c_deg, lat_c_deg, lsm_data, label):
    empty = counts == 0
    n_empty = int(np.sum(empty))
    if n_empty == 0:
        return land_fraction
    print(f"  {n_empty} {label} cells got no ERA5 samples (near-pole cells); filling via nearest-neighbor")
    fallback = nearest_era5_value(query_lon_rad[empty], query_lat_rad[empty], lon_c_deg, lat_c_deg, lsm_data)
    land_fraction = land_fraction.copy()
    land_fraction[empty] = fallback
    return land_fraction


def bounds_1d_deg(bounds_rad):
    """CF-style 2-value bounds (n, 2) in degrees, for a plain rectilinear axis."""
    return np.rad2deg(np.stack([bounds_rad[:-1], bounds_rad[1:]], axis=-1))


def corner_bounds_deg(lon_bounds_rad, lat_bounds_rad):
    """
    Build the 4-corner quadrilateral bounds of every cell in a regular
    lon/lat grid defined by ascending lon_bounds_rad (size n1+1) and
    lat_bounds_rad (size n0+1). Returns (lon_corners_deg, lat_corners_deg),
    each of shape (n0, n1, 4), corners ordered
    (lower-left, lower-right, upper-right, upper-left) -- i.e. the same
    convention as JCMGrid.py / rotating_coordinate_generation.py.
    """
    lon_b0 = lon_bounds_rad[:-1]
    lon_b1 = lon_bounds_rad[1:]
    lat_b0 = lat_bounds_rad[:-1]
    lat_b1 = lat_bounds_rad[1:]

    n0 = len(lat_b0)
    n1 = len(lon_b0)

    lon_corners = np.stack([
        np.broadcast_to(lon_b0[None, :], (n0, n1)),
        np.broadcast_to(lon_b1[None, :], (n0, n1)),
        np.broadcast_to(lon_b1[None, :], (n0, n1)),
        np.broadcast_to(lon_b0[None, :], (n0, n1)),
    ], axis=-1)
    lat_corners = np.stack([
        np.broadcast_to(lat_b0[:, None], (n0, n1)),
        np.broadcast_to(lat_b0[:, None], (n0, n1)),
        np.broadcast_to(lat_b1[:, None], (n0, n1)),
        np.broadcast_to(lat_b1[:, None], (n0, n1)),
    ], axis=-1)

    return np.rad2deg(lon_corners), np.rad2deg(lat_corners)


def jcm_bounds_rad(nlon, nlat):
    lon_bounds_deg = np.linspace(0, 360, nlon + 1)
    lat_bounds_deg = np.linspace(-90, 90, nlat + 1)
    return np.deg2rad(lon_bounds_deg), np.deg2rad(lat_bounds_deg)


def rg_prerotation_bounds_rad(resolution_deg):
    nlat = int(round(180 / resolution_deg))
    nlon = int(round(360 / resolution_deg))
    lat_bounds_deg = np.linspace(-90, 90, nlat + 1)
    lon_bounds_deg = np.linspace(0, 360, nlon + 1)
    return np.deg2rad(lon_bounds_deg), np.deg2rad(lat_bounds_deg)


def save_mask(output_file, dim0_deg, dim1_deg, land_fraction, counts, dim_names, bounds_specs=None, extra_coords=None):
    """
    dim0_deg, dim1_deg: 1-D coordinate values (degrees) for dim_names[0] and
    dim_names[1] respectively -- caller controls axis order (e.g. lon,lat
    for JCM; lat,lon for the rotated grid).

    bounds_specs: list of (coord_name, bnds_var_name, bnds_dims, bnds_data, bnds_attrs).
    For each entry, ds[coord_name].attrs["bounds"] is set to bnds_var_name (CF-style
    linkage), as long as coord_name is one of this dataset's coordinates.
    """
    binary_mask = np.where(np.isnan(land_fraction), 0, land_fraction >= LAND_THRESHOLD).astype(np.int8)

    data_vars = dict(
        land_fraction=(dim_names, land_fraction, {"long_name": "ERA5-derived land fraction", "units": "1"}),
        land_sea_mask=(dim_names, binary_mask, {"long_name": "binary land-sea mask (1=land, 0=sea)", "units": "1", "threshold": LAND_THRESHOLD}),
        sample_count=(dim_names, counts, {"long_name": "number of ERA5 0.25deg pixels averaged into this cell"}),
    )

    coords = {}
    if dim0_deg.ndim == 1:
        coords[dim_names[0]] = dim0_deg
        coords[dim_names[1]] = dim1_deg
    if extra_coords:
        coords.update(extra_coords)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    for coord_name, bnds_name, bnds_dims, bnds_data, bnds_attrs in (bounds_specs or []):
        ds[bnds_name] = (bnds_dims, bnds_data, bnds_attrs)
        if coord_name in ds.coords:
            ds[coord_name].attrs["bounds"] = bnds_name

    ds.attrs["source"] = ERA5_FILE
    ds.attrs["method"] = "simple area-averaging of ERA5 pixel centers"
    ds.to_netcdf(output_file)
    print(f"Wrote {output_file}")


def process_jcm(lon_rad, lat_rad, values, lon_c_deg, lat_c_deg, lsm_data):
    grid = generate_JCMGrid(JCM_RESOLUTION)
    nlon, nlat = grid.binary_mask.shape

    lon_bounds_rad, lat_bounds_rad = jcm_bounds_rad(nlon, nlat)
    land_fraction, counts = bin_average(lon_rad, lat_rad, values, lon_bounds_rad, lat_bounds_rad)
    # bin_average always returns (n_lat, n_lon); transpose to JCMGrid.py's
    # native (lon, lat) axis order.
    land_fraction = land_fraction.T
    counts = counts.T

    lat_centers_deg = np.rad2deg((lat_bounds_rad[:-1] + lat_bounds_rad[1:]) / 2)
    lon_centers_deg = np.rad2deg((lon_bounds_rad[:-1] + lon_bounds_rad[1:]) / 2)
    LON_Q, LAT_Q = np.meshgrid(np.deg2rad(lon_centers_deg), np.deg2rad(lat_centers_deg), indexing="ij")

    land_fraction = fill_empty_cells(
        land_fraction, counts, LON_Q, LAT_Q, lon_c_deg, lat_c_deg, lsm_data, f"JCM T{JCM_RESOLUTION}"
    )

    bounds_specs = [
        ("lon", "lon_bnds", ["lon", "bnds"], bounds_1d_deg(lon_bounds_rad), {"units": "degrees_east"}),
        ("lat", "lat_bnds", ["lat", "bnds"], bounds_1d_deg(lat_bounds_rad), {"units": "degrees_north"}),
    ]

    output_file = OUTPUT_DIR / f"landsea_mask_JCM_T{JCM_RESOLUTION}.nc"
    save_mask(output_file, lon_centers_deg, lat_centers_deg, land_fraction, counts, ["lon", "lat"], bounds_specs=bounds_specs)


def process_rotating_grid(lon_rad, lat_rad, values, resolution_deg, lon_c_deg, lat_c_deg, lsm_data):
    # Map every ERA5 point back to the pre-rotation plain lat/lon grid.
    pts = np.stack([np.ones_like(lon_rad), lon_rad, lat_rad], axis=0)
    pts_pre = my_rotate_in_spherical(pts, RG_ROTATION_ALONG_LONGITUDE_DEGREE, -RG_ROTATION_DEGREE)
    lon_pre, lat_pre = pts_pre[1], pts_pre[2]

    lon_bounds_rad, lat_bounds_rad = rg_prerotation_bounds_rad(resolution_deg)
    land_fraction, counts = bin_average(lon_pre, lat_pre, values, lon_bounds_rad, lat_bounds_rad)
    # bin_average always returns (n_lat, n_lon); transpose to lon,lat order.
    land_fraction = land_fraction.T
    counts = counts.T

    # True geographic lon/lat of each rotated cell center, for reference/plotting
    # and as the query point for nearest-neighbor fallback on empty cells.
    lat_pre_centers = (lat_bounds_rad[:-1] + lat_bounds_rad[1:]) / 2
    lon_pre_centers = (lon_bounds_rad[:-1] + lon_bounds_rad[1:]) / 2
    LON_PRE, LAT_PRE = np.meshgrid(lon_pre_centers, lat_pre_centers, indexing="ij")
    centers_pre = np.stack([np.ones(LAT_PRE.shape), LON_PRE, LAT_PRE], axis=0)
    centers_true = my_rotate_in_spherical(centers_pre, RG_ROTATION_ALONG_LONGITUDE_DEGREE, RG_ROTATION_DEGREE)
    true_lon_rad = centers_true[1]
    true_lat_rad = centers_true[2]
    true_lon_deg = np.rad2deg(true_lon_rad) % 360
    true_lat_deg = np.rad2deg(true_lat_rad)

    land_fraction = fill_empty_cells(
        land_fraction, counts, true_lon_rad, true_lat_rad, lon_c_deg, lat_c_deg, lsm_data, f"RG {resolution_deg}deg"
    )

    # Un-rotated (pre-rotation) cell bounds: plain per-axis CF-style bounds,
    # since the "j"/"i" grid is rectilinear before rotation.
    lat_bnds_prerotation = bounds_1d_deg(lat_bounds_rad)
    lon_bnds_prerotation = bounds_1d_deg(lon_bounds_rad)

    # True (rotated) cell bounds: each cell is a quadrilateral in true
    # geographic lon/lat, so it needs all 4 corners, obtained by forward-
    # rotating the pre-rotation corners.
    lon_corner_pre_deg, lat_corner_pre_deg = corner_bounds_deg(lon_bounds_rad, lat_bounds_rad)
    # corner_bounds_deg always returns (n_lat, n_lon, 4); transpose to lon,lat order.
    lon_corner_pre_deg = lon_corner_pre_deg.transpose(1, 0, 2)
    lat_corner_pre_deg = lat_corner_pre_deg.transpose(1, 0, 2)
    corners_pre = np.stack([
        np.ones_like(lon_corner_pre_deg),
        np.deg2rad(lon_corner_pre_deg),
        np.deg2rad(lat_corner_pre_deg),
    ], axis=0)
    corners_true = my_rotate_in_spherical(corners_pre, RG_ROTATION_ALONG_LONGITUDE_DEGREE, RG_ROTATION_DEGREE)
    true_lon_bnds_deg = np.rad2deg(corners_true[1]) % 360
    true_lat_bnds_deg = np.rad2deg(corners_true[2])

    bounds_specs = [
        ("i", "lon_bnds", ["i", "bnds"], lon_bnds_prerotation, {"units": "degrees_east", "long_name": "pre-rotation (un-rotated) longitude bounds"}),
        ("j", "lat_bnds", ["j", "bnds"], lat_bnds_prerotation, {"units": "degrees_north", "long_name": "pre-rotation (un-rotated) latitude bounds"}),
        ("true_lon", "true_lon_bnds", ["i", "j", "grid_corners"], true_lon_bnds_deg, {"units": "degrees_east", "long_name": "true (rotated) longitude of cell corners"}),
        ("true_lat", "true_lat_bnds", ["i", "j", "grid_corners"], true_lat_bnds_deg, {"units": "degrees_north", "long_name": "true (rotated) latitude of cell corners"}),
    ]

    output_file = OUTPUT_DIR / f"landsea_mask_RG_{resolution_deg:.2f}deg.nc"
    save_mask(
        output_file,
        np.rad2deg(lon_pre_centers), np.rad2deg(lat_pre_centers),
        land_fraction, counts, ["i", "j"],
        bounds_specs=bounds_specs,
        extra_coords={
            "true_lon": (["i", "j"], true_lon_deg, {"units": "degrees_east"}),
            "true_lat": (["i", "j"], true_lat_deg, {"units": "degrees_north"}),
        },
    )


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Loading {ERA5_FILE} ...")
    lon_c_deg, lat_c_deg, lsm_data = load_era5_landsea_mask(ERA5_FILE)  # lsm_data shape (nlon, nlat)

    lon_rad_1d = np.deg2rad(lon_c_deg)
    lat_rad_1d = np.deg2rad(lat_c_deg)
    LON, LAT = np.meshgrid(lon_rad_1d, lat_rad_1d, indexing="ij")  # shape (nlon, nlat)

    lon_rad = LON.ravel()
    lat_rad = LAT.ravel()
    values = lsm_data.ravel()

    print(f"Processing JCM T{JCM_RESOLUTION} ...")
    process_jcm(lon_rad, lat_rad, values, lon_c_deg, lat_c_deg, lsm_data)

    for resolution_deg in RG_RESOLUTIONS_DEG:
        print(f"Processing rotated grid {resolution_deg}deg ...")
        process_rotating_grid(lon_rad, lat_rad, values, resolution_deg, lon_c_deg, lat_c_deg, lsm_data)

